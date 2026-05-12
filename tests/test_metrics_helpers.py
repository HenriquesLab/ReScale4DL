from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tifffile import imwrite

from rescale4dl.metrics import fov_estimates, metrics, metrics3d, properties


@pytest.mark.parametrize("module", [properties, metrics])
def test_2d_pixel_coverage_and_diameter(module):
    image = np.zeros((5, 5), dtype=bool)
    image[1:4, 1:4] = True

    assert module.pixel_coverage_percent(image) == pytest.approx(100 / 9)
    diameters = module.object_diameter(image)
    assert len(diameters) == 4
    assert all(value > 0 for value in diameters)


def test_metrics_bbox_padding_and_time_helpers():
    assert metrics.bbox_points_for_crop([2, 2, 4, 4], xmax=10, ymax=10) == (0, 0, 6, 6)
    assert metrics.bbox_points_for_crop([0, 0, 1, 1], xmax=3, ymax=3) == (0, 0, 2, 2)

    padded = metrics.pad_br_with_zeroes(np.zeros((4, 5)), np.ones((2, 3)))
    assert padded.shape == (4, 5)
    assert padded[:2, :3].sum() == 6
    assert metrics.compact_time_string("1hour(s) 2min(s) 3sec(s)") == "01:02:03"


def test_metrics_region_properties_extra_and_normalization():
    labels = np.zeros((5, 5), dtype=np.uint16)
    labels[1:4, 1:4] = 1

    df = metrics.region_properties(labels)
    df = metrics.extra_properties(df)
    df = metrics.add_file_name_to_dataframe("img.tif", df)
    df = metrics.add_parent_folder(
        df,
        given_dir="/tmp/root",
        root="/tmp/root/downsampling_2/GT",
        folder_sampling_dict={"downsampling_2": 0.5},
    )
    df = metrics.normalize_to_sampling(df, ["area", "perimeter", "axis_major_length"])

    assert df.loc[0, "File_name"] == "img.tif"
    assert df.loc[0, "Grand_Parent_Folder"] == "downsampling_2"
    assert df.loc[0, "norm_area"] == pytest.approx(df.loc[0, "area"] / 0.25)
    assert "Circularity" in df


def test_metrics_object_props_walks_tiffs(tmp_path):
    root = tmp_path / "Dataset"
    gt = root / "OG" / "GT"
    gt.mkdir(parents=True)
    labels = np.zeros((5, 5), dtype=np.uint16)
    labels[1:4, 1:4] = 1
    imwrite(gt / "img.tif", labels)

    df = metrics.object_props(str(root), properties=["label", "area", "area_filled"])

    assert len(df) == 1
    assert df.loc[0, "Grand_Parent_Folder"] == "OG"
    assert df.loc[0, "norm_area"] == df.loc[0, "area"]


def test_2d_semantic_and_binary_statistics(segmentation_dataset, tmp_path):
    result_dir = tmp_path / "Results"
    result_dir.mkdir()
    semantic_gt = np.zeros((6, 6), dtype=np.uint16)
    semantic_gt[1:3, 1:3] = 1
    semantic_gt[3:5, 3:5] = 2
    imwrite(segmentation_dataset / "OG" / "GT" / "semantic.tif", semantic_gt)
    imwrite(segmentation_dataset / "OG" / "Prediction" / "semantic.tif", semantic_gt)

    semantic = metrics.semantic_statistics(str(segmentation_dataset), str(result_dir), ["OG"])
    binary = metrics.binary_mask_statistics(
        str(segmentation_dataset),
        str(result_dir),
        ["OG"],
        max_samples_to_save=1,
        save_images=True,
    )

    assert set(semantic["GT_Label"]) == {1, 2, "ALL"}
    assert binary.iloc[0]["GT_Label"] == "mask"
    assert list((result_dir / "OG").glob("*_TP_binary.tif"))


def test_2d_per_object_statistics_on_tiny_dataset(segmentation_dataset, tmp_path):
    result_dir = tmp_path / "PerObjectResults"
    result_dir.mkdir()

    summary, per_obj = metrics.per_object_statistics(
        str(segmentation_dataset),
        str(result_dir),
        ["OG"],
        save_images=False,
    )

    assert {"GT_Label", "Prediction_Label", "IoU", "GT_area"}.issubset(per_obj.columns)
    assert {"true_positives_count", "false_negatives_count", "false_positives_count"}.issubset(summary.columns)
    assert (result_dir / "Dataset_IoU_per_obj_stats.csv").exists()


def test_percentage_variation_metrics(tmp_path, monkeypatch):
    dataset = tmp_path / "Dataset"
    results = dataset / "Results"
    results.mkdir(parents=True)
    pd.DataFrame({"Grand_Parent_Folder": ["OG", "downsampling_2"], "IoU": [0.8, 0.4]}).to_csv(results / "a_per_obj.csv", index=False)
    pd.DataFrame({"Grand_Parent_Folder": ["OG", "downsampling_2"], "IoU": [0.9, 0.45]}).to_csv(results / "b_binary_mask.csv", index=False)
    pd.DataFrame({"Grand_Parent_Folder": ["OG", "downsampling_2"], "GT_Label": ["ALL", "ALL"], "IoU": [1.0, 0.5]}).to_csv(results / "c_semantic.csv", index=False)
    pd.DataFrame({"Grand_Parent_Folder": ["OG", "downsampling_2"], "IoU": [0.8, 0.4]}).to_csv(results / "z_summary_stats.csv", index=False)

    monkeypatch.setattr(
        metrics,
        "get_csv_dict",
        __import__("rescale4dl.utils", fromlist=["get_csv_dict"]).get_csv_dict,
        raising=False,
    )

    metrics.percentage_variation_metrics(str(tmp_path), "Dataset", instance_segmentation=False)

    out = dataset / "Dataset_percent_var_dict.csv"
    assert out.exists()
    assert "BN OG vs downsampling_2" in pd.read_csv(out)["Comparison"].tolist()


def test_fov_estimates_from_csvs(tmp_path):
    metrics_csv = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "Grand_Parent_Folder": ["OG", "downsampling_2"],
            "Dimensions": ["(10, 20)", "(5, 10)"],
        }
    ).to_csv(metrics_csv, index=False)

    assert fov_estimates.microscope_FOV_area(str(metrics_csv)) == 200

    dataset = tmp_path / "Dataset"
    results = dataset / "Results"
    results.mkdir(parents=True)
    pd.DataFrame(
        {
            "Grand_Parent_Folder": ["OG", "OG"],
            "File_name": ["a", "a"],
            "GT_area": [10, 20],
        }
    ).to_csv(results / "Dataset_per_obj.csv", index=False)

    obj_df = fov_estimates.obj_per_microscope_FOV(200, str(tmp_path), "Dataset")
    assert obj_df.iloc[0]["Obj_per_FOV_mean"] == 13


def test_3d_basic_helpers(volume_3d):
    assert metrics3d.voxel_coverage_percent_3d(volume_3d == 1) == pytest.approx(12.5)
    assert metrics3d.bbox_points_for_crop_3d([1, 1, 1, 2, 2, 2], 5, 5, 5) == (0, 0, 0, 3, 3, 3)
    assert all(value >= 0 for value in metrics3d.object_diameter_3d(volume_3d == 1))
    assert np.isfinite(metrics3d.surface_area_3d(volume_3d == 1)) or np.isnan(metrics3d.surface_area_3d(volume_3d == 1))

    padded = metrics3d.pad_br_with_zeroes_3d(np.zeros((3, 4, 5)), np.ones((2, 2, 2)))
    assert padded.shape == (3, 4, 5)
    assert padded[:2, :2, :2].sum() == 8


def test_3d_region_properties_and_normalization(volume_3d):
    df = metrics3d.region_properties_3d(volume_3d, properties=["label", "area", "equivalent_diameter_area"])
    df = metrics3d.add_file_name_to_dataframe("vol.tif", df)
    df = metrics3d.add_parent_folder(
        df,
        given_dir="GT",
        root="/tmp/Dataset/OG",
        folder_sampling_dict={"Dataset": 1, "OG": 1},
    )
    df = metrics3d.extra_properties(df)
    df = metrics3d.normalize_to_sampling(df, ["area", "equivalent_diameter_area"])

    assert "File_name" in df
    assert "area_normalized" in df


def test_3d_semantic_and_binary_statistics(segmentation_volume_dataset, tmp_path):
    result_dir = tmp_path / "Results3D"
    result_dir.mkdir()

    semantic = metrics3d.semantic_statistics_3d(str(segmentation_volume_dataset), str(result_dir), ["OG"])
    binary = metrics3d.binary_mask_statistics_3d(
        str(segmentation_volume_dataset),
        str(result_dir),
        ["OG"],
        max_samples_to_save=1,
        save_volumes=True,
    )

    assert "ALL" in semantic["GT_Label"].tolist()
    assert binary.iloc[0]["GT_Label"] == "BINARY"
    assert (result_dir / "OG" / "volume_TP_binary.tif").exists()


def test_3d_per_object_statistics_on_tiny_dataset(segmentation_volume_dataset, tmp_path):
    result_dir = tmp_path / "PerObject3D"
    result_dir.mkdir()

    summary, per_obj = metrics3d.per_object_statistics_3d(
        str(segmentation_volume_dataset),
        str(result_dir),
        ["OG"],
        save_volumes=False,
    )

    assert {"GT_Label", "Prediction_Label", "IoU", "GT_volume"}.issubset(per_obj.columns)
    assert {"true_positives_count", "false_negatives_count", "false_positives_count"}.issubset(summary.columns)
    assert (result_dir / "Dataset3D_IoU_per_obj_stats.csv").exists()
