from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tifffile import imwrite
import importlib

from rescale4dl import plot


def test_load_2d_or_3d_slice(tmp_path):
    img2d = np.arange(9, dtype=np.uint8).reshape(3, 3)
    vol = np.arange(2 * 5 * 6, dtype=np.uint8).reshape(2, 5, 6)
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    imwrite(tmp_path / "img.tif", img2d)
    imwrite(tmp_path / "vol.tif", vol)
    imwrite(tmp_path / "rgb.tif", rgb)

    np.testing.assert_array_equal(plot._load_2d_or_3d_slice(str(tmp_path / "img.tif")), img2d)
    np.testing.assert_array_equal(plot._load_2d_or_3d_slice(str(tmp_path / "vol.tif"), slice_axis=1, slice_index=2), np.moveaxis(vol, 1, 0)[2])
    np.testing.assert_array_equal(plot._load_2d_or_3d_slice(str(tmp_path / "rgb.tif")), rgb)

    with pytest.raises(ValueError, match="slice_axis"):
        plot._load_2d_or_3d_slice(str(tmp_path / "vol.tif"), slice_axis=9)
    with pytest.raises(ValueError, match="slice_index"):
        plot._load_2d_or_3d_slice(str(tmp_path / "vol.tif"), slice_axis=0, slice_index=99)


def _write_plot_csvs(root):
    dataset = root / "Dataset"
    ss = root / "Semantic"
    for folder in (dataset / "Results", ss / "Results"):
        folder.mkdir(parents=True)

    per_obj = pd.DataFrame(
        {
            "Grand_Parent_Folder": ["OG", "OG", "downsampling_2", "downsampling_2"],
            "File_name": ["a", "b", "a", "b"],
            "GT_area": [16, 25, 9, 16],
        }
    )
    summary = pd.DataFrame(
        {
            "Grand_Parent_Folder": ["OG", "downsampling_2", "OG", "downsampling_2"],
            "File_name": ["a", "a", "b", "b"],
            "GT_diameter_median": [4.0, 3.0, 5.0, 4.0],
            "pred_diameter_median": [4.2, 2.5, 4.8, 3.5],
            "GT_area": [16, 9, 25, 16],
            "pred_area": [18, 8, 22, 14],
            "IoU": [0.9, 0.7, 0.85, 0.65],
            "Dimensions": ["(10, 10)", "(5, 5)", "(10, 10)", "(5, 5)"],
        }
    )
    binary = pd.DataFrame(
        {
            "Grand_Parent_Folder": ["OG", "downsampling_2"],
            "File_name": ["a", "a"],
            "IoU": [0.9, 0.7],
        }
    )
    semantic = pd.DataFrame(
        {
            "Grand_Parent_Folder": ["OG", "downsampling_2", "OG", "downsampling_2"],
            "File_name": ["a", "a", "a", "a"],
            "GT_Label": ["ALL", "ALL", 1, 1],
            "IoU": [0.8, 0.6, 0.7, 0.5],
        }
    )
    per_obj["pred_area"] = [15, 24, 8, 14]
    per_obj.to_csv(dataset / "Results" / "Dataset_per_obj_stats.csv", index=False)
    summary.to_csv(dataset / "Results" / "Dataset_summary_stats.csv", index=False)
    binary.to_csv(ss / "Results" / "Semantic_binary_mask_stats.csv", index=False)
    semantic.to_csv(ss / "Results" / "Semantic_semantic_stats.csv", index=False)
    return dataset, ss


def _write_plot_csvs_3d(root):
    dataset = root / "Dataset3D"
    ss = root / "Semantic3D"
    for folder in (dataset / "Results", ss / "Results"):
        folder.mkdir(parents=True)
    summary = pd.DataFrame(
        {
            "Grand_Parent_Folder": ["OG", "upsampling_xyz_2"],
            "File_name": ["v", "v"],
            "GT_diameter_median": [4.0, 8.0],
            "IoU": [0.9, 0.7],
            "Dimensions": ["(4, 5, 5)", "(8, 10, 10)"],
        }
    )
    per_obj = pd.DataFrame(
        {
            "Grand_Parent_Folder": ["OG", "upsampling_xyz_2"],
            "File_name": ["v", "v"],
            "GT_area": [16, 64],
        }
    )
    binary = pd.DataFrame(
        {
            "Grand_Parent_Folder": ["OG", "upsampling_xyz_2"],
            "File_name": ["v", "v"],
            "IoU": [0.9, 0.7],
        }
    )
    semantic = pd.DataFrame(
        {
            "Grand_Parent_Folder": ["OG", "upsampling_xyz_2"],
            "File_name": ["v", "v"],
            "GT_Label": ["ALL", "ALL"],
            "IoU": [0.8, 0.6],
        }
    )
    per_obj.to_csv(dataset / "Results" / "Dataset3D_per_obj_stats.csv", index=False)
    summary.to_csv(dataset / "Results" / "Dataset3D_summary_stats.csv", index=False)
    binary.to_csv(ss / "Results" / "Semantic3D_binary_mask_stats.csv", index=False)
    semantic.to_csv(ss / "Results" / "Semantic3D_semantic_stats.csv", index=False)
    return dataset, ss


def test_mean_obj_diam_dict_for_round_and_non_round(tmp_path):
    dataset, _ = _write_plot_csvs(tmp_path)
    csv_dict = plot.get_csv_dict(str(tmp_path))

    non_round = plot.mean_obj_diam_dict("Dataset", csv_dict, is_round_obj=False)
    round_obj = plot.mean_obj_diam_dict("Dataset", csv_dict, is_round_obj=True)

    assert non_round["OG"] == 4.5
    assert round_obj["OG"] > 0


def test_selected_plot_functions_run_with_temporary_csvs(tmp_path, monkeypatch):
    _write_plot_csvs(tmp_path)
    monkeypatch.setattr(plot.plt, "show", lambda: None)
    monkeypatch.setattr(plot.plt, "savefig", lambda *args, **kwargs: None)

    plot.generate_binary_semantic_box_plot(
        str(tmp_path),
        dataset_SS="Semantic",
        dataset_name="Dataset",
        fig_name="demo",
        y_axis="IoU",
        output_path=str(tmp_path),
    )
    plot.generate_semantic_gt_pred_bar_plot(
        str(tmp_path),
        dataset_name="Dataset",
        fig_name="demo",
        output_path=str(tmp_path),
    )
    plot.generate_instance_box_plot(
        str(tmp_path),
        dataset_name="Dataset",
        fig_name="demo",
        y_axis="IoU",
        output_path=str(tmp_path),
    )


def test_remaining_plot_functions_run_with_temporary_csvs(tmp_path, monkeypatch):
    _write_plot_csvs(tmp_path)
    _write_plot_csvs_3d(tmp_path)
    monkeypatch.setattr(plot.plt, "show", lambda: None)
    monkeypatch.setattr(plot.plt, "savefig", lambda *args, **kwargs: None)

    plot.generate_binary_semantic_box_plot_3D(
        str(tmp_path),
        dataset_SS="Semantic3D",
        dataset_name="Dataset3D",
        fig_name="demo",
        y_axis="IoU",
        output_path=str(tmp_path),
    )
    plot.generate_instance_gt_pred_bar_plot(
        str(tmp_path),
        dataset_name="Dataset",
        fig_name="demo",
        output_path=str(tmp_path),
    )
    plot.generate_instance_wt_treatment_bar_plot(
        str(tmp_path),
        dataset_name="Dataset",
        fig_name="demo",
        subset_filenames_treatment=["b"],
        output_path=str(tmp_path),
    )
    plot.generate_throughput_line_plot(
        str(tmp_path),
        dataset_name_list=["Dataset"],
        fig_name="demo",
        output_path=str(tmp_path),
    )
    plot.custom_boxplot(
        str(tmp_path),
        dataset_name="Dataset",
        fig_name="demo",
        y_axis="IoU",
        thoughput_plot=True,
        output_path=str(tmp_path),
    )


def test_plot_data_distributions(tmp_path, monkeypatch):
    results = tmp_path / "Dataset" / "Results"
    results.mkdir(parents=True)
    pd.DataFrame(
        {
            "condition": ["A", "A", "B", "B"],
            "value": [1.0, 1.1, 2.0, 2.1],
        }
    ).to_csv(results / "Dataset_summary_stats.csv", index=False)
    monkeypatch.setattr(plot.plt, "show", lambda: None)

    plot.plot_data_distributions(
        str(tmp_path),
        "Dataset",
        quantitative_column="value",
        categorical_columns=["condition"],
    )

    assert (tmp_path / "Dataset" / "Plots" / "Dataset_value_distributions_summary_stats.csv.png").exists()


def test_plot_segmentation_example_loads_expected_files(tmp_path, monkeypatch):
    root = tmp_path / "analysis"
    gt_dir = root / "Dataset" / "OG" / "GT"
    pred_dir = root / "Dataset" / "OG" / "Prediction"
    results_dir = root / "Dataset" / "Results" / "OG"
    for folder in (gt_dir, pred_dir, results_dir):
        folder.mkdir(parents=True)
    arr = np.ones((3, 3), dtype=np.uint8)
    for path in [
        gt_dir / "sample.tif",
        pred_dir / "sample.tif",
        results_dir / "sample_true_positives.tif",
        results_dir / "sample_false_positives.tif",
        results_dir / "sample_false_negatives.tif",
        results_dir / "sample_TP_binary.tif",
        results_dir / "sample_FP_binary.tif",
        results_dir / "sample_FN_binary.tif",
    ]:
        imwrite(path, arr)
    monkeypatch.setattr(plot.plt, "show", lambda: None)

    plot.plot_segmentation_example(str(root), "Dataset", "OG")


def test_analyse_dispatches_to_2d_or_3d(monkeypatch):
    analyse = importlib.import_module("rescale4dl.analyse")
    calls = []
    monkeypatch.setattr(analyse, "morphology_2d", lambda **kwargs: calls.append(("2d", kwargs)))
    monkeypatch.setattr(analyse, "morphology_3d", lambda **kwargs: calls.append(("3d", kwargs)))

    analyse.analyse("root", is_3d=False, save_images=False)
    analyse.analyse("root", is_3d=True, save_images=True)

    assert calls[0][0] == "2d"
    assert calls[0][1]["save_images"] is False
    assert calls[1][0] == "3d"
    assert calls[1][1]["save_volumes"] is True
