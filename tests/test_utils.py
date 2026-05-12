from __future__ import annotations

import numpy as np

from rescale4dl import utils


def test_check_crop_img_crops_to_factor():
    arr = np.arange(35).reshape(5, 7)

    cropped = utils.check_crop_img(arr, 3)

    assert cropped.shape == (3, 6)
    np.testing.assert_array_equal(cropped, arr[:3, :6])


def test_crop_with_padding_crops_and_centers_padding():
    small = np.ones((2, 2), dtype=np.uint8)
    padded = utils.crop_with_padding(small, (4, 4))
    assert padded.sum() == 4
    np.testing.assert_array_equal(padded[1:3, 1:3], small)

    large = np.arange(36).reshape(6, 6)
    cropped = utils.crop_with_padding(large, (2, 4))
    np.testing.assert_array_equal(cropped, large[2:4, 1:5])


def test_score_and_find_matching_labels(label_image_2d, prediction_image_2d):
    assert utils.score_label_overlap(label_image_2d, prediction_image_2d, 1, 10) == 1.0

    scores = utils.compute_labels_matching_scores(label_image_2d, prediction_image_2d)
    assert scores[1][0] == [10, 1.0]
    assert scores[2][0][0] == 20

    matches = list(utils.find_matching_labels(label_image_2d, prediction_image_2d))
    assert (1, 10, 1.0) in matches
    assert any(match[0] == 0 and match[1] == 0 for match in matches)


def test_find_matching_labels_background_only_prediction(label_image_2d):
    matches = list(utils.find_matching_labels(label_image_2d, np.zeros_like(label_image_2d)))

    assert matches == [(0, 0, 0), (1, 0, 0), (2, 0, 0)]


def test_remove_duplicates_prefers_highest_scoring_ground_truth():
    scores = {
        1: [(5, 0.4), (6, 0.2)],
        2: [(5, 0.9), (7, 0.1)],
    }

    matches = utils.remove_duplicates(scores, np.array([0, 5, 6, 7, 8]))

    assert matches[:2] == [(2, 5, 0.9), (1, 6, 0.2)]
    assert (0, 8, 0.0) in matches


def test_3d_matching_and_padding(volume_3d):
    pred = np.zeros_like(volume_3d)
    pred[volume_3d == 1] = 4

    scores = utils.compute_labels_matching_scores_3d(volume_3d, pred)

    assert scores[1][0] == [4, 1.0]
    assert utils.score_label_overlap_3d(volume_3d, pred, 2, 4) == 0.0
    assert any(match[0] == 1 and match[1] == 4 for match in utils.find_matching_labels_3d(volume_3d, pred))


def test_incremental_dir_creation_and_pad_with_zeroes(tmp_path):
    first = utils.incremental_dir_creation(tmp_path, "Results")
    second = utils.incremental_dir_creation(tmp_path, "Results")

    assert first.endswith("Results")
    assert second.endswith("Results_01")

    padded = utils.pad_with_zeroes(np.zeros((3, 4)), np.ones((2, 2)))
    assert padded.shape == (3, 4)
    assert padded[:2, :2].sum() == 4


def test_get_csv_dict_uses_latest_results_folder(tmp_path, capsys):
    dataset = tmp_path / "Dataset"
    (dataset / "Results").mkdir(parents=True)
    (dataset / "Results_01").mkdir()
    (dataset / "Results" / "old.csv").write_text("a\n1\n")
    latest = dataset / "Results_01" / "new.csv"
    latest.write_text("a\n2\n")

    csv_dict = utils.get_csv_dict(str(tmp_path))

    assert csv_dict == {"Dataset": [str(latest)]}
    assert "DONE!" in capsys.readouterr().out


def test_parse_scaling_2d_and_3d_common_cases():
    assert utils.parse_scaling_2d("downsampling_4") == {
        "direction": "down",
        "type": "Rescaling",
        "factor": 4.0,
        "dims_detected": {"scale": 4.0},
    }
    assert utils.parse_scaling_2d("OG")["direction"] == "original"
    assert utils.parse_scaling_2d("misc") is None

    parsed_3d = utils.parse_scaling_3d("upsampling_xyz_2")
    assert parsed_3d["direction"] == "up"
    assert parsed_3d["type"] == "XYZ"
    assert parsed_3d["factor"] == 2.0
