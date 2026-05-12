from __future__ import annotations

import pandas as pd

from rescale4dl.metrics import metrics, metrics3d, metrics_skeleton


def test_morphology_dispatches_selected_2d_workflows(tmp_path, monkeypatch):
    dataset = tmp_path / "Dataset"
    (dataset / "OG" / "GT").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(metrics, "per_object_statistics", lambda **kwargs: calls.append(("obj", kwargs)))
    monkeypatch.setattr(metrics, "semantic_statistics", lambda **kwargs: calls.append(("sem", kwargs)))
    monkeypatch.setattr(metrics, "binary_mask_statistics", lambda **kwargs: calls.append(("bin", kwargs)))

    metrics.morphology(
        str(tmp_path),
        sampling_dir_list=["OG"],
        run_per_object_stats=True,
        run_semantic_stats=False,
        run_binary_mask_stats=True,
        save_images=False,
    )

    assert [call[0] for call in calls] == ["obj", "bin"]
    assert (dataset / "Results").exists()


def test_morphology_dispatches_selected_3d_workflows(tmp_path, monkeypatch):
    dataset = tmp_path / "Dataset3D"
    (dataset / "OG" / "GT").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(metrics3d, "per_object_statistics_3d", lambda **kwargs: calls.append(("obj", kwargs)))
    monkeypatch.setattr(metrics3d, "semantic_statistics_3d", lambda **kwargs: calls.append(("sem", kwargs)))
    monkeypatch.setattr(metrics3d, "binary_mask_statistics_3d", lambda **kwargs: calls.append(("bin", kwargs)))

    metrics3d.morphology_3d(
        str(tmp_path),
        sampling_dir_list=["OG"],
        run_per_object_stats=False,
        run_semantic_stats=True,
        run_binary_mask_stats=True,
        save_volumes=False,
    )

    assert [call[0] for call in calls] == ["sem", "bin"]
    assert calls[0][1]["sampling_dir_list"] == ["OG"]


def test_metrics_skeleton_empty_and_missing_paths(tmp_path, capsys):
    dataset = tmp_path / "Dataset"
    dataset.mkdir()

    metrics_skeleton.per_dataset(str(dataset), {"OG": 1})
    assert (dataset / "results").exists()
    assert "No sampling folders" in capsys.readouterr().out

    empty = metrics_skeleton.per_sampling(str(dataset / "OG"), 1)
    assert isinstance(empty, pd.DataFrame)
    assert empty.empty
