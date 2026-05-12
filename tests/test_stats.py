from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rescale4dl import stats


def test_format_and_safe_log_p_values():
    assert stats.format_p_value(0.0005) == "< 0.001"
    assert stats.format_p_value(0.012345) == "0.01235"

    matrix = np.array([[1.0, 0.01], [0.0, 0.5]])
    logged = stats.safe_log10_p_values(matrix)

    assert logged[0, 0] == pytest.approx(-np.log10(0.999))
    assert logged[0, 1] == pytest.approx(2)
    assert np.isfinite(logged[1, 0])


def test_create_pvalue_matrix_is_symmetric():
    results = pd.DataFrame(
        [
            {"Hierarchy1": "A", "Hierarchy2": "B", "p-value": 0.03},
            {"Hierarchy1": "B", "Hierarchy2": "C", "p-value": 0.2},
        ]
    )

    matrix = stats.create_pvalue_matrix(results)

    assert matrix.loc["A", "A"] == 1.0
    assert matrix.loc["A", "B"] == 0.03
    assert matrix.loc["C", "B"] == 0.2
    assert np.isnan(matrix.loc["A", "C"])


@pytest.mark.parametrize("test_type", ["t-test", "Welch's t-test", "Kolmogorov-Smirnov"])
def test_run_statistical_test_returns_statistic_and_pvalue(test_type):
    statistic, pvalue = stats.run_statistical_test(np.arange(5), np.arange(5) + 0.2, test_type)

    assert np.isfinite(statistic)
    assert 0 <= pvalue <= 1


def test_choose_statistical_test_with_monkeypatched_scipy(monkeypatch):
    monkeypatch.setattr(stats.stats, "shapiro", lambda group: (0.0, 0.9))
    monkeypatch.setattr(stats.stats, "levene", lambda group1, group2: (0.0, 0.9))
    assert stats.choose_statistical_test([1, 2, 3], [1, 2, 4]) == "t-test"

    monkeypatch.setattr(stats.stats, "levene", lambda group1, group2: (0.0, 0.01))
    assert stats.choose_statistical_test([1, 2, 3], [1, 2, 4]) == "Welch's t-test"

    monkeypatch.setattr(stats.stats, "shapiro", lambda group: (0.0, 0.01))
    assert stats.choose_statistical_test([1, 2, 3], [1, 2, 4]) == "Kolmogorov-Smirnov"


def test_compute_statistical_analysis_from_pandas_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(stats.plt, "show", lambda: None)
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "value": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
        }
    )

    result = stats.compute_statistical_analysis_from_pandas(
        df,
        tmp_path,
        quantitative_column="value",
        categorical_columns=["group"],
        test_type="Kolmogorov-Smirnov",
    )

    assert list(result["Test"]) == ["Kolmogorov-Smirnov"]
    assert (tmp_path / "p-values.csv").exists()
    assert (tmp_path / "p-values.png").exists()


def test_compute_statistical_analysis_reads_expected_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(stats.plt, "show", lambda: None)
    dataset = tmp_path / "Dataset" / "Results"
    dataset.mkdir(parents=True)
    pd.DataFrame(
        {
            "condition": ["A", "A", "A", "B", "B", "B"],
            "score": [1, 2, 3, 4, 5, 6],
        }
    ).to_csv(dataset / "Dataset_summary_stats.csv", index=False)

    result = stats.compute_statistical_analysis(
        str(tmp_path),
        "Dataset",
        quantitative_column="score",
        categorical_columns=["condition"],
        test_type="t-test",
    )

    assert result.iloc[0]["N Hierarchy1"] == 3
    assert (dataset / "stats" / "Dataset_p-values_summary_stats.csv").exists()
