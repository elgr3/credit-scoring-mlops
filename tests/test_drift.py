from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from credit_scoring import config, drift


def test_psi_identical_is_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert drift.psi(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref, cur = rng.normal(0, 1, 5000), rng.normal(1, 1, 5000)
    assert drift.psi(ref, cur) > 0.25


def test_psi_small_noise_is_stable():
    rng = np.random.default_rng(1)
    assert drift.psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.1


@pytest.mark.parametrize(
    ("value", "label"), [(0.05, "stable"), (0.15, "a_surveiller"), (0.4, "drift")]
)
def test_psi_label(value, label):
    assert drift.psi_label(value) == label


def test_simulate_drift(dataset):
    out = drift.simulate_production_drift(dataset)
    assert out.shape == dataset.shape
    assert out["credit_amount"].mean() > dataset["credit_amount"].mean() * 1.3
    assert out["age"].min() >= 18
    assert (out["checking_status"] == "<0").mean() > (dataset["checking_status"] == "<0").mean()
    # l'original n'est pas modifié
    assert not out["duration"].equals(dataset["duration"])


@pytest.fixture()
def isolated_reports(tmp_path, monkeypatch, trained_model):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    return tmp_path


def test_run_drifted_scenario(isolated_reports):
    result = drift.run("drifted")
    assert (isolated_reports / "drift_report.html").stat().st_size > 10_000
    saved = json.loads((isolated_reports / "drift_summary.json").read_text("utf-8"))
    assert saved["n_drifted_columns"] == result["n_drifted_columns"]
    for col in ("credit_amount", "duration", "age", "checking_status"):
        assert saved["columns"][col]["drifted"], col
    assert saved["score_psi"] > 0.1


def test_run_reference_vs_holdout_is_stable(isolated_reports):
    """Le test set non modifié ne doit pas déclencher de drift global."""
    result = drift.run("none")
    assert result["dataset_drift"] is False
    assert result["score_psi_status"] != "drift"
    assert isinstance(pd.read_csv(config.DATA_DIR / "current.csv"), pd.DataFrame)
