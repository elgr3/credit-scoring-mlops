from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_scoring import config
from credit_scoring.data import split, to_xy
from credit_scoring.train import build_pipeline, business_cost, evaluate, train


def test_build_pipeline_fits(dataset):
    train_df, test_df = split(dataset)
    x, y = to_xy(train_df)
    pipe = build_pipeline({"n_estimators": 20}).fit(x, y)
    proba = pipe.predict_proba(to_xy(test_df)[0])[:, 1]
    assert proba.shape == (200,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_exported_model_quality(trained_model):
    model = joblib.load(trained_model)
    test_df = pd.read_csv(config.DATA_DIR / "current.csv")
    x, y = to_xy(test_df)
    auc = roc_auc_score(y, model.predict_proba(x)[:, 1])
    assert auc > 0.70, f"AUC trop faible : {auc:.3f}"


def test_model_handles_unknown_category(trained_model):
    model = joblib.load(trained_model)
    x, _ = to_xy(pd.read_csv(config.DATA_DIR / "current.csv").head(1))
    x.loc[x.index[0], "purpose"] = "jamais vu"
    assert 0 <= model.predict_proba(x)[0, 1] <= 1


def test_business_cost():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0])  # 1 FN (coût 5) + 1 FP (coût 1)
    assert business_cost(y_true, y_pred) == 6 / 4


def test_threshold_from_cost_matrix():
    assert abs(config.DECISION_THRESHOLD - 1 / 6) < 1e-9


def test_evaluate_keys(dataset):
    train_df, test_df = split(dataset)
    pipe = build_pipeline({"n_estimators": 20}).fit(*to_xy(train_df))
    m = evaluate(pipe, *to_xy(test_df))
    assert {"auc", "gini", "brier", "business_cost_per_case"} <= set(m)
    assert abs(m["gini"] - (2 * m["auc"] - 1)) < 1e-3


def test_train_end_to_end(tmp_path, monkeypatch):
    """Entraînement complet isolé : MLflow, export du modèle et des métadonnées."""
    monkeypatch.setattr(config, "MLRUNS_DIR", tmp_path / "mlruns")
    monkeypatch.setattr(config, "MLFLOW_DB", tmp_path / "mlflow.db")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "model.joblib")
    monkeypatch.setattr(config, "MODEL_META_PATH", tmp_path / "models" / "meta.json")

    meta = train({"n_estimators": 30}, register=False)

    assert (tmp_path / "models" / "model.joblib").exists()
    assert (tmp_path / "reference.csv").exists()
    saved = json.loads((tmp_path / "models" / "meta.json").read_text("utf-8"))
    assert saved["mlflow_run_id"] == meta["mlflow_run_id"]
    assert saved["metrics"]["auc"] > 0.6
    assert (tmp_path / "mlruns").exists()


def test_fast_scorer_matches_pipeline(trained_model, dataset):
    """Le chemin d'inférence optimisé doit reproduire exactement le pipeline sklearn."""
    from credit_scoring.scoring import FastScorer

    model = joblib.load(trained_model)
    x, _ = to_xy(dataset)
    x.loc[x.index[0], "purpose"] = "jamais vu"
    expected = model.predict_proba(x)[:, 1]
    got = FastScorer(model).predict_proba(x.to_dict(orient="records"))
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_fast_scorer_rejects_drop(trained_model):
    import copy

    import pytest

    from credit_scoring.scoring import FastScorer

    model = copy.deepcopy(joblib.load(trained_model))
    model.named_steps["preprocess"].named_transformers_["cat"].drop = "first"
    with pytest.raises(ValueError, match="drop"):
        FastScorer(model)
