"""Entraînement du modèle de scoring, tracking MLflow et export dans models/.

Usage : python -m credit_scoring.train
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from credit_scoring import config
from credit_scoring.data import fetch_dataset, split, to_xy, validate_dataframe

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "reg_lambda": 1.0,
}


def build_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    """Préprocessing (one-hot sur les catégorielles) + XGBoost."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def business_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coût moyen par dossier selon la matrice de coûts German Credit."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = fp * config.COST_FALSE_POSITIVE + fn * config.COST_FALSE_NEGATIVE
    return float(total / len(y_true))


def evaluate(pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = pipeline.predict_proba(x)[:, 1]
    pred = (proba >= config.DECISION_THRESHOLD).astype(int)
    auc = roc_auc_score(y, proba)
    return {
        "auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "brier": round(float(brier_score_loss(y, proba)), 4),
        "accuracy_at_threshold": round(float(accuracy_score(y, pred)), 4),
        "business_cost_per_case": round(business_cost(y.to_numpy(), pred), 4),
        "rejection_rate": round(float(pred.mean()), 4),
        # Référence naïve : tout accepter (chaque mauvais payeur coûte 5)
        "baseline_cost_approve_all": round(
            business_cost(y.to_numpy(), np.zeros(len(y), dtype=int)), 4
        ),
    }


def setup_mlflow() -> None:
    """Backend SQLite local (runs + model registry), artefacts dans mlruns/.

    MLflow 3.x a placé le backend « fichiers seuls » en maintenance : on suit la
    recommandation officielle (base SQLite) tout en gardant les artefacts sur disque.
    """
    mlflow.set_tracking_uri(f"sqlite:///{config.MLFLOW_DB.resolve().as_posix()}")
    if mlflow.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            config.EXPERIMENT_NAME, artifact_location=config.MLRUNS_DIR.resolve().as_uri()
        )
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def train(params: dict[str, Any] | None = None, register: bool = True) -> dict[str, Any]:
    df = validate_dataframe(fetch_dataset())
    train_df, test_df = split(df)
    x_train, y_train = to_xy(train_df)
    x_test, y_test = to_xy(test_df)

    setup_mlflow()

    pipeline = build_pipeline(params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    with mlflow.start_run() as run:
        cv_auc = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc")
        pipeline.fit(x_train, y_train)
        metrics = evaluate(pipeline, x_test, y_test)
        metrics["cv_auc_mean"] = round(float(cv_auc.mean()), 4)
        metrics["cv_auc_std"] = round(float(cv_auc.std()), 4)

        mlflow.log_params({**DEFAULT_PARAMS, **(params or {})})
        mlflow.log_params(
            {
                "decision_threshold": round(config.DECISION_THRESHOLD, 4),
                "n_train": len(x_train),
                "n_test": len(x_test),
            }
        )
        mlflow.log_metrics(metrics)

        signature = infer_signature(x_test.head(20), pipeline.predict_proba(x_test.head(20)))
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=x_test.head(3),
            # MLflow 3.x sérialise via skops (pas de pickle arbitraire) : on déclare
            # explicitement les types XGBoost comme fiables.
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
            registered_model_name=config.REGISTERED_MODEL_NAME if register else None,
        )
        version = getattr(model_info, "registered_model_version", None) or "unregistered"

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)

    meta = {
        "model_name": config.REGISTERED_MODEL_NAME,
        "model_version": str(version),
        "mlflow_run_id": run.info.run_id,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision_threshold": round(config.DECISION_THRESHOLD, 4),
        "features": config.FEATURES,
        "categories": {c: sorted(df[c].unique().tolist()) for c in config.CATEGORICAL_FEATURES},
        "metrics": metrics,
    }
    config.MODEL_META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    (config.REPORTS_DIR / "train_metrics.json").write_text(json.dumps(metrics, indent=2), "utf-8")

    # Jeux de référence / production pour le monitoring de drift
    train_df.to_csv(config.DATA_DIR / "reference.csv", index=False)
    test_df.to_csv(config.DATA_DIR / "current.csv", index=False)
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    meta = train()
    logger.info("Modèle %s v%s", meta["model_name"], meta["model_version"])
    logger.info("Métriques test : %s", json.dumps(meta["metrics"]))


if __name__ == "__main__":
    main()
