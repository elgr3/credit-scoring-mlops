"""Configuration centrale : chemins, colonnes, seuil métier."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("CREDIT_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"  # artefacts MLflow
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"  # backend store (runs, params, registry)

DATASET_CSV = DATA_DIR / "credit-g.csv"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.joblib"))
MODEL_META_PATH = Path(os.getenv("MODEL_META_PATH", MODELS_DIR / "model_meta.json"))

EXPERIMENT_NAME = "credit-scoring"
REGISTERED_MODEL_NAME = "credit-scoring-xgb"
RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "class"
POSITIVE_LABEL = "bad"  # 1 = défaut de paiement

NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Matrice de coûts du jeu German Credit (Hofmann, 1994) :
# accorder un crédit à un mauvais payeur coûte 5, refuser un bon payeur coûte 1.
# Seuil bayésien optimal sur P(défaut) : C_fp / (C_fp + C_fn) = 1 / (1 + 5).
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0
DECISION_THRESHOLD = COST_FALSE_POSITIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE)
