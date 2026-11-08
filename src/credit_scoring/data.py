"""Chargement, cache et validation du jeu German Credit (OpenML credit-g)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scoring import config


class DataValidationError(ValueError):
    """Le DataFrame ne respecte pas le contrat de données attendu."""


def fetch_dataset(csv_path: Path = config.DATASET_CSV) -> pd.DataFrame:
    """Retourne le dataset, téléchargé depuis OpenML au premier appel puis lu en cache CSV."""
    if csv_path.exists():
        return pd.read_csv(csv_path)

    from sklearn.datasets import fetch_openml

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(
        "credit-g", version=1, as_frame=True, data_home=str(csv_path.parent / "openml")
    )
    df = bunch.frame.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str)
    df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path)


def validate_dataframe(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Contrôles qualité minimaux avant entraînement / scoring."""
    expected = list(config.FEATURES) + ([config.TARGET] if require_target else [])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise DataValidationError(f"Colonnes manquantes : {missing}")
    if df.empty:
        raise DataValidationError("DataFrame vide")

    nulls = df[expected].isna().sum()
    if nulls.any():
        raise DataValidationError(f"Valeurs manquantes : {nulls[nulls > 0].to_dict()}")

    for col in config.NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Colonne numérique attendue : {col}")
        if (df[col] < 0).any():
            raise DataValidationError(f"Valeurs négatives dans {col}")

    if (df["age"] < 18).any():
        raise DataValidationError("Âge < 18 ans détecté")

    if require_target:
        labels = set(df[config.TARGET].unique())
        if not labels <= {"good", "bad"}:
            raise DataValidationError(f"Labels inattendus : {labels - {'good', 'bad'}}")
    return df


def to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare features et cible binaire (1 = défaut)."""
    x = df[config.FEATURES].copy()
    y = (df[config.TARGET] == config.POSITIVE_LABEL).astype(int)
    return x, y


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split stratifié reproductible train / test."""
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[config.TARGET],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
