from __future__ import annotations

import typing

import pytest

from credit_scoring import config, schemas
from credit_scoring.data import DataValidationError, split, to_xy, validate_dataframe


def test_dataset_shape(dataset):
    assert dataset.shape == (1000, 21)
    assert set(dataset[config.TARGET].unique()) == {"good", "bad"}


def test_validate_ok(dataset):
    assert validate_dataframe(dataset) is dataset


def test_validate_without_target(dataset):
    validate_dataframe(dataset.drop(columns=[config.TARGET]), require_target=False)


def test_missing_column(dataset):
    with pytest.raises(DataValidationError, match="manquantes"):
        validate_dataframe(dataset.drop(columns=["age"]))


def test_empty_frame(dataset):
    with pytest.raises(DataValidationError, match="vide"):
        validate_dataframe(dataset.iloc[0:0])


def test_nulls_detected(dataset):
    df = dataset.copy()
    df.loc[0, "duration"] = None
    with pytest.raises(DataValidationError, match="manquantes"):
        validate_dataframe(df)


def test_non_numeric(dataset):
    df = dataset.copy()
    df["duration"] = df["duration"].astype(str)
    with pytest.raises(DataValidationError, match="numérique"):
        validate_dataframe(df)


def test_negative_values(dataset):
    df = dataset.copy()
    df.loc[0, "credit_amount"] = -1
    with pytest.raises(DataValidationError, match="négatives"):
        validate_dataframe(df)


def test_minor_age(dataset):
    df = dataset.copy()
    df.loc[0, "age"] = 16
    with pytest.raises(DataValidationError, match="18"):
        validate_dataframe(df)


def test_unexpected_label(dataset):
    df = dataset.copy()
    df.loc[0, config.TARGET] = "unknown"
    with pytest.raises(DataValidationError, match="Labels"):
        validate_dataframe(df)


def test_split_is_stratified(dataset):
    train_df, test_df = split(dataset)
    assert len(train_df) == 800 and len(test_df) == 200
    rate = lambda d: (d[config.TARGET] == "bad").mean()  # noqa: E731
    assert abs(rate(train_df) - rate(test_df)) < 0.01


def test_to_xy(dataset):
    x, y = to_xy(dataset)
    assert list(x.columns) == config.FEATURES
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 300  # 30 % de mauvais payeurs dans German Credit


@pytest.mark.parametrize("column", config.CATEGORICAL_FEATURES)
def test_api_schema_covers_all_categories(dataset, column):
    """Le contrat d'API accepte exactement les modalités vues à l'entraînement."""
    annotation = schemas.CreditApplication.model_fields[column].annotation
    allowed = set(typing.get_args(annotation))
    assert allowed == set(dataset[column].unique())


def test_fetch_downloads_and_caches(tmp_path, monkeypatch, dataset):
    """Premier appel : téléchargement OpenML (simulé), conversion des catégories, cache CSV."""
    from types import SimpleNamespace

    import sklearn.datasets

    from credit_scoring.data import fetch_dataset

    frame = dataset.head(5).copy()
    frame["purpose"] = frame["purpose"].astype("category")
    calls = []

    def fake_fetch_openml(name, version, as_frame, data_home):
        calls.append((name, version))
        return SimpleNamespace(frame=frame)

    monkeypatch.setattr(sklearn.datasets, "fetch_openml", fake_fetch_openml)
    csv = tmp_path / "sub" / "credit.csv"
    first = fetch_dataset(csv)
    second = fetch_dataset(csv)
    assert calls == [("credit-g", 1)]
    assert csv.exists() and first.shape == second.shape == (5, 21)
    assert first["purpose"].dtype == object
