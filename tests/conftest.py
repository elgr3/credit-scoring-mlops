from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from credit_scoring import config
from credit_scoring.data import fetch_dataset
from credit_scoring.schemas import EXAMPLE_APPLICANT


@pytest.fixture(scope="session")
def dataset():
    return fetch_dataset()


@pytest.fixture(scope="session")
def trained_model():
    """Garantit qu'un modèle exporté existe (entraîne si besoin, ex. CI à froid)."""
    if not config.MODEL_PATH.exists() or not (config.DATA_DIR / "current.csv").exists():
        from credit_scoring.train import train

        train(register=False)
    return config.MODEL_PATH


@pytest.fixture()
def client(trained_model):
    from credit_scoring.api import app, metrics

    metrics.reset()
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def applicant():
    return dict(EXAMPLE_APPLICANT)
