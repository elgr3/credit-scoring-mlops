"""Service de scoring FastAPI.

Lancement : uvicorn credit_scoring.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from credit_scoring import config
from credit_scoring.schemas import (
    BatchRequest,
    BatchResponse,
    CreditApplication,
    Health,
    Prediction,
)
from credit_scoring.scoring import FastScorer

logger = logging.getLogger("credit_scoring.api")


class Metrics:
    """Compteurs minimalistes exposés au format texte Prometheus."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.requests: dict[tuple[str, str, int], int] = defaultdict(int)
            self.latency_sum: dict[str, float] = defaultdict(float)
            self.latency_count: dict[str, int] = defaultdict(int)
            self.decisions: dict[str, int] = defaultdict(int)

    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests[(method, path, status)] += 1
            self.latency_sum[path] += seconds
            self.latency_count[path] += 1

    def observe_decisions(self, decisions: list[str]) -> None:
        with self._lock:
            for d in decisions:
                self.decisions[d] += 1

    def render(self, model_version: str | None) -> str:
        with self._lock:
            lines = [
                "# HELP http_requests_total Nombre de requêtes HTTP.",
                "# TYPE http_requests_total counter",
            ]
            for (method, path, status), n in sorted(self.requests.items()):
                lines.append(
                    f'http_requests_total{{method="{method}",path="{path}",status="{status}"}} {n}'
                )
            lines += [
                "# HELP http_request_duration_seconds Latence des requêtes.",
                "# TYPE http_request_duration_seconds summary",
            ]
            for path in sorted(self.latency_count):
                lines.append(
                    f'http_request_duration_seconds_sum{{path="{path}"}} '
                    f"{self.latency_sum[path]:.6f}"
                )
                lines.append(
                    f'http_request_duration_seconds_count{{path="{path}"}} '
                    f"{self.latency_count[path]}"
                )
            lines += [
                "# HELP credit_decisions_total Décisions de crédit rendues.",
                "# TYPE credit_decisions_total counter",
            ]
            for decision in ("approve", "reject"):
                lines.append(
                    f'credit_decisions_total{{decision="{decision}"}} {self.decisions[decision]}'
                )
            lines += [
                "# HELP model_info Modèle actuellement servi.",
                "# TYPE model_info gauge",
                f'model_info{{version="{model_version or "none"}"}} {1 if model_version else 0}',
            ]
        return "\n".join(lines) + "\n"


class ModelService:
    """Encapsule le pipeline sklearn et ses métadonnées."""

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.scorer: FastScorer | None = None
        self.meta: dict[str, Any] = {}

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    @property
    def version(self) -> str | None:
        if not self.loaded:
            return None
        return f"{self.meta.get('model_name', 'model')}:{self.meta.get('model_version', '?')}"

    @property
    def threshold(self) -> float:
        return float(self.meta.get("decision_threshold", config.DECISION_THRESHOLD))

    def load(self, model_path: Path, meta_path: Path) -> None:
        if not model_path.exists():
            logger.warning("Modèle introuvable : %s", model_path)
            self.pipeline, self.scorer, self.meta = None, None, {}
            return
        self.pipeline = joblib.load(model_path)
        self.scorer = FastScorer(self.pipeline)
        self.meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
        logger.info("Modèle chargé : %s", self.version)

    def predict(self, applications: list[CreditApplication]) -> list[Prediction]:
        if not self.loaded:
            raise HTTPException(status_code=503, detail="Modèle non chargé")
        records = [a.model_dump() for a in applications]
        if self.scorer is not None:
            probas = self.scorer.predict_proba(records)
        else:  # repli : pipeline sklearn complet
            frame = pd.DataFrame(records, columns=config.FEATURES)
            probas = self.pipeline.predict_proba(frame)[:, 1]
        threshold = self.threshold
        version = self.version or "unknown"
        return [
            Prediction(
                probability_default=round(float(p), 6),
                decision="reject" if p >= threshold else "approve",
                threshold=round(threshold, 4),
                model_version=version,
            )
            for p in probas
        ]


service = ModelService()
metrics = Metrics()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load(config.MODEL_PATH, config.MODEL_META_PATH)
    yield


class MetricsMiddleware:
    """Middleware ASGI pur (plus léger que @app.middleware("http")) : compte et chronomètre."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Le routeur Starlette renseigne scope["route"] : on agrège par template de route
            path = getattr(scope.get("route"), "path", "unmatched")
            metrics.observe_request(scope["method"], path, status, time.perf_counter() - start)


app = FastAPI(
    title="Credit Scoring API",
    version="1.0.0",
    description="Scoring du risque de défaut (German Credit) — XGBoost servi par FastAPI.",
    lifespan=lifespan,
)
app.add_middleware(MetricsMiddleware)


@app.get("/health", response_model=Health)
async def health() -> Health:
    return Health(
        status="ok" if service.loaded else "degraded",
        model_loaded=service.loaded,
        model_version=service.version,
    )


@app.post("/predict", response_model=Prediction)
async def predict(application: CreditApplication) -> Prediction:
    prediction = service.predict([application])[0]
    metrics.observe_decisions([prediction.decision])
    return prediction


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(batch: BatchRequest) -> BatchResponse:
    predictions = service.predict(batch.applications)
    metrics.observe_decisions([p.decision for p in predictions])
    return BatchResponse(
        predictions=predictions, count=len(predictions), model_version=service.version or ""
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    return metrics.render(service.version)
