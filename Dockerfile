# syntax=docker/dockerfile:1.7
# Image de service : uniquement les dépendances d'inférence (pas de MLflow / Evidently).
# Prérequis : models/model.joblib doit exister (python -m credit_scoring.train).

ARG PYTHON_VERSION=3.12

# ---------- Étape 1 : construction du virtualenv ----------
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# ---------- Étape 2 : image d'exécution ----------
FROM python:${PYTHON_VERSION}-slim AS runtime
# libgomp1 : runtime OpenMP requis par XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app --home /app app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    CREDIT_PROJECT_ROOT=/app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src/ src/
COPY --chown=app:app models/model.joblib models/model_meta.json models/

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "credit_scoring.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
