"""Test de charge asynchrone de l'API de scoring.

Par défaut, lance lui-même uvicorn en sous-processus, attend /health, envoie N requêtes
/predict avec une concurrence fixe, puis écrit reports/load_test.json.

    python scripts/load_test.py --requests 2000 --concurrency 20
    python scripts/load_test.py --url http://localhost:8000 --no-spawn
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_scoring import config  # noqa: E402


def load_payloads(n: int, seed: int = 0) -> list[dict]:
    df = pd.read_csv(config.DATA_DIR / "current.csv")[config.FEATURES]
    records = df.to_dict(orient="records")
    rng = random.Random(seed)
    return [rng.choice(records) for _ in range(n)]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


async def wait_ready(url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{url}/health", timeout=2)
                if r.status_code == 200 and r.json().get("model_loaded"):
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError(f"API non disponible sur {url} après {timeout}s")


def server_side_mean_ms(metrics_text: str, path: str = "/predict") -> float | None:
    """Latence moyenne mesurée côté serveur (middleware) à partir de /metrics."""
    values = {}
    for line in metrics_text.splitlines():
        for kind in ("sum", "count"):
            prefix = f'http_request_duration_seconds_{kind}{{path="{path}"}} '
            if line.startswith(prefix):
                values[kind] = float(line[len(prefix) :])
    if not values.get("count"):
        return None
    return values["sum"] / values["count"] * 1000


async def run(url: str, n_requests: int, concurrency: int, warmup: int) -> dict:
    payloads = load_payloads(n_requests + warmup)
    latencies: list[float] = []
    errors = 0
    queue: asyncio.Queue[dict] = asyncio.Queue()
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(base_url=url, limits=limits, timeout=30) as client:
        for p in payloads[:warmup]:
            await client.post("/predict", json=p)
        for p in payloads[warmup:]:
            queue.put_nowait(p)

        async def worker() -> None:
            nonlocal errors
            while True:
                try:
                    payload = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                t0 = time.perf_counter()
                try:
                    r = await client.post("/predict", json=payload)
                    ok = r.status_code == 200
                except httpx.HTTPError:
                    ok = False
                latencies.append((time.perf_counter() - t0) * 1000)
                if not ok:
                    errors += 1

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        elapsed = time.perf_counter() - start
        server_ms = server_side_mean_ms((await client.get("/metrics")).text)

    return {
        "requests": n_requests,
        "concurrency": concurrency,
        "errors": errors,
        "duration_s": round(elapsed, 3),
        "rps": round(n_requests / elapsed, 1),
        "p50_ms": round(percentile(latencies, 0.50), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "p99_ms": round(percentile(latencies, 0.99), 2),
        "mean_ms": round(statistics.fmean(latencies), 2),
        # inclut le warm-up ; temps passé dans l'application, hors réseau / file d'attente
        "server_side_mean_ms": round(server_ms, 2) if server_ms is not None else None,
        "server": "uvicorn 1 worker, localhost",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8799")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--no-spawn", action="store_true", help="Utiliser une API déjà lancée")
    parser.add_argument("--output", default=str(config.REPORTS_DIR / "load_test.json"))
    args = parser.parse_args()

    proc = None
    if not args.no_spawn:
        port = args.url.rsplit(":", 1)[-1]
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "credit_scoring.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                port,
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env,
        )
    try:
        asyncio.run(wait_ready(args.url))
        result = asyncio.run(run(args.url, args.requests, args.concurrency, args.warmup))
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=10)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), "utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
