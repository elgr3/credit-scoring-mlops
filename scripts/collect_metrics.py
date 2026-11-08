"""Agrège les mesures réelles produites par la pipeline dans reports/metrics.json.

Sources (toutes générées par les étapes précédentes, aucune valeur saisie à la main) :
- reports/train_metrics.json  (python -m credit_scoring.train)
- reports/drift_summary.json  (python -m credit_scoring.drift)
- reports/load_test.json      (python scripts/load_test.py)
- reports/load_test_sequential.json (optionnel, --concurrency 1)
- reports/coverage.json       (pytest --cov --cov-report=json:reports/coverage.json)
- reports/junit.xml           (pytest --junitxml=reports/junit.xml)
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def _load(name: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        raise SystemExit(f"Fichier manquant : {path} — lancer l'étape correspondante d'abord.")
    return json.loads(path.read_text("utf-8"))


def _test_count() -> dict:
    root = ET.parse(REPORTS / "junit.xml").getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    total = sum(int(s.get("tests", 0)) for s in suites)
    failed = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    return {"test_count": total, "tests_failed": failed, "tests_skipped": skipped}


def main() -> None:
    train = _load("train_metrics.json")
    drift = _load("drift_summary.json")
    load = _load("load_test.json")
    cov = _load("coverage.json")

    metrics = {
        "auc": train["auc"],
        "gini": train["gini"],
        "cv_auc_mean": train["cv_auc_mean"],
        "business_cost_per_case": train["business_cost_per_case"],
        "p50_ms": load["p50_ms"],
        "p95_ms": load["p95_ms"],
        "rps": load["rps"],
        "load_test_requests": load["requests"],
        "load_test_concurrency": load["concurrency"],
        "load_test_errors": load["errors"],
        **_test_count(),
        "coverage_pct": round(cov["totals"]["percent_covered"], 1),
        "drift_detected": drift["dataset_drift"],
        "drift_share": drift["drift_share"],
        "drifted_columns": drift["n_drifted_columns"],
        "score_psi": drift["score_psi"],
    }
    seq = REPORTS / "load_test_sequential.json"
    if seq.exists():  # optionnel : latence unitaire sans file d'attente (concurrence 1)
        s = json.loads(seq.read_text("utf-8"))
        metrics |= {"sequential_p50_ms": s["p50_ms"], "sequential_p95_ms": s["p95_ms"]}
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), "utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
