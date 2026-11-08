"""Monitoring de drift : référence (train) vs lot de production simulé et dérivé.

Produit :
- reports/drift_report.html  (rapport Evidently interactif)
- reports/drift_summary.json (résumé machine : drift global, colonnes dérivées, PSI)

Usage : python -m credit_scoring.drift
"""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

from credit_scoring import config

logger = logging.getLogger(__name__)

PSI_BINS = 10


def simulate_production_drift(df: pd.DataFrame, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Simule un lot de production dont la population a changé.

    Scénario : inflation des montants demandés (+40 %), allongement des durées (+6 mois),
    clientèle plus jeune, et afflux de dossiers sans compte courant positif.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["credit_amount"] = (out["credit_amount"] * 1.4).round()
    out["duration"] = out["duration"] + 6
    out["age"] = np.clip(out["age"] - rng.integers(5, 12, len(out)), 18, None)
    mask = rng.random(len(out)) < 0.35
    out.loc[mask, "checking_status"] = "<0"
    return out


def psi(reference: np.ndarray, current: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population Stability Index sur des déciles de la référence.

    Règle usuelle en risque de crédit : < 0,1 stable ; 0,1-0,25 à surveiller ; > 0,25 drift.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    eps = 1e-6
    ref_pct, cur_pct = np.clip(ref_pct, eps, None), np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_label(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "a_surveiller"
    return "drift"


def evidently_drift(reference: pd.DataFrame, current: pd.DataFrame, html_path) -> dict[str, Any]:
    """Lance le preset DataDrift d'Evidently, sauvegarde le HTML, renvoie le résumé."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(
        numerical_columns=config.NUMERIC_FEATURES,
        categorical_columns=config.CATEGORICAL_FEATURES,
    )
    ref_ds = Dataset.from_pandas(reference[config.FEATURES], data_definition=definition)
    cur_ds = Dataset.from_pandas(current[config.FEATURES], data_definition=definition)

    report = Report([DataDriftPreset(drift_share=0.5)])
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)
    snapshot.save_html(str(html_path))

    columns: dict[str, dict[str, Any]] = {}
    n_drifted, share = None, None
    for metric in snapshot.dict()["metrics"]:
        name = metric.get("metric_name", "")
        value = metric.get("value")
        cfg = metric.get("config", {})
        if name.startswith("DriftedColumnsCount"):
            n_drifted, share = int(value["count"]), float(value["share"])
        elif name.startswith("ValueDrift"):
            col = cfg.get("column")
            threshold = cfg.get("threshold")
            method = cfg.get("method")
            score = float(value)
            # Les tests statistiques (p-value) dérivent quand score < seuil,
            # les distances (Wasserstein, Jensen-Shannon...) quand score >= seuil.
            is_pvalue = method is not None and ("p_value" in str(method) or "test" in str(method))
            drifted = score < threshold if is_pvalue else score >= threshold
            columns[col] = {
                "method": method,
                "score": float(f"{score:.4g}"),
                "threshold": threshold,
                "drifted": bool(drifted),
            }
    if n_drifted is None:
        raise RuntimeError("Métrique DriftedColumnsCount absente du rapport Evidently")
    return {"n_drifted_columns": n_drifted, "drift_share": round(share, 4), "columns": columns}


def run(scenario: str = "drifted") -> dict[str, Any]:
    reference = pd.read_csv(config.DATA_DIR / "reference.csv")
    current = pd.read_csv(config.DATA_DIR / "current.csv")
    if scenario == "drifted":
        current = simulate_production_drift(current)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = config.REPORTS_DIR / "drift_report.html"
    summary = evidently_drift(reference, current, html_path)

    # Compléments "risque de crédit" : PSI par variable numérique et PSI du score
    feature_psi = {c: round(psi(reference[c], current[c]), 4) for c in config.NUMERIC_FEATURES}
    model = joblib.load(config.MODEL_PATH)
    score_ref = model.predict_proba(reference[config.FEATURES])[:, 1]
    score_cur = model.predict_proba(current[config.FEATURES])[:, 1]
    score_psi = round(psi(score_ref, score_cur), 4)

    result = {
        "scenario": scenario,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "dataset_drift": summary["drift_share"] >= 0.5,
        "drift_share_threshold": 0.5,
        **summary,
        "psi_numeric": feature_psi,
        "score_psi": score_psi,
        "score_psi_status": psi_label(score_psi),
        "mean_pd_reference": round(float(score_ref.mean()), 4),
        "mean_pd_current": round(float(score_cur.mean()), 4),
        "html_report": html_path.name,
    }
    (config.REPORTS_DIR / "drift_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), "utf-8"
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run()
    logger.info(
        "Drift : %s colonnes / %s (share=%.2f), PSI score=%.3f",
        result["n_drifted_columns"],
        len(config.FEATURES),
        result["drift_share"],
        result["score_psi"],
    )


if __name__ == "__main__":
    main()
