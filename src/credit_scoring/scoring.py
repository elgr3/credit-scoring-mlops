"""Chemin d'inférence « compilé » pour la latence unitaire.

Le pipeline sklearn (ColumnTransformer + OneHotEncoder) coûte ~7-10 ms par appel sur une
seule ligne, essentiellement en validations pandas. On extrait une fois pour toutes les
modalités apprises par l'encodeur et on construit directement la matrice numpy attendue
par XGBoost. L'équivalence avec le pipeline complet est vérifiée par les tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from credit_scoring import config


class FastScorer:
    def __init__(self, pipeline: Any) -> None:
        preprocess = pipeline.named_steps["preprocess"]
        encoder = preprocess.named_transformers_["cat"]
        if getattr(encoder, "drop", None) is not None:
            raise ValueError("OneHotEncoder avec drop non supporté par FastScorer")

        self.model = pipeline.named_steps["model"]
        self.numeric = list(config.NUMERIC_FEATURES)
        self.categorical = list(config.CATEGORICAL_FEATURES)
        offset = len(self.numeric)
        self.cat_index: list[dict[str, int]] = []
        for categories in encoder.categories_:
            self.cat_index.append({str(c): offset + i for i, c in enumerate(categories)})
            offset += len(categories)
        self.width = offset

    def transform(self, records: Iterable[Mapping[str, Any]]) -> np.ndarray:
        records = list(records)
        x = np.zeros((len(records), self.width), dtype=np.float32)
        for row, record in enumerate(records):
            x[row, : len(self.numeric)] = [record[c] for c in self.numeric]
            for col, mapping in zip(self.categorical, self.cat_index, strict=True):
                idx = mapping.get(str(record[col]))
                if idx is not None:  # modalité inconnue -> vecteur nul (handle_unknown=ignore)
                    x[row, idx] = 1.0
        return x

    def predict_proba(self, records: Iterable[Mapping[str, Any]]) -> np.ndarray:
        """Probabilité de défaut (classe 1) pour chaque dossier."""
        return self.model.predict_proba(self.transform(records))[:, 1]
