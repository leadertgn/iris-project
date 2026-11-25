import json

import joblib
import pandas as pd

from . import config

_pipeline = joblib.load(config.MODEL_PATH)

with open(config.MODEL_METADATA_PATH, "r", encoding="utf-8") as file:
    _metadata = json.load(file)


def predict(measures: dict) -> dict:
    missing = set(_metadata["features_names"]) - set(measures)
    if missing:
        raise ValueError(f"Mesures manquantes : {sorted(missing)}")
    X = pd.DataFrame([measures], columns=_metadata["features_names"])
    proba = _pipeline.predict_proba(X)[0]
    classes = _pipeline.classes_

    return {
        "species": str(classes[proba.argmax()]),
        "probabilities": {str(c): float(p) for c, p in zip(classes, proba)},
    }
