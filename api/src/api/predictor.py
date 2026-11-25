import json

import joblib
import pandas as pd


class Predictor:
    def __init__(self, model_path, metadata_path):
        self.pipeline = joblib.load(model_path)
        with open(metadata_path, encoding="utf-8") as f:
            self.features = json.load(f)["features_names"]

    def predict(self, measures: dict) -> dict:
        missing = set(self.features) - set(measures)
        if missing:
            raise ValueError(f"Mesures manquantes : {sorted(missing)}")
        X = pd.DataFrame([measures], columns=self.features)
        proba = self.pipeline.predict_proba(X)[0]
        classes = self.pipeline.classes_
        return {
            "species": str(classes[proba.argmax()]),
            "probabilities": {str(c): float(p) for c, p in zip(classes, proba)},
        }
