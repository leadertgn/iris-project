import json
from datetime import datetime

import joblib
import sklearn

from . import config


def export_model_and_metadata(model, metric, features_names):
    metadata = {
        "accuracy": metric, 
        "features_names": features_names,
        "classes_names": model.classes_.tolist(),
        "version": config.MODEL_VERSION,
        "trained_on": datetime.now().isoformat(),
        "scikit-learn_version": sklearn.__version__
    }
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    print(f" Version {config.MODEL_VERSION} sauvegardé avec succès ! ")

    metrics_path = config.MODEL_METADATA_PATH
    with metrics_path.open("w", encoding="utf-8") as file :
        json.dump(metadata, file, indent=4, ensure_ascii=False)
        print("Méta données sauvegardées avec succès !")
    