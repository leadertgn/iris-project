from pathlib import Path



SEED = 42
TEST_SIZE = 0.2 
MODEL_PARAMS = {'C': 0.1,
                 'gamma': 'scale',
                   'kernel': 'linear'
                                                         }

BASE_DIR = Path(__file__).resolve().parents[1] 

DATA_PATH = BASE_DIR / "data" / "raw" / "iris.csv"

TARGET = "species"


MODELS_DIR = BASE_DIR / "models"
MODEL_VERSION = 1
MODEL_PATH = MODELS_DIR / f"iris_svc_model_v{(MODEL_VERSION)}.joblib"
MODEL_METADATA_PATH = MODELS_DIR / f"iris_svc_metrics_v{(MODEL_VERSION)}.json"

