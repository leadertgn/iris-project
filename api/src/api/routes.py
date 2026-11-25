from fastapi import APIRouter, Depends

from .deps import get_predictor
from .predictor import Predictor
from .schemas import IrisInput, IrisOutput

router = APIRouter(prefix="/predict", tags=["Predict"])

FEATURE_MAP = {
    "sepal_length": "sepal length (cm)",
    "sepal_width": "sepal width (cm)",
    "petal_length": "petal length (cm)",
    "petal_width": "petal width (cm)",
}


@router.post("/", response_model=IrisOutput)
def predict(data: IrisInput, predictor: Predictor = Depends(get_predictor)):
    measures = {FEATURE_MAP[k]: v for k, v in data.model_dump().items()}
    return predictor.predict(measures)
