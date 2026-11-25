from fastapi import Request

from .predictor import Predictor


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor
