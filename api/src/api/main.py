from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .predictor import Predictor
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor(settings.model_path, settings.metadata_path)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="API du Projet Iris (FastApi)",
    version="1.0.0",
    description="Serveur exposant le modèle conçu dans le cadre du renforcement des connaissances acquis lors du cours d'apprentissage supervisée avec scikit-learn",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_headers=["*"],
    allow_methods=["*"],
)


@router.get("/")
def root():
    return {"message": "Serveur contacté ! "}


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "predictor")}


app.include_router(router)
