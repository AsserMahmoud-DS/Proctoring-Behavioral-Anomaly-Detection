from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from .schemas import PredictionRequest, PredictionResponse
from .service import PredictionService
from .dependencies import get_predictor

@asynccontextmanager
async def lifespan(app):
    get_predictor()
    yield

app = FastAPI(title="CheatDetect Inference API", version="0.1.0",lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictionRequest, service: PredictionService = Depends(get_predictor)) -> PredictionResponse:
    prediction = service.predict(req)
    return prediction




