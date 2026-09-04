from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class Event(BaseModel):
    model_config = ConfigDict(extra = "ignore")
    x: float | None = None
    y: float | None = None
    time: float = Field(ge=0)
    event_type: str 
    action: str | None = None

class PredictionRequest(BaseModel):
    session_id: str
    events: list[Event] = Field(min_length=1, description="events fed into the classification model" \
    " of at least n=chunk_size")

class ChunkPrediction(BaseModel):
    chunk_index: int
    score: float
    isanomalous: bool

class PredictionResponse(BaseModel):
    session_id: str
    verdict: Literal["anomalous" , "normal"]
    chunks_pred: list[ChunkPrediction] = Field(description="preds of n chunks sent, in order") # 0 normal 1 anomalous

class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra = "ignore")
    model: str
    chunk_size: int = Field(gt=0)
    step_size: int = Field(gt=0)
    cheating_threshold: float  # Percentage of anomalous events to assume the chunk as anomalous
    decision_threshold: float  # The threshold on the model's score to decide anomalies if below
    features_to_keep: list[str]

