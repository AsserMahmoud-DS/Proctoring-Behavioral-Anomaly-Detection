from .schemas import PredictionRequest, PredictionResponse, InferenceConfig, ChunkPrediction
from .buffer import WindowBuffer
from cheatdetect.models import AnomalyDetector
from cheatdetect.data import clean_session_data, merge_window_switch_events, extract_features_from_chunk
import pandas as pd

COLUMN_MAP = {
        "time":"Time (seconds)",
        "event_type":"Event Type",
        "x": "X Coordinate",
        "y": "Y Coordinate",
        "action":"Action"
    }

class PredictionService:

    def __init__(self, detector: AnomalyDetector, config: InferenceConfig):
        self._buffers: dict[str, WindowBuffer]  = {}
        self._detector = detector
        self._config = config

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        chunks_preds = []
        anomaly_flag = 0
        
        buf = self._buffers.setdefault(req.session_id, WindowBuffer(self._config.chunk_size, self._config.step_size))
        windows = buf.add_events(req.events) # returns packed chunks of events
        if not windows:
            return PredictionResponse(session_id=req.session_id, verdict="normal", chunks_pred=[])


        feature_rows = []
        for i, window in enumerate(windows):
            df = pd.DataFrame([e.model_dump() for e in window])
            df = df.rename(columns = COLUMN_MAP)

            df = clean_session_data(df)
            new_features = extract_features_from_chunk(df, self._config.cheating_threshold)
            feature_rows.append(new_features)

        X_all = merge_window_switch_events(pd.DataFrame(feature_rows))
        X = X_all[self._config.features_to_keep]
        scores = self._detector.decision_function(X)

        for i, score in enumerate(scores):
            is_anom = score > self._config.threshold
            chunks_preds.append(ChunkPrediction(chunk_index=i, score=score, isanomalous=is_anom))
            if is_anom:
                anomaly_flag = 1

        verdict="anomalous" if anomaly_flag == 1 else "normal"
        return PredictionResponse(session_id=req.session_id, verdict=verdict, chunks_pred=chunks_preds)