import numpy as np
import pytest

from cheatdetect.app.schemas import Event, InferenceConfig, PredictionRequest, PredictionResponse, ChunkPrediction
from cheatdetect.app.service import PredictionService
from cheatdetect.models.base import AnomalyDetector


class FakeDetector(AnomalyDetector):
    """Returns predefined scores, ignoring input data."""

    def __init__(self, scores: list[float]):
        self._scores = iter(scores)

    def fit(self, X):
        return self

    def decision_function(self, X):
        return np.array([next(self._scores) for _ in range(len(X))])


@pytest.fixture
def fake_config():
    return InferenceConfig(
        model="fake",
        chunk_size=4,
        step_size=2,
        cheating_threshold=0.5,
        threshold=0.0,
        features_to_keep=[
            "elapsed_time",
            "mouse_path_length",
            "mouse_click_count",
            "keyboard_typing_rate",
            "window_switch_events",
        ],
    )


def make_events(n: int) -> list[Event]:
    """Generate n mousemove events for testing."""
    return [
        Event(time=float(i), event_type="mousemove", x=float(i), y=float(i))
        for i in range(n)
    ]

class TestPredictionService:
    def test_less_than_chunk_request(self, fake_config):
        service = PredictionService(detector = FakeDetector([0.0]), config=fake_config)

        prediction = service.predict(PredictionRequest(session_id = "123", events = [
                        Event(x = 2, y = 3, time = 10, event_type = 'click',
                        action='copy')]
                        ))
        
        assert prediction == PredictionResponse(session_id="123", verdict="normal", chunks_pred=[])

    def test_anomalous_chunks(self, fake_config):
        service = PredictionService(detector = FakeDetector([1.0, 1.0, 1.0]), config=fake_config)

        prediction = service.predict(PredictionRequest(session_id = "123", events = [
                        Event(x = 2, y = 3, time = 10, event_type = 'click',
                        action='copy')] * 8
                        ))
        assert prediction == PredictionResponse(session_id="123", verdict="anomalous", chunks_pred=
                                                [ChunkPrediction(chunk_index = i, score = 1, isanomalous = True) for i in range(3)])


    def test_normal_chunks(self, fake_config):
        service = PredictionService(detector = FakeDetector([0.0, 0.0, 0.0]), config=fake_config)

        prediction = service.predict(PredictionRequest(session_id = "123", events = [
                        Event(x = 2, y = 3, time = 10, event_type = 'click',
                        action='copy')] * 8
                        ))
        assert prediction == PredictionResponse(session_id="123", verdict="normal", chunks_pred=
                                                [ChunkPrediction(chunk_index = i, score = 0, isanomalous = False) for i in range(3)])


    def test_mixed_sessions(self, fake_config):
        service = PredictionService(detector = FakeDetector([0.0, 1.0, 0.0]), config=fake_config)

        prediction = service.predict(PredictionRequest(session_id = "123", events = [
                        Event(x = 2, y = 3, time = 10, event_type = 'click',
                        action='copy')] * 8
                        ))
        assert prediction == PredictionResponse(session_id="123", verdict="anomalous", chunks_pred=
                                                [ChunkPrediction(chunk_index = 0, score = 0, isanomalous = False),
                                                 ChunkPrediction(chunk_index = 1, score = 1, isanomalous = True),
                                                 ChunkPrediction(chunk_index = 2, score = 0, isanomalous = False)])

    def test_service_buffering_calls(self, fake_config):
        service = PredictionService(detector = FakeDetector([0.0]), config=fake_config)

        req1 = service.predict(PredictionRequest(session_id = "123", events = [
                        Event(x = 2, y = 3, time = 10, event_type = 'click',
                        action='copy')]
                        ))                    # first request less than chunk_size, returns default empty
        
        assert req1 == PredictionResponse(session_id="123", verdict="normal", chunks_pred=[])


        req2 = service.predict(PredictionRequest(session_id = "123", events = [
                        Event(x = 2, y = 3, time = 10, event_type = 'click',
                        action='copy')] * 3
                        ))                    # second request complements the first, leading to a chunksize of events, returns 1 prediction

        assert req2 == PredictionResponse(session_id="123", verdict="normal", chunks_pred=[
            ChunkPrediction(chunk_index = 0, score = 0, isanomalous = False)
        ])
