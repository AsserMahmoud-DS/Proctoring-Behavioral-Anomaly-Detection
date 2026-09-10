from cheatdetect.app.schemas import (Event, PredictionRequest, ChunkPrediction, 
                                     PredictionResponse, InferenceConfig)
import pytest
from pydantic import ValidationError

class TestEvent:
    def test_valid_event_all_fields(self):
        event = Event(x = 2, y = 3, time = 10, event_type = 'click',
                     action='copy') 

        assert event.x == 2
        assert event.y == 3
        assert event.time == 10
        assert event.event_type == "click"
        assert event.action == "copy"

    def test_rejects_negative_time(self):
        with pytest.raises(ValidationError):
            Event(time=-5, event_type='click')

    def test_valid_event_required_only(self):
        event = Event(time=0.0, event_type='keydown')
        assert event.time == 0.0
        assert event.event_type == 'keydown'
        assert event.x is None
        assert event.y is None
        assert event.action is None

    def test_extra_fields_ignored(self):
        event = Event(time=1.0, event_type='click', nonsense='hello')
        assert not hasattr(event, 'nonsense')


class TestPredictionRequest:
    def test_valid_request(self):
        req = PredictionRequest(
            session_id='s1',
            events=[Event(time=0, event_type='click')],
        )
        assert req.session_id == 's1'
        assert len(req.events) == 1

    def test_empty_events_rejected(self):
        with pytest.raises(ValidationError):
            PredictionRequest(session_id='s1', events=[])


class TestChunkPrediction:
    def test_valid_chunk(self):
        chunk = ChunkPrediction(chunk_index=0, score=0.5, isanomalous=True)
        assert chunk.chunk_index == 0
        assert chunk.score == 0.5
        assert chunk.isanomalous is True


class TestPredictionResponse:
    def test_valid_response(self):
        resp = PredictionResponse(
            session_id='s1', verdict='anomalous', chunks_pred=[]
        )
        assert resp.verdict == 'anomalous'

    def test_invalid_verdict_rejected(self):
        with pytest.raises(ValidationError):
            PredictionResponse(session_id='s1', verdict='wrong', chunks_pred=[])


class TestInferenceConfig:
    def test_valid_config(self):
        config = InferenceConfig(
            model='IF',
            chunk_size=50,
            step_size=25,
            cheating_threshold=0.5,
            threshold=-0.1,
            features_to_keep=['elapsed_time'],
        )
        assert config.chunk_size == 50
        assert config.model == 'IF'

    def test_rejects_zero_chunk_size(self):
        with pytest.raises(ValidationError):
            InferenceConfig(
                model='IF',
                chunk_size=0,
                step_size=25,
                cheating_threshold=0.5,
                threshold=-0.1,
                features_to_keep=['elapsed_time'],
            )

    def test_extra_fields_ignored(self):
        config = InferenceConfig(
            model='IF',
            chunk_size=50,
            step_size=25,
            cheating_threshold=0.5,
            threshold=-0.1,
            features_to_keep=['elapsed_time'],
            nonsense='hello',
        )
        assert not hasattr(config, 'nonsense')