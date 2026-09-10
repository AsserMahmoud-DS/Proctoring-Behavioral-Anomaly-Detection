import pytest
from fastapi.testclient import TestClient

from cheatdetect.app.app import app
from cheatdetect.app.dependencies import get_predictor
from cheatdetect.app.schemas import InferenceConfig
from cheatdetect.app.service import PredictionService
from test_service import FakeDetector, make_events


def fake_get_predictor():
    return PredictionService(
        detector=FakeDetector([0.0, 0.0, 0.0]),
        config=InferenceConfig(
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
        ),
    )


@pytest.fixture(autouse=True)
def override_get_predictor():
    app.dependency_overrides[get_predictor] = fake_get_predictor
    yield
    app.dependency_overrides.clear()


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_200():
    resp = client.post("/predict", json={
        "session_id": "test",
        "events": [{"time": 0, "event_type": "click"}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "test"
    assert data["verdict"] == "normal"
    assert isinstance(data["chunks_pred"], list)


def test_empty_events_returns_422():
    resp = client.post("/predict", json={
        "session_id": "test",
        "events": [],
    })
    assert resp.status_code == 422


def test_missing_session_id_returns_422():
    resp = client.post("/predict", json={
        "events": [{"time": 0, "event_type": "click"}],
    })
    assert resp.status_code == 422
