from uuid import uuid4
from locust import HttpUser, task, constant
from cheatdetect.app import PredictionRequest, PredictionService

def make_events(start, count = 25):
    events = [
        {
        "time": i * 0.2,
        "event_type": "mousemove",
        "x": 100 + i % 13,
        "y": 80 + i % 7,
        }
        for i in range(start, start + count)
    ]
    return events    


class StudentUser(HttpUser):
    wait_time = constant(5)

    @task
    def send_batch(self):

        first_batch = self.next_event == 0
        expected_windows = 0 if first_batch else 1
        request_name = "predict: buffer" if first_batch else "predict: score"

        request = PredictionRequest(
            session_id = self.session_id,
            events = make_events(self.next_event, 25)
        )
        self.next_event += len(request.events)

        with self.client.post(
            "/predict",
            name = request_name,
            json = request.model_dump(mode = "json"),
            timeout = 10,
            catch_response=True
        ) as response:
            if response.status_code!=200:
                response.failure(f"Expected HTTP 200, got {response.status_code}")
                return

            try:
                windows = response.json()['chunks_pred']
            except (ValueError, KeyError, TypeError):
                response.failure("Response did not contain a valid chunks_pred field")
                return

            if not isinstance(windows, list) or len(windows) != expected_windows:
                response.failure(f"Expected {expected_windows} scored windows")

    def on_start(self):
        self.next_event = 0
        self.session_id = uuid4().hex
