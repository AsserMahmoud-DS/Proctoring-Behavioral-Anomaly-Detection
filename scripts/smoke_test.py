"""Check that a running CheatDetect API buffers events and scores a window."""

import argparse
import json
import os
from urllib.request import Request, urlopen
from uuid import uuid4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="Base URL of the running API")
    args = parser.parse_args()

    url = f"{args.base_url.rstrip('/')}/predict"
    headers = {"Content-Type": "application/json"}
    if token := os.getenv("SMOKE_TEST_ID_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"

    session_id = uuid4().hex
    for batch, expected_chunks in enumerate((0, 1), start=1):
        start = (batch - 1) * 25
        events = [
            {
                "time": i * 0.2,
                "event_type": "mousemove",
                "x": 100 + i % 13,
                "y": 80 + i % 7,
            }
            for i in range(start, start + 25)
        ]
        request = Request(
            url,
            data=json.dumps({"session_id": session_id, "events": events}).encode(),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            result = json.load(response)

        actual = len(result["chunks_pred"])
        if result["session_id"] != session_id or actual != expected_chunks:
            raise SystemExit(
                f"Batch {batch}: expected session {session_id} and "
                f"{expected_chunks} scored windows, got {result}"
            )
        print(f"Batch {batch}: {actual} scored window(s)")


if __name__ == "__main__":
    main()
