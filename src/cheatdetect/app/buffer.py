from .schemas import Event

class WindowBuffer():
    """A rolling window that emits completed windows of chunk_size and shifts by step_size"""
    def __init__(self, chunk_size: int, step_size: int):
        self.chunk_size = chunk_size
        self.step_size = step_size
        self._buffer : list[Event] = []

    def add_events(self, events: list[Event]) -> list[list[Event]]:
        """Add events to the buffer, returning events of completed windows of chunk_size each"""
        self._buffer.extend(events)
        windows = []

        while len(self._buffer) >= self.chunk_size:
            windows.append(self._buffer[:self.chunk_size])
            self._buffer = self._buffer[self.step_size:]

        return windows


