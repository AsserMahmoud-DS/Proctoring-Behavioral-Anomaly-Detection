from cheatdetect.app.buffer import WindowBuffer
from cheatdetect.app.schemas import Event

class TestBuffer:
    def test_return_empty_less_than_chunksize(self):
        window_buffer = (WindowBuffer(chunk_size=4, step_size=2))
        events = [Event(x = 2, y = 3, time = 10, event_type = 'click',
                        action='copy')] * 3  

        windows = window_buffer.add_events(events)
        assert windows == []


    def test_return_multiple_windows(self):
        """Tests windows greater than chunk size are buffered, with sliding logic that leads to correct length """
        window_buffer = (WindowBuffer(chunk_size=4, step_size=2))
        events = [Event(x = 2, y = 3, time = 10, event_type = 'click',
                        action='copy')] * 8
        
        windows = window_buffer.add_events(events)
        assert len(windows) == 3


    def test_state_accross_calls(self):
        window_buffer = (WindowBuffer(chunk_size=4, step_size=2))
        # less than chunksize, no windows yet
        events_1 = [Event(x = i, y = i+1, time = i+2, event_type = 'click',
                        action='copy') for i in range(3)]
        windows = window_buffer.add_events(events_1)
        assert windows == []

        # greater than chunk size after the new events arrive
        events_2 = [Event(x = i+1, y = i+2, time = i+4, event_type = 'click',
                        action='copy') for i in range(3)]
        windows = window_buffer.add_events(events_2)
        assert len(windows) == 2

    def test_overlapping_windows(self):
        """tests if windows returned are overlapped correctly with stride"""
        window_buffer = (WindowBuffer(chunk_size=4, step_size=2))
        events = [Event(x = 2, y = 3, time = 10, event_type = 'click',
                        action='copy')] * 6

        windows = window_buffer.add_events(events)

        assert windows[0][2:4] == windows[1][0:2] 
        
    

