import json

from scripts.me3_fdbench_agent import FdbenchAudio


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


def test_direct_feed_adapter_sends_stage_marks_over_websocket():
    ws = FakeWebSocket()
    audio = FdbenchAudio(ws)

    audio.send_mark({
        "event": "mark",
        "stage": "asr_final",
        "t": 12.5,
        "text": "test transcript",
    })

    assert len(ws.sent) == 1
    assert json.loads(ws.sent[0]) == {
        "event": "mark",
        "stage": "asr_final",
        "t": 12.5,
        "text": "test transcript",
    }
