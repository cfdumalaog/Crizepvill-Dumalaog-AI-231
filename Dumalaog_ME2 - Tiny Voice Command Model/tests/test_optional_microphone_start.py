"""Keep the Pi UI/model available so a microphone can be selected later."""
import vcm_app


class FakePredictor:
    binary_wake_enabled = True
    wake_threshold = 0.9
    model_name = "test paired model"
    model_size_kb = 75.0


class FakeDevices:
    live_weather = False
    message = ""

    def __init__(self, **_kwargs):
        pass

    def tick(self):
        pass


class MissingMicrophone:
    stream = None
    device = None
    name = "not started"
    status = ""
    blocks_received = 0
    dropped = 0

    def __init__(self, **_kwargs):
        pass

    def start(self):
        raise RuntimeError("No input device is attached")

    def read(self, timeout=0.3):
        return None

    def close(self):
        pass


class DummyThread:
    def __init__(self, **_kwargs):
        self.started = False

    def start(self):
        self.started = True

    def join(self, timeout=None):
        pass


def test_assistant_start_keeps_ui_available_without_a_microphone(monkeypatch):
    monkeypatch.setattr(vcm_app, "VCMPredictor", lambda **_kwargs: FakePredictor())
    monkeypatch.setattr(vcm_app, "Devices", FakeDevices)
    monkeypatch.setattr(vcm_app, "Microphone", MissingMicrophone)
    monkeypatch.setattr(vcm_app.threading, "Thread", DummyThread)

    assistant = vcm_app.VCMAssistant()
    assistant.start()

    assert assistant.state == "STANDBY"
    assert assistant.worker.started
    assert assistant.mic.name == "No microphone connected"
    assert "No input device" in assistant.mic.status
    assert "Connect or select a microphone" in assistant.devices.message
    assert assistant.error is None

