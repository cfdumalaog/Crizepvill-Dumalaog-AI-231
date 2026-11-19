from __future__ import annotations

import threading

from me3.audio_io import AudioDeviceError
from scripts.me3_pi_server import AgentController


class _Status:
    def __init__(self):
        self.errors = []
        self.reasons = []

    def report_error(self, stage, message):
        self.errors.append((stage, str(message)))

    def report_state_reason(self, reason):
        self.reasons.append(str(reason))


class _RecoveringLoop:
    def __init__(self):
        self.calls = 0
        self._stop_event = threading.Event()

    def run(self):
        self.calls += 1
        if self.calls == 1:
            raise AudioDeviceError("microphone absent")


def test_pi_controller_retries_when_usb_microphone_appears():
    status = _Status()
    loop = _RecoveringLoop()
    controller = AgentController({}, None, status, None)
    controller.MIC_RETRY_S = 0.001
    controller.loop = loop

    controller._run(loop)

    assert loop.calls == 2
    assert status.errors == [("AUDIO", "microphone absent")]
    assert "retrying" in status.reasons[-1]


def test_pi_controller_does_not_retry_unrelated_loop_failure():
    status = _Status()

    class BrokenLoop(_RecoveringLoop):
        def run(self):
            self.calls += 1
            raise RuntimeError("brain failed")

    loop = BrokenLoop()
    controller = AgentController({}, None, status, None)
    controller.MIC_RETRY_S = 0.001
    controller.loop = loop

    controller._run(loop)

    assert loop.calls == 1
    assert status.errors == [("LOOP", "brain failed")]


def test_browser_output_defaults_to_automatic_full_duplex_barge_in():
    controller = AgentController(
        {"agent": {"barge_in": True}}, None, _Status(), None,
        output_mode="browser")
    assert controller.duplex_mode == "automatic"
