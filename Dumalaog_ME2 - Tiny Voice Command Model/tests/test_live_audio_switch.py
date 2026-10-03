"""A connected microphone can be changed without reloading either model."""
import pytest

import vcm_app


class Predictor:
    binary_wake_enabled = True
    wake_threshold = 0.95
    model_name = 'test models'
    model_size_kb = 75
    labels = ['intent']


class Devices:
    live_weather = False
    message = ''
    ducked = False

    def __init__(self, **_kwargs):
        pass

    def snapshot(self):
        return {}


class Mic:
    def __init__(self, device=None):
        self.requested_device = device
        self.device = None
        self.stream = None
        self.name = 'not started'
        self.status = ''
        self.blocks_received = self.dropped = 0
        self.start_count = self.close_count = 0

    def start(self):
        self.start_count += 1
        self.device = self.requested_device if self.requested_device is not None else 7
        self.stream = type('Stream', (), {'active': True})()
        self.name = f'mic {self.device}'

    def close(self):
        self.close_count += 1
        self.stream = None


def test_live_selection_keeps_predictor_and_changes_capture(monkeypatch):
    monkeypatch.setattr(vcm_app, 'VCMPredictor', lambda **_: Predictor())
    monkeypatch.setattr(vcm_app, 'Devices', Devices)
    monkeypatch.setattr(vcm_app, 'Microphone', Mic)
    assistant = vcm_app.VCMAssistant(device=2)
    predictor = assistant.predictor
    assistant._try_start_microphone()
    assistant.trigger_wake(source='test')
    result = assistant.select_microphone('5')
    assert result['capture_ready'] is True
    assert assistant.predictor is predictor
    assert assistant.mic.device == 5
    assert assistant.mic.requested_device == 5
    assert assistant.mic.close_count == 1
    assert assistant.state == 'STANDBY'
    assert assistant.snapshot()['microphone_requested_device'] == 5
    assistant.select_microphone('auto')
    assert assistant.mic.device == 7
    assert assistant.mic.requested_device is None


def test_live_selection_rejects_unlisted_input_text(monkeypatch):
    monkeypatch.setattr(vcm_app, 'VCMPredictor', lambda **_: Predictor())
    monkeypatch.setattr(vcm_app, 'Devices', Devices)
    monkeypatch.setattr(vcm_app, 'Microphone', Mic)
    assistant = vcm_app.VCMAssistant()
    with pytest.raises(ValueError, match='listed microphone'):
        assistant.select_microphone('1;whoami')


def test_default_predictor_loads_current_two_model_pair():
    predictor = vcm_app.VCMPredictor()
    assert predictor.binary_wake_enabled
    assert predictor.model_path.name == 'intent_int8.onnx'
    assert predictor.binary_wake_model_path.name == 'binary_wake_int8.onnx'
    assert len(predictor.labels) == 31
    assert predictor.wake_threshold == 0.95
