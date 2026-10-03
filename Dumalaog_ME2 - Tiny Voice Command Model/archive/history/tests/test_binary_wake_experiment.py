import sys
from pathlib import Path
import numpy as np
import time
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_binary_wake as experiment
import antigrav_demo
from tinyvcm_antigrav.config import LABELS, SAMPLES


def test_human_recordings_are_split_deterministically_by_group():
    groups = [{"hash": f"clip-{i}", "audio": None, "speaker": "person-01"} for i in range(16)]
    train, val, test = experiment.split_wake_groups(groups, seed=231)
    train2, val2, test2 = experiment.split_wake_groups(groups, seed=231)

    assert (len(train), len(val), len(test)) == (10, 3, 3)
    assert [g["hash"] for g in train] == [g["hash"] for g in train2]
    assert [g["hash"] for g in val] == [g["hash"] for g in val2]
    assert [g["hash"] for g in test] == [g["hash"] for g in test2]
    split_sets = [{g["hash"] for g in split} for split in (train, val, test)]
    assert not (split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2])


def test_validation_threshold_respects_false_accept_cap_and_prioritizes_recall():
    threshold, metrics = experiment.validation_threshold(
        wake_max_probs=[0.9, 0.8, 0.7],
        nonwake_probs=[0.1, 0.8, 0.1, 0.1],
    )

    assert threshold == 0.7
    assert metrics == {"wake_recall": 1.0, "false_count": 1, "false_budget": 1}


def test_calibration_reader_returns_named_inputs_and_resets():
    sample = np.zeros((1, 40, 251), dtype=np.float32)
    reader = experiment.CalibrationReader([sample])

    first = reader.get_next()
    assert set(first) == {"input"}
    assert first["input"].shape == (1, 1, 40, 251)
    assert reader.get_next() is None

    reader.reset()
    assert reader.get_next()["input"].shape == (1, 1, 40, 251)


def test_hybrid_predictor_loads_threshold_and_routes_wake_vs_command(tmp_path, monkeypatch):
    command_path = tmp_path / "command.onnx"
    wake_path = tmp_path / "run" / "models" / "wake.onnx"
    command_path.write_bytes(b"command model placeholder")
    wake_path.parent.mkdir(parents=True)
    wake_path.write_bytes(b"wake model placeholder")
    (wake_path.parent.parent / "export_summary.json").write_text(
        '{"validation_selected_threshold": 0.9751818776130676}', encoding="utf-8"
    )

    class FakeSession:
        def __init__(self, path, _options, providers):
            assert providers == ["CPUExecutionProvider"]
            self.is_wake = Path(path) == wake_path
            self.width = 2 if self.is_wake else len(LABELS)

        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def get_outputs(self):
            return [SimpleNamespace(shape=[None, self.width])]

        def run(self, _outputs, _feed):
            logits = np.zeros((1, self.width), dtype=np.float32)
            if self.is_wake:
                logits[0] = [-4.0, 4.0]
            else:
                logits[0, LABELS.index("WEATHER")] = 4.0
            return [logits]

    monkeypatch.setattr(antigrav_demo.ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(antigrav_demo, "Frontend", lambda: lambda _audio: np.zeros((1, 40, 251), dtype=np.float32))
    predictor = antigrav_demo.AntigravPredictor(command_path, binary_wake_model_path=wake_path)

    wake = predictor.predict(np.zeros(SAMPLES, dtype=np.float32), mode="wake")
    command = predictor.predict(np.zeros(SAMPLES, dtype=np.float32), mode="command")

    assert predictor.binary_wake_enabled is True
    assert predictor.wake_threshold == 0.9751818776130676
    assert wake["label"] == "WAKE_WORD"
    assert wake["wake_probability"] > predictor.wake_threshold
    assert command["label"] == "WEATHER"
    assert command["wake_probability"] is None


def test_assistant_routes_standby_to_binary_wake_then_command():
    assistant = antigrav_demo.AntigravAssistant(timeout_sec=10.0)
    assistant.predictor.binary_wake_enabled = True
    assistant.predictor.wake_threshold = 0.9
    modes = []

    def fake_predict(_buffer, mode):
        modes.append(mode)
        if mode == "wake":
            return {"label": "WAKE_WORD", "confidence": 0.99, "margin": 0.98, "wake_probability": 0.99, "latency_ms": 0.2, "top3": {"WAKE_WORD": 0.99, "NON_WAKE": 0.01}}
        return {"label": "WEATHER", "confidence": 0.95, "margin": 0.80, "wake_probability": None, "latency_ms": 0.3, "top3": {"WEATHER": 0.95, "TIME": 0.03, "PLAY_MUSIC": 0.02}}

    assistant.predictor.predict = fake_predict
    chunk = np.full(1600, 0.05, dtype=np.float32)
    reads = 0

    silence = np.zeros_like(chunk)
    sequence = [chunk, silence, silence, silence, silence, chunk, silence, silence, silence, silence]
    def mock_read():
        nonlocal reads
        if reads >= len(sequence):
            assistant.stop_event.set()
            return None
        result = sequence[reads]
        reads += 1
        if reads == 6:
            # Simulate the normal wake cooldown elapsing before the command turn.
            assistant.cooldown_until = time.monotonic() - 0.1
        return result

    assistant.mic.read = mock_read
    try:
        assistant.loop()
        assert modes == ["wake", "command"]
        assert assistant.inference_count == 2
        assert assistant.state == "LISTENING"
        assert any("[WEATHER]" in event for event in assistant.events)
    finally:
        assistant.close()


def test_command_classifier_waits_for_utterance_end_and_runs_once():
    assistant = antigrav_demo.AntigravAssistant(timeout_sec=10.0)
    assistant.trigger_wake(source="test_runner")
    calls = []
    assistant.predictor.predict = lambda _buffer: (
        calls.append("command") or {
            "label": "LIGHT_OFF", "confidence": 0.82, "margin": 0.61,
            "wake_probability": None, "latency_ms": 0.4, "top3": {},
        }
    )
    voice = np.full(1600, 0.05, dtype=np.float32)
    silence = np.zeros_like(voice)
    sequence = [voice, silence, silence, silence, silence, silence, silence]
    reads = 0

    def mock_read():
        nonlocal reads
        if reads == len(sequence):
            assistant.stop_event.set()
            return None
        chunk = sequence[reads]
        reads += 1
        return chunk

    assistant.mic.read = mock_read
    try:
        assistant.loop()
        assert calls == ["command"]
        assert assistant.inference_count == 1
        assert sum("[LIGHT_OFF]" in event for event in assistant.events) == 1
        assert not any("[STOP]" in event for event in assistant.events)
    finally:
        assistant.close()


def test_two_wake_confirmations_ignore_a_single_high_window():
    assistant = antigrav_demo.AntigravAssistant(wake_confirmations=2, inference_interval_sec=0.12)
    assistant.predictor.binary_wake_enabled = True
    assistant.predictor.wake_threshold = 0.9
    assistant.predictor.predict = lambda _buffer, mode: {
        "label": "WAKE_WORD", "confidence": 0.99, "margin": 0.98,
        "wake_probability": 0.99, "latency_ms": 0.2, "top3": {},
    }
    chunk = np.full(1600, 0.05, dtype=np.float32)
    reads = 0

    def one_read():
        nonlocal reads
        reads += 1
        if reads == 1:
            return chunk
        assistant.stop_event.set()
        return None

    assistant.mic.read = one_read
    try:
        assistant.loop()
        assert assistant.state == "STANDBY"
        assert assistant.wake_streak == 1
        assert assistant.inference_count == 1

        assistant.stop_event.clear()
        assistant.last_inference = 0.0
        assistant.mic.read = lambda: (assistant.stop_event.set(), chunk)[1]
        assistant.loop()
        assert assistant.state == "LISTENING"
        assert assistant.wake_streak == 0
        assert any("[WAKE]" in event for event in assistant.events)
    finally:
        assistant.close()


def test_assistant_keeps_conservative_default_audio_settings():
    assistant = antigrav_demo.AntigravAssistant()
    try:
        state = assistant.snapshot()
        assert state["vad_threshold"] == 0.012
        assert state["inference_interval_sec"] == 0.35
    finally:
        assistant.close()


def test_assistant_exposes_audio_gate_and_inference_diagnostics():
    assistant = antigrav_demo.AntigravAssistant(vad_threshold=0.006, inference_interval_sec=0.12)
    try:
        assistant.rms = 0.0045
        assistant.inference_count = 7
        state = assistant.snapshot()
        assert state["vad_threshold"] == 0.006
        assert state["inference_interval_sec"] == 0.12
        assert state["rms"] == 0.0045
        assert state["inference_count"] == 7
    finally:
        assistant.close()


def test_assistant_rejects_invalid_vad_threshold():
    for value in (-0.1, 1.1, float("nan")):
        try:
            antigrav_demo.AntigravAssistant(vad_threshold=value)
        except ValueError as exc:
            assert "VAD RMS threshold" in str(exc)
        else:
            raise AssertionError(f"Invalid VAD threshold was accepted: {value}")


def test_assistant_rejects_invalid_inference_interval():
    for value in (0.0, -0.1, float("inf")):
        try:
            antigrav_demo.AntigravAssistant(inference_interval_sec=value)
        except ValueError as exc:
            assert "inference interval" in str(exc)
        else:
            raise AssertionError(f"Invalid inference interval was accepted: {value}")
