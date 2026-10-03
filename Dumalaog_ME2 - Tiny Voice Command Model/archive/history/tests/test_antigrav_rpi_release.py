import hashlib
import io
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from antigrav_demo import AntigravAssistant, AntigravPredictor
from tinyvcm.devices import Devices, fetch_live_weather
from tinyvcm_antigrav.config import LABELS, SAMPLES, WAKE_LABELS


def test_current_scratch_and_wake_models_have_exact_output_maps():
    for filename, expected in (
        ("antigrav_optionb_int8.onnx", LABELS),
        ("antigrav_wake32_int8.onnx", WAKE_LABELS),
    ):
        predictor = AntigravPredictor(ROOT / "models" / filename)
        result = predictor.predict(np.zeros(SAMPLES, dtype=np.float32))
        assert predictor.labels == expected
        assert len(result["top3"]) == 3
        assert np.isfinite(result["confidence"])
        assert 0.0 <= result["confidence"] <= 1.0
        if len(expected) == 32:
            assert 0.0 <= result["wake_probability"] <= 1.0
        else:
            assert result["wake_probability"] is None


def test_weather_action_stays_local(monkeypatch):
    def must_not_call_network(*args, **kwargs):
        raise AssertionError("Offline VCM weather action attempted a network call")

    monkeypatch.setattr("tinyvcm.devices.fetch_live_weather", must_not_call_network)
    assert "offline" in Devices().execute("WEATHER").lower()
    assert "no live weather" in Devices().execute("question_weather").lower()


def test_live_weather_intents_call_provider_only_when_opted_in(monkeypatch):
    calls = []

    def fake_weather():
        calls.append(True)
        return "Live weather from a mocked provider."

    monkeypatch.setattr("tinyvcm.devices.fetch_live_weather", fake_weather)
    devices = Devices(live_weather=True)
    assert devices.execute("WEATHER") == "Live weather from a mocked provider."
    assert devices.execute("question_weather") == "Live weather from a mocked provider."
    assert len(calls) == 2


def test_live_weather_uses_current_api_values(monkeypatch):
    def fake_urlopen(request, timeout):
        assert request.full_url.startswith("https://api.open-meteo.com/v1/forecast?")
        assert "current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m" in request.full_url
        assert timeout == 3
        return io.BytesIO(b'{"current":{"temperature_2m":32.8,"relative_humidity_2m":57,"weather_code":1,"wind_speed_10m":6.9}}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = fetch_live_weather()
    assert "32.8°C" in result
    assert "57% humidity" in result
    assert "Mainly clear" in result
    assert "6.9 km/h" in result
    assert "live Open-Meteo API" in result


def test_live_weather_failure_never_invents_a_cached_report(monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("connection unavailable")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    result = fetch_live_weather()
    assert "unavailable" in result.lower()
    assert "cached" not in result.lower()
    assert "°C" not in result


def test_live_weather_rejects_missing_current_values(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: io.BytesIO(b'{"current":{"temperature_2m":32.8,"weather_code":1}}'),
    )
    result = fetch_live_weather()
    assert "unavailable" in result.lower()
    assert "°C" not in result


def test_live_weather_lookup_does_not_block_audio_loop(monkeypatch):
    entered = threading.Event()
    release_provider = threading.Event()

    def blocked_weather():
        entered.set()
        if not release_provider.wait(timeout=5):
            raise AssertionError("Mock weather provider was never released")
        return "Diliman, QC: mocked live weather."

    monkeypatch.setattr("tinyvcm.devices.fetch_live_weather", blocked_weather)
    assistant = AntigravAssistant(live_weather=True)
    try:
        started = time.perf_counter()
        initial_message = assistant.start_weather_lookup("WEATHER", 0.95, 0.4)
        elapsed = time.perf_counter() - started
        assert initial_message == "Checking live weather..."
        assert elapsed < 1.0
        assert entered.wait(timeout=2)
        assert assistant.weather_lock.locked()
        assert assistant.start_weather_lookup("WEATHER", 0.95, 0.4) == "Live weather lookup already in progress."
    finally:
        release_provider.set()

    assert assistant.weather_lock.acquire(timeout=2)
    assistant.weather_lock.release()
    assert "Diliman, QC: mocked live weather." in assistant.events[0]
    assert assistant.devices.message == "Diliman, QC: mocked live weather."


def test_default_predictor_selects_the_wake_capable_model():
    predictor = AntigravPredictor()
    assert Path(predictor.model_path).name == "antigrav_wake32_int8.onnx"
    assert predictor.labels == WAKE_LABELS


def test_missing_model_fails_with_a_direct_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model not found"):
        AntigravPredictor(tmp_path / "missing.onnx")


def test_antigrav_cli_exposes_safe_host_and_model_selection():
    result = subprocess.run(
        [sys.executable, str(ROOT / "antigrav_demo.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--host" in result.stdout
    assert "--model" in result.stdout
    assert "--binary-wake-model" in result.stdout
    assert "--wake-threshold" in result.stdout
    assert "--timeout" in result.stdout


def test_release_builder_creates_verified_current_bundle(tmp_path):
    output = tmp_path / "bundle"
    archive = tmp_path / "bundle.zip"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "deployment" / "package_antigrav_rpi.py"),
            "--output-dir",
            str(output),
            "--archive",
            str(archive),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (output / "bundle_manifest.json").is_file()
    assert (output / "models" / "antigrav_optionb_int8.onnx").stat().st_size == 39315
    assert (output / "models" / "antigrav_wake32_int8.onnx").stat().st_size == 39384
    assert archive.is_file() and archive.stat().st_size > 0

    # Rebuilding without --overwrite into an existing path must fail and preserve output.
    before = archive.stat().st_size
    repeated = subprocess.run(
        [sys.executable, str(ROOT / "deployment" / "package_antigrav_rpi.py"), "--output-dir", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0
    assert archive.stat().st_size == before


def test_assistant_timeout_defaults_to_ten_seconds():
    assistant = AntigravAssistant()
    assert assistant.timeout_sec == 10.0
    snap = assistant.snapshot()
    assert snap["timeout_sec"] == 10.0
    assert snap["state"] == "STANDBY"
    assert snap["classes_count"] == len(assistant.predictor.labels)
    assert snap["classes_count"] == 32


def test_assistant_state_machine_stays_listening_before_ten_seconds_and_times_out():
    assistant = AntigravAssistant(timeout_sec=10.0)
    assert assistant.state == "STANDBY"

    # Manual wake triggers LISTENING with a full 10-second deadline
    assistant.trigger_wake(source="test_runner")
    assert assistant.state == "LISTENING"
    assert assistant.inactivity_deadline > time.monotonic()

    # Before 10s: snapshot confirms LISTENING and remaining seconds scaled to 10s
    now = time.monotonic()
    remaining = assistant.inactivity_deadline - now
    assert 9.0 <= remaining <= 10.1
    snap = assistant.snapshot()
    assert snap["state"] == "LISTENING"
    assert snap["remaining_sec"] >= 9.0

    # Advance deadline to the past to simulate 10 seconds elapsed
    assistant.inactivity_deadline = time.monotonic() - 0.1
    assistant.trigger_timeout()
    assert assistant.state == "STANDBY"
    snap_timeout = assistant.snapshot()
    assert snap_timeout["state"] == "STANDBY"
    assert snap_timeout["remaining_sec"] == 0.0


def test_assistant_accepted_command_resets_ten_second_deadline():
    assistant = AntigravAssistant(timeout_sec=10.0)
    assistant.trigger_wake(source="test_runner")
    t0 = time.monotonic()

    # Artificially decay deadline so only 3.0s remain before timeout
    assistant.inactivity_deadline = t0 + 3.0
    initial_remaining = assistant.inactivity_deadline - t0
    assert 2.5 <= initial_remaining <= 3.5

    # Supply a complete command utterance followed by enough silence to end it.
    audio_chunk = np.ones(1600, dtype=np.float32) * 0.05
    silence_chunk = np.zeros_like(audio_chunk)
    feed_count = 0

    def mock_read():
        nonlocal feed_count
        sequence = [audio_chunk, silence_chunk, silence_chunk, silence_chunk, silence_chunk]
        if feed_count >= len(sequence):
            assistant.stop_event.set()
            return None
        chunk = sequence[feed_count]
        feed_count += 1
        return chunk

    assistant.mic.read = mock_read

    # Mock predictor to return an accepted command (confidence >= 0.68, margin >= 0.15)
    assistant.predictor.predict = lambda _buf: {
        "label": "WEATHER",
        "confidence": 0.95,
        "margin": 0.80,
        "wake_probability": 0.001,
        "latency_ms": 1.2,
        "top3": {"WEATHER": 0.95, "TIME": 0.03, "PLAY_MUSIC": 0.02},
    }

    # Execute through the actual assistant loop
    assistant.loop()

    # Verify loop state, device dispatch, and that deadline was reset to 10.0s
    assert assistant.state == "LISTENING"
    assert feed_count == 5
    assert len(assistant.events) > 0
    assert any("[WEATHER]" in ev for ev in assistant.events)
    now = time.monotonic()
    remaining = assistant.inactivity_deadline - now
    assert 9.5 <= remaining <= 10.1, f"Expected ~10.0s remaining after command reset, got {remaining:.2f}s"
    snap = assistant.snapshot()
    assert snap["state"] == "LISTENING"
    assert snap["remaining_sec"] >= 9.5


def test_bundle_preserves_both_model_sha256_hashes(tmp_path):
    output = tmp_path / "bundle_hash_check"
    archive = tmp_path / "bundle_hash_check.zip"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "deployment" / "package_antigrav_rpi.py"),
            "--output-dir",
            str(output),
            "--archive",
            str(archive),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    # Compare packaged model hashes against source models/
    for rel_path, expected_hash in (
        ("models/antigrav_optionb_int8.onnx", "cf3c393b99bcbe991baf6c529b18d47c044a3e8c898fa41348762f7be8d24a0a"),
        ("models/antigrav_wake32_int8.onnx", "551836d42f33a34c9a6d13e37e963bcdf82470ba9ae7d71f080941306e351bf3"),
    ):
        src_bytes = (ROOT / rel_path).read_bytes()
        pkg_bytes = (output / rel_path).read_bytes()
        src_hash = hashlib.sha256(src_bytes).hexdigest()
        pkg_hash = hashlib.sha256(pkg_bytes).hexdigest()
        assert src_hash == expected_hash, f"Source model {rel_path} hash changed!"
        assert pkg_hash == expected_hash, f"Packaged model {rel_path} hash does not match expected!"
        assert src_hash == pkg_hash, f"Hash divergence between source and package for {rel_path}!"
