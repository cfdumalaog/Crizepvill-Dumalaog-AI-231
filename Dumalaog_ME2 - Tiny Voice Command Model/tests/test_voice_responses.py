from pathlib import Path

from tinyvcm.devices import Devices
from tinyvcm_model.config import LABELS
from vcm_app import BENCHMARK_INTENT_SLOT, benchmark_log_record


class QuietMedia:
    def get_current_track(self):
        return {"title": "Test track", "artist": "Test artist"}

    def next_track(self):
        return {"title": "Next test track"}

    def play_query(self, _query):
        return {"title": "Test track", "artist": "Test artist"}


def test_every_intent_returns_a_spoken_response():
    devices = Devices(gpio=False, live_weather=False)
    devices.media_engine = QuietMedia()
    responses = {label: devices.execute(label) for label in LABELS}
    assert len(responses) == 31
    assert all(isinstance(message, str) and message.strip() for message in responses.values())
    assert "Lights off" in responses["LIGHT_OFF"]
    assert "red" in responses["COLOR_RED"].lower()
    assert "green" in responses["COLOR_GREEN"].lower()
    assert "blue" in responses["COLOR_BLUE"].lower()


def test_wake_timeout_and_reject_have_spoken_responses():
    devices = Devices(gpio=False)
    assert "listening" in devices.event({"event": "WAKE"}).lower()
    assert "timed out" in devices.event({"event": "TIMEOUT"}).lower()
    assert (
        devices.event({"event": "REJECT", "message": "Command not recognized."})
        == "Command not recognized."
    )


def test_assistant_speaks_on_each_new_event_with_one_locked_voice():
    project = Path(__file__).resolve().parents[1]
    app = (project / "vcm_app.py").read_text(encoding="utf-8")
    assert "speechVoiceLocked" in app
    assert "const latestEvent = Array.isArray(data.events)" in app
    assert "else if (latestEvent !== lastSpeechEvent)" in app
    assert "if (data.message) speakText(data.message);" in app
    assert "else if (messageChanged) speakText(data.message); // timer/alarm and async device updates" in app
    assert "data.message.includes('Lights')" not in app


def test_new_microphone_speech_cancels_current_reply_and_mute_is_immediate():
    project = Path(__file__).resolve().parents[1]
    app = (project / "vcm_app.py").read_text(encoding="utf-8")
    assert "function cancelAssistantSpeech()" in app
    assert "const newUtteranceStarted = active && !lastMicrophoneSpeechActive;" in app
    assert "if (newUtteranceStarted && window.speechSynthesis?.speaking) cancelAssistantSpeech();" in app
    assert "stopSpeechOnMicrophoneActivity(data.speech_active);" in app
    assert "if (!voiceOutputEnabled) cancelAssistantSpeech();" in app
    assert "localStorage.setItem('vcm.voiceOutputEnabled'" in app


def test_every_class_maps_to_the_agreed_benchmark_intents_and_slots():
    assert set(BENCHMARK_INTENT_SLOT) == set(LABELS)
    assert len({intent for intent, _slot in BENCHMARK_INTENT_SLOT.values()}) == 19
    for label in LABELS:
        record = benchmark_log_record(label, {"latency_ms": 5.25}, 2500)
        assert record["intent"] == BENCHMARK_INTENT_SLOT[label][0]
        assert record["infer_ms"] == 5.25
        assert record["audio_ms"] == 2500
        assert ("slot" in record) == (BENCHMARK_INTENT_SLOT[label][1] is not None)
