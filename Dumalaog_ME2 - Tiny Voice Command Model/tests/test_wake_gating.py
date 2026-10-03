from collections import deque
from types import SimpleNamespace

from vcm_app import (
    MAX_SPEECH_TIMEOUT_GRACE_SEC,
    VCMAssistant,
    accepts_intent_prediction,
)


def _listening_assistant(deadline):
    assistant = object.__new__(VCMAssistant)
    assistant.state = "LISTENING"
    assistant.inactivity_deadline = deadline
    assistant.timeout_sec = 10.0
    assistant.wake_streak = 0
    assistant.ignore_current_utterance = False
    assistant.command_armed = True
    assistant.command_consumed = False
    assistant.pending_command = False
    assistant.speech_active = False
    assistant.devices = SimpleNamespace(ducked=True, message="Listening")
    assistant.events = deque(maxlen=5)
    return assistant


def test_active_utterance_is_not_cut_off_at_inactivity_deadline():
    assistant = _listening_assistant(deadline=10.0)
    assistant.speech_active = True

    expired = assistant._expire_session_if_due(now=10.01)

    assert expired is False
    assert assistant.state == "LISTENING"
    assert assistant.devices.ducked is True


def test_pending_utterance_gets_a_command_inference_opportunity():
    assistant = _listening_assistant(deadline=10.0)
    assistant.pending_command = True

    expired = assistant._expire_session_if_due(now=10.01)

    assert expired is False
    assert assistant.state == "LISTENING"


def test_sustained_speech_or_noise_cannot_extend_session_forever():
    assistant = _listening_assistant(deadline=10.0)
    assistant.speech_active = True

    expired = assistant._expire_session_if_due(
        now=10.0 + MAX_SPEECH_TIMEOUT_GRACE_SEC + 0.01
    )

    assert expired is True
    assert assistant.state == "STANDBY"
    assert assistant.devices.ducked is False
    assert assistant.command_armed is False


def test_wake_session_remains_open_until_its_deadline():
    assistant = _listening_assistant(deadline=10.0)
    assistant.speech_active = True

    expired = assistant._expire_session_if_due(now=9.99)

    assert expired is False
    assert assistant.state == "LISTENING"
    assert assistant.inactivity_deadline == 10.0


def test_silence_still_expires_at_the_deadline():
    assistant = _listening_assistant(deadline=10.0)

    expired = assistant._expire_session_if_due(now=10.0)

    assert expired is True
    assert assistant.state == "STANDBY"


def test_each_speech_utterance_queues_at_most_one_command():
    assistant = _listening_assistant(deadline=10.0)

    assistant._on_speech_start()
    assistant._on_utterance_end()
    assert assistant.pending_command is True
    assert assistant.command_armed is False
    assert assistant.command_consumed is True

    # Processing the pending inference does not re-arm on trailing silence.
    assistant.pending_command = False
    assistant._on_utterance_end()
    assert assistant.pending_command is False
    assert assistant.command_armed is False

    # A distinct new speech onset is eligible as the next command.
    assistant._on_speech_start()
    assert assistant.command_armed is True
    assistant._on_utterance_end()
    assert assistant.pending_command is True


def test_live_intent_gate_requires_both_confidence_and_margin():
    assert accepts_intent_prediction(0.68, 0.15)
    assert not accepts_intent_prediction(0.679, 0.90)
    assert not accepts_intent_prediction(0.99, 0.149)
