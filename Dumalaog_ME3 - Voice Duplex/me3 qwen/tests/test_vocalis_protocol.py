from __future__ import annotations

import time
from collections import deque

import numpy as np

from me3.agent import LocalVoiceLoop, TurnStats
from me3.audio_io import InjectedDuplexAudio
from me3.config import load_config
from me3.vocalis import EventChannel, TTSAction, TurnState, VocalisOrchestrator


def channels(agent):
    return [event.channel for event in agent.events]


def test_vocalis_starts_always_listening_and_streams_transcript_to_thinking():
    agent = VocalisOrchestrator()
    assert agent.state is TurnState.LISTENING
    agent.on_user_vad(True, probability=0.91)
    agent.on_transcript_delta("build a web app", final=False)
    agent.on_transcript_delta("build a web app", final=True)
    assert agent.state is TurnState.THINKING
    assert EventChannel.USER_VAD.value in channels(agent)
    assert EventChannel.USER_TRANSCRIPT.value in channels(agent)


def test_vocalis_barge_in_flushes_and_fades_tts_before_listening():
    agent = VocalisOrchestrator()
    agent.on_assistant_text_delta("Hello", final=False)
    assert agent.state is TurnState.SPEAKING
    agent.on_user_vad(True, probability=0.88)
    assert agent.state is TurnState.LISTENING
    controls = [event.payload["action"] for event in agent.events
                if event.channel == EventChannel.TTS_CONTROL.value]
    assert controls == [TTSAction.PLAY.value, TTSAction.FLUSH_QUEUE.value,
                        TTSAction.FADE_OUT.value]
    assert any(event.channel == EventChannel.BARGE_IN.value
               for event in agent.events)


def test_vocalis_emits_json_rpc_harness_dispatch_and_async_result():
    received = []

    def dispatch(payload):
        received.append(payload)
        return {"ok": True, "task_id": "task-1"}

    agent = VocalisOrchestrator(harness_dispatch=dispatch)
    payload = agent.dispatch_harness(
        task="build a web app",
        arguments={"path": "agent_workspace/demo"},
    )
    assert payload["jsonrpc"] == "2.0"
    assert payload["method"] == "tools/call"
    assert payload["params"]["name"] == "harness_dispatch"
    assert received == [payload]
    assert any(event.channel == EventChannel.HARNESS_DISPATCH.value
               for event in agent.events)
    assert any(event.channel == EventChannel.HARNESS_EVENT.value
               and event.payload["phase"] == "completed"
               for event in agent.events)


def test_local_voice_loop_bridges_qwen_tool_progress_without_audio_hardware():
    callbacks = []

    def original_callback(kind, name, payload):
        callbacks.append((kind, name))

    class DummyBrain:
        name = "dummy"

        def __init__(self):
            self.on_tool = original_callback

    cfg = load_config()
    audio = InjectedDuplexAudio(
        sample_rate=int(cfg["audio"]["sample_rate"]),
        frame_ms=int(cfg["audio"]["frame_ms"]),
    )
    loop = LocalVoiceLoop(cfg, brain=DummyBrain(), audio=audio)
    loop.brain.on_tool("start", "write_file", {"path": "redacted"})
    loop.brain.on_tool("finish", "write_file", {"ok": True})

    assert callbacks == [("start", "write_file"), ("finish", "write_file")]
    progress = [event.payload for event in loop.orchestrator.events
                if event.channel == EventChannel.HARNESS_EVENT.value]
    assert progress == [
        {"phase": "tool_started", "name": "write_file"},
        {"phase": "tool_finished", "name": "write_file", "ok": True},
    ]
    audio.close()


def test_barge_in_preserves_an_accepted_background_task():
    """Barge-in stops speech, not the already accepted tool request."""
    cfg = load_config()
    audio = InjectedDuplexAudio(
        sample_rate=int(cfg["audio"]["sample_rate"]),
        frame_ms=int(cfg["audio"]["frame_ms"]),
    )

    class DummyBrain:
        name = "dummy"

    loop = LocalVoiceLoop(cfg, brain=DummyBrain(), audio=audio)
    marker = object()
    loop._pending.append((marker, {"reply": "finished"}, 0.0, 0.0,
                          "build the requested app"))

    loop._preserve_pending()

    assert len(loop._pending) == 1
    assert loop._pending[0][1]["reply"] == "finished"
    audio.close()


def test_protected_playback_recognizes_explicit_voice_controls_only():
    assert LocalVoiceLoop._voice_command("stop processing") == "stop"
    assert LocalVoiceLoop._voice_command("Skip this") == "skip"
    assert LocalVoiceLoop._voice_command("steer toward the latest request") == "steer"
    assert LocalVoiceLoop._voice_command("please keep explaining") is None


def test_action_clarification_is_marked_as_a_followup_but_generic_greeting_is_not():
    assert LocalVoiceLoop._is_clarification_reply(
        "What message would you like to send, and via which channel?"
    )
    assert not LocalVoiceLoop._is_clarification_reply(
        "Good morning. How can I help you today?"
    )


def test_followup_input_is_explicitly_bound_to_pending_action():
    loop = LocalVoiceLoop.__new__(LocalVoiceLoop)
    loop._pending_followup = {"question": "What message should I send?"}
    prepared = loop._prepare_brain_input("good morning")
    assert "FOLLOW-UP TO THE PREVIOUS ASSISTANT QUESTION" in prepared
    assert "good morning" in prepared
    assert loop._pending_followup is None


def test_default_profile_uses_full_duplex_barge_in_and_command_control():
    cfg = load_config()
    assert cfg["audio"]["full_duplex"] is True
    assert cfg["agent"]["barge_in"] is True
    assert cfg["agent"]["command_control"] is True
    assert cfg["agent"]["protected_playback_ignore_noncontrol"] is True


def test_automatic_barge_in_stops_tts_and_starts_new_turn():
    loop = LocalVoiceLoop.__new__(LocalVoiceLoop)
    events = []
    loop.state = "speaking"
    loop.vad_thr = 0.5
    loop.barge_in = True
    loop.command_control = True
    loop._control_uttr = []
    loop.stats = TurnStats()
    loop.status = None
    loop.orchestrator = type("Orchestrator", (), {
        "on_user_vad": lambda _self, speech, **_kwargs: events.append(speech),
    })()
    loop.audio = type("Audio", (), {
        "stop_playback": lambda _self: events.append("stop"),
    })()
    loop._preserve_pending = lambda: events.append("preserve")
    loop._start_turn = lambda _t: events.append("start")

    loop._on_vad_frame(0.9, np.zeros(480, dtype=np.int16))

    assert loop.stats.n_barge_in == 1
    assert events == [True, "stop", "preserve", "start"]


def test_recent_me3_output_is_recognized_as_echo():
    loop = LocalVoiceLoop.__new__(LocalVoiceLoop)
    loop._spoken_transcripts = deque(
        [("i am processing your request now", time.perf_counter() + 10)],
        maxlen=8,
    )
    assert loop._is_recent_spoken_echo("I am processing your request now.")


def _protected_loop_for_echo_test(rejected):
    loop = LocalVoiceLoop.__new__(LocalVoiceLoop)
    loop.ignore_noncontrol_during_playback = True
    loop._spoken_transcripts = deque(maxlen=8)
    loop._transcribe_control_audio = lambda _audio: ("this is speaker echo", 1.0)
    loop._voice_command = LocalVoiceLoop._voice_command
    loop.orchestrator = type("Orchestrator", (), {
        "on_transcript_delta": lambda *_args, **_kwargs: None,
    })()
    loop.status = None
    loop.audio = type("Audio", (), {
        "capture": type("Capture", (), {
            "note_transcript_echo_rejected": lambda _self: rejected.append(True),
        })(),
    })()
    loop.stats = type("Stats", (), {"n_control": 0})()
    loop._deferred_text = None
    return loop


def test_real_overlap_is_queued_during_protected_playback():
    loop = _protected_loop_for_echo_test([])
    loop._transcribe_control_audio = lambda _audio: ("build the requested website", 1.0)

    loop._process_protected_utterance(
        np.zeros(480, dtype=np.int16), 0.0, echo_suspect_frames=0,
        captured_frames=10)

    assert loop._deferred_text == "build the requested website"


def test_echo_overlap_is_rejected_during_protected_playback():
    rejected = []
    loop = _protected_loop_for_echo_test(rejected)

    loop._process_protected_utterance(
        np.zeros(480, dtype=np.int16), 0.0, echo_suspect_frames=8,
        captured_frames=10)

    assert loop._deferred_text is None
    assert rejected == [True]
