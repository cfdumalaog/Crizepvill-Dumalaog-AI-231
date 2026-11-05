from __future__ import annotations

from me3.agent import LocalVoiceLoop
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


def test_default_profile_uses_protected_playback_and_command_control():
    cfg = load_config()
    assert cfg["agent"]["barge_in"] is False
    assert cfg["agent"]["command_control"] is True
