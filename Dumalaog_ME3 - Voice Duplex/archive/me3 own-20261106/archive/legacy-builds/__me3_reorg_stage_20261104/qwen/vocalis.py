"""Vocalis-Pi full-duplex voice orchestration protocol.

This module is the small, transport-independent contract between the live
audio loop and an optional background MCP harness.  It does not own PortAudio,
ASR, TTS, or the Qwen client; those remain replaceable services.  It makes the
important realtime behavior explicit:

* VAD and transcript deltas can arrive while the agent is speaking;
* barge-in immediately emits ``FLUSH_QUEUE`` and ``FADE_OUT`` controls; and
* long-running work can be represented as a JSON-RPC MCP dispatch plus an
  asynchronous harness event stream.

The default ME3 workstation/Pi path still executes the bounded local Qwen
tool loop.  A Pipecat or Onit adapter can consume ``events`` and provide
``harness_dispatch`` without changing the audio state machine.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class TurnState(str, Enum):
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    BACKCHANNELING = "BACKCHANNELING"


class EventChannel(str, Enum):
    USER_VAD = "user_vad_state"
    USER_TRANSCRIPT = "user_transcript_delta"
    BARGE_IN = "barge_in_event"
    ASSISTANT_TEXT = "assistant_text_delta"
    TTS_CONTROL = "tts_control_signal"
    HARNESS_DISPATCH = "harness_mcp_dispatch"
    HARNESS_EVENT = "harness_event_stream"
    TURN_STATE = "turn_state"


class TTSAction(str, Enum):
    PLAY = "PLAY"
    PAUSE = "PAUSE"
    FLUSH_QUEUE = "FLUSH_QUEUE"
    FADE_OUT = "FADE_OUT"


@dataclass(frozen=True)
class VocalisEvent:
    """One serializable event on an input/output protocol channel."""

    channel: str
    payload: dict[str, Any]
    timestamp: float
    sequence: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }


class VocalisOrchestrator:
    """State machine and event bus for the Vocalis-Pi protocol.

    ``emit`` is called synchronously for every event.  Keep the callback fast:
    a Pi transport can forward events to a websocket, an MCP bridge, or a
    metrics queue.  ``harness_dispatch`` is optional; when provided it receives
    the JSON-RPC payload and may return a result that is then published as a
    harness event.  No network or provider call is made by this class itself.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        backchannel: str = "Let me work on that.",
        emit: Callable[[VocalisEvent], None] | None = None,
        harness_dispatch: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self.sample_rate = int(sample_rate)
        self.backchannel = str(backchannel)
        self._emit_callback = emit
        self._harness_dispatch = harness_dispatch
        self._sequence = itertools.count(1)
        self._request_ids = itertools.count(1)
        self.events: list[VocalisEvent] = []
        self.state = TurnState.LISTENING
        self._tts_playing = False
        self._last_transcript = ""
        self._publish_state("startup")

    def _publish(self, channel: EventChannel | str,
                 payload: dict[str, Any]) -> VocalisEvent:
        event = VocalisEvent(
            channel=channel.value if isinstance(channel, EventChannel) else str(channel),
            payload=dict(payload),
            timestamp=time.time(),
            sequence=next(self._sequence),
        )
        self.events.append(event)
        if self._emit_callback:
            self._emit_callback(event)
        return event

    def _publish_state(self, reason: str) -> VocalisEvent:
        return self._publish(EventChannel.TURN_STATE, {
            "state": self.state.value,
            "reason": reason,
        })

    def set_state(self, state: TurnState | str, reason: str = "") -> TurnState:
        target = state if isinstance(state, TurnState) else TurnState(str(state))
        if target != self.state:
            self.state = target
            self._publish_state(reason or "transition")
        return self.state

    def on_user_vad(self, active: bool, *, timestamp: float | None = None,
                    probability: float | None = None) -> None:
        """Accept the 20-30 ms VAD update and handle automatic barge-in."""
        payload: dict[str, Any] = {
            "state": "ACTIVE" if active else "INACTIVE",
            "timestamp": timestamp if timestamp is not None else time.time(),
        }
        if probability is not None:
            payload["probability"] = float(probability)
        self._publish(EventChannel.USER_VAD, payload)
        if not active:
            return
        if self.state in {
            TurnState.SPEAKING,
            TurnState.THINKING,
            TurnState.BACKCHANNELING,
        }:
            self.on_barge_in(payload["timestamp"])
        elif self.state == TurnState.INTERRUPTED:
            self.set_state(TurnState.LISTENING, "new speech after interruption")

    def on_barge_in(self, timestamp: float | None = None) -> None:
        """Stop queued/playing speech immediately and return to listening."""
        t = timestamp if timestamp is not None else time.time()
        self._publish(EventChannel.BARGE_IN, {"timestamp": t})
        self.set_state(TurnState.INTERRUPTED, "user speech overlap")
        self.tts_control(TTSAction.FLUSH_QUEUE, reason="barge-in")
        self.tts_control(TTSAction.FADE_OUT, reason="barge-in")
        self._tts_playing = False
        self.set_state(TurnState.LISTENING, "barge-in ready")

    def on_transcript_delta(self, text: str, *, final: bool = False,
                            timestamp: float | None = None) -> None:
        """Publish an ASR delta; final text advances the turn to reasoning."""
        text = str(text or "")
        if final:
            self._last_transcript = text
        self._publish(EventChannel.USER_TRANSCRIPT, {
            "text": text,
            "final": bool(final),
            "timestamp": timestamp if timestamp is not None else time.time(),
        })
        if final and text.strip():
            self.set_state(TurnState.THINKING, "final transcript")

    def begin_backchannel(self, text: str | None = None) -> str:
        """Emit a short immediate acknowledgement while work continues."""
        text = str(text or self.backchannel)
        self.set_state(TurnState.BACKCHANNELING, "background task started")
        self._publish(EventChannel.ASSISTANT_TEXT, {
            "text": text, "final": True, "kind": "backchannel",
        })
        self.tts_control(TTSAction.PLAY, reason="backchannel")
        return text

    def dispatch_harness(self, *, task: str, arguments: dict[str, Any] | None = None,
                         request_id: int | None = None) -> dict[str, Any]:
        """Build and optionally send one JSON-RPC MCP harness dispatch."""
        rid = request_id if request_id is not None else next(self._request_ids)
        params = {"task": str(task)}
        if arguments:
            params["arguments"] = dict(arguments)
        payload = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": "tools/call",
            "params": {
                "name": "harness_dispatch",
                "arguments": params,
            },
        }
        self.set_state(TurnState.THINKING, "MCP harness dispatch")
        self._publish(EventChannel.HARNESS_DISPATCH, payload)
        if self._harness_dispatch is not None:
            result = self._harness_dispatch(payload)
            if result is not None:
                self.on_harness_event({"request_id": rid, "phase": "completed",
                                       "result": result})
        return payload

    def on_harness_event(self, event: dict[str, Any]) -> None:
        """Forward an asynchronous harness status/result without blocking audio."""
        self._publish(EventChannel.HARNESS_EVENT, dict(event))

    def on_assistant_text_delta(self, text: str, *, final: bool = False) -> None:
        """Stream text to the TTS adapter and mark the agent as speaking."""
        text = str(text or "")
        if not text:
            return
        if self.state != TurnState.SPEAKING:
            self.set_state(TurnState.SPEAKING, "assistant text available")
        self._publish(EventChannel.ASSISTANT_TEXT, {
            "text": text, "final": bool(final),
        })
        if not self._tts_playing:
            self._tts_playing = True
            self.tts_control(TTSAction.PLAY, reason="assistant text delta")

    def tts_control(self, action: TTSAction | str, *, reason: str = "") -> None:
        action_value = action.value if isinstance(action, TTSAction) else str(action)
        self._publish(EventChannel.TTS_CONTROL, {
            "action": action_value,
            "reason": reason,
        })

    def on_tts_complete(self) -> None:
        self._tts_playing = False
        self.set_state(TurnState.LISTENING, "TTS complete")

    def snapshot(self) -> dict[str, Any]:
        return {
            "agent": "Vocalis-Pi",
            "state": self.state.value,
            "sample_rate": self.sample_rate,
            "last_transcript": self._last_transcript,
            "tts_playing": self._tts_playing,
            "event_count": len(self.events),
        }
