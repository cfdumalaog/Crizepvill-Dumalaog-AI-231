"""ME3 Full-Duplex Voice Agent.

Wires VAD -> ASR -> LLM -> TTS with barge-in (interruptible), running the
local dev loop on the Windows dev box (ME2-style launcher) and the same
control logic on the RPi (Pipecat pipeline).

  mic -> [Silero VAD] --speech--> [faster-whisper ASR] -> [LLM] -> [Piper TTS] -> speaker
                                        ^^^^^^^ barge-in: new speech cancels TTS ^^^^^

Run locally (the voice loop; also what launchers/Start-ME3.bat starts):
    python scripts/run_me3_module.py me3.agent                    # live loop
    python scripts/run_me3_module.py me3.agent --check            # verify deps
    python scripts/run_me3_module.py me3.agent --check --load-models
    python scripts/run_me3_module.py me3.agent --device "Fifine"
    python scripts/run_me3_module.py me3.agent --ssh-mic dalmacio@cfdfnjrpi5.local \
        --remote-device hw:2,0  # Pi mic -> this host's speakers
    python scripts/run_me3_module.py me3.agent --brain stub       # offline

The LLM brain is remote (OpenAI-compatible vLLM endpoint from config.yaml)
when the configured API-key env var is set; otherwise a stub brain keeps the
full voice pipeline running end-to-end. The key is read from the environment
only and never written to files or logs.

The MCP servers (me3.mcp_asr / mcp_tts / mcp_vad / mcp_tasks) share the same
engine loaders, so the local loop and the Pi deployment use identical model
paths.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import logging
import os
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audio_io import FullDuplexAudio, RemoteMicAudio, resample_to
from .brain import (ToolBrain, build_system_prompt, build_tool_brain,
                    resolve_llm_config)
from .config import load_config, project_root
from .agent_status import LoopStatus, StatusServer, load_release_info
from .vocalis import TurnState, VocalisOrchestrator
from . import me2_commands

log = logging.getLogger("me3.agent")

PROJECT = project_root()


# ---------------------------------------------------------------------------
# Pipecat-facing control logic (harness-agnostic, unit-testable).
# Kept intact for the Pi transport wiring; the local loop below reuses the
# same state machine inline for a synchronous realtime path.
# ---------------------------------------------------------------------------
class VoiceDuplexAgent:
    """High-level controller for the full-duplex voice loop.

    In production this drives a Pipecat Pipeline. Here we implement the
    control logic (turn state machine + barge-in) that is harness-agnostic
    and unit-testable, plus a Pipecat adapter.
    """

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.a = self.cfg["audio"]
        self.vad = self.cfg["vad"]
        self.agent = self.cfg["agent"]
        self.sr = int(self.a["sample_rate"])
        self.frame_n = int(self.sr * self.a["frame_ms"] / 1000)
        self.state = "idle"          # idle | listening | thinking | speaking
        self._speaking_task: asyncio.Task | None = None
        self._barge_in = self.agent.get("barge_in", True)
        self.vocalis = VocalisOrchestrator(
            sample_rate=self.sr,
            backchannel=self.agent.get(
                "backchannel", "I am processing your request now."
            ),
        )

    # ---- Turn state machine ------------------------------------------------
    def on_frame(self, frame, vad_speech: bool) -> str:
        """Process one audio frame; returns an action string.

        Actions: 'continue', 'start_turn', 'end_turn', 'barge_in'
        """
        self.vocalis.on_user_vad(vad_speech)
        if self.state == "speaking" and vad_speech and self._barge_in:
            # User interrupted the agent -> stop TTS, switch to listening.
            self._cancel_speaking()
            self.state = "listening"
            return "barge_in"
        if self.state == "idle" and vad_speech:
            self.state = "listening"
            return "start_turn"
        if self.state == "listening" and not vad_speech:
            # Endpointing handled by caller via silence_ms accumulation.
            return "continue"
        return "continue"

    def on_silence_endpoint(self) -> str:
        """Called when VAD reports enough trailing silence during listening."""
        if self.state == "listening":
            self.state = "thinking"
            self.vocalis.set_state(TurnState.THINKING, "silence endpoint")
            return "end_turn"
        return "continue"

    def on_reply_ready(self, text: str) -> str:
        """LLM produced a reply; start speaking."""
        self.state = "speaking"
        self.vocalis.on_assistant_text_delta(text, final=True)
        self._speak(text)
        return "speaking"

    def on_speech_done(self) -> str:
        """TTS finished; back to idle."""
        self.state = "idle"
        self.vocalis.on_tts_complete()
        return "idle"

    # ---- Speaking / barge-in ----------------------------------------------
    def _speak(self, text: str):
        """Kick off TTS playback (async). Overridable for Pipecat sink."""
        async def _run():
            # In Pipecat this is a TTSSink; here a placeholder hook.
            log.info("TTS: %s", text[:60])
            await asyncio.sleep(0)  # yield
        self._speaking_task = asyncio.create_task(_run())

    def _cancel_speaking(self):
        if self._speaking_task and not self._speaking_task.done():
            self._speaking_task.cancel()
            log.info("barge-in: TTS cancelled")
        self._speaking_task = None

    # ---- Pipecat adapter ---------------------------------------------------
    def build_pipecat_pipeline(self):
        """Construct the Pipecat Pipeline (imported lazily so the control
        logic stays testable without pipecat installed)."""
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.tts import OpenAITTSService
        from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport
        from pipecat.processors.aggregators.llm_context import LLMContext

        selection = self.cfg.get("llm", {}).get("active_provider", "auto")
        _provider, llm_cfg = resolve_llm_config(self.cfg, selection)
        if llm_cfg is None:
            raise RuntimeError(
                "no configured Pipecat LLM provider; set VLLM_API_KEY for Qwen "
                "or OPENAI_API_KEY for GPT-5.6 Luna")
        ctx = LLMContext([
            {"role": "system", "content": self.agent["system_prompt"]}
        ])
        api_key = os.environ.get(llm_cfg["api_key_env"], "")
        if not api_key:
            raise RuntimeError(
                f"{llm_cfg['api_key_env']} is required for the Pipecat "
                "pipeline; use LocalVoiceLoop --brain stub for an "
                "offline smoke test"
            )
        settings_kwargs = {"model": llm_cfg["model"]}
        if llm_cfg.get("temperature") is not None:
            settings_kwargs["temperature"] = llm_cfg["temperature"]
        if llm_cfg.get("max_completion_tokens") is not None:
            settings_kwargs["max_completion_tokens"] = int(
                llm_cfg["max_completion_tokens"])
        elif llm_cfg.get("max_tokens") is not None:
            settings_kwargs["max_tokens"] = int(llm_cfg["max_tokens"])
        # Keep the voice path functional and direct: do not request hidden
        # reasoning from Luna or any OpenAI-compatible provider.
        llm = OpenAILLMService(
            api_key=api_key,
            base_url=llm_cfg["base_url"],
            settings=OpenAILLMService.Settings(**settings_kwargs),
        )
        vad_analyzer = SileroVADAnalyzer(
            sample_rate=self.sr,
            params=VADParams(
                confidence=self.vad["threshold"],
                start_secs=self.vad["min_speech_ms"] / 1000.0,
                stop_secs=self.vad["min_silence_ms"] / 1000.0,
            ),
        )
        # NOTE: full transport wiring (mic/speaker via sounddevice) is in
        # me3/audio_io.py. This returns the core analysis->llm chain.
        return Pipeline, PipelineRunner, PipelineTask, llm, ctx, vad_analyzer


# ---------------------------------------------------------------------------
# LLM brains
# ---------------------------------------------------------------------------
class StubBrain:
    """Canned-response brain so the full voice pipeline runs without an LLM."""

    name = "stub"

    def chat(self, history: list[dict], user_text: str) -> str:
        return ("No paid or remote LLM key is configured, so I am using the "
                "offline stub brain. Set VLLM_API_KEY for Qwen or "
                "OPENAI_API_KEY for GPT-5.6 Luna. I heard: "
                f"{user_text}")


class RemoteBrain:
    """OpenAI-compatible chat brain (the vLLM endpoint in config.yaml)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_url = cfg.get("base_url", "").rstrip("/")
        self.model = cfg.get("model", "")
        self.api_key_env = cfg.get("api_key_env", "")
        self.temperature = cfg.get("temperature", 0.4)
        self.max_tokens = cfg.get("max_tokens", 256)
        self.timeout_s = float(cfg.get("timeout_s", 60))
        self.api_key = os.environ.get(self.api_key_env, "")
        self.name = "remote" if self.api_key else "stub"
        if not self.api_key:
            log.warning("LLM env var %s not set - falling back to stub brain",
                        self.api_key_env)

    def chat(self, history: list[dict], user_text: str) -> str:
        if not self.api_key:
            return StubBrain().chat(history, user_text)
        import requests
        body = {
            "model": self.model,
            "messages": history + [{"role": "user", "content": user_text}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body, timeout=self.timeout_s,
        )
        r.raise_for_status()
        message = r.json()["choices"][0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            # Some OpenAI-compatible backends can return a successful choice
            # with null/empty content (for example after a provider timeout or
            # an unsupported tool/reasoning response).  Keep the voice loop
            # alive and let its turn-level error accounting mark the trial
            # invalid; never send the literal string "None" to TTS.
            raise RuntimeError("remote LLM returned empty message content")
        return content.strip()


# ---------------------------------------------------------------------------
# Local full-duplex voice loop (the live app)
# ---------------------------------------------------------------------------
@dataclass
class TurnStats:
    n_turns: int = 0
    n_barge_in: int = 0
    n_control: int = 0
    n_errors: int = 0
    asr_ms: list = field(default_factory=list)
    llm_ms: list = field(default_factory=list)
    tts_ms: list = field(default_factory=list)
    e2e_ms: list = field(default_factory=list)
    first_tool_ms: list = field(default_factory=list)

    def summary(self) -> str:
        def _ms(v):
            return f"{np.median(v):.0f}/{np.percentile(v, 95):.0f}" if v else "-"
        ft = f"{np.median(self.first_tool_ms):.0f} ms" if self.first_tool_ms else "-"
        return (f"turns={self.n_turns} barge_in={self.n_barge_in} "
                f"controls={self.n_control} "
                f"errors={self.n_errors} "
                f"asr_med/p95={_ms(self.asr_ms)}ms llm={_ms(self.llm_ms)}ms "
                f"tts={_ms(self.tts_ms)}ms e2e={_ms(self.e2e_ms)}ms "
                f"first_tool_med={ft}")


class LocalVoiceLoop:
    """The live full-duplex loop: mic + speaker always open.

    State: idle -> (speech) -> listening -> (silence) -> thinking
           -> (reply) -> speaking -> (done | voice-control -> listening)

    Protected playback is the default.  Ordinary speech is transcribed while
    the response remains audible; only explicit voice controls stop playback,
    skip an announcement, or route the newest utterance next.
    """

    VAD_CHUNK = 512  # silero frame size @16 kHz

    def __init__(self, cfg: dict | None = None, brain=None, device=None,
                 audio=None, orchestrator: VocalisOrchestrator | None = None,
                 status: LoopStatus | None = None):
        self.cfg = cfg or load_config()
        a = self.cfg["audio"]
        self.sr = int(a["sample_rate"])
        self.frame_ms = int(a["frame_ms"])
        self.frame_n = int(self.sr * self.frame_ms / 1000)
        v = self.cfg["vad"]
        self.vad_thr = float(v.get("threshold", 0.5))
        self.min_speech_ms = int(v.get("min_speech_ms", 120))
        self.min_silence_ms = int(v.get("min_silence_ms", 450))
        self.pre_roll_ms = int(v.get("pre_roll_ms", 300))
        agent = self.cfg["agent"]
        self.barge_in = bool(agent.get("barge_in", True))
        self.command_control = bool(agent.get("command_control", True))
        self.command_min_speech_ms = int(
            agent.get("command_min_speech_ms", self.min_speech_ms))
        self.command_min_silence_ms = int(
            agent.get("command_min_silence_ms", self.min_silence_ms))
        self.system_prompt = agent["system_prompt"]
        self.device = device

        if device is not None:
            a = dict(self.cfg["audio"])
            a["input_device"] = device
            self.cfg = {**self.cfg, "audio": a}
        self.audio = audio if audio is not None else FullDuplexAudio(self.cfg)
        self.brain = brain
        self.history: list[dict] = []
        self.stats = TurnStats()
        self.status = status

        self.ack_wait_s = float(agent.get("ack_wait_s", 2.5))
        self.backchannel = str(
            agent.get("backchannel", "I am processing your request now.")
        )
        # Protected playback is intentionally uninterruptible for ordinary
        # speech. Only the narrow voice controls below are actionable while
        # the browser speaker is active; this prevents a residual acoustic
        # fragment from becoming a new task after the echo canceller.
        self.ignore_noncontrol_during_playback = bool(
            agent.get("protected_playback_ignore_noncontrol", True))
        # Long tasks survive barge-in.  The user may interrupt the spoken
        # backchannel or a reply, but an already accepted tool request must
        # not be discarded.  Completed results wait until the audio path is
        # idle before being announced, so they cannot cut off a newer reply.
        self._pending: deque[tuple] = deque()
        self._completed: deque[tuple] = deque()
        self._tts_lock = threading.Lock()
        self._progress_lock = threading.RLock()
        self._progress_holders: dict[int, dict] = {}
        self._spoken_transcripts: deque[tuple[str, float]] = deque(maxlen=8)
        self.orchestrator = orchestrator or VocalisOrchestrator(
            sample_rate=self.sr,
            backchannel=self.backchannel,
        )
        self._brain_on_tool = None
        self._brain_on_progress = None
        if self.brain is not None:
            callback = getattr(self.brain, "on_tool", None)
            if callable(callback):
                self._brain_on_tool = callback
                self.brain.on_tool = self._on_brain_tool
            callback = getattr(self.brain, "on_progress", None)
            if callable(callback):
                self._brain_on_progress = callback
                self.brain.on_progress = self._on_brain_progress

        self.state = "idle"
        self._pre: deque[np.ndarray] = deque(
            maxlen=max(1, self.pre_roll_ms // self.frame_ms))
        self._uttr: list[np.ndarray] = []
        self._speech_ms = 0
        self._silence_ms = 0
        self._vad_pending = np.zeros(0, dtype=np.float32)
        self._control_uttr: list[np.ndarray] = []
        self._control_speech_ms = 0
        self._control_silence_ms = 0
        self._control_started_at: float | None = None
        self._control_echo_suspect_frames = 0
        self._control_frame_count = 0
        self._deferred_text: str | None = None
        self._deferred_asr_ms = 0.0
        self._deferred_t_start: float | None = None
        self._engines = None   # (vad_model, unused, asr_engine, asr_cfg, tts_engine, tts_cfg)
        self._stop_event = threading.Event()

    def _on_brain_tool(self, kind: str, name: str, payload) -> None:
        """Bridge Qwen tool progress into Vocalis' asynchronous event stream.

        The original logging callback is retained.  Event payloads deliberately
        contain only the tool name and success flag; arguments and tool results
        can contain user data or credentials and must not enter the event log.
        """
        if self._brain_on_tool is not None:
            self._brain_on_tool(kind, name, payload)
        if kind == "start":
            self.orchestrator.on_harness_event({
                "phase": "tool_started",
                "name": str(name),
            })
            return
        ok = payload.get("ok") if isinstance(payload, dict) else True
        self.orchestrator.on_harness_event({
            "phase": "tool_finished",
            "name": str(name),
            "ok": bool(ok),
        })

    def _on_brain_progress(self, stage: str, detail: str) -> None:
        """Bridge safe model/tool phase summaries into status telemetry."""
        if self._brain_on_progress is not None:
            self._brain_on_progress(stage, detail)
        ident = threading.get_ident()
        with self._progress_lock:
            holder = self._progress_holders.get(ident)
            if holder is not None:
                holder["progress_stage"] = str(stage)[:80]
                holder["progress_detail"] = str(detail)[:240]
                holder["last_progress_at"] = time.time()
        if self.status is not None:
            self.status.report_progress(stage, detail)

    @staticmethod
    def _long_task_update(stage: str, detail: str) -> str:
        """Build a short professional heartbeat without exposing reasoning."""
        clean_stage = str(stage or "work").replace("_", " ").strip()
        clean_detail = " ".join(str(detail or "working").split())[:160]
        if clean_detail.lower().startswith("waiting for"):
            return ("I am still working on your request. I am waiting for the "
                    "model to return the next step, and I will report the result.")
        if clean_detail:
            return (f"I am still working on your request. Current stage: "
                    f"{clean_detail}. I will continue and report the result.")
        return (f"I am still working on your request during the {clean_stage} "
                "stage. I will continue and report the result.")

    @staticmethod
    def _normalise_spoken_text(text: str) -> str:
        return " ".join(re.sub(r"[^a-zA-Z0-9À-ž\s'-]", " ",
                               str(text).lower()).split())

    def _note_transcript_echo_rejected(self, text: str, reason: str) -> None:
        capture = getattr(self.audio, "capture", self.audio)
        note = getattr(capture, "note_transcript_echo_rejected", None)
        if callable(note):
            note()
        message = f"Rejected protected playback speech ({reason})"
        log.info("%s: %r", message, str(text)[:120])
        if self.status is not None:
            self.status.report_progress("echo_guard", message)
            self.status.report_state_reason("protected playback speech rejected")

    def _is_recent_spoken_echo(self, text: str) -> bool:
        """Reject ASR text that matches audio ME3 recently sent to speakers."""
        candidate = self._normalise_spoken_text(text)
        if len(candidate) < 6:
            return False
        now = time.perf_counter()
        kept: deque[tuple[str, float]] = deque(maxlen=8)
        matched = False
        for spoken, expires in self._spoken_transcripts:
            if expires <= now:
                continue
            kept.append((spoken, expires))
            if candidate == spoken:
                matched = True
            elif len(candidate) >= 12 and len(spoken) >= 12:
                ratio = difflib.SequenceMatcher(None, candidate, spoken).ratio()
                if ratio >= 0.82:
                    matched = True
        self._spoken_transcripts = kept
        return matched

    def _refresh_background_status(self) -> None:
        """Publish live agent-thread and tool-process telemetry to the UI."""
        long_tasks = []
        now = time.perf_counter()
        for thread, holder, _t_start, _asr_ms, request_text in list(self._pending):
            started = float(holder.get("started_perf", now) or now)
            long_tasks.append({
                "status": "running" if thread.is_alive() else "completed",
                "thread": thread.name,
                "elapsed_s": round(max(0.0, now - started), 1),
                "request": str(request_text)[:160],
                "stage": str(holder.get("progress_stage", "thinking"))[:80],
                "detail": str(holder.get("progress_detail", "working"))[:240],
                "heartbeat_count": int(holder.get("heartbeat_count", 0) or 0),
                "last_update_at": holder.get("last_progress_at"),
            })
        try:
            from .tools import TASKS
            processes = TASKS.snapshot()
        except Exception as exc:  # noqa: BLE001 - telemetry must not stop voice
            log.debug("background process snapshot unavailable: %s", exc)
            processes = []
        if self.status is not None:
            self.status.report_background(long_tasks=long_tasks,
                                          processes=processes)

    def _emit_long_task_heartbeats(self) -> None:
        """Queue one spoken progress update per minute for each long turn."""
        now = time.perf_counter()
        for _thread, holder, _t_start, _asr_ms, _request_text in list(self._pending):
            started = float(holder.get("started_perf", now) or now)
            last = float(holder.get("last_heartbeat_perf", started) or started)
            if holder.get("cancel_requested") or now - started < 60.0:
                continue
            if now - last < 60.0 or holder.get("heartbeat_text"):
                continue
            stage = str(holder.get("progress_stage", "working"))
            detail = str(holder.get("progress_detail", "working"))
            holder["heartbeat_text"] = self._long_task_update(stage, detail)
            holder["last_heartbeat_perf"] = now
            holder["heartbeat_count"] = int(holder.get("heartbeat_count", 0) or 0) + 1
            holder["last_progress_at"] = time.time()
            if self.status is not None:
                self.status.report_progress("heartbeat", holder["heartbeat_text"])

    def _announce_progress_if_idle(self) -> None:
        """Speak one queued heartbeat only at a safe audio boundary."""
        if self.state not in ("idle", "awaiting") or not self.audio.playback_done():
            return
        for _thread, holder, t_start, asr_ms, _request_text in list(self._pending):
            text = holder.get("heartbeat_text")
            if not text or holder.get("cancel_requested"):
                continue
            holder["heartbeat_text"] = None
            if self.status is not None:
                self.status.report_reply(text, "progress")
            self.orchestrator.on_assistant_text_delta(text, final=True)
            self._speak_text(self._engines[4], text, t_start, asr_ms, 0.0,
                             count_e2e=False, voice_role="announcer")
            return

    # ---- Engine loading -----------------------------------------------------
    def _load_engines(self) -> None:
        from . import mcp_asr, mcp_tts
        from .vad_onnx import load_vad_model

        t0 = time.perf_counter()
        vad_model = load_vad_model()
        t1 = time.perf_counter()
        asr_engine, asr_cfg = mcp_asr._get_engine()
        t2 = time.perf_counter()
        tts_engine, tts_cfg = mcp_tts._get_engine()
        t3 = time.perf_counter()
        # Every spoken output, including the immediate backchannel and the
        # later completion announcement, uses one standard announcer voice.
        # QwenRemoteTTS exposes ``voice``; Piper is already loaded from the
        # configured standard voice and therefore needs no mutation here.
        self.announcer_voice = str(
            tts_cfg.get("announcer_voice") or tts_cfg.get("voice") or ""
        ).strip()
        if self.announcer_voice and hasattr(tts_engine, "voice"):
            tts_engine.voice = self.announcer_voice
        self._engines = (vad_model, None, asr_engine, asr_cfg,
                         tts_engine, tts_cfg)
        log.info("models loaded: vad=%.0f ms asr=%.0f ms (%s) tts=%.0f ms (%s @ %d Hz)",
                 (t1 - t0) * 1e3, (t2 - t1) * 1e3, asr_cfg["model"],
                 (t3 - t2) * 1e3, tts_cfg["voice"],
                 getattr(getattr(tts_engine, "config", None), "sample_rate", 0) or 22050)
        if self.status is not None:
            self.status.report_models(
                vad_ms=round((t1 - t0) * 1e3, 1),
                asr_ms=round((t2 - t1) * 1e3, 1),
                asr_model=asr_cfg.get("model"),
                asr_backend=getattr(asr_engine, "last_backend", asr_cfg.get("engine")),
                asr_fallback_model=getattr(asr_engine, "fallback_model", None),
                tts_ms=round((t3 - t2) * 1e3, 1),
                tts_backend=getattr(tts_engine, "provider", tts_cfg.get("engine")),
                tts_model=tts_cfg.get("model"),
                tts_voice=tts_cfg.get("voice"),
                tts_hz=int(getattr(getattr(tts_engine, "config", None),
                                   "sample_rate", 0) or 22050),
            )

    def _reset_vad(self) -> None:
        vad_model = self._engines[0]
        if hasattr(vad_model, "reset_states"):
            vad_model.reset_states()

    # ---- VAD ----------------------------------------------------------------
    def _vad_probs(self, frame: np.ndarray) -> list[float]:
        """Feed one 480-sample frame into the 512-sample silero window."""
        vad_model = self._engines[0]
        from .vad_onnx import frame_probability
        self._vad_pending = np.concatenate(
            [self._vad_pending, frame.astype(np.float32) / 32768.0])
        probs = []
        while self._vad_pending.shape[0] >= self.VAD_CHUNK:
            chunk, self._vad_pending = (
                self._vad_pending[: self.VAD_CHUNK],
                self._vad_pending[self.VAD_CHUNK :],
            )
            p = frame_probability(vad_model, chunk, self.sr)
            probs.append(p)
        return probs

    # ---- Turn processing ----------------------------------------------------
    def _process_turn(self, utterance: np.ndarray | None, t_start: float,
                      text_override: str | None = None,
                      asr_ms_override: float | None = None) -> None:
        _vad_model, _unused, asr_engine, asr_cfg, tts_engine, tts_cfg = self._engines
        self.state = "thinking"
        if self.status is not None:
            self.status.report_state_reason("transcribing + thinking")

        if text_override is None:
            t0 = time.perf_counter()
            try:
                segments, _info = asr_engine.transcribe(
                    utterance.astype(np.float32) / 32768.0,
                    beam_size=asr_cfg.get("beam_size", 1),
                    language=asr_cfg.get("language", "en"),
                    vad_filter=False,
                )
            except Exception as exc:  # noqa: BLE001 - keep listening after provider failure
                self._report_runtime_error("ASR", exc)
                raise
            text = " ".join(s.text.strip() for s in segments).strip()
            asr_ms = (time.perf_counter() - t0) * 1e3
            if self.status is not None:
                self.status.report_models(
                    asr_backend=getattr(asr_engine, "last_backend", "qwen_remote"),
                    asr_fallback_model=getattr(asr_engine, "fallback_model", None),
                )
        else:
            text = text_override.strip()
            asr_ms = float(asr_ms_override or 0.0)
        self.stats.asr_ms.append(asr_ms)
        if not text:
            log.info("turn ignored (empty transcript) [asr %.0f ms]", asr_ms)
            self.state = "idle"
            if self.status is not None:
                self.status.report_state_reason("empty transcript discarded")
            self.orchestrator.set_state(TurnState.LISTENING, "empty transcript")
            return
        if self._is_recent_spoken_echo(text):
            self._note_transcript_echo_rejected(text, "matches recent ME3 output")
            self.state = "idle"
            self.orchestrator.set_state(TurnState.LISTENING,
                                         "speaker echo rejected")
            return
        log.info("heard: %r [asr %.0f ms]", text, asr_ms)
        if self.status is not None:
            self.status.clear_error()
        self.orchestrator.on_transcript_delta(text, final=True,
                                              timestamp=t_start)
        if self.status is not None:
            self.status.report_transcript(text)

        t0 = time.perf_counter()
        if self.ack_wait_s > 0:
            # Long-task path: run the (tool-calling) brain on a worker
            # thread. If it finishes within ack_wait_s, speak it like a
            # normal turn; otherwise acknowledge with a backchannel and
            # announce the result when it completes. Barge-in during the
            # wait stops only the current audio; the background work and its
            # completion reply remain queued for a later idle boundary.
            holder: dict = {}
            holder.update({
                "started_perf": time.perf_counter(),
                "started_at": time.time(),
                "progress_stage": "thinking",
                "progress_detail": f"Waiting for {self.brain.name} model inference",
                "last_progress_at": time.time(),
                "last_heartbeat_perf": time.perf_counter(),
                "heartbeat_count": 0,
            })

            history_snapshot = list(self.history)

            def _run_brain() -> None:
                ident = threading.get_ident()
                with self._progress_lock:
                    self._progress_holders[ident] = holder
                try:
                    holder["reply"] = self.brain.chat(history_snapshot, text)
                except Exception as exc:  # noqa: BLE001 - announce safely later
                    holder["error"] = type(exc).__name__
                    log.exception("background brain task failed")
                finally:
                    with self._progress_lock:
                        self._progress_holders.pop(ident, None)

            th = threading.Thread(target=_run_brain, daemon=True,
                                  name="me3-turn")
            th.start()
            th.join(self.ack_wait_s)
            if th.is_alive():
                self._pending.append((th, holder, t_start, asr_ms, text))
                self.history.append({"role": "user", "content": text})
                self.stats.llm_ms.append((time.perf_counter() - t0) * 1e3)
                log.info("long task running in background; backchannel")
                self.orchestrator.begin_backchannel(self.backchannel)
                if self.status is not None:
                    self.status.report_reply(self.backchannel, "backchannel")
                self._speak_text(tts_engine, self.backchannel, t_start,
                                 asr_ms, (time.perf_counter() - t0) * 1e3,
                                 voice_role="announcer")
                self.state = "awaiting"
                if self.status is not None:
                    self.status.report_state_reason("long task running")
                return
            if holder.get("error"):
                raise RuntimeError(f"background brain task failed: {holder['error']}")
            reply = holder.get("reply", "")
        else:
            reply = self.brain.chat(self.history, text)
        llm_ms = (time.perf_counter() - t0) * 1e3
        self.stats.llm_ms.append(llm_ms)
        m = getattr(self.brain, "metrics", None)
        if isinstance(m, dict) and m.get("first_tool_ms") is not None:
            self.stats.first_tool_ms.append(m["first_tool_ms"])
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": reply})
        self.history = self.history[-8:]
        log.info("reply (%s, %.0f ms): %r", self.brain.name, llm_ms,
                 reply[:120])
        self.orchestrator.on_assistant_text_delta(reply, final=True)
        if self.status is not None:
            self.status.report_reply(reply, "reply")
        self._speak_text(tts_engine, reply, t_start, asr_ms, llm_ms)

    def _report_runtime_error(self, stage: str, exc: Exception) -> None:
        """Turn provider failures into useful, secret-free operator status."""
        raw = " ".join(str(exc).split()).lower()
        if stage == "ASR":
            if "401" in raw or "unauthorized" in raw:
                message = "Qwen ASR authorization failed; refresh the protected audio credential."
            elif any(token in raw for token in (
                    "connection", "empty reply", "timed out", "timeout",
                    "max retries", "502", "503", "504")):
                message = "Qwen ASR provider unavailable; restore the VPN/SSH audio tunnel and DGX ASR service."
            else:
                message = "Qwen ASR failed; check the protected audio provider configuration."
        else:
            message = f"{stage} failed; check the ME3 provider status."
        if self.status is not None:
            self.status.report_error(stage, message)
            self.status.report_state_reason(message)
        log.error("%s provider failure: %s", stage, message)

    def _speak_text(self, tts_engine, text: str, t_start: float,
                    asr_ms: float, llm_ms: float,
                    count_e2e: bool = True,
                    voice_role: str = "response") -> None:
        """Synthesize + queue one reply (thread-safe playback start)."""
        t0 = time.perf_counter()
        chunks = [c.audio_int16_bytes for c in tts_engine.synthesize(text)]
        pcm = np.frombuffer(b"".join(chunks), dtype=np.int16)
        tts_ms = (time.perf_counter() - t0) * 1e3
        self.stats.tts_ms.append(tts_ms)
        if count_e2e:
            self.stats.e2e_ms.append((time.perf_counter() - t_start) * 1e3)
        if pcm.size == 0:
            log.warning("TTS produced no audio")
            self.state = "idle"
            self.orchestrator.on_tts_complete()
            return
        tts_sr = int(getattr(getattr(tts_engine, "config", None),
                             "sample_rate", 22050) or 22050)
        out = resample_to(pcm, tts_sr, self.sr)
        self._spoken_transcripts.append((
            self._normalise_spoken_text(text),
            time.perf_counter() + len(out) / max(1, self.sr) +
            float(self.cfg["audio"].get("echo_guard", {}).get("tail_ms", 250)) /
            1000.0 + 2.0,
        ))
        with self._tts_lock:
            backend = getattr(tts_engine, "provider", "piper")
            event = getattr(tts_engine, "last_event", None) or {}
            log.info(
                "speaking backend=%s voice=%s role=%s duration=%.2f s [tts %.0f "
                "ms e2e %.0f ms sample_rate=%d event=%s]",
                backend, self.announcer_voice or tts_cfg.get("voice"),
                voice_role, len(out) / self.sr, tts_ms,
                (time.perf_counter() - t_start) * 1e3, tts_sr,
                {k: event[k] for k in ("ok", "attempts", "latency_ms",
                                       "bytes", "error_class", "model",
                                       "voice_requested", "language")
                 if k in event})
            self.audio.play(out, blocking=False)
        self.state = "speaking"
        if self.status is not None:
            self.status.report_state_reason("speaking")

    def _preserve_pending(self) -> None:
        """Keep accepted background work alive when the user barges in.

        Barge-in stops only the currently playing audio.  The original brain
        request and its tool side effects continue, then its final reply is
        queued for an idle audio boundary.
        """
        if self._pending:
            log.info("pending task retained (barge-in); completion will be announced when idle")

    def _reset_control_buffer(self) -> None:
        self._control_uttr = []
        self._control_speech_ms = 0
        self._control_silence_ms = 0
        self._control_started_at = None
        self._control_echo_suspect_frames = 0
        self._control_frame_count = 0

    def _current_frame_echo_suspect(self) -> bool:
        """Return echo provenance attached to the most recently read frame."""
        capture = getattr(self.audio, "capture", self.audio)
        return bool(getattr(capture, "_last_echo_suspect", False))

    def _transcribe_control_audio(self, utterance: np.ndarray) -> tuple[str, float]:
        """Transcribe speech captured during protected playback.

        This is intentionally a short, local control-path transcription.  It
        lets the user say ``stop processing``, ``skip``, or ``steer`` without
        interrupting every ordinary sentence that happens to overlap TTS.
        """
        _vad_model, _unused, asr_engine, asr_cfg, _tts_engine, _tts_cfg = self._engines
        t0 = time.perf_counter()
        segments, _info = asr_engine.transcribe(
            utterance.astype(np.float32) / 32768.0,
            beam_size=asr_cfg.get("beam_size", 1),
            language=asr_cfg.get("language", "en"),
            vad_filter=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, (time.perf_counter() - t0) * 1e3

    @staticmethod
    def _clean_control_text(text: str) -> str:
        # Qwen/Whisper can occasionally wrap a short audio result in a
        # transcription preamble.  Remove that wrapper before classification.
        m = re.search(
            r"transcript(?: of the audio)?\s+is\s*:\s*[\"'](.+?)[\"']\s*\.?$",
            text,
            re.IGNORECASE,
        )
        if m:
            text = m.group(1)
        return " ".join(
            re.sub(r"[^a-zA-Z0-9À-ž\s'-]", " ", text.lower()).split()
        )

    @classmethod
    def _voice_command(cls, text: str) -> str | None:
        """Return a protected-playback command, if the utterance is one."""
        clean = cls._clean_control_text(text)
        if clean in {
            "stop", "stop processing", "cancel processing", "cancel that",
            "stop it", "stop now", "tigil", "ihinto ang pagproseso",
        }:
            return "stop"
        if clean in {
            "skip", "skip this", "skip response", "next", "next prompt",
            "go to latest prompt", "use latest prompt", "laktaw",
        }:
            return "skip"
        if clean in {
            "steer", "steer this", "change direction", "redirect",
            "baguhin ang direksyon",
        } or clean.startswith("steer "):
            return "steer"
        return None

    def _capture_protected_speech(self, speech: bool, frame: np.ndarray,
                                  t: float) -> None:
        """Buffer speech while TTS is protected and endpoint it for control."""
        if speech:
            if not self._control_uttr:
                self._control_started_at = t
            self._control_frame_count += 1
            if self._current_frame_echo_suspect():
                self._control_echo_suspect_frames += 1
            self._control_uttr.append(frame)
            self._control_speech_ms += self.frame_ms
            self._control_silence_ms = 0
            return
        if not self._control_uttr:
            return
        self._control_frame_count += 1
        if self._current_frame_echo_suspect():
            self._control_echo_suspect_frames += 1
        self._control_silence_ms += self.frame_ms
        self._control_uttr.append(frame)
        if (self._control_speech_ms >= self.command_min_speech_ms and
                self._control_silence_ms >= self.command_min_silence_ms):
            utterance = np.concatenate(self._control_uttr)
            started = self._control_started_at or t
            suspect_frames = self._control_echo_suspect_frames
            frame_count = self._control_frame_count
            self._reset_control_buffer()
            self._process_protected_utterance(
                utterance, started, suspect_frames, frame_count)

    def _cancel_pending_announcements(self) -> None:
        """Suppress accepted results after an explicit stop command.

        A Python worker already inside a remote request cannot be safely
        killed.  Marking its holder keeps the tool side effects observable but
        prevents its eventual reply from being spoken after the user said
        ``stop processing``.
        """
        for _thread, holder, *_rest in list(self._pending) + list(self._completed):
            holder["cancel_requested"] = True
        self._completed.clear()

    def _dispatch_deferred_if_ready(self, force: bool = False) -> None:
        """Run the newest buffered prompt once protected audio is clear."""
        if not self._deferred_text or self.state in ("speaking", "thinking"):
            return
        if not force and not self.audio.playback_done():
            return
        text = self._deferred_text
        asr_ms = self._deferred_asr_ms
        started = self._deferred_t_start or time.perf_counter()
        self._deferred_text = None
        self._deferred_asr_ms = 0.0
        self._deferred_t_start = None
        log.info("dispatching latest protected-playback prompt %r", text[:120])
        self._process_turn(None, started, text_override=text,
                           asr_ms_override=asr_ms)

    def _handle_voice_command(self, kind: str) -> None:
        """Apply an explicit voice command without exposing model reasoning."""
        self.audio.stop_playback(silent=True)
        if kind == "stop":
            self._cancel_pending_announcements()
            self._deferred_text = None
            self._deferred_asr_ms = 0.0
            self._deferred_t_start = None
            self.state = "idle"
            self.orchestrator.on_tts_complete()
            if self.status is not None:
                self.status.report_state_reason("stopped by voice command")
            log.info("voice control: stop processing")
            return

        # Skip/steer stop current audio, but do not discard an already accepted
        # background task.  The newest buffered utterance takes priority.
        self._preserve_pending()
        self.state = "idle"
        self.orchestrator.on_tts_complete()
        if self.status is not None:
            self.status.report_state_reason(f"{kind} command accepted")
        log.info("voice control: %s", kind)
        self._dispatch_deferred_if_ready(force=True)

    def _process_protected_utterance(self, utterance: np.ndarray,
                                     t_start: float,
                                     echo_suspect_frames: int = 0,
                                     captured_frames: int = 0) -> None:
        text, asr_ms = self._transcribe_control_audio(utterance)
        if not text:
            return
        self.orchestrator.on_transcript_delta(text, final=True,
                                              timestamp=t_start)
        if self.status is not None:
            self.status.report_transcript(text)
        kind = self._voice_command(text)
        if kind:
            self.stats.n_control += 1
            self._handle_voice_command(kind)
            return
        ratio = (float(echo_suspect_frames) / float(captured_frames)
                 if captured_frames > 0 else 0.0)
        recent_echo = (self._is_recent_spoken_echo(text)
                       if self.ignore_noncontrol_during_playback else False)
        if self.ignore_noncontrol_during_playback and (recent_echo or ratio >= 0.55):
            reason = ("matches recent ME3 output" if recent_echo else
                      f"speaker-reference evidence {ratio:.0%}")
            self._note_transcript_echo_rejected(
                text, reason)
            return
        # Genuine user speech during protected playback is retained as the
        # newest prompt and dispatched once the current TTS reaches a safe
        # boundary. This preserves overlap handling without admitting echoes.
        self._deferred_text = text
        self._deferred_asr_ms = asr_ms
        self._deferred_t_start = t_start
        log.info("protected playback: queued latest prompt %r", text[:120])
        if self.status is not None:
            self.status.report_state_reason("latest prompt queued behind protected playback")

    def _collect_completed(self) -> None:
        """Move finished background tasks to the ordered announcement queue."""
        if not self._pending:
            return
        active: deque[tuple] = deque()
        while self._pending:
            item = self._pending.popleft()
            if item[0].is_alive():
                active.append(item)
            else:
                self._completed.append(item)
        self._pending = active

    def _announce_completed_if_idle(self) -> None:
        """Announce one completed task without interrupting a newer turn."""
        if not self._completed or self.state not in ("idle", "awaiting"):
            return
        if not self.audio.playback_done():
            return
        _thread, holder, t_start, asr_ms, request_text = self._completed.popleft()
        if holder.get("cancel_requested"):
            log.info("background completion suppressed by voice stop")
            return
        reply = holder.get("reply", "")
        if not reply and holder.get("error"):
            reply = "I could not complete that request. Please try again."
        if reply:
            self.history.append({"role": "assistant", "content": reply})
            self.history = self.history[-8:]
            m = getattr(self.brain, "metrics", None)
            if isinstance(m, dict) and m.get("first_tool_ms") is not None:
                self.stats.first_tool_ms.append(m["first_tool_ms"])
            log.info("background task finished; announcing request=%r",
                     request_text[:120])
            if self.status is not None:
                self.status.report_reply(reply, "announce")
            self.orchestrator.on_assistant_text_delta(reply, final=True)
            self._speak_text(self._engines[4], reply, t_start, asr_ms, 0.0,
                             count_e2e=False, voice_role="announcer")
        elif self.state == "awaiting":
            self.state = "idle"
            self.orchestrator.set_state(
                TurnState.LISTENING, "background task returned empty")

    def _start_turn(self, t: float) -> None:
        self._uttr = list(self._pre)
        self._speech_ms = 0
        self._silence_ms = 0
        self._reset_vad()
        self.state = "listening"
        if self.status is not None:
            self.status.report_state_reason("user speech detected")
        log.info("speech start (pre-roll %d frames)", len(self._pre))

    def _end_turn(self, t: float) -> None:
        if not self._uttr:
            self.state = "idle"
            return
        utterance = np.concatenate(self._uttr)
        self._uttr = []
        if len(utterance) < self.sr * 0.15:  # <150 ms of audio: ignore
            self.state = "idle"
            self.orchestrator.set_state(TurnState.LISTENING, "short audio ignored")
            return
        t_end = time.perf_counter()
        try:
            self._process_turn(utterance, t)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            self.stats.n_errors += 1
            self.state = "idle"
            self.orchestrator.set_state(TurnState.LISTENING, "turn error")
            if self.status is not None and not self.status.last_error:
                self._report_runtime_error("TURN", exc)
            log.exception("turn processing failed: %s", exc)

    # ---- Main loop ----------------------------------------------------------
    def stop(self) -> None:
        """Request a clean stop; used by the Pi web device selector."""
        self._stop_event.set()
        try:
            self.audio.stop_playback(silent=True)
        except Exception:  # noqa: BLE001
            pass

    def run(self) -> None:
        a = self.cfg["audio"]
        log.info("ME3 Voice Duplex - local full-duplex loop")
        log.info("  sr=%d frame=%d ms protected_playback=%s command_control=%s brain=%s",
                 self.sr, self.frame_ms, not self.barge_in,
                 self.command_control, self.brain.name)
        self._load_engines()
        self.audio.start_capture()
        self.audio.startup_signal_check()
        self.orchestrator.set_state(TurnState.LISTENING, "audio capture started")
        log.info("LISTENING. Playback is protected; say 'stop processing', "
                 "'skip', or 'steer' for voice control. Ctrl+C stops.")
        if self.status is not None:
            self.status.report_started()

        try:
            while not self._stop_event.is_set():
                self._collect_completed()
                self._emit_long_task_heartbeats()
                self._refresh_background_status()
                self._announce_progress_if_idle()
                self._announce_completed_if_idle()
                self._dispatch_deferred_if_ready()
                frame = self.audio.read_frame(timeout=0.5)
                if frame is None:
                    if self.state == "speaking" and self.audio.playback_done():
                        self.state = "idle"
                        self.orchestrator.on_tts_complete()
                        log.info("agent finished speaking; idle")
                        if self.status is not None:
                            self.status.report_state_reason("idle")
                        self._dispatch_deferred_if_ready()
                    continue
                for p in self._vad_probs(frame):
                    self._on_vad_frame(p, frame)
                if self.state == "speaking" and self.audio.playback_done():
                    self.state = "idle"
                    self.orchestrator.on_tts_complete()
                    log.info("agent finished speaking; idle")
                    if self.status is not None:
                        self.status.report_state_reason("idle")
                    self._dispatch_deferred_if_ready()
        except KeyboardInterrupt:
            pass
        finally:
            self.audio.close()
            if self.status is not None:
                self.status.report_stopped()
            log.info("stopped. %s", self.stats.summary())

    def _on_vad_frame(self, p: float, frame: np.ndarray) -> None:
        speech = p >= self.vad_thr
        t = time.perf_counter()
        if self.status is not None:
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            self.status.report_vad(p, rms)
        self.orchestrator.on_user_vad(
            speech, timestamp=t, probability=p)

        if self.state in ("speaking", "awaiting") or self._control_uttr:
            if speech and self.barge_in:
                self.stats.n_barge_in += 1
                if self.status is not None:
                    self.status.report_barge_in()
                self.audio.stop_playback()
                self._preserve_pending()
                log.info("BARGE-IN: interrupting agent (state was %s)",
                         self.state)
                self._start_turn(t)
                self._uttr = [frame]  # replace pre-roll with live frame
            elif self.command_control:
                self._capture_protected_speech(speech, frame, t)
            return

        if self.state == "listening":
            if speech:
                self._uttr.append(frame)
                self._speech_ms += self.frame_ms
                self._silence_ms = 0
            else:
                self._silence_ms += self.frame_ms
                self._uttr.append(frame)  # keep trailing silence in the slice
                if self._speech_ms >= self.min_speech_ms and \
                        self._silence_ms >= self.min_silence_ms:
                    self._end_turn(t)
                    return
                if self._speech_ms < self.min_speech_ms and \
                        self._silence_ms >= 4000:
                    log.info("no valid speech; discarding")
                    self._uttr = []
                    self.state = "idle"
            return

        # idle
        if speech:
            self.stats.n_turns += 1
            self._pre.append(frame)
            self._start_turn(t)
            self._uttr = [frame]
        else:
            self._pre.append(frame)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def _tool_event(kind: str, name: str, payload) -> None:
    """Log tool progress without logging file contents or credentials."""
    if kind == "start":
        log.info("tool start: %s", name)
    else:
        ok = payload.get("ok") if isinstance(payload, dict) else True
        log.info("tool finish: %s ok=%s", name, ok)


def cmd_check(args) -> int:
    import importlib

    print("ME3 agent --check")
    cfg = load_config()
    print(f"  config OK: {len(cfg)} top-level keys; target={cfg['deployment']['target']}; "
          f"audio_variant={cfg.get('runtime_variant', 'current')}")
    if cfg.get("runtime_variant") == "qwen_instruct":
        asr = cfg.get("asr", {})
        tts = cfg.get("tts", {})
        asr_key = bool(os.environ.get(asr.get("api_key_env", ""), "").strip())
        tts_key = bool(os.environ.get(tts.get("api_key_env", ""), "").strip())
        print("  Qwen audio: ASR model=%s (%s); TTS model=%s (%s)" % (
            asr.get("model", ""),
            "endpoint/key configured" if asr_key and asr.get("base_url_env") else "not configured",
            tts.get("model", ""),
            "endpoint/key configured" if tts_key and tts.get("base_url_env") else "not configured",
        ))
    for mod in ("pipecat", "mcp", "fastmcp", "faster_whisper", "piper",
                "onnxruntime", "sounddevice", "numpy", "yaml", "requests"):
        importlib.import_module(mod)
        print(f"  import OK: {mod}")
    import sounddevice as sd
    devs = sd.query_devices()
    print(f"  PortAudio OK: {len(devs)} devices")
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            print(f"    in : [{i}] {d['name']}")
    for i, d in enumerate(devs):
        if d.get("max_output_channels", 0) > 0:
            print(f"    out: [{i}] {d['name']}")
    for provider_name, selection in (("Qwen", "qwen"), ("GPT-5.6 Luna", "luna")):
        _provider, llm = resolve_llm_config(cfg, selection)
        if llm is None:
            print(f"  LLM brain: {provider_name} (not configured)")
            continue
        have_key = bool(os.environ.get(llm["api_key_env"], ""))
        print(f"  LLM brain: {llm['model']} @ {llm['base_url']} "
              f"({llm['api_key_env']} {'configured' if have_key else 'not set'}; direct stream; reasoning disabled)")
    from .channels import channel_health
    channel_state = channel_health()["channels"]
    print("  external channels: " + ", ".join(
        f"{name}={'configured' if state['configured'] else 'not configured'}"
        for name, state in channel_state.items()))
    if args.load_models:
        loop = LocalVoiceLoop(cfg)
        loop._load_engines()
        print("  engines loaded (VAD/ASR/TTS) OK")
    print("CHECK PASS")
    return 0


def main() -> None:
    # Windows consoles default to cp1252; device names can contain chars it
    # cannot encode. Never crash on output encoding.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="ME3 full-duplex voice agent (local dev loop).")
    parser.add_argument("--check", action="store_true",
                        help="verify config, imports and audio devices, then exit")
    parser.add_argument("--load-models", action="store_true",
                        help="with --check: also load the VAD/ASR/TTS engines")
    parser.add_argument("--device", default=None,
                        help="input device index or name substring (default: system default)")
    parser.add_argument("--speaker", default=None,
                        help="local output device index or name substring")
    parser.add_argument("--ssh-mic", default=None, metavar="HOST",
                        help="stream Pi arecord input over SSH; play TTS locally")
    parser.add_argument("--remote-device", default="hw:2,0",
                        help="Pi ALSA capture device for --ssh-mic (default: hw:2,0)")
    parser.add_argument("--brain", choices=("auto", "remote", "qwen", "luna", "stub"),
                        default="auto", help="LLM brain selection")
    parser.add_argument("--variant", choices=("current", "qwen_instruct"),
                        default=None,
                        help="audio path: current local/ElevenLabs or qwen_instruct DGX3 ASR/TTS")
    parser.add_argument("--text", default=None,
                        help="run one text turn through the selected brain and exit")
    parser.add_argument("--status-port", type=int, default=8126,
                        help="serve the live assistant/roadmap UI on "
                             "127.0.0.1:<port> (default 8126; 0 disables)")
    args = parser.parse_args()

    if args.variant:
        os.environ["ME3_VARIANT"] = args.variant
    cfg = load_config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = None if args.check else PROJECT / "runs" / f"local-run-{stamp}.log"
    _setup_logging(log_file)

    if args.check:
        raise SystemExit(cmd_check(args))

    if log_file:
        log.info("log file: %s", log_file)
    log.info("audio variant: %s", cfg.get("runtime_variant", "current"))

    if args.brain == "stub":
        brain = StubBrain()
    else:
        try:
            selection = "qwen" if args.brain == "remote" else args.brain
            system_prompt = build_system_prompt(cfg["agent"]["system_prompt"])
            brain = build_tool_brain(
                cfg, selection, system_prompt, on_tool=_tool_event)
            if selection in {"qwen", "luna"}:
                cfg.setdefault("llm", {})["active_provider"] = selection
        except RuntimeError as exc:
            if args.brain != "auto":
                raise SystemExit(str(exc))
            brain = StubBrain()

    if args.text is not None:
        try:
            reply = brain.chat([], args.text)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"text turn failed: {exc}")
        m = getattr(brain, "metrics", None)
        if isinstance(m, dict):
            if m.get("tool_calls"):
                print("tools: " + ", ".join(
                    f"{c['name']}({'ok' if c['ok'] else 'ERR'}) {c['ms']}ms"
                    for c in m["tool_calls"]))
            print(f"first_tool_ms={m['first_tool_ms'] if m['first_tool_ms'] is not None else '-'} "
                  f"llm_calls={m['n_llm_calls']} "
                  f"tokens={m['prompt_tokens']}+{m['completion_tokens']}")
        print(reply)
        return

    run_cfg = cfg
    if args.speaker is not None:
        audio_cfg = dict(cfg["audio"])
        audio_cfg["output_device"] = args.speaker
        run_cfg = {**cfg, "audio": audio_cfg}
    status = None
    if args.status_port:
        status = LoopStatus(
            release_info=load_release_info(PROJECT),
            brain_name=str(getattr(brain, "name", "brain")),
            variant=str(cfg.get("runtime_variant", "current")),
        )
    if args.ssh_mic:
        remote_audio = RemoteMicAudio(
            run_cfg, ssh_host=args.ssh_mic, remote_device=args.remote_device)
        loop = LocalVoiceLoop(run_cfg, brain=brain, audio=remote_audio,
                              status=status)
        log.info("split-device mode: Pi %s/%s -> local speaker %s",
                 args.ssh_mic, args.remote_device,
                 args.speaker or "system default")
    else:
        loop = LocalVoiceLoop(run_cfg, brain=brain, device=args.device,
                              status=status)
    status_server = None
    if args.status_port:
        status_server = StatusServer(
            status,
            html_dir=PROJECT / "web",
            project=PROJECT,
            get_state=lambda: loop.state,
            get_stats=lambda: loop.stats,
            port=args.status_port,
            demo_light_get=me2_commands.simulated_light_state,
            demo_light_control=lambda payload: me2_commands.control_lights(**payload),
        )
        if status_server.start():
            log.info("status UI: http://127.0.0.1:%d/ (roadmap: /roadmap)",
                     status_server.port)
            print(f"Assistant UI : http://127.0.0.1:{status_server.port}/")
            print(f"Roadmap UI   : http://127.0.0.1:{status_server.port}/roadmap")
        else:
            log.warning("status UI disabled: 127.0.0.1:%d unavailable",
                        args.status_port)
    try:
        loop.run()
    finally:
        if status_server is not None:
            status_server.stop()


if __name__ == "__main__":
    main()
