"""ME3 Full-Duplex Voice Agent.

Wires VAD -> ASR -> LLM -> TTS with barge-in (interruptible), running the
local dev loop on the Windows dev box (ME2-style launcher) and the same
control logic on the RPi (Pipecat pipeline).

  mic -> [Silero VAD] --speech--> [faster-whisper ASR] -> [LLM] -> [Piper TTS] -> speaker
                                        ^^^^^^^ barge-in: new speech cancels TTS ^^^^^

Run locally (the voice loop; also what launchers/Start-ME3.bat starts):
    python -m me3.agent                    # live full-duplex loop
    python -m me3.agent --check            # verify config/deps/audio devices
    python -m me3.agent --check --load-models   # + load VAD/ASR/TTS engines
    python -m me3.agent --device "Fifine"  # pick a specific input device
    python -m me3.agent --ssh-mic dalmacio@cfdfnjrpi5.local \
        --remote-device hw:2,0  # Pi mic -> this host's speakers
    python -m me3.agent --brain stub       # canned replies (no LLM needed)

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
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audio_io import FullDuplexAudio, RemoteMicAudio, resample_to
from .brain import ToolBrain, build_system_prompt
from .config import load_config
from .vocalis import TurnState, VocalisOrchestrator

log = logging.getLogger("me3.agent")

PROJECT = Path(__file__).resolve().parent.parent


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
            backchannel=self.agent.get("backchannel", "Let me work on that."),
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

        llm_cfg = self.cfg["llm"]["remote"]
        ctx = LLMContext([
            {"role": "system", "content": self.agent["system_prompt"]}
        ])
        api_key = os.environ.get(llm_cfg["api_key_env"], "")
        if not api_key:
            raise RuntimeError(
                f"{llm_cfg['api_key_env']} is required for the Pipecat "
                "remote pipeline; use LocalVoiceLoop --brain stub for an "
                "offline smoke test"
            )
        llm = OpenAILLMService(
            api_key=api_key,
            base_url=llm_cfg["base_url"],
            settings=OpenAILLMService.Settings(
                model=llm_cfg["model"],
                temperature=llm_cfg.get("temperature"),
                max_tokens=llm_cfg.get("max_tokens"),
            ),
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
        return ("The LLM endpoint is not configured, so I am using the stub "
                "brain. Set VLLM_API_KEY to enable the Qwen brain. I heard: "
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
                f"errors={self.n_errors} "
                f"asr_med/p95={_ms(self.asr_ms)}ms llm={_ms(self.llm_ms)}ms "
                f"tts={_ms(self.tts_ms)}ms e2e={_ms(self.e2e_ms)}ms "
                f"first_tool_med={ft}")


class LocalVoiceLoop:
    """The live full-duplex loop: mic + speaker always open, barge-in capable.

    State: idle -> (speech) -> listening -> (silence) -> thinking
           -> (reply) -> speaking -> (done | barge-in -> listening)
    """

    VAD_CHUNK = 512  # silero frame size @16 kHz

    def __init__(self, cfg: dict | None = None, brain=None, device=None,
                 audio=None, orchestrator: VocalisOrchestrator | None = None):
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

        self.ack_wait_s = float(agent.get("ack_wait_s", 2.5))
        self.backchannel = str(agent.get("backchannel", "Let me work on that."))
        self._pending = None  # (thread, holder, t_start, asr_ms) long task
        self._tts_lock = threading.Lock()
        self.orchestrator = orchestrator or VocalisOrchestrator(
            sample_rate=self.sr,
            backchannel=self.backchannel,
        )
        self._brain_on_tool = None
        if self.brain is not None:
            callback = getattr(self.brain, "on_tool", None)
            if callable(callback):
                self._brain_on_tool = callback
                self.brain.on_tool = self._on_brain_tool

        self.state = "idle"
        self._pre: deque[np.ndarray] = deque(
            maxlen=max(1, self.pre_roll_ms // self.frame_ms))
        self._uttr: list[np.ndarray] = []
        self._speech_ms = 0
        self._silence_ms = 0
        self._vad_pending = np.zeros(0, dtype=np.float32)
        self._engines = None   # (vad_model, unused, asr_engine, asr_cfg, tts_engine, tts_cfg)

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
        self._engines = (vad_model, None, asr_engine, asr_cfg,
                         tts_engine, tts_cfg)
        log.info("models loaded: vad=%.0f ms asr=%.0f ms (%s) tts=%.0f ms (%s @ %d Hz)",
                 (t1 - t0) * 1e3, (t2 - t1) * 1e3, asr_cfg["model"],
                 (t3 - t2) * 1e3, tts_cfg["voice"],
                 getattr(getattr(tts_engine, "config", None), "sample_rate", 0) or 22050)

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
    def _process_turn(self, utterance: np.ndarray, t_start: float) -> None:
        _vad_model, _unused, asr_engine, asr_cfg, tts_engine, tts_cfg = self._engines
        self.state = "thinking"

        t0 = time.perf_counter()
        segments, _info = asr_engine.transcribe(
            utterance.astype(np.float32) / 32768.0,
            beam_size=asr_cfg.get("beam_size", 1),
            language=asr_cfg.get("language", "en"),
            vad_filter=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        asr_ms = (time.perf_counter() - t0) * 1e3
        self.stats.asr_ms.append(asr_ms)
        if not text:
            log.info("turn ignored (empty transcript) [asr %.0f ms]", asr_ms)
            self.state = "idle"
            self.orchestrator.set_state(TurnState.LISTENING, "empty transcript")
            return
        log.info("heard: %r [asr %.0f ms]", text, asr_ms)
        self.orchestrator.on_transcript_delta(text, final=True,
                                              timestamp=t_start)

        t0 = time.perf_counter()
        if self.ack_wait_s > 0:
            # Long-task path: run the (tool-calling) brain on a worker
            # thread. If it finishes within ack_wait_s, speak it like a
            # normal turn; otherwise acknowledge with a backchannel and
            # announce the result when it completes. Barge-in during the
            # wait discards the pending announcement (the background work
            # is allowed to finish; e.g. a started web server keeps running).
            holder: dict = {}

            def _run_brain() -> None:
                holder["reply"] = self.brain.chat(self.history, text)

            th = threading.Thread(target=_run_brain, daemon=True,
                                  name="me3-turn")
            th.start()
            th.join(self.ack_wait_s)
            if th.is_alive():
                self._pending = (th, holder, t_start, asr_ms)
                self.history.append({"role": "user", "content": text})
                self.stats.llm_ms.append((time.perf_counter() - t0) * 1e3)
                log.info("long task running in background; backchannel")
                self.orchestrator.begin_backchannel(self.backchannel)
                self._speak_text(tts_engine, self.backchannel, t_start,
                                 asr_ms, (time.perf_counter() - t0) * 1e3)
                self.state = "awaiting"
                return
            reply = holder["reply"]
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
        self._speak_text(tts_engine, reply, t_start, asr_ms, llm_ms)

    def _speak_text(self, tts_engine, text: str, t_start: float,
                    asr_ms: float, llm_ms: float,
                    count_e2e: bool = True) -> None:
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
        with self._tts_lock:
            backend = getattr(tts_engine, "provider", "piper")
            event = getattr(tts_engine, "last_event", None) or {}
            log.info(
                "speaking backend=%s duration=%.2f s [tts %.0f ms e2e %.0f "
                "ms sample_rate=%d event=%s]",
                backend, len(out) / self.sr, tts_ms,
                (time.perf_counter() - t_start) * 1e3, tts_sr,
                {k: event[k] for k in ("ok", "attempts", "latency_ms",
                                       "bytes", "error_class") if k in event})
            self.audio.play(out, blocking=False)
        self.state = "speaking"

    def _drop_pending(self) -> None:
        """User barged in: discard the pending announcement.

        The worker thread keeps finishing (its side effects, e.g. a web
        server, are useful); only the spoken reply is abandoned.
        """
        if self._pending is not None:
            self._pending = None
            log.info("pending reply discarded (barge-in)")

    def _start_turn(self, t: float) -> None:
        self._uttr = list(self._pre)
        self._speech_ms = 0
        self._silence_ms = 0
        self._reset_vad()
        self.state = "listening"
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
            log.exception("turn processing failed: %s", exc)

    # ---- Main loop ----------------------------------------------------------
    def run(self) -> None:
        a = self.cfg["audio"]
        log.info("ME3 Voice Duplex - local full-duplex loop")
        log.info("  sr=%d frame=%d ms barge_in=%s brain=%s",
                 self.sr, self.frame_ms, self.barge_in, self.brain.name)
        self._load_engines()
        self.audio.start_capture()
        self.audio.startup_signal_check()
        self.orchestrator.set_state(TurnState.LISTENING, "audio capture started")
        log.info("LISTENING. Speak to the microphone; interrupt me mid-sentence "
                 "to barge in. Ctrl+C stops.")

        try:
            while True:
                if self._pending is not None and \
                        self.state in ("awaiting", "idle", "speaking"):
                    th, holder, t_start, asr_ms = self._pending
                    if not th.is_alive():
                        self._pending = None
                        reply = holder.get("reply", "")
                        self.history.append(
                            {"role": "assistant", "content": reply})
                        self.history = self.history[-8:]
                        m = getattr(self.brain, "metrics", None)
                        if isinstance(m, dict) \
                                and m.get("first_tool_ms") is not None:
                            self.stats.first_tool_ms.append(
                                m["first_tool_ms"])
                        log.info("background task finished; announcing")
                        if reply:
                            self.orchestrator.on_assistant_text_delta(
                                reply, final=True)
                            self._speak_text(self._engines[3], reply,
                                             t_start, asr_ms, 0.0,
                                             count_e2e=False)
                        elif self.state in ("awaiting",):
                            self.state = "idle"
                            self.orchestrator.set_state(
                                TurnState.LISTENING,
                                "background task returned empty",
                            )
                frame = self.audio.read_frame(timeout=0.5)
                if frame is None:
                    if self.state == "speaking" and self.audio.playback_done():
                        self.state = "idle"
                        self.orchestrator.on_tts_complete()
                        log.info("agent finished speaking; idle")
                    continue
                for p in self._vad_probs(frame):
                    self._on_vad_frame(p, frame)
                if self.state == "speaking" and self.audio.playback_done():
                    self.state = "idle"
                    self.orchestrator.on_tts_complete()
                    log.info("agent finished speaking; idle")
        except KeyboardInterrupt:
            pass
        finally:
            self.audio.close()
            log.info("stopped. %s", self.stats.summary())

    def _on_vad_frame(self, p: float, frame: np.ndarray) -> None:
        speech = p >= self.vad_thr
        t = time.perf_counter()
        self.orchestrator.on_user_vad(
            speech, timestamp=t, probability=p)

        if self.state in ("speaking", "awaiting"):
            if speech and self.barge_in:
                self.stats.n_barge_in += 1
                self.audio.stop_playback()
                self._drop_pending()
                log.info("BARGE-IN: interrupting agent (state was %s)",
                         self.state)
                self._start_turn(t)
                self._uttr = [frame]  # replace pre-roll with live frame
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
    print(f"  config OK: {len(cfg)} top-level keys; target={cfg['deployment']['target']}")
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
    llm = cfg["llm"]["remote"]
    have_key = bool(os.environ.get(llm["api_key_env"], ""))
    print(f"  LLM brain: {llm['model']} @ {llm['base_url']} "
          f"({llm['api_key_env']} {'configured' if have_key else 'NOT set -> stub'})")
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
    parser.add_argument("--brain", choices=("auto", "remote", "stub"),
                        default="auto", help="LLM brain selection")
    parser.add_argument("--text", default=None,
                        help="run one text turn through the selected brain and exit")
    args = parser.parse_args()

    cfg = load_config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = None if args.check else PROJECT / "runs" / f"local-run-{stamp}.log"
    _setup_logging(log_file)

    if args.check:
        raise SystemExit(cmd_check(args))

    if log_file:
        log.info("log file: %s", log_file)

    if args.brain == "stub":
        brain = StubBrain()
    else:
        brain_cfg = dict(cfg["llm"]["remote"])
        brain_cfg["system_prompt"] = build_system_prompt(cfg["agent"]["system_prompt"])
        try:
            brain = ToolBrain(brain_cfg, brain_cfg["system_prompt"], on_tool=_tool_event)
        except RuntimeError:
            env = cfg["llm"]["remote"]["api_key_env"]
            if args.brain == "remote":
                raise SystemExit(f"--brain remote requires env var {env}")
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
    if args.ssh_mic:
        remote_audio = RemoteMicAudio(
            run_cfg, ssh_host=args.ssh_mic, remote_device=args.remote_device)
        loop = LocalVoiceLoop(run_cfg, brain=brain, audio=remote_audio)
        log.info("split-device mode: Pi %s/%s -> local speaker %s",
                 args.ssh_mic, args.remote_device,
                 args.speaker or "system default")
    else:
        loop = LocalVoiceLoop(run_cfg, brain=brain, device=args.device)
    loop.run()


if __name__ == "__main__":
    main()
