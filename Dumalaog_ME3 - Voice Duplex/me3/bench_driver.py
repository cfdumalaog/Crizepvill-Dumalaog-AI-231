"""ME3 engineering latency benchmark driver (internal loopback run).

Drives the REAL LocalVoiceLoop (VAD -> ASR -> LLM -> TTS, barge-in capable)
end-to-end over this laptop's Stereo Mix loopback: the driver plays the
canonical stimulus WAVs into the output device, the agent hears them through
the loopback capture exactly as it would hear a user, and the driver stamps
every FDB boundary event with time.perf_counter().

Method (echoed into results.json "protocol"):
  * One LocalVoiceLoop for the whole run; input device = "Stereo Mix"
    (WDM-KS), output = the default working speakers.
  * Loopback round-trip latency (RT) is calibrated once per run with tone
    pings (play a 1 kHz tone; first audible capture frame).
  * Echo gating: while the agent is speaking (non-barge-in scenarios) the
    driver does NOT feed captured frames into the VAD, so the agent never
    "hears" its own TTS. For barge-in trials the feed is re-enabled exactly
    when the interrupt stimulus starts, right after a fresh queue drain that
    discards stale frames containing the agent's own speech.
  * Boundary events per trial (perf_counter domain):
      T_stim_play    driver calls play(stimulus)              (speaker side)
      T_vad_trigger  first VAD speech frame fed into the loop (capture side)
      T_endpoint     VAD endpoint that closed the turn        (capture side)
      T_asr_done     T_endpoint + asr_ms
      T_llm_done     T_asr_done + llm_ms
      T_tts_first    first TTS audio chunk produced (instrumented)
      T_agent_play   agent TTS sent to speaker, first byte (instrumented)
      T_agent_stop   stop_playback() triggered by barge-in   (instrumented)
  * Primary metrics per scenario (config.yaml benchmark section):
      wake_to_first_word     = T_tts_first - T_vad_trigger
      utterance_to_llm_token = llm_ms (PROXY: the synchronous brain has no
                                TTFT instrumentation; llm_ms is full-reply)
      llm_token_to_tts_audio = T_tts_first - T_llm_done
      end_to_end             = T_agent_play - (T_stim_play + stim_dur)
                               (user's last audible ms -> agent's first ms)
      barge_in               = T_agent_stop - T_interrupt
  * Legacy FDB-v1 behavior metrics (me3/benchmark.py, aligned to the FDB repo):
      agent segments = each spoken reply (known text + measured audio span);
      ME3 has no mid-turn backchannel mechanism, so n_bc = 0, Freq = 0 and
      JSD = 1.0 (FDB repo convention for zero backchannel predictions).

This driver is not the paper-native FULL-DUPLEX-BENCH v1.5 run.  The
paper-native overlap contract and offline evaluator live in ``me3/fdb_v15.py``
and ``scripts/fdb_v15_eval.py``.  Keeping the engineering latency run
separate prevents a five-scenario latency smoke test from being reported as
the paper's four-scenario, 499-utterance overlap evaluation.

Outputs (contract: subagent_briefs/fdb_benchmark.md):
  runs/fdb_benchmark/<UTC-stamp>/results.json
  runs/fdb_benchmark/<UTC-stamp>/summary.md
  runs/fdb_benchmark/<UTC-stamp>/driver.log
  runs/fdb_benchmark_latest.json
  subagent_status/fdb_benchmark.json

The vLLM key is taken from env VLLM_API_KEY when set, else from the DPAPI
stored helper (C:\\Users\\danda\\.codex\\Get-VllmApiKey.ps1). It is kept in
process memory only and never written to any file or log line.

Usage:
  python -m me3.bench_driver --smoke          # 1 scenario, 1 warmup + 2 trials
  python -m me3.bench_driver                  # full run (config defaults)
  python -m me3.bench_driver --trials 5 --scenarios end_to_end barge_in
  python -m me3.bench_driver --brain stub --trials 5  # offline/local brain
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from me3.agent import LocalVoiceLoop, RemoteBrain, StubBrain  # noqa: E402
from me3.benchmark import (  # noqa: E402
    DATASET,
    evaluate_full_duplex,
    fdb_jsd_distance,
    jensen_shannon_divergence,
    session_takeover_flag,
    takeover_rate,
)
from me3.config import load_config  # noqa: E402
from me3.vad_onnx import frame_probability  # noqa: E402

log = logging.getLogger("me3.bench_driver")

CANONICAL = list(DATASET)

SCENARIOS = [
    "wake_to_first_word",
    "utterance_to_llm_token",
    "llm_token_to_tts_audio",
    "end_to_end",
    "barge_in",
]
BARGE_STIMULUS = "tell_me_system_status"
BARGE_INTERRUPT = "stop"
# "stop" is a control command (barge-in interrupt), not a conversational
# latency stimulus; keep it out of the latency-scenario cycle.
LATENCY_CANONICAL = [n for n in CANONICAL if n != BARGE_INTERRUPT]
ICC_PATH = (PROJECT / "refs" / "full-duplex-bench" / "v1_v1.5"
            / "evaluation" / "icc_gt_distribution.json")
HUMAN_MANIFEST = PROJECT / "data" / "bench" / "human_manifest.json"


def load_human_set() -> dict:
    """ME2-recorded human stimulus sets (scripts/curate_human_bench.py)."""
    if not HUMAN_MANIFEST.exists():
        raise SystemExit(f"human manifest missing: {HUMAN_MANIFEST}")
    with open(HUMAN_MANIFEST, encoding="utf-8") as f:
        d = json.load(f)
    return {
        "latency": [e["name"] for e in d["entries"]
                    if e["kind"] == "latency"],
        "stop": [e["name"] for e in d["entries"] if e["kind"] == "stop"],
    }


def load_vllm_key() -> str:
    """VLLM_API_KEY from env, else DPAPI helper. Never printed/persisted."""
    key = os.environ.get("VLLM_API_KEY", "").strip()
    if key:
        return key
    helper = Path.home() / ".codex" / "Get-VllmApiKey.ps1"
    if not helper.exists():
        raise SystemExit(f"VLLM_API_KEY not set and helper missing: {helper}")
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(helper)],
        capture_output=True, text=True, timeout=30)
    key = (r.stdout or "").strip()
    if not key:
        raise SystemExit("VLLM_API_KEY not set and DPAPI helper returned "
                         f"empty (rc={r.returncode})")
    return key


def resolve_device(sd, wanted, *, input_required=True):
    """Resolve an explicit sounddevice index or name substring."""
    if wanted is None:
        return None
    devs = sd.query_devices()
    if isinstance(wanted, int) and 0 <= wanted < len(devs):
        d = devs[wanted]
        channels = (d.get("max_input_channels", 0) if input_required
                    else d.get("max_output_channels", 0))
        if channels > 0:
            return int(wanted)
    text = str(wanted).strip().lower()
    for i, d in enumerate(devs):
        channels = (d.get("max_input_channels", 0) if input_required
                    else d.get("max_output_channels", 0))
        if channels > 0 and (text == str(i) or
                             text in str(d.get("name", "")).lower()):
            return i
    return None


def pick_stereo_mix(sd):
    """Index of the first 'Stereo Mix' input device, else None."""
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0 and \
                "stereo mix" in str(d.get("name", "")).lower():
            return i
    return None


def load_wav16(path: Path) -> np.ndarray:
    """Load a mono int16 WAV, resampled to 16 kHz (bench fixture format)."""
    import wave
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1, f"{path}: not mono"
        assert w.getsampwidth() == 2, f"{path}: not int16"
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16)
    if sr != 16000:
        from me3.audio_io import resample_to
        pcm = resample_to(pcm, sr, 16000)
    return pcm


def resize_dist(vec, n: int) -> list[float]:
    """Linearly resize a distribution to n bins (FDB repo interp1d step)."""
    x = np.linspace(0.0, 1.0, len(vec))
    y = np.asarray(vec, dtype=np.float64)
    xn = np.linspace(0.0, 1.0, n)
    return list(np.interp(xn, x, y))



class BenchDriver:
    """In-process benchmark harness around one real LocalVoiceLoop."""

    def __init__(self, cfg: dict, in_idx: int, brain: RemoteBrain):
        self.cfg = cfg
        self.sr = int(cfg["audio"]["sample_rate"])
        self.frame_ms = int(cfg["audio"]["frame_ms"])
        self.b = cfg["benchmark"]
        self.loop = LocalVoiceLoop(cfg, brain=brain, device=in_idx)
        self.loop._load_engines()
        self.vad_model = self.loop._engines[0]
        self.tts_eng = self.loop._engines[4]

        # ---- instrumentation (harness-side; agent.py stays untouched) -----
        self.t_vad_trigger: float | None = None
        self.t_endpoint: float | None = None
        self.t_tts_first: float | None = None
        self.t_agent_play: float | None = None
        self.t_agent_stop: float | None = None
        self.agent_play_samples = 0
        self._play_context = "idle"
        self.main_play_abs: float | None = None
        self.main_play_samples = 0
        self.main_stop_abs: float | None = None
        self.fu_play_abs: float | None = None
        self.fu_samples = 0
        self.fixtures: dict[str, np.ndarray] = {}
        self.fixture_spans: dict[str, tuple[float, float]] = {}
        self.rt_ms: float | None = None
        self._last_frame: np.ndarray | None = None

        audio = self.loop.audio
        orig_play = audio.play

        def wrapped_play(pcm, blocking=False):
            if self._play_context == "agent":
                self.t_agent_play = time.perf_counter()
                self.agent_play_samples = int(len(pcm))
            return orig_play(pcm, blocking=blocking)
        audio.play = wrapped_play

        orig_stop = audio.stop_playback

        def wrapped_stop(silent=False):
            if not silent:
                self.t_agent_stop = time.perf_counter()
            return orig_stop(silent=silent)
        audio.stop_playback = wrapped_stop

        orig_end_turn = self.loop._end_turn

        def wrapped_end_turn(t):
            self.t_endpoint = time.perf_counter()
            return orig_end_turn(t)
        self.loop._end_turn = wrapped_end_turn

        orig_syn = self.tts_eng.synthesize

        def wrapped_syn(text):
            gen = orig_syn(text)
            it = iter(gen)
            try:
                first = next(it)
            except StopIteration:
                return
            self.t_tts_first = time.perf_counter()
            yield first
            yield from it
        self.tts_eng.synthesize = wrapped_syn

        # open the capture stream (LocalVoiceLoop.run() does this in the
        # live app; the driver owns its own frame loop)
        self.loop.audio.start_capture()

    # ---- helpers -----------------------------------------------------------
    def _drain(self, max_s: float = 1.0) -> None:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < max_s:
            if self.loop.audio.read_frame(timeout=0.02) is None:
                return

    def fixture(self, name: str) -> np.ndarray:
        if name not in self.fixtures:
            p = PROJECT / "data" / "bench" / f"{name}.wav"
            if not p.exists():
                raise SystemExit(f"missing fixture: {p}")
            self.fixtures[name] = load_wav16(p)
        return self.fixtures[name]

    def fixture_span(self, name: str) -> tuple[float, float]:
        """Silero speech span (start_s, end_s) of a fixture, computed once."""
        if name not in self.fixture_spans:
            pcm = self.fixture(name)
            x = pcm.astype(np.float32) / 32768.0
            if hasattr(self.vad_model, "reset_states"):
                self.vad_model.reset_states()
            idx = []
            for i in range(0, max(1, len(x) - 511), 512):
                p = frame_probability(self.vad_model, x[i:i + 512], self.sr)
                if p >= self.loop.vad_thr:
                    idx.append(i)
            if idx:
                self.fixture_spans[name] = (idx[0] * 512 / self.sr,
                                            (idx[-1] + 1) * 512 / self.sr)
            else:
                self.fixture_spans[name] = (0.0, len(x) / self.sr)
        return self.fixture_spans[name]

    def reset_trial(self) -> None:
        lp = self.loop
        lp.state = "idle"
        lp._uttr = []
        lp._pre.clear()
        lp._speech_ms = 0
        lp._silence_ms = 0
        lp._vad_pending = np.zeros(0, dtype=np.float32)
        lp._reset_vad()
        self.t_vad_trigger = None
        self.t_endpoint = None
        self.t_tts_first = None
        self.t_agent_play = None
        self.t_agent_stop = None
        self.agent_play_samples = 0
        self.main_play_abs = None
        self.main_play_samples = 0
        self.main_stop_abs = None
        self.fu_play_abs = None
        self.fu_samples = 0
        self._drain(1.0)
        # Clear any remaining stale capture backlog and wait for the
        # previous agent playback to finish draining before the next
        # stimulus (real-time with the non-blocking input queue).
        t_wait = time.perf_counter() + 5.0
        while time.perf_counter() < t_wait:
            if (lp.audio.read_frame(timeout=0.1) is None
                    and lp.audio.playback_done()):
                break
        # Audible-tail gate: after the driver buffer drains, the endpoint
        # keeps emitting for roughly the loopback RT (device latency).
        # Require a continuous quiet window (RMS < 500, same threshold as
        # the RT calibration) so residual agent speech cannot seed a
        # spurious VAD turn on the next trial.
        quiet_need = max(0.5, (self.rt_ms or 450.0) / 1000.0 + 0.2)
        quiet_have = 0.0
        t_gate_end = time.perf_counter() + 20.0
        while time.perf_counter() < t_gate_end:
            if quiet_have >= quiet_need:
                break
            f = lp.audio.read_frame(timeout=0.05)
            if f is None:
                quiet_have += 0.05
                continue
            rms = float(np.sqrt(np.mean(f.astype(np.float64) ** 2)))
            if rms < 500.0:
                quiet_have += 0.03
            else:
                quiet_have = 0.0

    def calibrate_rt(self, n_pings: int = 3) -> float:
        """Loopback round-trip: tone play -> first audible capture frame."""
        sr = self.sr
        t = np.arange(int(sr * 0.25), dtype=np.float64) / sr
        tone = 0.25 * np.sin(2 * np.pi * 1000 * t)
        fade = int(sr * 0.01)
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
        tone_pcm = (tone * 32767).astype(np.int16)
        ests = []
        for _ in range(n_pings):
            self._drain()
            self._play_context = "calibration"
            t_play = time.perf_counter()
            self.loop.audio.play(tone_pcm, blocking=False)
            first = None
            frames_seen = 0
            max_rms = 0.0
            t_end = t_play + 1.2
            while time.perf_counter() < t_end:
                f = self.loop.audio.read_frame(timeout=0.05)
                if f is None:
                    continue
                frames_seen += 1
                tf = time.perf_counter()
                # frames are int16 (audio_io resamples inside the callback);
                # digital-silence noise floor is RMS < ~2, the 0.25-amp tone
                # reads RMS > 5000 through the loopback. Detect on energy.
                rms = float(np.sqrt(np.mean(f.astype(np.float64) ** 2)))
                max_rms = max(max_rms, rms)
                if rms > 500.0:
                    first = tf
                    break
            if first is not None:
                ests.append((first - t_play) - self.frame_ms / 2000.0)
            self._drain()
            time.sleep(0.2)
        if not ests:
            in_idx = self.loop.audio.in_idx
            out_idx = self.loop.audio.out_idx
            raise SystemExit(
                "RT calibration failed: tone never detected in the loopback "
                f"capture (input_index={in_idx}, output_index={out_idx}, "
                f"capture_frames={frames_seen}, max_frame_rms="
                f"{max_rms:.1f}). Run scripts/pi_audio_preflight.py and "
                "verify that the selected capture device receives the "
                "selected playback signal before running trials.")
        self.rt_ms = float(statistics.median(ests)) * 1000.0
        log.info("loopback RT calibrated: %.1f ms (est: %s ms)", self.rt_ms,
                 ", ".join(f"{e * 1000:.0f}" for e in ests))
        return self.rt_ms

    def _stats_snapshot(self) -> dict:
        s = self.loop.stats
        return {"asr": len(s.asr_ms), "llm": len(s.llm_ms),
                "tts": len(s.tts_ms), "e2e": len(s.e2e_ms),
                "hist": len(self.loop.history)}

    def _stats_diff(self, snap: dict) -> dict:
        s = self.loop.stats
        out = {}
        for attr in ("asr", "llm", "tts", "e2e"):
            out[attr] = list(getattr(s, attr + "_ms")[snap[attr]:])
        hist = self.loop.history
        out["heard"] = [h["content"] for h in hist[snap["hist"]:]
                        if h["role"] == "user"]
        out["replies"] = [h["content"] for h in hist[snap["hist"]:]
                          if h["role"] == "assistant"]
        return out

    def _feed_frame(self, frame, tf: float) -> None:
        """Dispatch one captured frame through the loop's VAD path."""
        prev = self._play_context
        self._play_context = "agent"
        try:
            for p in self.loop._vad_probs(frame):
                if p >= self.loop.vad_thr and self.t_vad_trigger is None:
                    self.t_vad_trigger = tf
                self.loop._on_vad_frame(p, frame)
        finally:
            self._play_context = prev

    def _idle_transition(self) -> None:
        if self.loop.state == "speaking" and self.loop.audio.playback_done():
            self.loop.state = "idle"

    # ---- trial runner --------------------------------------------------------
    def run_trial(self, scenario: str, stim_name: str, warmup: bool,
                  interrupt_name: str | None = None) -> dict:
        lp = self.loop
        stim = self.fixture(stim_name)
        stim_dur = len(stim) / self.sr
        span_s, span_e = self.fixture_span(stim_name)
        rec = {
            "scenario": scenario, "stimulus": stim_name,
            "warmup": warmup, "valid": True, "invalid_reason": None,
            "stim_dur_s": round(stim_dur, 3),
            "stim_speech_span_s": [round(span_s, 3), round(span_e, 3)],
        }
        self.reset_trial()
        snap = self._stats_snapshot()
        self._play_context = "stimulus"
        t_stim = time.perf_counter()
        rec["t_stim_abs"] = t_stim
        lp.audio.play(stim, blocking=False)

        is_barge = scenario == "barge_in"
        interrupt = (self.fixture(interrupt_name or BARGE_INTERRUPT)
                     if is_barge else None)
        interrupt_dur = len(interrupt) / self.sr if interrupt is not None \
            else 0.0
        interrupt_delay = float(self.b.get("interrupt_delay_s", 1.2))
        deadline = t_stim + 40.0
        t_interrupt: float | None = None
        barge_watchdog: float | None = None
        interrupt_played = False
        n_diag = 0
        diag_peak = 0.0
        t_diag = time.perf_counter()

        while time.perf_counter() < deadline:
            tf_hb = time.perf_counter()
            if tf_hb - t_diag >= 2.0:
                log.info(
                    "[hb %s] frame=%s n=%d peak=%.0f state=%s q=%d "
                    "vad_trigger=%s t+%.1fs",
                    scenario,
                    ("yes" if self._last_frame is not None else "NO"),
                    n_diag, diag_peak, lp.state, lp.audio._q.qsize(),
                    (f"{self.t_vad_trigger - t_stim:+.2f}s"
                     if self.t_vad_trigger is not None else "-"),
                    tf_hb - t_stim)
                n_diag = 0
                diag_peak = 0.0
                t_diag = tf_hb
            frame = lp.audio.read_frame(timeout=0.2)
            self._last_frame = frame
            if frame is None:
                self._idle_transition()
                continue
            tf = time.perf_counter()
            self._idle_transition()
            n_diag += 1
            pk = float(np.abs(frame).max())
            if pk > diag_peak:
                diag_peak = pk

            if is_barge and self.t_agent_stop is None:
                # --- pre-barge-in: wait for the agent, then interrupt -------
                if not interrupt_played:
                    if lp.state == "speaking":
                        if (tf - (self.t_agent_play or t_stim)
                                >= interrupt_delay):
                            self.main_play_abs = self.t_agent_play
                            self.main_play_samples = self.agent_play_samples
                            self._drain(0.3)  # stale agent-speech frames
                            self._play_context = "interrupt"
                            t_interrupt = time.perf_counter()
                            lp.audio.play(interrupt, blocking=False)
                            self._play_context = "idle"
                            interrupt_played = True
                            barge_watchdog = time.perf_counter() + 3.0
                    elif lp.state in ("idle", "listening", "thinking"):
                        self._feed_frame(frame, tf)
                        if (lp.state == "idle"
                                and self.t_endpoint is not None
                                and not lp._uttr
                                and tf - self.t_endpoint > 1.0):
                            rec["valid"] = False
                            rec["invalid_reason"] = (
                                "no agent speech to interrupt")
                            break
                    # state == "speaking" before window: keep draining
                else:
                    # interrupt playing; feed so VAD can fire barge-in
                    self._feed_frame(frame, tf)
                    if (barge_watchdog is not None
                            and time.perf_counter() > barge_watchdog):
                        rec["valid"] = False
                        rec["invalid_reason"] = (
                            "barge-in not detected in 3 s")
                        break
            else:
                # --- post-barge-in (or normal scenarios): gate own TTS ------
                # Break on idle BEFORE feeding the frame, so the tail of the
                # agent's own speech (still in the queue) can never trigger
                # an echo turn and burn a wasted ASR/LLM pass.
                if lp.state == "idle" and self.t_agent_play is not None:
                    break
                if (lp.state == "idle" and self.t_endpoint is not None
                        and not lp._uttr and self.t_agent_play is None
                        and (tf - self.t_endpoint) > 1.0):
                    rec["valid"] = False
                    rec["invalid_reason"] = (
                        "turn ignored by pipeline (empty transcript)")
                    break
                if lp.state == "speaking":
                    if is_barge and self.fu_play_abs is None \
                            and self.t_agent_play is not None \
                            and self.t_agent_play != self.main_play_abs:
                        self.fu_play_abs = self.t_agent_play
                        self.fu_samples = self.agent_play_samples
                    continue
                self._feed_frame(frame, tf)
        rec["t_end_abs"] = time.perf_counter()

        # barge-in success is the stop event itself (follow-up is bonus)
        if is_barge and self.t_agent_stop is not None \
                and self.main_play_abs is None:
            self.main_play_abs = self.t_agent_play
        if is_barge and self.t_agent_stop is not None:
            self.main_stop_abs = self.t_agent_stop
        if rec["valid"] and self.t_agent_play is None and not is_barge:
            rec["valid"] = False
            rec["invalid_reason"] = "timeout: agent never spoke"
        if rec["valid"] and is_barge and self.t_agent_stop is None \
                and rec["invalid_reason"] is None:
            rec["valid"] = False
            rec["invalid_reason"] = "no barge-in stop captured"

        diff = self._stats_diff(snap)
        asr_ms = diff["asr"][0] if diff["asr"] else None
        llm_ms = diff["llm"][0] if diff["llm"] else None
        tts_ms = diff["tts"][0] if diff["tts"] else None
        e2e_int_ms = diff["e2e"][0] if diff["e2e"] else None
        heard = diff["heard"][0] if diff["heard"] else None
        reply = diff["replies"][0] if diff["replies"] else None
        followup_reply = diff["replies"][1] if len(diff["replies"]) > 1 \
            else None

        def ms(a, b):
            return round((b - a) * 1000.0, 1) if (a is not None
                                                  and b is not None) else None

        t_llm_done = None
        if self.t_endpoint is not None and asr_ms is not None \
                and llm_ms is not None:
            t_llm_done = self.t_endpoint + (asr_ms + llm_ms) / 1000.0

        if is_barge and self.main_play_abs is not None:
            main_play_samples = self.main_play_samples
        else:
            main_play_samples = self.agent_play_samples
        full_dur_s = main_play_samples / self.sr if main_play_samples else None
        interrupted_s = None
        if is_barge and self.main_play_abs is not None \
                and self.main_stop_abs is not None:
            interrupted_s = self.main_stop_abs - self.main_play_abs

        rec.update({
            "heard": heard, "reply": reply,
            "followup_reply": followup_reply,
            "reply_n_words": len(reply.split()) if reply else None,
            "reply_full_dur_s": round(full_dur_s, 3) if full_dur_s else None,
            "interrupted_s": round(interrupted_s, 3)
            if interrupted_s is not None else None,
            "rt_ms": round(self.rt_ms, 1) if self.rt_ms else None,
            "vad_trigger_ms": ms(t_stim, self.t_vad_trigger),
            "endpoint_ms": ms(t_stim, self.t_endpoint),
            "asr_ms": asr_ms, "llm_ms_full": llm_ms,
            "tts_ttfb_ms": ms(t_llm_done, self.t_tts_first),
            "tts_total_ms": tts_ms,
            "audio_out_ms": ms(self.t_tts_first,
                               self.main_play_abs if is_barge
                               else self.t_agent_play),
            "e2e_internal_ms": e2e_int_ms,
            "wake_to_first_word_ms": ms(self.t_vad_trigger, self.t_tts_first),
            "end_to_end_ms": ms(t_stim + stim_dur,
                                self.main_play_abs if is_barge
                                else self.t_agent_play),
            "t_agent_play_abs": (self.main_play_abs if is_barge
                                 else self.t_agent_play),
            "main_play_samples": main_play_samples,
            "fu_play_abs": self.fu_play_abs, "fu_samples": self.fu_samples,
            "main_stop_abs": self.main_stop_abs,
        })
        if is_barge:
            rec.update({
                "interrupt_delay_s": interrupt_delay,
                "interrupt_onset_ms": ms(t_stim, t_interrupt),
                "interrupt_dur_s": round(interrupt_dur, 3),
                "barge_in_latency_ms": ms(t_interrupt, self.main_stop_abs),
            })
        try:
            import psutil
            rec["cpu_pct"] = round(psutil.cpu_percent(interval=0.02), 1)
            rec["ram_mb"] = round(psutil.virtual_memory().used / 1e6, 1)
        except Exception:  # noqa: BLE001
            pass
        rec["frame_ms"] = float(self.cfg["audio"]["frame_ms"])

        log.info("[%s|%s|%s] valid=%s e2e=%s barge=%s heard=%r reply=%r",
                 scenario, "warmup" if warmup else "trial", stim_name,
                 rec["valid"], rec.get("end_to_end_ms"),
                 rec.get("barge_in_latency_ms"), heard,
                 (reply[:60] + "...") if reply and len(reply) > 60 else reply)
        return rec


    # ---- FDB behavior --------------------------------------------------------
    def agent_segments_for(self, rec: dict, t0: float) -> list[dict]:
        """Agent spoken segments from one valid trial, on the scenario
        timeline starting at t0. Segments come from the known reply text +
        measured audio span (speaker-side times): exact for what the agent
        emitted. For barge-in trials the interrupted reply is truncated at
        T_agent_stop, with n_words scaled proportionally (an estimate)."""
        segs = []
        if not rec.get("valid") or rec.get("t_agent_play_abs") is None:
            return segs
        s0 = rec["t_agent_play_abs"] - t0
        full = rec.get("reply_full_dur_s") or 0.0
        n_words = rec.get("reply_n_words") or 0
        if rec["scenario"] == "barge_in" and rec.get("interrupted_s") \
                is not None and full > 0:
            d = min(full, rec["interrupted_s"])
            w = max(1, round(n_words * d / full)) if d > 0 else 0
        else:
            d = full
            w = n_words
        if d > 0 and w > 0:
            segs.append({"start_s": round(s0, 3), "end_s": round(s0 + d, 3),
                         "n_words": int(w)})
        if rec["scenario"] == "barge_in" and rec.get("fu_play_abs") \
                and rec.get("followup_reply") and rec.get("fu_samples"):
            s1 = rec["fu_play_abs"] - t0
            d1 = rec["fu_samples"] / self.sr
            w1 = len(rec["followup_reply"].split())
            segs.append({"start_s": round(s1, 3),
                         "end_s": round(s1 + d1, 3), "n_words": int(w1)})
        return segs

    def fdb_for_scenario(self, name: str, recs: list, q: list[float]) -> dict:
        valid = [r for r in recs if r.get("valid") and not r.get("warmup")]
        if not valid:
            return {"scenario": name, "n_trials_valid": 0,
                    "note": "no valid trials"}
        t0 = valid[0]["t_stim_abs"]
        t1 = max(r["t_end_abs"] for r in valid)
        total_audio_s = max(t1 - t0, 1.0)
        segs = []
        for r in valid:
            segs.extend(self.agent_segments_for(r, t0))
        user_speech_s = sum(
            (r["stim_speech_span_s"][1] - r["stim_speech_span_s"][0])
            for r in valid)
        jsd_fdb = None
        jsd_true = None
        if q:
            rep = evaluate_full_duplex(
                name, segs, user_speech_s,
                human_bc_times=list(q), t_max=total_audio_s)
            jsd_fdb = rep.backchannel_jsd
        out = {
            "scenario": name,
            "n_trials_valid": len(valid),
            "total_audio_s": round(total_audio_s, 2),
            "user_speech_s": round(user_speech_s, 2),
            "agent_segments": segs,
            "takeover_rate_per_segment": round(takeover_rate(segs), 3),
            "session_takeover_flag": session_takeover_flag(segs),
            "backchannel_count": sum(
                1 for s in segs
                if s["end_s"] - s["start_s"] < 1.0 and s["n_words"] <= 2),
            "backchannel_freq_per_user_speech_s":
                round(sum(1 for s in segs
                          if s["end_s"] - s["start_s"] < 1.0
                          and s["n_words"] <= 2) / user_speech_s, 4)
                if user_speech_s > 0 else 0.0,
            "backchannel_freq_per_total_audio_s":
            round(sum(1 for s in segs
                      if s["end_s"] - s["start_s"] < 1.0
                      and s["n_words"] <= 2) / total_audio_s, 4),
            "backchannel_jsd_fdb_convention": jsd_fdb,
            "note": ("ME3 has no mid-turn backchannel mechanism; JSD uses the "
                     "FDB repo convention (1.0 for zero backchannel "
                     "predictions). Q = mean of the 55 ICC ground-truth "
                     "speaker distributions, resized to the scenario "
                     "horizon."),
        }
        return out

    # ---- run ------------------------------------------------------------------
    def run(self, scenarios: list[str], n_trials: int, warmup: int,
            human: bool = False) -> dict:
        hs = load_human_set() if human else None
        lat_stims = hs["latency"] if human else LATENCY_CANONICAL
        stop_stims = hs["stop"] if human else []
        q: list[float] = []
        if ICC_PATH.exists():
            try:
                with open(ICC_PATH, encoding="utf-8") as f:
                    icc = json.load(f)
                arrays = [np.asarray(v, dtype=np.float64)
                          for v in icc.values() if isinstance(v, list)]
                if arrays:
                    resized = [resize_dist(a, 200) for a in arrays]
                    q = [float(x) for x in np.mean(
                        np.asarray(resized), axis=0)]
                    log.info("ICC GT loaded: %d speakers -> Q(%d bins)",
                             len(arrays), len(q))
            except Exception as exc:  # noqa: BLE001
                log.warning("ICC GT load failed: %s", exc)

        all_recs: list[dict] = []
        t_run0 = time.perf_counter()
        for sc in scenarios:
            t_sc0 = time.perf_counter()
            log.info("=== scenario %s: %d warmup + %d trials ===",
                     sc, warmup, n_trials)
            for w in range(warmup):
                stim = (lat_stims[w % len(lat_stims)]
                        if (sc == "barge_in" and human)
                        else (BARGE_STIMULUS if sc == "barge_in"
                              else lat_stims[w % len(lat_stims)]))
                it = (stop_stims[w % len(stop_stims)]
                      if (sc == "barge_in" and stop_stims) else None)
                all_recs.append(self.run_trial(sc, stim, warmup=True,
                                               interrupt_name=it))
            for i in range(n_trials):
                stim = (lat_stims[i % len(lat_stims)]
                        if (sc == "barge_in" and human)
                        else (BARGE_STIMULUS if sc == "barge_in"
                              else lat_stims[i % len(lat_stims)]))
                it = (stop_stims[i % len(stop_stims)]
                      if (sc == "barge_in" and stop_stims) else None)
                all_recs.append(self.run_trial(sc, stim, warmup=False,
                                               interrupt_name=it))
            log.info("scenario %s done in %.1f s", sc,
                     time.perf_counter() - t_sc0)
        elapsed = time.perf_counter() - t_run0
        fdb = {}
        for sc in scenarios:
            fdb[sc] = self.fdb_for_scenario(
                sc, [r for r in all_recs if r["scenario"] == sc], q)
        return {"records": all_recs, "fdb": fdb,
                "elapsed_s": round(elapsed, 1)}



PRIMARY_METRIC = {
    "wake_to_first_word": "wake_to_first_word_ms",
    "utterance_to_llm_token": "llm_ms_full",
    "llm_token_to_tts_audio": "tts_ttfb_ms",
    "end_to_end": "end_to_end_ms",
    "barge_in": "barge_in_latency_ms",
}


def summarize_scenario(sc: str, recs: list[dict]) -> dict:
    metric = PRIMARY_METRIC.get(sc, "end_to_end_ms")
    vals = sorted(r[metric] for r in recs
                  if not r.get("warmup") and r.get(metric) is not None)
    if not vals:
        return {"scenario": sc, "metric": metric, "n": 0}
    out = {
        "scenario": sc, "metric": metric, "n": len(vals),
        "median_ms": round(statistics.median(vals), 1),
        "p95_ms": round(vals[int(0.95 * (len(vals) - 1))], 1),
        "min_ms": round(min(vals), 1), "max_ms": round(max(vals), 1),
        "mean_ms": round(statistics.mean(vals), 1),
    }
    if metric == "llm_ms_full":
        out["note"] = ("PROXY: llm_ms is the FULL brain-call time "
                       "(synchronous, non-streaming; no TTFT instrumentation)")
    return out


def redact_cfg(cfg: dict) -> dict:
    out = json.loads(json.dumps(cfg))
    def walk(d):
        if isinstance(d, dict):
            for k in list(d.keys()):
                if str(k).lower() in ("api_key", "key", "token", "secret"):
                    d.pop(k)
                else:
                    walk(d[k])
        elif isinstance(d, list):
            for x in d:
                walk(x)
    walk(out)
    return out


def load_icc_q() -> list[float]:
    if not ICC_PATH.exists():
        return []
    with open(ICC_PATH, encoding="utf-8") as f:
        icc = json.load(f)
    arrays = [np.asarray(v, dtype=np.float64)
              for v in icc.values() if isinstance(v, list)]
    if not arrays:
        return []
    resized = [resize_dist(a, 200) for a in arrays]
    return [float(x) for x in np.mean(np.asarray(resized), axis=0)]


def write_results(out_dir: Path, run_meta: dict, res: dict,
                  summaries: list[dict]) -> None:
    if run_meta.get("stimulus_set") == "human":
        stim_protocol = (
            "Stimuli are recorded human utterances curated from the ME2 "
            "dataset (data/bench/human_manifest.json: speaker, condition, "
            "label, provenance); the barge-in interrupt is a recorded human "
            "STOP command. Agent voice = Piper en_GB-alba-medium.")
        stim_limit = (
            "Human stimuli come from ME2 class recordings (7 speakers, "
            "quiet-near, fixed 1.5-2.5 s clip lengths); the agent voice is "
            "synthetic (Piper).")
    else:
        stim_protocol = (
            "Stimuli are Piper-synthesized fixtures (data/bench, see "
            "manifest.json); single TTS voice en_GB-alba-medium for both "
            "'user' and 'agent'.")
        stim_limit = (
            "Synthetic TTS stimuli (no human voices); single voice for both "
            "roles.")
    results = {
        "run": run_meta,
        "protocol": (
            "In-process driver (me3/bench_driver.py) runs the real "
            "LocalVoiceLoop over the Windows Stereo Mix loopback: stimuli are "
            "played to the output device and heard by the agent through "
            "capture, exactly like a user's voice. Echo gating suppresses "
            "feeding captured frames while the agent speaks (except during "
            "barge-in interrupts, after a fresh queue drain). All latency "
            "values are perf_counter deltas; RT is the calibrated loopback "
            f"round-trip. {stim_protocol} {run_meta['brain_description']}"),
        "summary": summaries,
        "fdb_behavior": res["fdb"],
        "trials": [r for r in res["records"] if not r.get("warmup")],
        "warmup": [r for r in res["records"] if r.get("warmup")],
        "limitations": [
            stim_limit,
            "Internal loopback capture (Stereo Mix), not a physical mic; "
            "RT = calibrated loopback round-trip is embedded in capture-side "
            "boundary events.",
            "utterance_to_llm_token uses full-reply llm_ms as a proxy: the "
            "synchronous brain has no time-to-first-token "
            "instrumentation.",
            "llm_token_to_tts_audio measures first TTS chunk of a BATCH "
            "synthesis (config sentence_split streaming is not used in the "
            "local loop path).",
            "ME3 has no mid-turn backchannel mechanism: backchannel count = "
            "0, Freq = 0, JSD = 1.0 (FDB repo convention).",
            "Barge-in under loopback: the feed is re-enabled at interrupt "
            "onset; the agent's own TTS is also present in the capture at "
            "that moment (no echo cancellation in the local loop), so the "
            "measured latency is dominated by loopback RT + frame "
            "quantization, not by user-speech disambiguation. stop_playback "
            "itself is synchronous (<1 ms).",
            "ICC ground-truth Q is the mean of 55 human speakers resized to "
            "the scenario horizon (the FDB ICC set has no synthetic "
            "speaker).",
        ],
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def write_summary_md(out_dir: Path, run_meta: dict, res: dict,
                     summaries: list[dict]) -> None:
    L = []
    L.append("# ME3 FDB-aligned internal benchmark - summary")
    L.append("")
    L.append(f"- Run: {run_meta['stamp']} (UTC) - elapsed "
             f"{res['elapsed_s']} s")
    L.append(f"- Brain: {run_meta['brain']} ({run_meta['model']})")
    if run_meta.get("brain") == "stub":
        L.append("- Brain note: offline stub brain; no remote LLM network call.")
    L.append(f"- Loopback RT (calibrated): {run_meta['rt_ms']} ms")
    L.append(f"- Input: {run_meta['input_device']} | Output: "
             f"{run_meta['output_device']}")
    if run_meta.get("stimulus_set") == "human":
        L.append("- Stimuli: ME2 recorded human voices (7 speakers, "
                 "quiet-near); data/bench/human_manifest.json")
    else:
        L.append("- Stimuli: Piper synthetic fixtures (en_GB-alba-medium), "
                 "data/bench (see manifest.json)")
    L.append("")
    L.append("## Latency scenarios (ms, valid trials only)")
    L.append("")
    L.append("| scenario | metric | n | median | p95 | min | max | mean |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        if s.get("n", 0) == 0:
            L.append(f"| {s['scenario']} | {s['metric']} | 0 | - | - | - |"
                     " - | - |")
            continue
        L.append(f"| {s['scenario']} | {s['metric']} | {s['n']} | "
                 f"{s['median_ms']} | {s['p95_ms']} | {s['min_ms']} | "
                 f"{s['max_ms']} | {s['mean_ms']} |")
    L.append("")
    L.append("Notes: utterance_to_llm_token = full-reply LLM time (no TTFT "
             "instrumentation in the sync brain). end_to_end = user's last "
             "audible ms -> agent's first audible ms (speaker side).")
    L.append("")
    L.append("## Full-duplex behavior (FDB-aligned)")
    L.append("")
    L.append("| scenario | valid | TOR/seg | session TOR | BC | Freq/user-s |"
             " Freq/total-s | JSD (FDB conv) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for sc, f in res["fdb"].items():
        if "n_trials_valid" not in f:
            continue
        L.append(f"| {sc} | {f.get('n_trials_valid', 0)} | "
                 f"{f.get('takeover_rate_per_segment', '-')} | "
                 f"{f.get('session_takeover_flag', '-')} | "
                 f"{f.get('backchannel_count', '-')} | "
                 f"{f.get('backchannel_freq_per_user_speech_s', '-')} | "
                 f"{f.get('backchannel_freq_per_total_audio_s', '-')} | "
                 f"{f.get('backchannel_jsd_fdb_convention', '-')} |")
    L.append("")
    L.append("ME3 has no mid-turn backchannel mechanism: JSD = 1.0 is the "
             "FDB repo convention for zero backchannel predictions, and Freq "
             "= 0 is a design property, not a tuning result.")
    L.append("")
    L.append("## Protocol / limitations")
    L.append("")
    for lim in results_limitations_text(res):
        L.append(f"- {lim}")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")


def results_limitations_text(res: dict) -> list[str]:
    return [
        "Synthetic TTS stimuli (no human voices); single voice for both "
        "roles.",
        "Internal loopback capture (Stereo Mix), not a physical mic; RT = "
        "calibrated loopback round-trip embedded in capture-side events.",
        "utterance_to_llm_token = full-reply proxy (no TTFT in sync brain).",
        "tts_ttfb = first chunk of a batch synthesis (no streaming in the "
        "local loop path).",
        "No mid-turn backchannel mechanism in ME3 (BC=0, Freq=0, JSD=1.0 "
        "convention).",
        "Barge-in measured under loopback gating; no echo cancellation in "
        "the local loop (see results.json limitations).",
        "ICC Q = mean of 55 human speakers, resized to the scenario "
        "horizon.",
    ]


def update_status_file(path: Path, run_meta: dict, summaries: list[dict],
                       res: dict) -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    status = {
        "task": "fdb_benchmark",
        "status": "done",
        "started_at": run_meta["started_at"],
        "finished_at": now,
        "fdb_notes": ("FDB v1/v1.5 (arXiv 2503.04721) from "
                      "refs/full-duplex-bench/v1_v1.5/evaluation/: BC = dur"
                      "<1.0s AND words<=2; formal-speech time_threshold=3.0s "
                      "(session TOR flag); JSD via scipy.spatial.distance."
                      "jensenshannon (base-e, i.e. sqrt of the base-e "
                      "divergence) over 0.2s bins, interval-counted, JSD=1.0 "
                      "when zero BC predictions; Freq = n_events / "
                      "total_audio_seconds. me3/benchmark.py aligned to "
                      "these conventions (2026-10-20)."),
        "evidence": [
            f"python -m me3.bench_driver -> run {run_meta['stamp']}: "
            f"{res['elapsed_s']} s, RT {run_meta['rt_ms']} ms",
        ] + [f"summary {s['scenario']}: {s.get('median_ms', '-')} ms "
             f"(n={s.get('n', 0)})" for s in summaries],
        "artifacts": [
            f"runs/fdb_benchmark/{run_meta['stamp']}/results.json",
            f"runs/fdb_benchmark/{run_meta['stamp']}/summary.md",
            f"runs/fdb_benchmark/{run_meta['stamp']}/driver.log",
            "runs/fdb_benchmark_latest.json",
        ],
        "changes": [
            "me3/bench_driver.py (new): in-process FDB-aligned loopback "
            "benchmark driver with instrumented boundary events",
            "me3/benchmark.py: JSD base-2 -> base-e; interval binning; "
            "fdb_jsd_distance (scipy/FDB-repo convention); "
            "session_takeover_flag; BC_FORMAL_SPEECH_S; JSD=1.0 on zero BC",
        ],
        "limitations": [
            "Synthetic TTS fixtures; single voice for both roles.",
            "No TTFT instrumentation (utterance_to_llm_token = full-reply "
            "proxy).",
            "No mid-turn backchannel mechanism (BC=0, Freq=0, JSD=1.0 "
            "convention).",
            "Loopback RT embedded in capture-side events; barge-in measured "
            "under echo gating without echo cancellation.",
        ],
        "next_step": None,
    }
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(
        description="ME3 FDB-aligned in-process loopback benchmark")
    ap.add_argument("--smoke", action="store_true",
                    help="1 scenario (end_to_end), 1 warmup + 2 trials")
    ap.add_argument("--preflight", action="store_true",
                    help="open the configured loopback, calibrate RT, then "
                         "stop without running trials")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--scenarios", nargs="+", default=None,
                    choices=SCENARIOS)
    ap.add_argument("--interrupt-delay", type=float, default=None,
                    help="s after agent speech starts before the barge-in "
                         "interrupt is played")
    ap.add_argument("--stimulus-set", choices=["canonical", "human"],
                    default="canonical",
                    help="canonical = Piper fixtures; human = ME2 recorded "
                         "voices (data/bench/human_manifest.json)")
    ap.add_argument("--brain", choices=["remote", "stub"], default="remote",
                    help="brain used for the run; remote is the default, stub "
                         "avoids the network and is labeled in the artifacts")
    ap.add_argument("--input-device", default=None,
                    help="explicit input device index/name; Pi requires an "
                         "ALSA loopback or other explicit capture path for "
                         "speaker-stimulus benchmarking")
    ap.add_argument("--output-device", default=None,
                    help="explicit output device index/name")
    args = ap.parse_args()

    cfg = load_config()
    # Protocol pin (data/DATASET.md): class latency runs use the synchronous
    # brain so no backchannel audio is emitted and the five-scenario metric
    # definitions stay byte-identical. The live agent uses agent.ack_wait_s.
    cfg.setdefault("agent", {})["ack_wait_s"] = 0.0
    b = cfg["benchmark"]
    scenarios = args.scenarios or list(SCENARIOS)
    if args.smoke:
        scenarios = ["end_to_end"]
    n_trials = args.trials or (2 if args.smoke else int(b.get("n_trials", 20)))
    warmup = args.warmup if args.warmup is not None \
        else (1 if args.smoke else int(b.get("warmup", 3)))
    if args.interrupt_delay is not None:
        b["interrupt_delay_s"] = args.interrupt_delay

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_root = PROJECT / "runs" / "fdb_benchmark"
    out_dir = out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(out_dir / "driver.log",
                                    encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S", handlers=handlers)

    key = None
    if args.brain == "remote":
        # key -> env (memory only) before RemoteBrain reads it
        key = load_vllm_key()
        os.environ["VLLM_API_KEY"] = key

    import sounddevice as sd
    requested_input = args.input_device or os.environ.get("ME3_BENCH_INPUT")
    requested_output = args.output_device or os.environ.get("ME3_BENCH_OUTPUT")
    in_idx = resolve_device(sd, requested_input, input_required=True)
    if in_idx is None and requested_input is None:
        in_idx = pick_stereo_mix(sd)
    if in_idx is None:
        if platform.system() == "Windows":
            raise SystemExit("no 'Stereo Mix' input device found; pass "
                             "--input-device for an explicit loopback")
        raise SystemExit("no benchmark input device found; on Raspberry Pi "
                         "pass --input-device (for example an ALSA loopback) "
                         "or set ME3_BENCH_INPUT")
    if requested_output is not None:
        out_idx = resolve_device(sd, requested_output, input_required=False)
        if out_idx is None:
            raise SystemExit(f"requested output device not found: {requested_output}")
        cfg.setdefault("audio", {})["output_device"] = out_idx
    cfg.setdefault("audio", {})["input_device"] = in_idx

    brain = (RemoteBrain(cfg["llm"]["remote"])
             if args.brain == "remote" else StubBrain())

    if args.brain == "remote":
        if brain.name != "remote":
            raise SystemExit("VLLM key not visible to RemoteBrain")
        # preflight: endpoint alive, model present
        import requests
        try:
            r = requests.get(f"{brain.base_url}/models",
                             headers={"Authorization": f"Bearer {key}"},
                             timeout=15)
            r.raise_for_status()
            models = [m.get("id") for m in r.json().get("data", [])]
            log.info("LLM preflight OK: models=%s", models)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"LLM endpoint preflight failed: {exc}")
    else:
        log.info("brain preflight: offline stub (no remote network call)")

    driver = BenchDriver(cfg, in_idx, brain)
    try:
        rt = driver.calibrate_rt()
    except SystemExit:
        driver.loop.audio.close()
        raise
    if args.preflight:
        driver.loop.audio.close()
        print(f"AUDIO PREFLIGHT PASS: loopback RT calibrated at {rt:.1f} ms")
        return

    devs = sd.query_devices()
    in_name = str(devs[driver.loop.audio.in_idx].get("name"))
    out_name = str(devs[driver.loop.audio.out_idx].get("name"))
    pkg = {}
    for mod in ("faster_whisper", "piper", "onnxruntime", "sounddevice"):
        try:
            m = __import__(mod)
            pkg[mod] = getattr(m, "__version__", "n/a")
        except Exception:  # noqa: BLE001
            pkg[mod] = "import-failed"
    started = datetime.now(timezone.utc).astimezone().isoformat(
        timespec="seconds")
    run_meta = {
        "stamp": stamp,
        "started_at": started,
        "host": platform.node(),
        "python": platform.python_version(),
        "packages": pkg,
        "brain": args.brain,
        "model": getattr(brain, "model", "stub"),
        "endpoint": getattr(brain, "base_url", None),
        "brain_description": (
            "Brain = remote vLLM Qwen/Qwen3.8-27B (key kept in process "
            "memory, never persisted)."
            if args.brain == "remote"
            else "Brain = offline stub (no remote LLM network call)."
        ),
        "api_key": "<redacted: process memory only>",
        "input_device": f"[{driver.loop.audio.in_idx}] {in_name}",
        "output_device": f"[{driver.loop.audio.out_idx}] {out_name}",
        "rt_ms": round(rt, 1),
        "n_trials": n_trials, "warmup": warmup,
        "scenarios": scenarios,
        "stimulus_set": args.stimulus_set,
        "config": redact_cfg(cfg),
    }
    log.info("run %s: scenarios=%s trials=%s warmup=%s brain=%s@%s",
             stamp, scenarios, n_trials, warmup, brain.name,
             getattr(brain, "model", "stub"))

    t0 = time.perf_counter()
    res = driver.run(scenarios, n_trials, warmup,
                      human=args.stimulus_set == "human")
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    summaries = [summarize_scenario(sc, res["records"]) for sc in scenarios]
    driver.loop.audio.close()

    run_meta["finished_at"] = datetime.now(timezone.utc).astimezone(
    ).isoformat(timespec="seconds")
    run_meta["n_valid"] = sum(1 for r in res["records"]
                              if not r.get("warmup") and r.get("valid"))
    run_meta["n_total"] = sum(1 for r in res["records"]
                              if not r.get("warmup"))
    write_results(out_dir, run_meta, res, summaries)
    write_summary_md(out_dir, run_meta, res, summaries)
    # Keep the documented root-level copy and the historical nested copy for
    # consumers that already look below runs/fdb_benchmark/.
    shutil_copy = PROJECT / "runs" / "fdb_benchmark_latest.json"
    shutil_copy_nested = out_root / "fdb_benchmark_latest.json"
    import shutil
    shutil.copyfile(out_dir / "results.json", shutil_copy)
    shutil.copyfile(out_dir / "results.json", shutil_copy_nested)
    update_status_file(PROJECT / "subagent_status" / "fdb_benchmark.json",
                       run_meta, summaries, res)
    log.info("DONE in %.1f s: %d/%d valid trials; artifacts in %s",
             res["elapsed_s"], run_meta["n_valid"], run_meta["n_total"],
             out_dir)
    print(f"\nBENCH RUN COMPLETE: {out_dir}")
    for s in summaries:
        if s.get("n", 0):
            print(f"  {s['scenario']:<24} {s['metric']:<22} med="
                  f"{s['median_ms']} ms p95={s['p95_ms']} ms n={s['n']}")
        else:
            print(f"  {s['scenario']:<24} NO VALID DATA")


if __name__ == "__main__":
    main()
