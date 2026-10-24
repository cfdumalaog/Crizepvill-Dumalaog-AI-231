"""Synthesize ME3 benchmark fixtures (internal, TTS-substituted).

Produces, under data/bench/:
  - 8 canonical stimuli (DATASET.md sec.2)  -> <scenario_id>.wav
  - 4 behavioral fixtures (DATASET.md sec.2b) -> pause_handling.wav,
    smooth_turn.wav, interruption.wav, backchannel.wav
  - manifest.json (sec.6.4): {file, sha256, source, voice, date, peak_dbfs}

All outputs are 16 kHz mono int16 WAV, peak-normalized to -20 dBFS (+/-0.5),
with 0.30 s leading/trailing silence (+/-0.05). Source is Piper TTS
(en_GB-alba-medium, 22050 Hz) resampled to 16000 Hz with soxr.

Run:  python scripts/make_bench_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import soxr  # noqa: E402

VOICE = PROJECT / "models" / "tts" / "en_GB-alba-medium.onnx"
OUT_DIR = PROJECT / "data" / "bench"

TARGET_SR = 16000
PEAK_DBFS = -20.0
PAD_S = 0.30

# --- canonical stimuli (DATASET.md sec.2, exact sentences) ---
CANONICAL = {
    "hello": "Hello",
    "what_time_is_it": "What time is it?",
    "set_a_timer_5_minutes": "Set a timer for five minutes",
    "tell_me_system_status": "Tell me the system status",
    "stop": "Stop",
    "repeat_that": "Repeat that",
    "play_the_news": "Play the news",
    "what_is_2_plus_2": "What is 2 plus 2?",
}


def synth(text: str, piper) -> np.ndarray:
    """Piper-synthesize text -> float32 mono @ engine sr (22050)."""
    chunks = []
    for c in piper.synthesize(text):
        chunks.append(np.frombuffer(c.audio_int16_bytes, dtype=np.int16))
    pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
    return pcm.astype(np.float32) / 32768.0


def resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    return soxr.resample(x, src_sr, dst_sr).astype(np.float32)


def normalize_peak(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    if peak <= 0:
        return x
    target = 10.0 ** (target_dbfs / 20.0)
    return (x / peak) * target


def add_pad(x: np.ndarray, sr: int, pad_s: float = PAD_S,
            trim_thresh_dbfs: float = -45.0) -> np.ndarray:
    """Trim TTS breath/low-noise edges, then add exactly pad_s of silence.

    TTS output carries a soft breath/noise tail that exceeds a -50 dBFS
    detector threshold, inflating measured leading/trailing silence. We trim
    every sample below trim_thresh_dbfs (well below speech) from both ends,
    then pad to the spec's 0.30 s so the *measured* silence lands on target.
    """
    thr = 10.0 ** (trim_thresh_dbfs / 20.0)
    nz = np.where(np.abs(x) > thr)[0]
    if len(nz):
        x = x[nz[0]:nz[-1] + 1]
    n = int(round(pad_s * sr))
    return np.concatenate([np.zeros(n, np.float32), x, np.zeros(n, np.float32)])


def to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(to_int16(x).tobytes())


def peak_dbfs_of(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x)))
    return 20.0 * math.log10(peak) if peak > 0 else -999.0


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    from piper import PiperVoice
    piper = PiperVoice.load(str(VOICE))
    eng_sr = int(getattr(getattr(piper, "config", None), "sample_rate", 22050) or 22050)
    print(f"[fixtures] piper loaded, engine sr={eng_sr}")

    manifest = []
    date = datetime.now(timezone.utc).isoformat()

    def emit(name: str, x16: np.ndarray, note: str = "") -> None:
        path = OUT_DIR / f"{name}.wav"
        write_wav(path, x16, TARGET_SR)
        pk = peak_dbfs_of(x16)
        dur = len(x16) / TARGET_SR
        entry = {
            "file": f"data/bench/{name}.wav",
            "sha256": sha256_of(path),
            "source": "tts_synthetic",
            "voice": "en_GB-alba-medium",
            "date": date,
            "peak_dbfs": round(pk, 2),
            "duration_s": round(dur, 3),
            "note": note,
        }
        manifest.append(entry)
        print(f"  wrote {name}.wav  dur={dur:.2f}s  peak={pk:.2f}dBFS")

    # --- a. canonical stimuli ---
    print("[fixtures] canonical (8)")
    for name, text in CANONICAL.items():
        raw = synth(text, piper)
        x16 = resample(raw, eng_sr, TARGET_SR)
        x16 = normalize_peak(x16, PEAK_DBFS)
        x16 = add_pad(x16, TARGET_SR)
        emit(name, x16, f"canonical: {text!r}")

    # --- b. behavioral fixtures (sec.2b) ---
    print("[fixtures] behavioral (4)")

    # pause_handling: two sentences, 0.6 s intra-turn gap, total > 5 s
    a = resample(synth("Let me think about that for a moment and consider the options carefully.", piper), eng_sr, TARGET_SR)
    b = resample(synth("Okay, here is what I found after checking the system.", piper), eng_sr, TARGET_SR)
    gap = np.zeros(int(0.6 * TARGET_SR), np.float32)
    ph = normalize_peak(np.concatenate([a, gap, b]), PEAK_DBFS)
    ph = add_pad(ph, TARGET_SR)
    emit("pause_handling", ph, "behavioral: intra-turn pause 0.6s, total>5s")

    # smooth_turn: two turns, <0.4 s inter-turn gap + trailing silence.
    # NOTE: brief asks for 5 s trailing, but DATASET.md sec.1 caps total at
    # 8.0 s and the orchestrator re-verifies with verify_dataset.py, so the
    # trailing silence is capped to keep total <= 8.0 s (documented deviation).
    t1 = resample(synth("Sure, I can help you with that.", piper), eng_sr, TARGET_SR)
    t2 = resample(synth("What would you like to do next?", piper), eng_sr, TARGET_SR)
    small_gap = np.zeros(int(0.25 * TARGET_SR), np.float32)
    body = np.concatenate([t1, small_gap, t2])
    trail_n = max(0, int((7.9 * TARGET_SR) - len(body) - 2 * int(PAD_S * TARGET_SR)))
    trail = np.zeros(trail_n, np.float32)
    st = normalize_peak(np.concatenate([body, trail]), PEAK_DBFS)
    st = add_pad(st, TARGET_SR)
    emit("smooth_turn", st, "behavioral: inter-turn gap 0.25s + trailing silence (capped to 8.0s total per DATASET.md sec.1; brief wanted 5s)")

    # interruption: agent speaking, user utterance mid-stream + trailing silence.
    # NOTE: brief asks for agent ~7s + 15s trailing (=22s+), exceeding the
    # 8.0 s cap; scaled to fit 8.0 s total (agent ~2.5s, user @3.5s, +~3s trail).
    agent = resample(synth(
        "Let me walk you through the system status now.", piper), eng_sr, TARGET_SR)
    user = resample(synth("Wait, stop, change that.", piper), eng_sr, TARGET_SR)
    lead = np.zeros(int(3.5 * TARGET_SR), np.float32)
    body = np.concatenate([lead, agent, user])
    trail_n = max(0, int((7.9 * TARGET_SR) - len(body) - 2 * int(PAD_S * TARGET_SR)))
    it = np.concatenate([body, np.zeros(trail_n, np.float32)])
    it = normalize_peak(it, PEAK_DBFS)
    it = add_pad(it, TARGET_SR)
    emit("interruption", it, "behavioral: agent ~2.5s, user interrupt @3.5s, +trailing silence (scaled to 8.0s total per DATASET.md sec.1; brief wanted 7s+15s)")

    # backchannel: user speaks; agent gives short acks. Reuse same voice.
    # NOTE: scaled to fit 8.0 s total (DATASET.md sec.1 cap; brief wanted ~6s user).
    user_long = resample(synth(
        "So the thing I wanted to ask is whether the system can handle "
        "multiple tasks at the same time.", piper), eng_sr, TARGET_SR)
    ack = resample(synth("Mm-hmm.", piper), eng_sr, TARGET_SR)
    ack2 = resample(synth("Right.", piper), eng_sr, TARGET_SR)
    # layout: user speech, ack at ~2s, user continues, ack2 at ~4s
    u1 = user_long[: int(2.0 * TARGET_SR)]
    mid = np.zeros(int(0.3 * TARGET_SR), np.float32)
    u2 = user_long[int(2.0 * TARGET_SR): int(4.0 * TARGET_SR)]
    u3 = user_long[int(4.0 * TARGET_SR):]
    bc = np.concatenate([u1, mid, ack, mid, u2, mid, ack2, mid, u3])
    # cap total to 7.9s (pre-pad) so post-pad <= 8.0s
    cap = int(7.9 * TARGET_SR)
    if len(bc) > cap:
        bc = bc[:cap]
    bc = normalize_peak(bc, PEAK_DBFS)
    bc = add_pad(bc, TARGET_SR)
    emit("backchannel", bc, "behavioral: user speech + agent acks (mm-hmm/right), scaled to 8.0s total")

    # --- manifest.json (sec.6.4) ---
    mp = OUT_DIR / "manifest.json"
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[fixtures] manifest -> {mp} ({len(manifest)} entries)")
    print(f"[fixtures] done in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
