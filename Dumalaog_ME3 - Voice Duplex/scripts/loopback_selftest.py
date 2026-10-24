"""ME3 loopback self-test: TTS -> speaker -> capture -> VAD -> ASR -> transcript.

Proves the full voice pipeline (Piper TTS -> audio out -> audio in -> Silero
VAD -> faster-whisper ASR) WITHOUT a human microphone, by routing the agent's
own speech back through the machine's loopback capture device (Windows
"Stereo Mix"; on the Pi, a virtual cable / arecord loopback works too).

PASS = the ASR transcript of the captured audio contains the expected
keywords of the synthesized sentence.

Usage:
    python scripts/loopback_selftest.py                 # auto-pick loopback input
    python scripts/loopback_selftest.py --input "Stereo Mix"
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from me3.audio_io import FullDuplexAudio, resample_to  # noqa: E402
from me3.config import load_config  # noqa: E402

SENTENCE = "Hello. This is the ME3 voice agent. The full duplex pipeline is working."
EXPECT = ("me3", "pipeline")  # keywords expected in the transcript


def pick_loopback_input(sd, explicit):
    """Prefer an explicit device, else a Stereo Mix / loopback-style device."""
    devs = sd.query_devices()
    if explicit is not None:
        found = None
        for i, d in enumerate(devs):
            s = str(explicit).lower()
            if s == str(i) or s in str(d.get("name", "")).lower():
                found = i
                break
        if found is None:
            raise SystemExit(f"input device not found: {explicit}")
        return found
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0 and \
                "stereo mix" in str(d.get("name", "")).lower():
            return i
    return None  # fall back to FullDuplexAudio's own probing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None,
                        help="loopback input device (index or name substring; "
                             "default: first 'Stereo Mix' device)")
    args = parser.parse_args()

    cfg = load_config()
    sd = None
    import sounddevice as sd  # noqa: F811

    in_idx = pick_loopback_input(sd, args.input)
    audio_cfg = dict(cfg["audio"])
    if in_idx is not None:
        audio_cfg["input_device"] = in_idx
    else:
        print("NOTE: no 'Stereo Mix' device found; using normal device probing "
              "(this test then needs real mic input).")
    audio = FullDuplexAudio({**cfg, "audio": audio_cfg})

    from me3 import mcp_asr, mcp_tts
    print("loading engines ...")
    asr_engine, asr_cfg = mcp_asr._get_engine()
    tts_engine, _tts_cfg = mcp_tts._get_engine()
    tts_sr = int(getattr(getattr(tts_engine, "config", None),
                         "sample_rate", 22050) or 22050)
    sr = audio.sr

    t0 = time.perf_counter()
    raw = b"".join(c.audio_int16_bytes for c in tts_engine.synthesize(SENTENCE))
    pcm = resample_to(np.frombuffer(raw, dtype=np.int16), tts_sr, sr)
    tts_ms = (time.perf_counter() - t0) * 1e3
    print(f"TTS: '{SENTENCE}'  ({len(pcm)/sr:.2f} s, {tts_ms:.0f} ms)")

    audio.start_capture()
    frames: list[np.ndarray] = []
    print("playing agent speech; capturing the loopback ...")
    audio.play(pcm, blocking=False)
    t_play = time.perf_counter()
    while not audio.playback_done() or time.perf_counter() - t_play < 1.0:
        f = audio.read_frame(timeout=0.2)
        if f is not None:
            frames.append(f)
    # 0.5 s tail for endpointing silence
    t_end = time.perf_counter()
    while time.perf_counter() - t_end < 0.5:
        f = audio.read_frame(timeout=0.2)
        if f is not None:
            frames.append(f)
    audio.close()
    if not frames:
        print("FAIL: captured no audio at all")
        return 1
    stream = np.concatenate(frames)
    print(f"captured: {len(stream)/sr:.2f} s audio")

    import torch
    from silero_vad import load_silero_vad
    vad = load_silero_vad(sampling_rate=sr)
    vad.eval()
    x = (stream.astype(np.float32) / 32768.0)
    thr = float(cfg["vad"].get("threshold", 0.5))
    probs = []
    with torch.no_grad():
        for i in range(0, len(x) - 511, 512):
            probs.append(float(vad(torch.from_numpy(x[i:i+512]), sr).item()))
    speech_idx = [i for i, p in enumerate(probs) if p >= thr]
    if not speech_idx:
        print(f"FAIL: VAD found no speech in the capture (thr={thr})")
        return 1
    start_s, end_s = speech_idx[0] * 512, (speech_idx[-1] + 1) * 512
    utterance = stream[start_s:end_s]
    print(f"VAD: speech span {start_s/sr:.2f}s - {end_s/sr:.2f}s "
          f"({len(speech_idx)} of {len(probs)} frames above threshold)")

    t0 = time.perf_counter()
    segments, _info = asr_engine.transcribe(
        utterance.astype(np.float32) / 32768.0,
        beam_size=asr_cfg.get("beam_size", 1),
        language=asr_cfg.get("language", "en"),
        vad_filter=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    asr_ms = (time.perf_counter() - t0) * 1e3
    print(f"ASR: '{text}'  ({asr_ms:.0f} ms)")

    low = text.lower()
    # faster-whisper (small.en) spells brand tokens out: ME3 -> Mi 3 / My 3
    low = re.sub(r"\bmi\s*3\b|\bmy\s*3\b|\bm\s+three\b", "me3", low)
    missing = [k for k in EXPECT if k not in low]
    if missing:
        print(f"FAIL: expected keywords missing: {missing}")
        return 1
    print(f"SELF-TEST PASS: TTS -> capture -> VAD -> ASR loop verified "
          f"(tts {tts_ms:.0f} ms, asr {asr_ms:.0f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
