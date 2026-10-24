"""ME3 class-group online voice recorder (browser tool).

Lets students on the lab LAN record the 8 canonical ME3 benchmark stimuli
(`data/DATASET.md` §2) from a laptop or phone, with metadata + consent, and
saves spec-compliant 16 kHz mono int16 WAVs under ``data/bench/speakers/``
with a JSONL manifest.

Design notes
------------
- Binds ``0.0.0.0:7871`` ONLY (ME2 owns 7861-7865; do not touch).
- All testing is INTERNAL: server-side + a synthetic WAV produced by Piper.
  No browser mic, no physical hardware, no public tunnel.
- No secrets in code, logs, UI, or chat.
- The committed canonical set at ``data/bench/<id>.wav`` is left untouched;
  the class set lives under ``data/bench/speakers/<speaker_label>/``.

Run:  python -m me3.web_recorder
"""
from __future__ import annotations

import hashlib
import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

import gradio as gr

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_BENCH = PROJECT_ROOT / "data" / "bench"
SPEAKERS_DIR = DATA_BENCH / "speakers"
MANIFEST_PATH = DATA_BENCH / "manifest.jsonl"

TARGET_SR = 16000          # spec sample rate
TARGET_PEAK_DBFS = -20.0   # spec peak level
PAD_SECONDS = 0.30         # spec leading/trailing silence
MIN_TOTAL_S = 0.30
MAX_TOTAL_S = 8.00
PAD_RMS_MAX_DBFS = -45.0   # quiet-room proof (informational, not fatal)
MANILA_TZ = timezone(timedelta(hours=8))

# The 8 canonical stimuli (data/DATASET.md §2). Order = display order.
STIMULI: list[tuple[str, str]] = [
    ("hello", "Hello"),
    ("what_time_is_it", "What time is it?"),
    ("set_a_timer_5_minutes", "Set a timer for five minutes"),
    ("stop", "Stop"),
    ("what_is_2_plus_2", "What is 2 plus 2?"),
    ("tell_me_a_joke", "Tell me a joke"),
    ("play_some_music", "Play some music"),
    ("who_are_you", "Who are you?"),
]
STIM_IDS = [sid for sid, _ in STIMULI]


# --------------------------------------------------------------------------- #
# Audio normalization (pure functions — unit-testable, no I/O)
# --------------------------------------------------------------------------- #
def _dbfs_amplitude(dbfs: float) -> float:
    """Convert a dBFS value to a linear amplitude."""
    return 10.0 ** (dbfs / 20.0)


def _peak_dbfs(samples: np.ndarray) -> float:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0.0:
        return -120.0
    return 20.0 * np.log10(peak)


def _rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    if rms <= 0.0:
        return -120.0
    return 20.0 * np.log10(rms)


def _trim_leading_trailing_silence(
    samples: np.ndarray, sr: int, pad_s: float = PAD_SECONDS
) -> np.ndarray:
    """Trim to the voiced region, then add exactly ``pad_s`` of silence on
    each side. Returns float32 in [-1, 1]."""
    if samples.size == 0:
        return np.zeros(int(pad_s * sr), dtype=np.float32)

    # Voice gate: a sample is "active" if it exceeds ~ -50 dBFS.
    gate = _dbfs_amplitude(-50.0)
    active = np.abs(samples) > gate
    idx = np.flatnonzero(active)
    if idx.size == 0:
        # All silence: keep a short centered slice so we don't emit a huge file.
        keep = int(min(samples.size, 0.5 * sr))
        start = (samples.size - keep) // 2
        return samples[start : start + keep].copy()

    start, end = int(idx[0]), int(idx[-1]) + 1
    voiced = samples[start:end].astype(np.float32, copy=False)

    pad = int(round(pad_s * sr))
    return np.concatenate([np.zeros(pad, np.float32), voiced, np.zeros(pad, np.float32)])


def normalize_to_spec(
    samples: np.ndarray,
    sr: int,
    target_sr: int = TARGET_SR,
    target_peak_dbfs: float = TARGET_PEAK_DBFS,
    pad_s: float = PAD_SECONDS,
    min_total_s: float = MIN_TOTAL_S,
    max_total_s: float = MAX_TOTAL_S,
) -> tuple[np.ndarray, dict]:
    """Resample -> mono float32 -> trim+pad -> peak-normalize.

    Returns ``(pcm_int16, info)``. Raises ``ValueError`` if the result is
    outside the allowed total-duration window.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim > 1:  # collapse stereo -> mono
        x = x.mean(axis=1)

    # Resample to target rate.
    if int(sr) != int(target_sr):
        x = soxr.resample(x, int(sr), int(target_sr), quality="HQ")

    # Trim leading/trailing silence, add exact padding.
    x = _trim_leading_trailing_silence(x, int(target_sr), pad_s)

    total_s = x.size / float(target_sr)
    if total_s < min_total_s:
        raise ValueError(
            f"Recording too short after trimming: {total_s:.2f}s "
            f"(need >= {min_total_s:.2f}s). Speak a little louder/clearer."
        )
    if total_s > max_total_s:
        raise ValueError(
            f"Recording too long: {total_s:.2f}s (max {max_total_s:.2f}s). "
            "Say only the sentence."
        )

    # Peak-normalize to target dBFS.
    peak = float(np.max(np.abs(x)))
    if peak <= 0.0:
        raise ValueError("Recording is silent — no voice detected.")
    scale = _dbfs_amplitude(target_peak_dbfs) / peak
    x = x * scale
    x = np.clip(x, -1.0, 1.0)

    pcm = (x * 32767.0).astype(np.int16)
    info = {
        "sr": int(target_sr),
        "channels": 1,
        "duration_s": round(pcm.size / float(target_sr), 3),
        "peak_dbfs": round(float(_peak_dbfs(pcm.astype(np.float32) / 32768.0)), 2),
        "pad_rms_dbfs": round(float(_rms_dbfs(pcm[: int(pad_s * target_sr)].astype(np.float32) / 32768.0)), 2),
        "original_sr": int(sr),
    }
    return pcm, info


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_recording(
    pcm: np.ndarray,
    sr: int,
    scenario_id: str,
    *,
    student_id: str,
    speaker_label: str,
    condition: str = "class_lan",
    consent: bool = False,
    source: str = "class_web",
    mic: str = "unknown",
) -> dict:
    """Write a spec-compliant WAV and append a manifest row.

    Re-saving the same speaker+scenario overwrites the WAV and appends a NEW
    manifest row (history preserved).
    """
    if scenario_id not in STIM_IDS:
        raise ValueError(f"Unknown scenario_id: {scenario_id!r}")
    student_id = (student_id or "").strip()
    speaker_label = (speaker_label or "").strip()
    if not student_id:
        raise ValueError("student_id is required.")
    if not speaker_label:
        raise ValueError("speaker_label is required.")
    if not consent:
        raise ValueError("Consent is required to save a recording.")

    speaker_dir = SPEAKERS_DIR / speaker_label
    speaker_dir.mkdir(parents=True, exist_ok=True)
    wav_path = speaker_dir / f"{scenario_id}.wav"
    sf.write(str(wav_path), pcm, sr, subtype="PCM_16")

    rel_file = str(Path("data/bench/speakers") / speaker_label / f"{scenario_id}.wav")
    row = {
        "file": rel_file,
        "sha256": _sha256_file(wav_path),
        "recorder": "me3.web_recorder",
        "mic": mic,
        "date": datetime.now(MANILA_TZ).isoformat(timespec="seconds"),
        "peak_dbfs": round(float(_peak_dbfs(pcm.astype(np.float32) / 32768.0)), 2),
        "student_id": student_id,
        "speaker_label": speaker_label,
        "condition": condition,
        "consent": bool(consent),
        "sr": int(sr),
        "channels": 1,
        "duration_s": round(pcm.size / float(sr), 3),
        "source": source,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    row["absolute_path"] = str(wav_path)
    return row


def load_manifest_rows() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _latest_per_speaker() -> dict[str, set[str]]:
    """Map speaker_label -> set(scenario_ids) that have at least one row."""
    out: dict[str, set[str]] = {}
    for row in load_manifest_rows():
        sp = row.get("speaker_label")
        fid = row.get("file", "")
        sid = Path(fid).stem
        if sp and sid:
            out.setdefault(sp, set()).add(sid)
    return out


def refresh_status_table(speaker_label: str) -> list[list]:
    """Build the 8-row status table for a speaker label."""
    done = _latest_per_speaker().get((speaker_label or "").strip(), set())
    rows = []
    for sid, sentence in STIMULI:
        status = "saved for this speaker" if sid in done else "missing"
        rows.append([sid, sentence, status])
    return rows


# --------------------------------------------------------------------------- #
# LAN IP detection (never includes credentials)
# --------------------------------------------------------------------------- #
def lan_ips() -> list[str]:
    ips: set[str] = set()
    try:
        hostname = socket.gethostname()
        for _, _, _, _, sockaddr in socket.getaddrinfo(hostname, None):
            ip = sockaddr[0]
            if ip and not ip.startswith("::"):
                ips.add(ip)
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))  # no packets sent; just resolves local iface
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    ordered = sorted(ips)
    return ordered or ["127.0.0.1"]


# --------------------------------------------------------------------------- #
# Gradio handlers
# --------------------------------------------------------------------------- #
def on_save(
    audio,
    scenario_id: str,
    student_id: str,
    speaker_label: str,
    condition: str,
    consent: bool,
):
    """Normalize + persist one recording. Returns (message, status_table)."""
    try:
        if audio is None:
            return ("No audio received. Record or upload first.", refresh_status_table(speaker_label))
        arr, sr = audio  # gr.Audio(type="numpy") -> (ndarray, sr)
        pcm, info = normalize_to_spec(arr, sr)
        row = save_recording(
            pcm,
            info["sr"],
            scenario_id,
            student_id=student_id,
            speaker_label=speaker_label,
            condition=condition or "class_lan",
            consent=bool(consent),
            source="class_web",
        )
        msg = (
            f"Saved {scenario_id}.wav  |  {row['duration_s']}s  |  "
            f"peak {row['peak_dbfs']} dBFS  |  sha256 {row['sha256'][:12]}…\n"
            f"→ data/bench/speakers/{speaker_label}/{scenario_id}.wav"
        )
        return (msg, refresh_status_table(speaker_label))
    except ValueError as e:
        return (f"Rejected: {e}", refresh_status_table(speaker_label))
    except Exception as e:  # noqa: BLE001
        return (f"Error saving: {e.__class__.__name__}: {e}", refresh_status_table(speaker_label))


def on_upload(
    audio,
    scenario_id: str,
    student_id: str,
    speaker_label: str,
    condition: str,
    consent: bool,
):
    """File-upload fallback path (phones that struggle with in-browser rec)."""
    if audio is None:
        return ("No file received.", refresh_status_table(speaker_label))
    # audio is a path (str) or a temp file-like; load with soundfile.
    try:
        arr, sr = sf.read(str(audio), dtype="float32", always_2d=False)
    except Exception as e:  # noqa: BLE001
        return (f"Could not read uploaded file: {e}", refresh_status_table(speaker_label))
    return on_save((arr, sr), scenario_id, student_id, speaker_label, condition, consent)


def build_ui() -> gr.Blocks:
    ips = ", ".join(lan_ips())
    with gr.Blocks(title="ME3 Class Recorder") as demo:
        gr.Markdown(
            f"### ME3 Class-group Voice Recorder\n"
            f"Open this on your laptop/phone over the lab LAN: **"
            f"http://{ips.split(',')[0]}:7871**  (all IPs: {ips})\n"
            "Record each of the 8 sentences exactly as written. "
            "See `data/DATASET.md` §2 for the canonical list."
        )
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("**Your details**")
                student_id = gr.Textbox(label="Student ID", placeholder="e.g. 2026-1234", lines=1)
                speaker_label = gr.Textbox(label="Speaker label", placeholder="e.g. speaker_01", lines=1)
                condition = gr.Textbox(label="Condition", value="class_lan", lines=1)
                consent = gr.Checkbox(
                    label=(
                        "I consent to my voice being recorded and stored for the "
                        "ME3 benchmark dataset only."
                    ),
                    value=False,
                )
                status_table = gr.Dataframe(
                    headers=["scenario_id", "say exactly", "status"],
                    value=refresh_status_table(""),
                    interactive=False,
                    label="Stimulus status",
                )
            with gr.Column(scale=3):
                gr.Markdown("**Record or upload**")
                scenario_dd = gr.Dropdown(
                    choices=[f"{sid}  —  “{sent}”" for sid, sent in STIMULI],
                    value=STIMULI[0][0] + "  —  “" + STIMULI[0][1] + "”",
                    label="Choose the sentence to record",
                )
                mic_audio = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Microphone recording",
                )
                file_audio = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="…or upload a recording (phone fallback)",
                )
                btn_mic = gr.Button("Save microphone recording", variant="primary")
                btn_file = gr.Button("Save uploaded file")
                out_msg = gr.Markdown("")

        def _sid(dd_value: str) -> str:
            return dd_value.split("  —  ")[0].strip()

        def _mic(dd, aud, sid, sp, cond, cons):
            return on_save(aud, _sid(dd), sid, sp, cond, cons)

        def _file(dd, aud, sid, sp, cond, cons):
            return on_upload(aud, _sid(dd), sid, sp, cond, cons)

        btn_mic.click(
            _mic,
            inputs=[scenario_dd, mic_audio, student_id, speaker_label, condition, consent],
            outputs=[out_msg, status_table],
        )
        btn_file.click(
            _file,
            inputs=[scenario_dd, file_audio, student_id, speaker_label, condition, consent],
            outputs=[out_msg, status_table],
        )
    return demo


demo = build_ui()


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7871,
        inbrowser=False,
    )
