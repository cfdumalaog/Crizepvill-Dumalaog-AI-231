"""Offline ASR audit for personal command recordings.

Transcripts and phrase-based label suggestions are audit fields only. This
script never changes the recording manifest or any training labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "data" / "human" / "manifest.csv"
VARIATIONS = PROJECT / "data" / "ai231-me2-voice-commands-hf-a90b8d10" / "variations.csv"
OUTPUT_DIR = PROJECT / "runs" / f"personal-transcript-audit-{datetime.now().strftime('%Y%m%d')}"
SPECIAL_LABELS = {"wake_word", "_unknown_", "_background_noise_", "_silence_"}
SAMPLE_RATE = 16000

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from tinyvcm_model.recording_labels import map_recorded_intent  # noqa: E402


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"\b(\d{1,2}):00\s*(am|pm)\b", r"\1 \2", text)
    text = re.sub(r"\b(\d{1,2})\s*(am|pm)\b", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def similarity(left: str, right: str) -> float:
    a, b = normalize(left), normalize(right)
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    token = 2 * len(ta & tb) / max(len(ta) + len(tb), 1)
    return 0.5 * seq + 0.5 * token


def load_phrases() -> list[dict[str, str]]:
    with VARIATIONS.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 93:
        raise ValueError(f"Expected 93 canonical phrases in {VARIATIONS}, got {len(rows)}")
    slot_values = {
        ("ALARM", normalize("6:00 AM")): "ALARM_6_00AM",
        ("ALARM", normalize("8:00 AM")): "ALARM_8_00AM",
        ("ALARM", normalize("9:00 PM")): "ALARM_9_00PM",
        ("BRIGHTNESS", normalize("20 percent")): "BRIGHTNESS_20",
        ("BRIGHTNESS", normalize("60 percent")): "BRIGHTNESS_60",
        ("BRIGHTNESS", normalize("100 percent")): "BRIGHTNESS_100",
        ("COLOR", normalize("Blue")): "COLOR_BLUE",
        ("COLOR", normalize("Green")): "COLOR_GREEN",
        ("COLOR", normalize("Red")): "COLOR_RED",
        ("CREATE_REMINDER", normalize("Drink water")): "CREATE_REMINDER_DRINK_WATER",
        ("CREATE_REMINDER", normalize("Exercise")): "CREATE_REMINDER_EXERCISE",
        ("CREATE_REMINDER", normalize("Study")): "CREATE_REMINDER_STUDY",
        ("TEMPERATURE", normalize("18 degrees")): "TEMPERATURE_18",
        ("TEMPERATURE", normalize("22 degrees")): "TEMPERATURE_22",
        ("TEMPERATURE", normalize("26 degrees")): "TEMPERATURE_26",
        ("TIMER", normalize("10 seconds")): "TIMER_10s",
        ("TIMER", normalize("1 minute")): "TIMER_1m",
        ("TIMER", normalize("30 seconds")): "TIMER_30s",
    }
    fixed = {
        "CALL", "LIGHT_OFF", "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT",
        "PAUSE", "PLAY_MUSIC", "STOP", "TIME", "VOLUME_DOWN", "VOLUME_UP", "WEATHER",
    }
    for row in rows:
        parent = row["label"]
        row["leaf_label"] = parent if parent in fixed else slot_values.get((parent, normalize(row.get("value", ""))), "")
    if any(not row["leaf_label"] for row in rows):
        raise ValueError("Could not map every phrase row to a supported leaf label")
    return rows


def best_phrase(transcript: str, phrases: list[dict[str, str]]) -> tuple[str, str, float]:
    scored = [(similarity(transcript, row["phrase"]), row["leaf_label"], row["phrase"]) for row in phrases]
    score, label, phrase = max(scored, default=(0.0, "", ""))
    return label if score >= 0.30 else "", phrase if score >= 0.30 else "", float(score)


def review_status(mapped: str, transcript: str, no_speech: float, candidate: str, score: float) -> str:
    if not mapped:
        return "unmapped_legacy_label_manual_review"
    if not transcript or no_speech >= 0.5:
        return "asr_uncertain_manual_review"
    if candidate != mapped or score < 0.42:
        return "phrase_or_label_review_audio"
    return "asr_suggests_label_match_still_unverified"


def write_review_queue(output_dir: Path, output_rows: list[dict[str, Any]]) -> Path:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in output_rows:
        key = row.get("audio_sha256") or row.get("path") or f"error-{len(grouped)}"
        grouped.setdefault(key, []).append(row)
    queue = []
    for audio_hash, records in grouped.items():
        if all(r.get("review_status") == "asr_suggests_label_match_still_unverified" for r in records):
            continue
        representative = records[0]
        queue.append({
            "audio_sha256": audio_hash,
            "path_to_listen": representative.get("path", ""),
            "all_manifest_paths": " | ".join(sorted({r.get("path", "") for r in records if r.get("path")})),
            "folder_labels": " | ".join(sorted({r.get("folder_label", "") for r in records if r.get("folder_label")})),
            "mapped_training_labels": " | ".join(sorted({r.get("mapped_training_label", "") for r in records if r.get("mapped_training_label")})),
            "asr_transcript": representative.get("asr_transcript", ""),
            "phrase_candidate_label": representative.get("phrase_candidate_label", ""),
            "phrase_candidate": representative.get("phrase_candidate", ""),
            "phrase_similarity": representative.get("phrase_similarity", ""),
            "asr_avg_logprob": representative.get("asr_avg_logprob", ""),
            "asr_no_speech_probability": representative.get("asr_no_speech_probability", ""),
            "review_status": " | ".join(sorted({r.get("review_status", "") for r in records})),
            "reviewer_confirmed_label": "",
            "reviewer_notes": "",
        })
    queue.sort(key=lambda row: (row["review_status"], row["path_to_listen"]))
    path = output_dir / "manual_review_queue.csv"
    fields = list(queue[0]) if queue else ["audio_sha256", "path_to_listen", "all_manifest_paths", "folder_labels", "mapped_training_labels", "asr_transcript", "phrase_candidate_label", "phrase_candidate", "phrase_similarity", "asr_avg_logprob", "asr_no_speech_probability", "review_status", "reviewer_confirmed_label", "reviewer_notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(queue)
    return path


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"path", "speaker", "condition", "label", "suggested_phrase"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected recording manifest columns in {MANIFEST}")
    return rows


def resolve_audio(relative: str) -> Path:
    path = (PROJECT / relative).resolve()
    if not path.is_relative_to(PROJECT.resolve()):
        raise ValueError(f"Audio path escapes the project: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_audio(path: Path) -> tuple[np.ndarray, str]:
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"Audio must be nonempty mono after downmix: {audio.shape}")
    if int(rate) != SAMPLE_RATE:
        # Only rate conversion is performed; no gain, noise reduction, or label
        # prompt is supplied to ASR.
        from math import gcd
        divisor = gcd(int(rate), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, int(rate) // divisor).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / max(peak, 1.0)
    return np.asarray(audio, dtype=np.float32), sha256(path.read_bytes())


def transcribe(model: Any, audio: np.ndarray) -> dict[str, Any]:
    segments, info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    parts = list(segments)
    text = " ".join(segment.text.strip() for segment in parts).strip()
    avg_logprob = float(np.mean([segment.avg_logprob for segment in parts])) if parts else None
    no_speech = float(np.mean([segment.no_speech_prob for segment in parts])) if parts else 1.0
    return {
        "transcript": text,
        "asr_avg_logprob": avg_logprob,
        "asr_no_speech_probability": no_speech,
        "asr_language": getattr(info, "language", ""),
        "asr_language_probability": float(getattr(info, "language_probability", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", default="small.en", help="Cached faster-whisper model; loading is local-only")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test cap; zero processes all command-labeled rows")
    parser.add_argument("--relabel-existing", action="store_true", help="Recompute phrase-to-leaf suggestions from an existing audit CSV without running ASR")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "personal_transcript_audit.csv"
    summary_path = args.output_dir / "summary.json"
    phrases = load_phrases()
    if args.relabel_existing:
        with output_csv.open(encoding="utf-8-sig", newline="") as stream:
            output_rows = list(csv.DictReader(stream))
        for row in output_rows:
            candidate, phrase, score = best_phrase(row.get("asr_transcript", ""), phrases)
            row["phrase_candidate_label"] = candidate
            row["phrase_candidate"] = phrase
            row["phrase_similarity"] = round(score, 4)
            mapped = row.get("mapped_training_label", "")
            row["candidate_agrees_with_mapped_label"] = bool(mapped and candidate == mapped)
            try:
                no_speech = float(row.get("asr_no_speech_probability") or 1.0)
            except ValueError:
                no_speech = 1.0
            row["review_status"] = review_status(mapped, row.get("asr_transcript", ""), no_speech, candidate, score)
        with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]) if output_rows else [])
            writer.writeheader()
            writer.writerows(output_rows)
        queue_path = write_review_queue(args.output_dir, output_rows)
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        statuses = Counter(row["review_status"] for row in output_rows)
        mapped_rows = [row for row in output_rows if row.get("mapped_training_label")]
        summary.update({
            "phrase_crosswalk_corrected": True,
            "candidate_label_agreements": sum(row["candidate_agrees_with_mapped_label"] is True for row in mapped_rows),
            "candidate_label_disagreements": sum(row["candidate_agrees_with_mapped_label"] is False for row in mapped_rows),
            "review_status_counts": dict(sorted(statuses.items())),
            "label_changes_made": 0,
        })
        crosswalk_note = "Phrase matching uses the dataset parent-intent plus slot-value to derive a 31-class leaf; suggestions remain unverified and labels were not changed."
        warnings = list(dict.fromkeys(summary.setdefault("warnings", [])))
        if crosswalk_note not in warnings:
            warnings.append(crosswalk_note)
        summary["warnings"] = warnings
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"summary": summary, "csv": str(output_csv), "review_queue": str(queue_path), "summary_file": str(summary_path)}, indent=2))
        return

    from faster_whisper import WhisperModel

    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache = cache / "hub" if (cache / "hub").is_dir() else cache
    model_slug = "models--Systran--faster-whisper-" + args.model.replace(".", ".")
    ref_file = cache / model_slug / "refs" / "main"
    model_revision = ref_file.read_text(encoding="utf-8").strip() if ref_file.is_file() else "unknown-local-cache-revision"
    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=max(1, (os.cpu_count() or 4) - 2), download_root=str(cache), local_files_only=True)

    rows = read_manifest()
    command_rows = [row for row in rows if row["label"] not in SPECIAL_LABELS]
    if args.limit:
        command_rows = command_rows[:args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "personal_transcript_audit.csv"
    fields = [
        "path", "speaker", "condition", "folder_label", "mapped_training_label",
        "suggested_phrase", "audio_sha256", "asr_transcript", "asr_avg_logprob",
        "asr_no_speech_probability", "asr_language", "asr_language_probability",
        "phrase_candidate_label", "phrase_candidate", "phrase_similarity",
        "candidate_agrees_with_mapped_label", "review_status", "error",
    ]
    by_audio: dict[str, dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for index, row in enumerate(command_rows, 1):
        out: dict[str, Any] = {
            "path": row["path"], "speaker": row["speaker"], "condition": row["condition"],
            "folder_label": row["label"], "mapped_training_label": map_recorded_intent(row["label"]) or "",
            "suggested_phrase": row.get("suggested_phrase", ""), "audio_sha256": "",
            "asr_transcript": "", "asr_avg_logprob": "", "asr_no_speech_probability": "",
            "asr_language": "", "asr_language_probability": "", "phrase_candidate_label": "",
            "phrase_candidate": "", "phrase_similarity": "", "candidate_agrees_with_mapped_label": "",
            "review_status": "needs_manual_audio_review", "error": "",
        }
        try:
            path = resolve_audio(row["path"])
            audio, audio_hash = load_audio(path)
            out["audio_sha256"] = audio_hash
            if audio_hash in by_audio:
                result = dict(by_audio[audio_hash])
            else:
                result = transcribe(model, audio)
                by_audio[audio_hash] = result
            out.update({
                "asr_transcript": result["transcript"],
                "asr_avg_logprob": result["asr_avg_logprob"] if result["asr_avg_logprob"] is not None else "",
                "asr_no_speech_probability": result["asr_no_speech_probability"],
                "asr_language": result["asr_language"],
                "asr_language_probability": result["asr_language_probability"],
            })
            candidate, phrase, score = best_phrase(result["transcript"], phrases)
            out["phrase_candidate_label"] = candidate
            out["phrase_candidate"] = phrase
            out["phrase_similarity"] = round(score, 4)
            mapped = out["mapped_training_label"]
            out["candidate_agrees_with_mapped_label"] = bool(mapped and candidate == mapped)
            out["review_status"] = review_status(mapped, result["transcript"], result["asr_no_speech_probability"], candidate, score)
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
            out["review_status"] = "transcription_failed_manual_review"
            errors.append(row["path"])
        output_rows.append(out)
        if index % 50 == 0:
            print(f"Processed {index}/{len(command_rows)} command-labeled recordings", flush=True)

    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    queue_path = write_review_queue(args.output_dir, output_rows)

    statuses = Counter(row["review_status"] for row in output_rows)
    mapped_rows = [row for row in output_rows if row["mapped_training_label"]]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_revision_from_local_cache": model_revision,
        "runtime": "faster-whisper; CPU int8; local_files_only=true; language=en; no label prompt supplied",
        "manifest": str(MANIFEST),
        "manifest_rows": len(rows),
        "transcribed_command_labeled_rows": len(output_rows),
        "unique_waveforms_transcribed": len(by_audio),
        "duplicate_waveform_rows_reused": len(output_rows) - len(by_audio) - len(errors),
        "special_wake_noise_silence_rows_skipped": len(rows) - len(command_rows),
        "rows_with_transcription_error": len(errors),
        "review_status_counts": dict(sorted(statuses.items())),
        "mapped_rows": len(mapped_rows),
        "candidate_label_agreements": sum(bool(row["candidate_agrees_with_mapped_label"]) for row in mapped_rows),
        "candidate_label_disagreements": sum(row["candidate_agrees_with_mapped_label"] is False for row in mapped_rows),
        "unmapped_legacy_rows": sum(not row["mapped_training_label"] for row in output_rows),
        "label_changes_made": 0,
        "warnings": [
            "Whisper output and phrase similarity are suggestions, not verified ground truth.",
            "No folder labels or training manifests were modified.",
            "A phrase-based label disagreement is not proof of a wrong label; paraphrases and ASR errors must be distinguished by listening.",
            "Manually listen to every disagreement before changing any label or retraining.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "csv": str(output_csv), "review_queue": str(queue_path), "summary_file": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
