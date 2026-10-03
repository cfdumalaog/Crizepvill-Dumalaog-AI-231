"""Canonical labels and recording prompts for the ME2 Spoken Command Dataset.

The legacy ``tinyvcm.config.LABELS`` taxonomy remains available for historical
experiments. New human recordings use the current 31-class model labels here.
Each command prompt is loaded from ``docs/ground_truth_phrases/<LABEL>.txt`` so
the text files are the single source of truth. Wake and negative examples are
explicit special labels, not intent classes.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .config import LABELS as INTENT_LABELS

GROUND_TRUTH_PHRASE_DIR = Path(__file__).resolve().parents[1] / "docs" / "ground_truth_phrases"
VARIATIONS_PATH = GROUND_TRUTH_PHRASE_DIR / "ME2_variations.csv"

RECORDER_PHRASES = {
    label: (GROUND_TRUTH_PHRASE_DIR / f"{label}.txt").read_text(encoding="utf-8").strip()
    for label in INTENT_LABELS
}
RECORDER_PHRASES.update({
    "wake_word": "Hi Dandan / Hello Dandan",
    "_unknown_": "Say unrelated speech, such as 'I am reading a book'",
    "_background_noise_": "Record room noise: fan, typing, or TV; do not say a command",
    "_silence_": "Stay quiet for at least 2.5 seconds",
})

_SLOT_CLASSES = {
    "TIMER": {"10 seconds": "TIMER_10s", "30 seconds": "TIMER_30s", "1 minute": "TIMER_1m"},
    "ALARM": {"6:00 AM": "ALARM_6_00AM", "8:00 AM": "ALARM_8_00AM", "9:00 PM": "ALARM_9_00PM"},
    "TEMPERATURE": {"18 degrees": "TEMPERATURE_18", "22 degrees": "TEMPERATURE_22", "26 degrees": "TEMPERATURE_26"},
    "BRIGHTNESS": {"20 percent": "BRIGHTNESS_20", "60 percent": "BRIGHTNESS_60", "100 percent": "BRIGHTNESS_100"},
    "COLOR": {"Red": "COLOR_RED", "Green": "COLOR_GREEN", "Blue": "COLOR_BLUE"},
    "CREATE_REMINDER": {"Drink water": "CREATE_REMINDER_DRINK_WATER", "Study": "CREATE_REMINDER_STUDY", "Exercise": "CREATE_REMINDER_EXERCISE"},
}


def _load_variations() -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, str]]]:
    """Load the official three dataset prompts for each of the 31 leaf labels."""
    rows: dict[str, list[tuple[int, str]]] = {label: [] for label in INTENT_LABELS}
    with VARIATIONS_PATH.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            command, value = row["label"], row["value"].strip()
            label = _SLOT_CLASSES.get(command, {}).get(value, command)
            if label not in rows or not row["phrase"].strip():
                raise RuntimeError(f"Unknown dataset phrase mapping: {row}")
            rows[label].append((int(row["variation"]), row["phrase"].strip()))
    result, identifiers = {}, {}
    for label, phrases in rows.items():
        sorted_phrases = sorted(phrases)
        ordered = tuple(phrase for _, phrase in sorted_phrases)
        if len(ordered) != 3 or len(set(ordered)) != 3:
            raise RuntimeError(f"Expected exactly three unique dataset phrases for {label}; got {ordered}")
        result[label] = (RECORDER_PHRASES[label],) + tuple(p for p in ordered if p != RECORDER_PHRASES[label])
        identifiers[label] = {
            phrase: (f"v{number}" if phrase in ordered else "canonical")
            for number, phrase in sorted_phrases
        }
        if RECORDER_PHRASES[label] not in identifiers[label]:
            identifiers[label][RECORDER_PHRASES[label]] = "canonical"
    return result, identifiers


RECORDER_VARIATIONS, RECORDER_VARIANT_IDS = _load_variations()

WAKE_LABEL = "wake_word"
UNKNOWN_LABEL = "_unknown_"
BACKGROUND_LABEL = "_background_noise_"
SILENCE_LABEL = "_silence_"
NON_INTENT_LABELS = (WAKE_LABEL, UNKNOWN_LABEL, BACKGROUND_LABEL, SILENCE_LABEL)
RECORDER_LABELS = tuple(INTENT_LABELS) + NON_INTENT_LABELS

# Exact legacy intent mappings only. Near matches (for example 25% -> 20%,
# five minutes -> one minute, or generic alarm -> a fixed-time alarm) stay
# unmapped so old recordings are never silently mislabeled.
LEGACY_INTENT_MAP = {
    "call_mom": "CALL",
    "lights_off": "LIGHT_OFF",
    "lights_on": "LIGHT_ON",
    "media_next": "NEXT",
    "media_pause": "PAUSE",
    "play_music": "PLAY_MUSIC",
    "question_time": "TIME",
    "question_weather": "WEATHER",
    "reminders_check": "LIST_REMINDERS",
    "timer_1min": "TIMER_1m",
    "volume_down": "VOLUME_DOWN",
    "volume_up": "VOLUME_UP",
}


def map_recorded_intent(label: str) -> str | None:
    """Return a current model class for a canonical or exact legacy label."""
    if label in INTENT_LABELS:
        return label
    return LEGACY_INTENT_MAP.get(label)


def validate_recorder_taxonomy() -> None:
    """Fail early if labels, phrase files, and loaded prompts drift apart."""
    if len(INTENT_LABELS) != 31 or len(set(INTENT_LABELS)) != len(INTENT_LABELS):
        raise RuntimeError("The ME2 command taxonomy must contain 31 unique classes")
    if set(RECORDER_PHRASES) != set(RECORDER_LABELS):
        missing = sorted(set(RECORDER_LABELS) - set(RECORDER_PHRASES))
        extra = sorted(set(RECORDER_PHRASES) - set(RECORDER_LABELS))
        raise RuntimeError(f"Recorder prompt mismatch; missing={missing}, extra={extra}")
    phrase_files = {path.stem for path in GROUND_TRUTH_PHRASE_DIR.glob("*.txt")}
    if phrase_files != set(INTENT_LABELS):
        missing = sorted(set(INTENT_LABELS) - phrase_files)
        extra = sorted(phrase_files - set(INTENT_LABELS))
        raise RuntimeError(f"Ground-truth phrase file mismatch; missing={missing}, extra={extra}")
    for label in INTENT_LABELS:
        phrase = (GROUND_TRUTH_PHRASE_DIR / f"{label}.txt").read_text(encoding="utf-8").strip()
        if not phrase or "\n" in phrase or "\r" in phrase:
            raise RuntimeError(f"Ground-truth phrase must be one non-empty line: {label}")
        if RECORDER_PHRASES[label] != phrase:
            raise RuntimeError(f"Loaded recorder prompt differs from ground truth: {label}")
    if set(RECORDER_VARIATIONS) != set(INTENT_LABELS) or any(len(items) not in (3, 4) for items in RECORDER_VARIATIONS.values()):
        raise RuntimeError("Dataset prompt variations do not cover the 31 intent labels")
