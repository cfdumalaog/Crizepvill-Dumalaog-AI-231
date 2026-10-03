import csv
from collections import Counter, defaultdict

import numpy as np
import pytest
import soundfile as sf

import record_dataset
from tinyvcm_model.config import LABELS, MANIFEST_PATH, SAMPLES, SR
from tinyvcm_model.recording_labels import (
    BACKGROUND_LABEL,
    GROUND_TRUTH_PHRASE_DIR,
    RECORDER_LABELS,
    RECORDER_PHRASES,
    RECORDER_VARIATIONS,
    RECORDER_VARIANT_IDS,
    VARIATIONS_PATH,
    _SLOT_CLASSES,
    SILENCE_LABEL,
    UNKNOWN_LABEL,
    WAKE_LABEL,
    validate_recorder_taxonomy,
)


def test_recorder_taxonomy_is_the_exact_current_model_taxonomy_plus_specials():
    validate_recorder_taxonomy()
    assert RECORDER_LABELS[:len(LABELS)] == tuple(LABELS)
    assert set(RECORDER_LABELS[len(LABELS):]) == {
        WAKE_LABEL, UNKNOWN_LABEL, BACKGROUND_LABEL, SILENCE_LABEL,
    }
    assert set(RECORDER_PHRASES) == set(RECORDER_LABELS)
    phrase_files = {path.stem: path.read_text(encoding="utf-8").strip() for path in GROUND_TRUTH_PHRASE_DIR.glob("*.txt")}
    assert set(phrase_files) == set(LABELS)
    assert {label: RECORDER_PHRASES[label] for label in LABELS} == phrase_files
    assert set(RECORDER_VARIATIONS) == set(LABELS)
    assert all(RECORDER_PHRASES[label] in RECORDER_VARIATIONS[label] for label in LABELS)
    assert all(len(set(RECORDER_VARIATIONS[label])) in (3, 4) for label in LABELS)


def test_all_93_audited_dataset_variations_are_available_under_the_correct_leaf_label():
    with VARIATIONS_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 93
    for row in rows:
        value = row["value"].strip()
        leaf_label = _SLOT_CLASSES.get(row["label"], {}).get(value, row["label"])
        phrase = row["phrase"].strip()
        assert phrase in RECORDER_VARIATIONS[leaf_label], (leaf_label, phrase)
        assert RECORDER_VARIANT_IDS[leaf_label][phrase] == f"v{row['variation']}"

    # 22 official variations duplicate a canonical ground-truth prompt, so the
    # UI deduplicates them while still exposing all 93 source rows.
    assert sum(len(phrases) for phrases in RECORDER_VARIATIONS.values()) == 102


def test_intent_suggestions_are_the_most_frequent_manifest_transcripts():
    if not MANIFEST_PATH.is_file():
        pytest.skip("ME2 training manifest is not available in this checkout")
    transcripts = defaultdict(Counter)
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            transcripts[row["label"]][row["transcript"]] += 1

    assert len(transcripts) == len(LABELS) == 31
    for label in LABELS:
        assert RECORDER_PHRASES[label] in transcripts[label], label
        assert RECORDER_PHRASES[label] == transcripts[label].most_common(1)[0][0]


def test_recorder_saves_speech_in_the_current_2_5_second_model_window(tmp_path, monkeypatch):
    monkeypatch.setattr(record_dataset, "ROOT", tmp_path)
    samples = np.arange(SR, dtype=np.float32) / SR
    audio = (0.2 * np.sin(2 * np.pi * 440 * samples)).astype(np.float32)

    status, saved = record_dataset.save_recording(
        (SR, audio), "person-01", "quiet-near", "COLOR_RED", True,
    )

    assert "COLOR_RED" in status
    assert saved[0] == SR
    assert saved[1].shape == (SAMPLES,)
    manifest = (tmp_path / "data" / "human" / "manifest.csv").read_text(encoding="utf-8")
    assert ",COLOR_RED," in manifest
    with (tmp_path / "data" / "human" / "manifest.csv").open(encoding="utf-8", newline="") as f:
        saved_row = next(csv.DictReader(f))
    assert saved_row["suggested_phrase"] == RECORDER_PHRASES["COLOR_RED"]
    assert saved_row["phrase_variant"] == RECORDER_VARIANT_IDS["COLOR_RED"][RECORDER_PHRASES["COLOR_RED"]]
    saved_wav = next((tmp_path / "data" / "human").rglob("*.wav"))
    wav, rate = sf.read(saved_wav, dtype="float32")
    assert rate == SR
    assert wav.shape == (SAMPLES,)


def test_unrelated_speech_is_a_wake_negative_and_uses_the_speech_capture_path(tmp_path, monkeypatch):
    monkeypatch.setattr(record_dataset, "ROOT", tmp_path)
    samples = np.arange(SR, dtype=np.float32) / SR
    audio = (0.15 * np.sin(2 * np.pi * 330 * samples)).astype(np.float32)

    status, saved = record_dataset.save_recording(
        (SR, audio), "person-01", "quiet-near", UNKNOWN_LABEL, True,
    )

    assert "_unknown_" in status
    assert saved[1].shape == (SAMPLES,)
    manifest = (tmp_path / "data" / "human" / "manifest.csv").read_text(encoding="utf-8")
    assert ",_unknown_," in manifest


def test_saving_migrates_old_manifest_and_keeps_old_prompt_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(record_dataset, "ROOT", tmp_path)
    manifest = tmp_path / "data" / "human" / "manifest.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "speaker", "condition", "label", "source_id"])
        writer.writeheader()
        writer.writerow({"path": "old.wav", "speaker": "person-01", "condition": "quiet-near", "label": "lights_off", "source_id": "old-take"})

    samples = np.arange(SR, dtype=np.float32) / SR
    audio = (0.2 * np.sin(2 * np.pi * 440 * samples)).astype(np.float32)
    record_dataset.save_recording((SR, audio), "person-02", "quiet-near", "LIGHT_OFF", True)

    with manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["path"] == "old.wav"
    assert rows[0]["suggested_phrase"] == ""
    assert rows[1]["label"] == "LIGHT_OFF"
    assert rows[1]["suggested_phrase"] == RECORDER_PHRASES["LIGHT_OFF"]


def test_recorder_saves_selected_dataset_phrase_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(record_dataset, "ROOT", tmp_path)
    samples = np.arange(SR, dtype=np.float32) / SR
    audio = (0.2 * np.sin(2 * np.pi * 440 * samples)).astype(np.float32)
    phrase = "Kill the lights"

    record_dataset.save_recording((SR, audio), "person-03", "quiet-near", "LIGHT_OFF", True, phrase)

    with (tmp_path / "data" / "human" / "manifest.csv").open(encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["suggested_phrase"] == phrase
    assert row["phrase_variant"] == RECORDER_VARIANT_IDS["LIGHT_OFF"][phrase]
