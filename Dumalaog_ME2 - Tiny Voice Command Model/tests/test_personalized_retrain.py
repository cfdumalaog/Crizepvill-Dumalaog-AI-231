import csv

import numpy as np
import soundfile as sf

from tinyvcm_model.personalized_retrain import (
    _split_counts,
    _threshold_for_zero_false,
    load_personal_records,
)
from tinyvcm_model.config import LABELS
from tinyvcm_model.recording_labels import map_recorded_intent


def test_split_counts_keep_test_and_all_partitions_when_possible():
    assert _split_counts(1) == (1, 0, 0)
    assert _split_counts(2) == (1, 0, 1)
    assert _split_counts(3) == (1, 1, 1)
    assert sum(_split_counts(24)) == 24
    assert all(_split_counts(24))


def test_zero_false_accept_threshold_stays_in_probability_range():
    threshold, metrics = _threshold_for_zero_false(
        np.asarray([0.20], dtype=np.float32),
        np.asarray([0.80], dtype=np.float32),
    )

    assert 0.0 <= threshold <= 1.0
    assert metrics["false_accepts"] == 0
    assert metrics["wake_recall"] == 0.0


def test_recorded_intent_mapping_accepts_current_ids_and_only_exact_legacy_labels():
    assert map_recorded_intent("COLOR_GREEN") == "COLOR_GREEN"
    assert map_recorded_intent("question_time") == "TIME"
    assert map_recorded_intent("dim_lights_25") is None
    assert map_recorded_intent("timer_10min") is None
    assert map_recorded_intent("wake_word") is None
    assert map_recorded_intent("not_a_class") is None


def test_personal_split_manifest_is_written_only_to_requested_run_path(tmp_path):
    project = tmp_path / "project"
    human = project / "data" / "human"
    audio_dir = human / "recordings"
    audio_dir.mkdir(parents=True)
    wav = audio_dir / "wake.wav"
    sf.write(wav, np.zeros(24000, dtype=np.float32), 16000, subtype="PCM_16")
    manifest = human / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "speaker", "condition", "label", "source_id"])
        writer.writeheader()
        writer.writerow({
            "path": "data/human/recordings/wake.wav", "speaker": "person-01",
            "condition": "quiet-near", "label": "wake_word", "source_id": "wake-1",
        })

    groups, audit = load_personal_records(project)
    assert len(groups) == 1
    assert audit["split_manifest_path"] is None
    assert "speakers may occur in multiple partitions" in audit["split_policy"]
    assert not (project / "runs" / "_personalized_data_preflight.csv").exists()

    explicit_path = project / "runs" / "isolated-run" / "personal_split_manifest.csv"
    _, explicit_audit = load_personal_records(project, explicit_path)
    assert explicit_audit["split_manifest_path"] == str(explicit_path)
    assert explicit_path.is_file()
    assert not (project / "runs" / "_personalized_data_preflight.csv").exists()


def test_personal_loader_maps_option_b_recordings_without_renaming_legacy_files(tmp_path):
    project = tmp_path / "project"
    human = project / "data" / "human"
    recordings = human / "person-01" / "quiet-near"
    recordings.mkdir(parents=True)
    rows = []
    for index, label in enumerate(("COLOR_GREEN", "question_time", "dim_lights_25")):
        path = recordings / f"sample-{index}.wav"
        t = np.arange(40000, dtype=np.float32) / 16000
        audio = (0.2 * np.sin(2 * np.pi * (220 + index * 110) * t)).astype(np.float32)
        sf.write(path, audio, 16000, subtype="PCM_16")
        rows.append({
            "path": path.relative_to(project).as_posix(), "speaker": "person-01",
            "condition": "quiet-near", "label": label, "source_id": f"source-{index}",
        })
    manifest = human / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "speaker", "condition", "label", "source_id"])
        writer.writeheader()
        writer.writerows(rows)

    groups, audit = load_personal_records(project)
    mapped = {item["label"]: item["intent"] for item in groups}
    assert mapped["COLOR_GREEN"] == "COLOR_GREEN"
    assert mapped["question_time"] == "TIME"
    assert mapped["dim_lights_25"] is None
    assert "dim_lights_25" in audit["unmapped_labels"]
    assert len(LABELS) == 31


def test_personal_loader_excludes_conflicting_duplicate_audio_and_preserves_rows(tmp_path):
    project = tmp_path / "project"
    human = project / "data" / "human"
    recordings = human / "person-01" / "quiet-near"
    recordings.mkdir(parents=True)
    t = np.arange(40000, dtype=np.float32) / 16000
    collision_audio = (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    clean_audio = (0.2 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
    rows = []
    for filename, label, audio in (
        ("study.wav", "CREATE_REMINDER_STUDY", collision_audio),
        ("message.wav", "MESSAGE", collision_audio),
        ("volume.wav", "VOLUME_UP", clean_audio),
    ):
        path = recordings / filename
        sf.write(path, audio, 16000, subtype="PCM_16")
        rows.append({
            "path": path.relative_to(project).as_posix(), "speaker": "person-01",
            "condition": "quiet-near", "label": label, "source_id": filename,
        })
    manifest = human / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "speaker", "condition", "label", "source_id"])
        writer.writeheader()
        writer.writerows(rows)

    groups, audit = load_personal_records(project)

    assert [item["label"] for item in groups] == ["VOLUME_UP"]
    assert audit["raw_manifest_rows"] == 3
    assert audit["cross_label_collision_groups_excluded"] == 1
    collision = audit["cross_label_hash_collisions"][0]
    assert collision["labels"] == ["CREATE_REMINDER_STUDY", "MESSAGE"]
    assert set(collision["paths"]) == {rows[0]["path"], rows[1]["path"]}
    assert "original_audio_and_manifest_preserved" in collision["action"]
    assert all((project / row["path"]).is_file() for row in rows)


def test_training_loader_uses_frozen_manifest_when_recording_continues(tmp_path):
    project = tmp_path / 'project'
    human = project / 'data/human'
    human.mkdir(parents=True)
    sf.write(human / 'take.wav', np.zeros(40000, dtype=np.float32), 16000, subtype='PCM_16')
    snapshot = project / 'snapshot.csv'
    columns = ['path', 'speaker', 'condition', 'label', 'source_id']
    with snapshot.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow(dict(path='data/human/take.wav', speaker='person-01', condition='quiet-near', label='wake_word', source_id='one'))
    (human / 'manifest.csv').write_text('path,speaker,condition,label,source_id\nmissing.wav,person-02,quiet-near,wake_word,two\n', encoding='utf-8')
    groups, audit = load_personal_records(project, manifest_path=snapshot)
    assert audit['raw_manifest_rows'] == 1
    assert groups[0]['source_id'] == 'one'
