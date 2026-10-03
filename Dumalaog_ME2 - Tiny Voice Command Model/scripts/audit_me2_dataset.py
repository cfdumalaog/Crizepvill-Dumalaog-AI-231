"""Verify the recorder, model, manifest phrases, and ME2 command coverage."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tinyvcm_model.config import LABELS, MANIFEST_PATH  # noqa: E402
from tinyvcm_model.recording_labels import (  # noqa: E402
    GROUND_TRUTH_PHRASE_DIR, RECORDER_LABELS, RECORDER_PHRASES, map_recorded_intent,
    validate_recorder_taxonomy,
)

COMMAND_GROUPS = {
    'Play music': ('PLAY_MUSIC',),
    'Questions: weather and time': ('WEATHER', 'TIME'),
    'Lights on and off': ('LIGHT_ON', 'LIGHT_OFF'),
    'Brightness and color': ('BRIGHTNESS_20', 'BRIGHTNESS_60', 'BRIGHTNESS_100', 'COLOR_RED', 'COLOR_GREEN', 'COLOR_BLUE'),
    'Timers': ('TIMER_10s', 'TIMER_30s', 'TIMER_1m'),
    'Alarms': ('ALARM_6_00AM', 'ALARM_8_00AM', 'ALARM_9_00PM'),
    'Thermostat setpoints': ('TEMPERATURE_18', 'TEMPERATURE_22', 'TEMPERATURE_26'),
    'Media and volume': ('PAUSE', 'STOP', 'NEXT', 'VOLUME_UP', 'VOLUME_DOWN'),
    'Reminders and lists': ('CREATE_REMINDER_DRINK_WATER', 'CREATE_REMINDER_EXERCISE', 'CREATE_REMINDER_STUDY', 'LIST_REMINDERS'),
    'Calls and messaging': ('CALL', 'MESSAGE'),
}


def audit() -> dict:
    validate_recorder_taxonomy()
    with MANIFEST_PATH.open(newline='', encoding='utf-8') as f:
        dataset = list(csv.DictReader(f))
    human_manifest = PROJECT / 'data/human/manifest.csv'
    with human_manifest.open(newline='', encoding='utf-8') as f:
        human_reader = csv.DictReader(f)
        human_fields = set(human_reader.fieldnames or [])
        human = list(human_reader)
    transcripts = defaultdict(Counter)
    for row in dataset:
        transcripts[row['label']][row['transcript'].strip()] += 1
    dataset_labels = set(transcripts)
    recorder_intents = set(RECORDER_LABELS[:len(LABELS)])
    group_labels = {label for group in COMMAND_GROUPS.values() for label in group}
    phrase_misses = [label for label in LABELS if RECORDER_PHRASES[label].strip() not in transcripts[label]]
    phrase_not_most_frequent = [
        label for label in LABELS
        if not transcripts[label] or RECORDER_PHRASES[label] != transcripts[label].most_common(1)[0][0]
    ]
    mapped = Counter(map_recorded_intent(r['label']) for r in human if map_recorded_intent(r['label']))
    unmapped = Counter(r['label'] for r in human if map_recorded_intent(r['label']) is None and not r['label'].startswith('_') and r['label'] != 'wake_word')
    staged = PROJECT / 'deployment/current_vcm/metadata.json'
    model_labels = json.loads(staged.read_text(encoding='utf-8'))['intent']['classes'] if staged.is_file() else None
    report = {
        'dataset_name': 'ME2 Spoken Command Dataset',
        'dataset_rows': len(dataset),
        'dataset_speakers': len({r['speaker'] for r in dataset}),
        'dataset_split_counts': dict(Counter(r['split'] for r in dataset)),
        'intent_labels': LABELS,
        'staged_intent_labels_match': model_labels == LABELS if model_labels is not None else None,
        'recorder_intent_labels_match': recorder_intents == set(LABELS),
        'manifest_labels_match': dataset_labels == set(LABELS),
        'ground_truth_phrase_file_count': len(list(GROUND_TRUTH_PHRASE_DIR.glob('*.txt'))),
        'ground_truth_phrase_files_match': {
            path.stem: path.read_text(encoding='utf-8').strip()
            for path in GROUND_TRUTH_PHRASE_DIR.glob('*.txt')
        } == {label: RECORDER_PHRASES[label] for label in LABELS},
        'suggested_phrase_not_in_training_transcripts': phrase_misses,
        'suggested_phrase_not_most_frequent_training_transcript': phrase_not_most_frequent,
        'course_command_groups': {name: list(labels) for name, labels in COMMAND_GROUPS.items()},
        'course_groups_cover_all_intent_labels': group_labels == set(LABELS),
        'human_manifest_rows': len(human),
        'human_manifest_has_suggested_phrase_column': 'suggested_phrase' in human_fields,
        'human_rows_with_saved_suggested_phrase': sum(bool(r.get('suggested_phrase', '').strip()) for r in human),
        'human_rows_without_saved_suggested_phrase': sum(not r.get('suggested_phrase', '').strip() for r in human),
        'human_speaker_rows': dict(Counter(r['speaker'] for r in human)),
        'human_mapped_command_rows': dict(sorted(mapped.items())),
        'human_unmapped_legacy_rows': dict(sorted(unmapped.items())),
        'intent_classes_with_no_personal_rows': [label for label in LABELS if label not in mapped],
    }
    if not (report['recorder_intent_labels_match'] and report['manifest_labels_match'] and
            report['ground_truth_phrase_file_count'] == len(LABELS) and report['ground_truth_phrase_files_match'] and
            report['course_groups_cover_all_intent_labels'] and not phrase_misses and not phrase_not_most_frequent and
            report['human_manifest_has_suggested_phrase_column'] and
            report['staged_intent_labels_match'] is not False):
        raise SystemExit(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    result = audit()
    path = PROJECT / 'docs/ME2_DATASET_ALIGNMENT.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"PASS: {len(LABELS)} model/recorder/manifest labels; {result['ground_truth_phrase_file_count']} phrase files match the most frequent training transcript; 10 course groups")
    print(f"Human recordings: {result['human_manifest_rows']} rows; {len(result['intent_classes_with_no_personal_rows'])} classes lack personal examples")
    print(f"Prompt metadata: {result['human_rows_with_saved_suggested_phrase']} rows populated; {result['human_rows_without_saved_suggested_phrase']} rows blank (older rows cannot be reconstructed)")
    print(path)
