"""Prepare an auditable workbook view without inventing spoken transcripts."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import soundfile as sf
import numpy as np

from .config import LABELS, INTENT_CONFIDENCE_THRESHOLD, INTENT_MARGIN_THRESHOLD
from .recording_labels import RECORDER_LABELS, RECORDER_PHRASES, map_recorded_intent, validate_recorder_taxonomy


def workbook_payload(project: Path) -> dict:
    project = Path(project)
    validate_recorder_taxonomy()
    meta = json.loads((project / 'deployment/current_vcm/metadata.json').read_text(encoding='utf-8'))
    run = project / 'runs' / meta['source_run']
    report = json.loads((run / 'final_evaluation.json').read_text(encoding='utf-8'))
    audit = json.loads((run / 'dataset_audit.json').read_text(encoding='utf-8'))
    with (project / 'data/human/manifest.csv').open(newline='', encoding='utf-8') as f:
        raw = list(csv.DictReader(f))
    with (run / 'personal_split_manifest.csv').open(newline='', encoding='utf-8') as f:
        split_rows = {r['pcm_sha256']: r for r in csv.DictReader(f)}
    with (project / 'data/option_b/manifest.csv').open(newline='', encoding='utf-8') as f:
        reference = list(csv.DictReader(f))
    transcripts = {label: Counter() for label in LABELS}
    for r in reference:
        transcripts[r['label']][r['transcript'].strip()] += 1
    raw_mapped = Counter(map_recorded_intent(r['label']) or r['label'] for r in raw)
    intent = report['intent_model_comparison']['personalized_int8']
    personal = intent['personal_heldout_test']['classification_report']
    ref = intent['optionb_test']['classification_report']
    commands, performance = [], []
    for label in RECORDER_LABELS:
        phrase = RECORDER_PHRASES[label]
        commands.append([label, phrase, 'Intent' if label in LABELS else ('Wake positive' if label == 'wake_word' else 'Wake negative'),
                         transcripts[label][phrase] if label in LABELS else None, raw_mapped[label],
                         audit['unique_mapped_intents'].get(label, 0) if label in LABELS else audit['unique_label_groups'].get(label, 0),
                         f'docs/ground_truth_phrases/{label}.txt' if label in LABELS else 'tinyvcm_model/recording_labels.py'])
        if label in LABELS:
            p = personal.get(label, {})
            weak = p.get('f1-score', 0) < .95 or ref[label]['f1-score'] < .95
            performance.append([label, phrase, ref[label]['recall'], ref[label]['f1-score'], int(ref[label]['support']),
                                p.get('precision'), p.get('recall'), p.get('f1-score'), int(p.get('support', 0)),
                                'More recordings' if weak else 'Small-sample pass', INTENT_CONFIDENCE_THRESHOLD, INTENT_MARGIN_THRESHOLD, 'Offline: top-class argmax'])
    collisions = {r['pcm_sha256'] for r in audit['cross_label_hash_collisions']}
    recordings = []
    for index, r in enumerate(raw, 1):
        mapped = map_recorded_intent(r['label'])
        phrase_key = mapped or (r['label'] if r['label'] in RECORDER_PHRASES else None)
        phrase = RECORDER_PHRASES.get(phrase_key, '')
        saved = r.get('suggested_phrase', '')
        wave, _ = sf.read(project / r['path'], dtype='float32')
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        digest = hashlib.sha256(np.asarray(wave, dtype='<f4').reshape(-1).tobytes()).hexdigest()
        group = split_rows.get(digest, {})
        status = 'EXCLUDED: conflicting labels' if digest in collisions else group.get('split', 'NOT_IN_ACTIVE_RUN')
        recordings.append([index, r['label'], mapped or '', phrase, saved or None, r.get('phrase_variant') or None,
                           'Logged prompt' if saved else 'UNKNOWN historical prompt', 'UNVERIFIED spoken transcript',
                           r['speaker'], r['condition'], status, int(group['duplicate_count']) if group else None,
                           r['source_id'], r['path'], digest])
    point = meta['wake']['operating_point_replay']
    wake = [[meta['source_run'], point['threshold'], point['personal_wake_hits'], point['personal_wake_support'],
             point['binary_recall'], point['binary_precision'], point['binary_f1'], point['binary_accuracy'],
             point['personal_nonwake_false_accepts'], point['personal_nonwake_support'],
             point['dataset_command_false_accepts'], point['dataset_command_support']]]
    selected = report['wake_model_comparison']['binary']
    wake.append(['Validation-selected cutoff', selected['validation_selected_int8_threshold'],
                 selected['heldout_personal_wake_hits'], selected['heldout_personal_wake_support'], selected['heldout_personal_wake_recall'],
                 None, None, None, selected['false_wake_on_personal_nonwake_clips'], selected['personal_nonwake_support'],
                 selected['false_wake_on_optionb_command_clips'], selected['optionb_command_support']])
    return {'title': 'ME2 - VCM on Raspberry Pi 5', 'created_at': datetime.now().astimezone().isoformat(),
            'active_run': meta['source_run'], 'dataset': meta['dataset_name'], 'raw_rows': len(raw),
            'snapshot_rows': audit['raw_manifest_rows'], 'logged_prompt_rows': sum(bool(r.get('suggested_phrase')) for r in raw),
            'commands': commands, 'recordings': recordings, 'performance': performance, 'wake': wake,
            'provenance': meta.get('pair_provenance', {}),
            'reference_accuracy': intent['optionb_test']['accuracy'], 'personal_accuracy': intent['personal_heldout_test']['accuracy']}
