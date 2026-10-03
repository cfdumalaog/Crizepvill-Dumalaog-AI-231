"""Stage a completed scratch run as the single local/Pi two-model candidate."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import onnxruntime as ort

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tinyvcm_model.config import LABELS  # noqa: E402
from tinyvcm_model.config import INTENT_CONFIDENCE_THRESHOLD, INTENT_MARGIN_THRESHOLD
from tinyvcm_model.frontend import Frontend  # noqa: E402
from tinyvcm_model.personalized_retrain import (  # noqa: E402
    _predict_onnx, _wake_test_windows, load_optionb_test, load_personal_records,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(path: Path) -> dict:
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    return {'input_name': session.get_inputs()[0].name,
            'input_shape': session.get_inputs()[0].shape,
            'output_shape': session.get_outputs()[0].shape}


def operating_point(path: Path, threshold: float, manifest_path: Path | None = None) -> dict:
    human, _ = load_personal_records(PROJECT, manifest_path=manifest_path)
    wake = [item for item in human if item['split'] == 'test' and item['label'] == 'wake_word']
    other = [item for item in human if item['split'] == 'test' and item['label'] != 'wake_word']
    frontend = Frontend()
    wake_x, wake_groups = _wake_test_windows(wake, frontend)
    other_x, other_groups = _wake_test_windows(other, frontend)
    test_x, _ = load_optionb_test(PROJECT)
    wake_scores = _predict_onnx(path, wake_x, binary=True)[:, 1]
    other_scores = _predict_onnx(path, other_x, binary=True)[:, 1]
    command_scores = _predict_onnx(path, test_x, binary=True)[:, 1]
    wake_record_scores = np.array([wake_scores[wake_groups == j].max() for j in range(len(wake))])
    other_record_scores = np.array([other_scores[other_groups == j].max() for j in range(len(other))])
    tp = int((wake_record_scores >= threshold).sum())
    fp = int((other_record_scores >= threshold).sum() + (command_scores >= threshold).sum())
    n_pos = len(wake)
    n_neg = len(other) + len(command_scores)
    recall = tp / n_pos if n_pos else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        'threshold': threshold,
        'personal_wake_hits': tp,
        'personal_wake_support': len(wake),
        'personal_nonwake_false_accepts': int((other_record_scores >= threshold).sum()),
        'personal_nonwake_support': len(other),
        'dataset_command_false_accepts': int((command_scores >= threshold).sum()),
        'dataset_command_support': len(command_scores),
        'binary_precision': precision,
        'binary_recall': recall,
        'binary_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'binary_accuracy': (tp + n_neg - fp) / (n_pos + n_neg),
        'balanced_accuracy': (recall + (n_neg - fp) / n_neg) / 2 if n_neg else 0.0,
        'personal_nonwake_labels': dict(Counter(item['label'] for item in other)),
        'protocol': 'Fixed user-requested 0.95 cutoff; five temporal placements per personal WAV; reused speaker-held-out command partition; not a live microphone or false-activations-per-hour test.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='me2-vcm-20260930-full-negatives')
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--destination', type=Path, default=PROJECT / 'deployment' / 'current_vcm',
                        help='New candidate directory; existing candidates are never overwritten.')
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error('Threshold must be between zero and one.')
    run = PROJECT / 'runs' / args.run
    dest = args.destination.resolve()
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite current candidate: {dest}')
    report = json.loads((run / 'final_evaluation.json').read_text(encoding='utf-8'))
    wake_export = json.loads((run / 'wake_binary/models/export_summary.json').read_text(encoding='utf-8'))
    intent_export = json.loads((run / 'intent_personalized/models/export_summary.json').read_text(encoding='utf-8'))
    if intent_export['classes'] != LABELS or wake_export['classes'] != ['NON_WAKE', 'WAKE_WORD']:
        raise ValueError('Training output class order does not match the runtime')
    wake_src = run / 'wake_binary/models/wake_personalized_int8.onnx'
    intent_src = run / 'intent_personalized/models/intent_personalized_int8.onnx'
    snapshot = run / 'human_manifest_snapshot.csv'
    point = operating_point(wake_src, args.threshold, snapshot if snapshot.exists() else None)
    dest.mkdir(parents=True)
    models = dest / 'models'
    models.mkdir()
    wake = models / 'binary_wake_int8.onnx'
    intent = models / 'intent_int8.onnx'
    shutil.copy2(wake_src, wake)
    shutil.copy2(intent_src, intent)
    intent_result = report['intent_model_comparison']['personalized_int8']
    wake_result = report['wake_model_comparison']['binary']
    pair_provenance = report.get('pair_provenance', {})
    class_report = intent_result['optionb_test']['classification_report']
    per_label = {label: {'f1': class_report[label]['f1-score'], 'recall': class_report[label]['recall'],
                         'support': int(class_report[label]['support'])} for label in LABELS}
    metadata = {
        'title': 'ME2 - VCM on Raspberry Pi 5',
        'status': 'manual_launch_release_candidate',
        'source_run': args.run,
        'pair_provenance': pair_provenance,
        'created_from_scratch': all(
            pair_provenance.get(part, {}).get('training') == 'from_random_initialization'
            for part in ('wake', 'intent')
        ),
        'pair_assembled_with_retained_wake': pair_provenance.get('wake', {}).get('training') == 'retained_byte_identical',
        'dataset_name': 'ME2 Spoken Command Dataset',
        'frontend': {'sample_rate_hz': 16000, 'window_seconds': 2.5, 'feature_shape': [1, 40, 251]},
        'wake': {'labels': ['NON_WAKE', 'WAKE_WORD'], 'wake_class_index': 1,
                 'validation_selected_threshold': wake_export['validation_threshold_int8'],
                 'local_operating_threshold': args.threshold,
                 'operating_point_replay': point,
                 'validation': {
                     'selected_int8_threshold': wake_result['validation_selected_int8_threshold'],
                     'heldout_wake_hits': wake_result['heldout_personal_wake_hits'],
                     'heldout_wake_support': wake_result['heldout_personal_wake_support'],
                     'reference_command_false_accepts': wake_result['false_wake_on_optionb_command_clips'],
                     'reference_command_support': wake_result['optionb_command_support'],
                     'personal_nonwake_false_accepts': wake_result['false_wake_on_personal_nonwake_clips'],
                     'personal_nonwake_support': wake_result['personal_nonwake_support'],
                 },
                 'model': {'path': 'models/binary_wake_int8.onnx', 'sha256': digest(wake),
                           'size_bytes': wake.stat().st_size, 'signature': signature(wake)}},
        'intent': {'classes': LABELS,
                   'live_confidence_threshold': INTENT_CONFIDENCE_THRESHOLD,
                   'live_margin_threshold': INTENT_MARGIN_THRESHOLD,
                   'evaluation_policy': 'argmax without live rejection gates',
                   'dataset_test_accuracy': intent_result['optionb_test']['accuracy'],
                   'dataset_test_macro_f1': intent_result['optionb_test']['macro_f1_supported_classes'],
                   'dataset_test_per_label': per_label,
                   'personal_heldout': intent_result['personal_heldout_test'],
                   'model': {'path': 'models/intent_int8.onnx', 'sha256': digest(intent),
                             'size_bytes': intent.stat().st_size, 'signature': signature(intent)}},
        'limitations': [
            'The reference command test split has been used in earlier iterations and is a reused benchmark.',
            'Personal recordings are dominated by one speaker ID and personal train/test clips share speakers.',
            'A fixed 0.95 threshold replay is not live room false-activation testing.',
            'Natural-speech recognition and in-room wake false activations per hour require live testing.',
        ],
    }
    (dest / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    (dest / 'export_summary.json').write_text(json.dumps({
        'classes': ['NON_WAKE', 'WAKE_WORD'],
        'validation_selected_threshold': wake_export['validation_threshold_int8'],
        'local_operating_threshold': args.threshold,
        'int8_path': 'models/binary_wake_int8.onnx',
        'int8_sha256': digest(wake),
        'pretrained_weights_used': False,
    }, indent=2) + '\n', encoding='utf-8')
    (dest / 'README.md').write_text(
        '# ME2 - VCM on Raspberry Pi 5: current two-model candidate\n\n'
        f'Wake trained from random initialization in `{run.name}`. '
        f'Intent provenance: `{metadata["pair_provenance"].get("intent", {"training": "from_random_initialization", "source_run": run.name})}`. Separate binary wake and 31-class '
        'intent INT8 ONNX models are staged here with exact class ordering and SHA-256 hashes. '
        'The local operating wake threshold is fixed at 0.95. See `metadata.json` for held-out '
        'scores and limitations. This directory is the source for the next manual-launch Pi bundle; '
        'its presence alone does not mean the Pi has been updated.\n', encoding='utf-8')
    print(json.dumps({'staged': str(dest), 'combined_model_bytes': wake.stat().st_size + intent.stat().st_size,
                      'operating_point': point,
                      'intent_accuracy': metadata['intent']['dataset_test_accuracy'],
                      'intent_macro_f1': metadata['intent']['dataset_test_macro_f1'],
                      'min_class_f1': min(row['f1'] for row in per_label.values())}, indent=2))


if __name__ == '__main__':
    main()
