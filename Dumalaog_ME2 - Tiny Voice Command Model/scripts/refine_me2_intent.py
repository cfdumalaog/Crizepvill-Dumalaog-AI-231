"""Validation-driven scratch retraining aimed at weak intent classes."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tinyvcm_model.personalized_retrain import (  # noqa: E402
    _export_intent_candidate, evaluate_all, prepare_experiment, train_intent_candidate,
)


def main() -> None:
    prior = PROJECT / 'archive/runs/me2-vcm-20260930-full-negatives'
    current = PROJECT / 'runs/me2-vcm-20260930-refined-intent'
    data = prepare_experiment(current, PROJECT)
    # Keep the already completed all-negative binary wake model byte-for-byte.
    shutil.copytree(prior / 'wake_binary', current / 'wake_binary')
    wake_models = current / 'wake_binary/models'
    summary = json.loads((wake_models / 'export_summary.json').read_text(encoding='utf-8'))
    wake = {'summary': summary, 'val_threshold': summary['validation_threshold_int8'],
            'fp32_path': wake_models / 'wake_personalized_fp32.onnx',
            'int8_path': wake_models / 'wake_personalized_int8.onnx'}
    intent = train_intent_candidate(data, current / 'intent_personalized', epochs=60,
                                    seed=232, recipe='refined')
    _export_intent_candidate(intent, data)
    report = evaluate_all(data, {'binary': wake}, intent)
    result = report['intent_model_comparison']['personalized_int8']
    print(f"Refined run: {current}", flush=True)
    print(f"Reference test accuracy={result['optionb_test']['accuracy']:.4f}, "
          f"macro F1={result['optionb_test']['macro_f1_supported_classes']:.4f}; "
          f"personal accuracy={result['personal_heldout_test']['accuracy']:.4f}", flush=True)


if __name__ == '__main__':
    main()
