"""Train the current two-stage ME2 VCM from random initialization.

Only the dataset's training split supplies optimization examples. The
speaker-held-out command test split is opened once after both models and
thresholds have been frozen; personal clips are content-deduplicated first.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from tinyvcm_model.personalized_retrain import (  # noqa: E402
    _export_intent_candidate,
    evaluate_all,
    prepare_experiment,
    train_intent_candidate_from_data,
    train_wake_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wake-epochs', type=int, default=30)
    parser.add_argument('--intent-epochs', type=int, default=60)
    parser.add_argument('--wake-seed', type=int, default=231)
    parser.add_argument('--intent-seed', type=int, default=232)
    parser.add_argument('--intent-recipe', choices=('standard', 'refined'), default='refined')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--wake-only', action='store_true', help='Retrain wake from scratch; retain the active intent weights only if its personal corpus and splits are unchanged')
    parser.add_argument('--train-only', action='store_true', help='Explicitly skip the default latest-model Pi sync')
    parser.add_argument('--pi-host', default='cfdfnjrpi5.local')
    args = parser.parse_args()
    if args.wake_epochs < 1 or args.intent_epochs < 1:
        parser.error('Epoch counts must be positive.')
    name = args.run_name or f"me2-vcm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        parser.error('Run names may contain only letters, digits, underscores and hyphens.')
    run_dir = PROJECT / 'runs' / name
    active = json.loads((PROJECT / 'deployment/current_vcm/metadata.json').read_text(encoding='utf-8')) if args.wake_only else None
    data = prepare_experiment(run_dir, PROJECT)
    provenance = {'wake': {'training': 'from_random_initialization', 'source_run': name}}
    if args.wake_only:
        source = PROJECT / 'runs' / active['source_run']
        with (source / 'personal_split_manifest.csv').open(newline='', encoding='utf-8') as f:
            old = {(r['pcm_sha256'], r['mapped_intent'], r['split']) for r in csv.DictReader(f) if r['mapped_intent']}
        current = {(r['pcm_sha256'], r['intent'], r['split']) for r in data['human'] if r['intent']}
        if current != old:
            raise ValueError('Personal intent corpus/splits changed. Use a full paired run without --wake-only.')
        intent_dir = run_dir / 'intent_personalized'
        shutil.copytree(source / 'intent_personalized', intent_dir)
        intent_hash = hashlib.sha256((intent_dir / 'models/intent_personalized_int8.onnx').read_bytes()).hexdigest()
        if intent_hash != active['intent']['model']['sha256']:
            raise ValueError('Retained intent source hash does not match the active model.')
        origin = active.get('pair_provenance', {}).get('intent', {}).get('source_run', active['source_run'])
        provenance['intent'] = {'training': 'retained_byte_identical', 'source_run': origin, 'retained_from_pair': active['source_run'], 'int8_sha256': intent_hash}
        (intent_dir / 'REUSED_FROM.md').write_text(f"Intent weights and training history retained from {source.name}; no new intent optimization in {name}.\n", encoding='utf-8')
        print(f"Wake-only run: retained intent from {source.name}; mapped personal intent corpus and splits unchanged.", flush=True)
    else:
        provenance['intent'] = {'training': 'from_random_initialization', 'source_run': name}
    (run_dir / 'pair_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    wake = train_wake_model(data, run_dir / 'wake_binary', True, epochs=args.wake_epochs, seed=args.wake_seed)
    if args.wake_only:
        intent = {'output_dir': intent_dir}
    else:
        intent = train_intent_candidate_from_data(
            data, epochs=args.intent_epochs, seed=args.intent_seed, recipe=args.intent_recipe,
        )
        _export_intent_candidate(intent, data)
    report = evaluate_all(data, {'binary': wake}, intent)
    report['pair_provenance'] = provenance
    (run_dir / 'final_evaluation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"Completed scratch run: {run_dir}", flush=True)
    print(f"Final report: {run_dir / 'final_evaluation.json'}", flush=True)
    print(f"Wake: {report['wake_model_comparison']['binary']['heldout_personal_wake_hits']}/"
          f"{report['wake_model_comparison']['binary']['heldout_personal_wake_support']} personal wake clips", flush=True)
    print(f"Intent: {report['intent_model_comparison']['personalized_int8']['optionb_test']['accuracy']:.4f} "
          "reused speaker-held-out command accuracy", flush=True)
    print(f"Intent provenance: {provenance['intent']}", flush=True)
    if not args.train_only:
        from deploy_latest_vcm import synchronize
        try:
            synchronize(run_name=name, pi_host=args.pi_host)
        except Exception as exc:
            print(f'Training completed, but Pi synchronization is PENDING: {exc}', file=sys.stderr, flush=True)
            raise SystemExit(3)
        try:
            from export_manifest_excel import export_manifest
            print(f'Excel manifest refreshed: {export_manifest()}', flush=True)
        except Exception as exc:
            print(f'Pi sync succeeded, but Excel refresh needs retry: {exc}', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
