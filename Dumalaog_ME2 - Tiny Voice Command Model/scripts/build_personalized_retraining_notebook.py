"""Build the repeatable, visible-progress personalized ME2 retraining notebook."""
from __future__ import annotations

from pathlib import Path
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "notebooks" / "ME2_Personalized_Wake_Intent_Retraining.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python (AI 222 + AI 231)", "language": "python", "name": "ai222-231"},
    "language_info": {"name": "python", "version": "3.12"},
}
nb.cells = [
    nbf.v4.new_markdown_cell("""# ME2 - VCM on Raspberry Pi 5 — Personalized Wake and Intent Retraining

This experiment uses the current deduplicated human recording manifest. It compares two independently scratch-initialized two-class wake detectors:

1. **Broad negatives:** real wake takes as positives; Option B command speech, silence and background noise as non-wake.
2. **Personal hard negatives:** the same wake takes and broad negatives, plus the speaker's own non-wake commands and near-miss speech.

Training on wake recordings alone cannot teach a wake/non-wake decision; a binary detector needs negative examples. The intent candidate is scratch-trained on the Option B training partition plus a small, controlled fraction of exact-mapped personal command recordings. The canonical Option B INT8 model is the intent baseline. All splits are made after exact PCM deduplication. The Option B held-out test data are not loaded until the final cell, after all models and thresholds are frozen.

**Important limits:** the audit cell reports the current speaker IDs. Personal data are split by label and clip after exact PCM deduplication, not by speaker, so the same people may occur in train, validation, and test. These results do not establish unseen-speaker accuracy. Several fixed intent classes have very few unique personal takes. The Option B test split has been used in prior project experiments. No Pi performance claim is made by these notebook results.

Normal execution trains from scratch and evaluates once. Set `VCM_EXISTING_RUN_DIR` before executing to reconstruct the visible report from an already completed run without retraining or reopening the held-out Option B test."""),
    nbf.v4.new_code_cell("""from pathlib import Path
import os, sys, json, time
from datetime import datetime
import pandas as pd
from IPython.display import display

HERE = Path.cwd().resolve()
PROJECT_ROOT = next((p for p in (HERE, *HERE.parents) if (p / 'tinyvcm_model').is_dir()), None)
if PROJECT_ROOT is None:
    raise RuntimeError('Open this notebook from its ME2 project folder so tinyvcm_model can be found.')
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tinyvcm_model.personalized_retrain import (
    prepare_experiment, run_wake_pair, train_intent_candidate_from_data,
    _export_intent_candidate, evaluate_all,
    load_personal_records, load_optionb_train_val,
)
from tinyvcm_model.recording_labels import map_recorded_intent

REUSE_RUN_DIR = Path(os.environ['VCM_EXISTING_RUN_DIR']).resolve() if os.environ.get('VCM_EXISTING_RUN_DIR') else None
RUN_DIR = REUSE_RUN_DIR or (PROJECT_ROOT / 'runs' / f"personalized-vcm-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
print('Project:', PROJECT_ROOT)
print('Existing completed run for report reconstruction:' if REUSE_RUN_DIR else 'New scratch training output:', RUN_DIR)
print('Shared Python:', sys.executable)
"""),
    nbf.v4.new_markdown_cell("""## 1. Audit, deduplicate, split, and prepare training data

This cell validates the local manifest and WAV headers, removes exact duplicate PCM before splitting, writes a split manifest under the new run, and reads only the Option B training and validation feature arrays. It deliberately does not open the Option B test features."""),
nbf.v4.new_code_cell("""if REUSE_RUN_DIR:
    human, _ = load_personal_records(PROJECT_ROOT)
    audit = json.loads((RUN_DIR / 'dataset_audit.json').read_text(encoding='utf-8'))
    DATA = {**load_optionb_train_val(PROJECT_ROOT), 'project_root': PROJECT_ROOT,
            'human': human, 'human_audit': audit, 'output_dir': RUN_DIR}
    print('Reconstructed the deterministic personal data split. Train/validation features only; test is not opened.')
else:
    DATA = prepare_experiment(RUN_DIR, PROJECT_ROOT)
audit = DATA['human_audit']
display(pd.DataFrame([
    {'saved files': audit['raw_manifest_rows'], 'unique waveforms': audit['unique_pcm_groups'],
     'duplicate rows removed': audit['exact_duplicate_rows_collapsed'],
     'real speakers': ', '.join(audit['speakers']), 'cross-label PCM collisions': len(audit['cross_label_hash_collisions'])}
]))
display(pd.DataFrame([
    {'label': label, 'unique takes': count,
     'train': sum(i['label']==label and i['split']=='train' for i in DATA['human']),
     'validation': sum(i['label']==label and i['split']=='validation' for i in DATA['human']),
     'test': sum(i['label']==label and i['split']=='test' for i in DATA['human']),
     'mapped intent': map_recorded_intent(label) or 'wake / not mapped'}
    for label, count in audit['unique_label_groups'].items()
]))
print('Option B cache manifest SHA-256:', DATA['manifest_sha256'].item())
print('Personal manifest SHA-256:', audit['manifest_sha256'])
"""),
    nbf.v4.new_markdown_cell("""## 2. Train the wake-model comparison

Each variant sees the same content-deduplicated, clip-held-out wake recordings, Option B command pool, and training schedule. The personal split is not speaker-disjoint. The second variant adds only training-split personal command/near-miss recordings. Validation selects each INT8 threshold under a zero-validation-false-accept constraint. The final test clips remain unopened."""),
nbf.v4.new_code_cell("""if REUSE_RUN_DIR:
    WAKE_RUNS = {}
    for name, folder in (('broad', 'wake_broad_negatives'), ('hard', 'wake_with_personal_intents')):
        model_dir = RUN_DIR / folder / 'models'
        summary = json.loads((model_dir / 'export_summary.json').read_text(encoding='utf-8'))
        WAKE_RUNS[name] = {
            'summary': summary,
            'val_threshold': summary['validation_threshold_int8'],
            'fp32_path': model_dir / 'wake_personalized_fp32.onnx',
            'int8_path': model_dir / 'wake_personalized_int8.onnx',
        }
        history = json.loads((RUN_DIR / folder / 'history.json').read_text(encoding='utf-8'))['history']
        print(f"Loaded {name} wake model trained for {len(history)} epochs; selected epoch {summary['best_epoch']}.")
        display(pd.DataFrame(history))
else:
    WAKE_RUNS = run_wake_pair(DATA, epochs=30)
display(pd.DataFrame([
    {'variant': name, 'INT8 bytes': run['summary']['int8_size_bytes'],
     'validation threshold': run['val_threshold'],
     'FP32 validation wake recall': run['summary']['fp32_validation_threshold_metrics']['wake_recall'],
     'validation wake recall': run['summary']['int8_validation_threshold_metrics']['wake_recall'],
     'validation wake support': run['summary']['int8_validation_threshold_metrics']['wake_support'],
     'validation non-wake support': run['summary']['int8_validation_threshold_metrics']['nonwake_support'],
     'INT8 SHA-256': run['summary']['int8_sha256']}
    for name, run in WAKE_RUNS.items()
]))
"""),
    nbf.v4.new_markdown_cell("""## 3. Train the personalized intent candidate from scratch

This retains the 31 fixed Option B output classes. Only exact-mapped personal labels are added; unsupported phrases such as five-minute timers, thermostat direction, and the recorded seven-AM alarm are excluded instead of being mislabeled. The per-batch personal fraction is intentionally small. Checkpoint selection uses validation only."""),
nbf.v4.new_code_cell("""if REUSE_RUN_DIR:
    INTENT_DIR = RUN_DIR / 'intent_personalized'
    INTENT_RUN = {'output_dir': INTENT_DIR,
                  'run_info': json.loads((INTENT_DIR / 'run_info.json').read_text(encoding='utf-8'))}
    INTENT_EXPORT = json.loads((INTENT_DIR / 'models' / 'export_summary.json').read_text(encoding='utf-8'))
    history = json.loads((INTENT_DIR / 'history.json').read_text(encoding='utf-8'))['history']
    print(f"Loaded scratch-trained personalized intent model; selected epoch {INTENT_RUN['run_info']['best_epoch']}.")
    display(pd.DataFrame(history))
else:
    INTENT_RUN = train_intent_candidate_from_data(DATA, epochs=35)
    INTENT_EXPORT = _export_intent_candidate(INTENT_RUN, DATA)
display(pd.DataFrame([{
    'epochs': INTENT_RUN['run_info']['epochs'], 'best epoch': INTENT_RUN['run_info']['best_epoch'],
    'personal train clips': INTENT_RUN['run_info']['personal_unique_train_clips'],
    'parameter count': INTENT_RUN['run_info']['parameter_count'],
    'FP32 bytes': INTENT_EXPORT['fp32_size_bytes'], 'INT8 bytes': INTENT_EXPORT['int8_size_bytes'],
    'INT8 SHA-256': INTENT_EXPORT['int8_sha256']
}]))
"""),

    nbf.v4.new_markdown_cell("""## 4. One-time held-out evaluation

Only now are the Option B test features opened. The threshold and selected checkpoints are frozen. Personal results are at the unique-waveform level; repeated recorder saves cannot multiply the test support. During report reconstruction, this cell reads the saved one-time evaluation and does not re-open the test set."""),
    nbf.v4.new_code_cell("""if REUSE_RUN_DIR:
    RESULTS = json.loads((RUN_DIR / 'final_evaluation.json').read_text(encoding='utf-8'))
    print('Loaded the saved final evaluation; held-out Option B data were not evaluated again.')
else:
    RESULTS = evaluate_all(DATA, WAKE_RUNS, INTENT_RUN)
print('Wake comparison:')
display(pd.DataFrame([
    {'variant': name, 'threshold': r['validation_selected_int8_threshold'],
     'FP32 wake hits/support': f"{r['fp32_heldout_personal_wake_hits']}/{r['fp32_heldout_personal_wake_support']}",
     'FP32 wake recall': r['fp32_heldout_personal_wake_recall'],
     'personal wake recall': r['heldout_personal_wake_recall'],
     'personal wake hits/support': f"{r['heldout_personal_wake_hits']}/{r['heldout_personal_wake_support']}",
     'personal non-wake false accepts': f"{r['false_wake_on_personal_nonwake_clips']}/{r['personal_nonwake_support']}",
     'Option B command false accepts': f"{r['false_wake_on_optionb_command_clips']}/{r['optionb_command_support']}",
     'model bytes': r['int8_model_size_bytes'], 'PC CPU p95 ms': r['pc_cpu_latency']['p95_ms_pc_cpu_model_only']}
    for name, r in RESULTS['wake_model_comparison'].items()
]))
print('Intent comparison:')
display(pd.DataFrame([
    {'model': name,
     'Option B INT8/test accuracy': vals['optionb_test']['accuracy'],
     'Option B macro-F1': vals['optionb_test']['macro_f1_supported_classes'],
     'personal heldout count': vals['personal_heldout_test'].get('count', 0),
     'personal heldout accuracy': vals['personal_heldout_test'].get('accuracy'),
     'personal macro-F1': vals['personal_heldout_test'].get('macro_f1_supported_classes'),
     'size bytes': vals['model_size_bytes'], 'PC CPU p95 ms': vals['pc_cpu_latency']['p95_ms_pc_cpu_model_only']}
    for name, vals in RESULTS['intent_model_comparison'].items()
]))
print('All 31 intent classes on the held-out Option B speakers (canonical vs personalized INT8):')
CLASSIC = RESULTS['intent_model_comparison']['canonical_int8']['optionb_test']['classification_report']
PERSONAL = RESULTS['intent_model_comparison']['personalized_int8']['optionb_test']['classification_report']
display(pd.DataFrame([
    {'class': label, 'support': int(CLASSIC[label]['support']),
     'canonical recall': CLASSIC[label]['recall'], 'personalized recall': PERSONAL[label]['recall'],
     'canonical F1': CLASSIC[label]['f1-score'], 'personalized F1': PERSONAL[label]['f1-score']}
    for label in CLASSIC if label in PERSONAL and isinstance(CLASSIC[label], dict) and 'f1-score' in CLASSIC[label]
]))
print('Your held-out recordings, grouped by recorder label (unmapped labels are prediction-only):')
display(pd.DataFrame([
    {'model': model_name, 'recorded label': source_label,
     'held-out clips': row['heldout_clip_support'], 'mapped intent': row['mapped_intent'],
     'correct': row['correct_hits'], 'accuracy': row['accuracy'],
     'predicted classes': row['predicted_class_counts']}
    for model_name in ('canonical_int8', 'personalized_int8')
    for source_label, row in RESULTS['intent_model_comparison'][model_name]['personal_recorded_commands_all_heldout']['per_recorded_label'].items()
]))
print('Full report:', RUN_DIR / 'final_evaluation.json')
print(json.dumps(RESULTS['known_limitations'], indent=2))
"""),
    nbf.v4.new_markdown_cell("""## Interpretation and promotion gate

Choose the wake variant using the held-out personal wake and non-wake results, with the speaker-overlap limitation in mind. Keep the simpler broad-negative version if personal hard negatives do not improve wake recall without added false accepts. Keep the canonical intent model if the personalized model's personal gain damages the Option B speaker-disjoint score. These are local research candidates; exporting ONNX does not promote them to the UI or Raspberry Pi. Collect more speakers, sessions, wake near-misses, and room noise before treating them as general-purpose."""),
]
nbf.write(nb, OUT)
print(f"Wrote {OUT}")
