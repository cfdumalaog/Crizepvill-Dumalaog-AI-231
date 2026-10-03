"""Build and execute the Paired SLURP Speech Comparison Jupyter Notebook.

Notebook target: notebooks/ME2_SLURP_Speech_Comparison.ipynb
Implements:
1. Header, student declaration, CC BY-NC 4.0 license attribution.
2. SLURP Manifest and exclusion audit visualization.
3. Paired dataset preparation & random initialization parity verification.
4. Execution of Control training run (Run C: Option B baseline) for 35 epochs with live progress.
5. Execution of Treatment training run (Run T: Option B + SLURP 75:25 mix) for 35 epochs with live progress.
6. Side-by-side comparative analysis of FP32 and INT8 models across Option B test and SLURP test.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(r"C:\Users\danda\Desktop\MEng AI Notebooks\AI 222 231\AI 231\Dumalaog_ME2 - Tiny Voice Command Model")
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "ME2_SLURP_Speech_Comparison.ipynb"

nb = nbf.v4.new_notebook()

# Cell 0: Header Markdown
cell_header = nbf.v4.new_markdown_cell(r"""# AI 231 MEX2: Embedded Voice Command System (Option B + Wake Word)
## Task B: Paired Real Natural Speech Comparison (Option B Control vs. Option B + SLURP Real Treatment)

**Student**: Crizepvill F. Dumalaog (SN: 202521406)  
**Course**: AI 231 (Deep Learning Systems) — Machine Exercise 2  
**Experimental Design**: Paired from-scratch comparison on 32-class TinyDSCNN-48:
- **Control (Run C)**: Option B Spoken Commands (14,370 train clips) + Identical Wake Data (910 train clips)
- **Treatment (Run T)**: Option B Spoken Commands + SLURP Real Audio (75:25 mix for 13 supported classes) + Identical Wake Data (910 train clips)
- **Model Architecture**: Depthwise Separable CNN with Log-Mel Frontend ($1 \times 40 \times 251$), 14,576 parameters (~60.7 KB FP32 / ~39.5 KB Static INT8 ONNX)

---

### Academic Integrity & Experimental Declarations
1. **Real Human Speech Baseline**: Option B is human speech (84 LibriSpeech reference speakers + 16 SilencioPH speakers); it is not synthetic command speech. The experiment evaluates the effect of supplementing narrow fixed-phrase human commands with natural real-world spoken commands from the SLURP corpus.
2. **SLURP Attribution & License**: SLURP audio is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0; Bastian et al., "SLURP: A Spoken Language Understanding Resource Package", EMNLP 2020, DOI `10.18653/v1/2020.emnlp-main.588`). Raw audio, intermediate caches, and checkpoints are excluded from Git and Pi release bundles.
3. **Random Initialization Parity**: Both models are initialized from the EXACT SAME random parameter state (`seed=231`, zero pretrained weights, identical initial parameter SHA-256 hash `76ea309101b22a3efdc80fdc9629e7d82a2cf45a5fad88c54796d522914442a6`).
4. **Matched Step Budget & Schedule**: Exactly 35 epochs, 239 batches/epoch (batch size 64), 8,365 optimizer steps total for both runs using AdamW (`lr=3e-3`, `weight_decay=1e-4`) and `CosineAnnealingLR`.
5. **Supported Label Mapping & Exclusions**: Mapped strictly to 13 supported classes (`PLAY_MUSIC`, `WEATHER`, `TIME`, `LIGHT_ON`, `LIGHT_OFF`, `VOLUME_UP`, `VOLUME_DOWN`, `COLOR_RED`, `COLOR_GREEN`, `COLOR_BLUE`, `ALARM_6_00AM`, `ALARM_8_00AM`, `ALARM_9_00PM`). Generic list intents (`lists_*`) were strictly excluded (no reminder invention), and entity rules require exact singular colors and times. Clock-time requests only for `datetime_query` (262 mapped, 228 calendar/date requests excluded).
6. **Deterministic Speaker-Disjoint Split**: Merged 153 active SLURP speakers partitioned disjointly using `seed=231` (123 train / 14 val / 16 test) across 6,880 deduplicated source groups. Exactly one recording per `(usrid, recid)` was selected, prioritizing far-field/non-headset.
7. **Validation & Quantization Protocol**: Checkpoint selection is strictly evaluated on Option B validation split + wake validation positives. Static INT8 quantization is calibrated using 256 training-only stratified samples with source-diverse human and synthetic wake samples.
8. **Strict Test Isolation Protocol**: Test sets (Option B test and SLURP test) remain unopened and unfeaturized until both models have finished training, validation checkpointing, FP32 export, and static INT8 quantization. Both models are evaluated once in a single pass at validation-frozen threshold $\theta^* = 0.45$.
""")

# Cell 1: Environment & System Initialization
cell_env = nbf.v4.new_code_cell(r"""import sys
from pathlib import Path
import json
import hashlib
import numpy as np
import soundfile as sf
import torch
import onnx
import onnxruntime as ort

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinyvcm_antigrav.config import LABELS, WAKE_LABELS, SAMPLES, SR, CHANNELS, FEATURE_SHAPE
from tinyvcm_antigrav.model import TinyDSCNN
from tinyvcm_antigrav.paired_trainer import SUPPORTED_SLURP_CLASSES

print("=" * 80)
print("  SYSTEM & HARDWARE ENVIRONMENT CHECK")
print("=" * 80)
print(f"Python:            {sys.version.split()[0]}")
print(f"PyTorch:           {torch.__version__} (CUDA: {torch.cuda.is_available()})")
if torch.cuda.is_available():
    print(f"Compute Device:    {torch.cuda.get_device_name(0)}")
print(f"ONNX:              {onnx.__version__}")
print(f"ONNX Runtime:      {ort.__version__}")
print(f"Total Model Head:  {len(WAKE_LABELS)} classes (31 Option B Commands + 1 WAKE_WORD)")
print(f"Supported SLURP:   {len(SUPPORTED_SLURP_CLASSES)} classes: {SUPPORTED_SLURP_CLASSES}")
""")

# Cell 2: Manifest & Exclusion Audit
cell_manifest = nbf.v4.new_code_cell(r"""import json

slurp_dir = PROJECT_ROOT / 'data' / 'slurp'
audit_path = slurp_dir / 'slurp_exclusion_audit.json'

with audit_path.open('r', encoding='utf-8') as f:
    audit = json.load(f)

print("=" * 80)
print("  SLURP ANNOTATION PREFLIGHT & EXCLUSION AUDIT")
print("=" * 80)
print(f"Total SLURP Prompts Inspected:       {audit['total_prompts']:,}")
print(f"Strictly Mapped Prompts:             {audit['mapped_prompts']:,} (13 classes)")
print(f"DateTime Clock-Time Only Mapped:     {audit['datetime_clock_time_only']['mapped_count']:,} prompts")
print(f"DateTime Calendar/Day Excluded:      {audit['datetime_clock_time_only']['excluded_non_clock_or_calendar']:,} prompts")
print(f"Lists Rejected (No Reminder Inven.): {audit['exclusions'].get('lists_not_reminders', 0):,}")
print(f"Ambiguous Color Exclusions:          {audit['exclusions'].get('ambiguous_or_mixed_color_entities', 0):,}")
print(f"Ambiguous Time Exclusions:           {audit['exclusions'].get('ambiguous_alarm_time_entities', 0):,}")
print(f"Cross-Label Source Collisions:       {len(audit.get('cross_label_source_exclusions', [])):,}")

spk_summary = audit['decoded_deduplicated_split_summary']
print("\nSpeaker-Disjoint Partitions (Seed 231):")
print(f"  Train: {spk_summary['train']['speakers']} speakers | {spk_summary['train']['recordings']:,} clips")
print(f"  Val:   {spk_summary['val']['speakers']} speakers | {spk_summary['val']['recordings']:,} clips")
print(f"  Test:  {spk_summary['test']['speakers']} speakers | {spk_summary['test']['recordings']:,} clips (Held-Out)")

print("\nTop Unsupported Intents Excluded:")
unsupp = {k.replace('unsupported_intent:', ''): v for k, v in audit['exclusions'].items() if k.startswith('unsupported_intent:')}
for k, v in sorted(unsupp.items(), key=lambda item: -item[1])[:5]:
    print(f"  - {k:<25}: {v:,} prompts")
""")

# Cell 3: Dataset Preparation & Parity Verification
cell_prep = nbf.v4.new_code_cell(r"""from tinyvcm_antigrav.paired_trainer import prepare_paired_datasets

print("=" * 80)
print("  PREPARING PAIRED DATASETS & VERIFYING PARITY")
print("=" * 80)

datasets = prepare_paired_datasets(seed=231)

print("\nDataset Partition Sizes (Train & Validation Splits):")
print(f"  Control Train (Run C):   {len(datasets['control_train_y']):,} samples")
print(f"  Treatment Train (Run T): {len(datasets['treatment_train_y']):,} samples (Matched 1:1)")
print(f"  Option B Val:            {len(datasets['optb_val_y']):,} samples (10 speakers)")
print(f"  SLURP Val:               {len(datasets['slurp_val_y']):,} samples (14 speakers)")
print(f"  Test Splits:             [ISOLATED] Unopened and unfeaturized until post-training evaluation.")

# Verify Random Initialization Parity
seed = 231
torch.manual_seed(seed)
m_c = TinyDSCNN(classes=len(WAKE_LABELS), channels=CHANNELS)
hash_c = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in m_c.parameters())).hexdigest()

torch.manual_seed(seed)
m_t = TinyDSCNN(classes=len(WAKE_LABELS), channels=CHANNELS)
hash_t = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in m_t.parameters())).hexdigest()

print(f"\nInitial Parameters Verification:")
print(f"  Control Model Initial SHA-256:   {hash_c}")
print(f"  Treatment Model Initial SHA-256: {hash_t}")
assert hash_c == hash_t, "Initialization mismatch!"
assert hash_c == '76ea309101b22a3efdc80fdc9629e7d82a2cf45a5fad88c54796d522914442a6', "Unexpected initial hash!"
print("  [OK] Initialization parity verified: 100% bit-exact initial weights (76ea3091...).")
""")

# Cell 4: Control Model Training Run (Run C)
cell_train_c = nbf.v4.new_code_cell(r"""from tinyvcm_antigrav.paired_trainer import train_model

run_dir_c = PROJECT_ROOT / 'runs' / 'antigrav-wake32-control-matched-20260928'

summary_c = train_model(
    run_dir=run_dir_c,
    train_x=datasets['control_train_x'],
    train_y=datasets['control_train_y'],
    datasets=datasets,
    run_name='antigrav_wake32_control_matched',
    model_name='TinyDSCNN-48 Control (Option B Baseline)',
    is_treatment=False,
    seed=231,
    epochs=35,
    batch_size=64
)
print("\nControl Model Completed:")
print(f"  Best Epoch: {summary_c['best_epoch']} | Best Val Score: {summary_c['best_val_score']:.4f}")
print(f"  FP32 ONNX:  {summary_c['artifacts']['fp32_size_bytes']:,} B | SHA-256: {summary_c['artifacts']['fp32_sha256'][:8]}...")
print(f"  INT8 ONNX:  {summary_c['artifacts']['int8_size_bytes']:,} B | SHA-256: {summary_c['artifacts']['int8_sha256'][:8]}...")
""")

# Cell 5: Treatment Model Training Run (Run T)
cell_train_t = nbf.v4.new_code_cell(r"""run_dir_t = PROJECT_ROOT / 'runs' / 'antigrav-wake32-slurp-matched-20260928'

summary_t = train_model(
    run_dir=run_dir_t,
    train_x=datasets['treatment_train_x'],
    train_y=datasets['treatment_train_y'],
    datasets=datasets,
    run_name='antigrav_wake32_slurp_matched',
    model_name='TinyDSCNN-48 Treatment (Option B + SLURP Real)',
    is_treatment=True,
    seed=231,
    epochs=35,
    batch_size=64
)
print("\nTreatment Model Completed:")
print(f"  Best Epoch: {summary_t['best_epoch']} | Best Val Score: {summary_t['best_val_score']:.4f}")
print(f"  FP32 ONNX:  {summary_t['artifacts']['fp32_size_bytes']:,} B | SHA-256: {summary_t['artifacts']['fp32_sha256'][:8]}...")
print(f"  INT8 ONNX:  {summary_t['artifacts']['int8_size_bytes']:,} B | SHA-256: {summary_t['artifacts']['int8_sha256'][:8]}...")
""")

# Cell 6: Single-Pass Dual Evaluation & Comparative Analysis
cell_eval = nbf.v4.new_code_cell(r"""from tinyvcm_antigrav.paired_trainer import evaluate_paired_models, SUPPORTED_SLURP_CLASSES

run_dir_c = PROJECT_ROOT / 'runs' / 'antigrav-wake32-control-matched-20260928'
run_dir_t = PROJECT_ROOT / 'runs' / 'antigrav-wake32-slurp-matched-20260928'

# Execute single-pass dual evaluation strictly after both runs are complete
paired_metrics = evaluate_paired_models(
    control_dir=run_dir_c,
    treatment_dir=run_dir_t,
    datasets=datasets,
    seed=231
)

m_c = paired_metrics['control']
m_t = paired_metrics['treatment']

c_fp = m_c['fp32']
c_i8 = m_c['int8']
t_fp = m_t['fp32']
t_i8 = m_t['int8']

print("=" * 90)
print(f"{'PAIRED EXPERIMENT EVALUATION SUMMARY (Operating Threshold theta* = 0.45)':^90}")
print("=" * 90)
print(f"{'Evaluation Metric':<42} | {'Control (Option B)':<20} | {'Treatment (SLURP Mix)':<20}")
print(f"{'':<42} | {'FP32':<9} {'INT8':<10} | {'FP32':<9} {'INT8':<10}")
print("-" * 90)
print(f"{'Option B Test Accuracy (1,798 clips)':<42} | {c_fp['optb_test_acc']*100:6.2f}%   {c_i8['optb_test_acc']*100:6.2f}%   | {t_fp['optb_test_acc']*100:6.2f}%   {t_i8['optb_test_acc']*100:6.2f}%")
print(f"{'Option B Test Macro F1':<42} | {c_fp['optb_test_f1']*100:6.2f}%   {c_i8['optb_test_f1']*100:6.2f}%   | {t_fp['optb_test_f1']*100:6.2f}%   {t_i8['optb_test_f1']*100:6.2f}%")
print(f"{'SLURP Test Accuracy (636 natural clips)':<42} | {c_fp['slurp_test_acc']*100:6.2f}%   {c_i8['slurp_test_acc']*100:6.2f}%   | {t_fp['slurp_test_acc']*100:6.2f}%   {t_i8['slurp_test_acc']*100:6.2f}%")
print(f"{'SLURP Test Macro F1':<42} | {c_fp['slurp_test_f1']*100:6.2f}%   {c_i8['slurp_test_f1']*100:6.2f}%   | {t_fp['slurp_test_f1']*100:6.2f}%   {t_i8['slurp_test_f1']*100:6.2f}%")
print(f"{'Held-Out Human Wake Recall (2/2 groups)':<42} | {c_fp['hw_wake_test_recall']*100:6.1f}%   {c_i8['hw_wake_test_recall']*100:6.1f}%   | {t_fp['hw_wake_test_recall']*100:6.1f}%   {t_i8['hw_wake_test_recall']*100:6.1f}%")
print(f"{'Held-Out Synthetic Wake Recall (35 clips)':<42} | {c_fp['synth_wake_test_recall']*100:6.1f}%   {c_i8['synth_wake_test_recall']*100:6.1f}%   | {t_fp['synth_wake_test_recall']*100:6.1f}%   {t_i8['synth_wake_test_recall']*100:6.1f}%")
print(f"{'False Wakes on Option B Commands (1,798)':<42} | {c_fp['false_wakes_on_optb_commands']:4d}      {c_i8['false_wakes_on_optb_commands']:4d}       | {t_fp['false_wakes_on_optb_commands']:4d}      {t_i8['false_wakes_on_optb_commands']:4d}")
print(f"{'False Wakes on SLURP Commands (636)':<42} | {c_fp['slurp_false_wakes']:4d}      {c_i8['slurp_false_wakes']:4d}       | {t_fp['slurp_false_wakes']:4d}      {t_i8['slurp_false_wakes']:4d}")
print(f"{'Option B Prediction Parity (FP32 vs INT8)':<42} | {'-':<9} {m_c['optb_prediction_parity']*100:6.2f}%   | {'-':<9} {m_t['optb_prediction_parity']*100:6.2f}%")
print(f"{'SLURP Prediction Parity (FP32 vs INT8)':<42} | {'-':<9} {m_c['slurp_prediction_parity']*100:6.2f}%   | {'-':<9} {m_t['slurp_prediction_parity']*100:6.2f}%")
print(f"{'INT8 ONNX Model Size (Bytes)':<42} | {'-':<9} {m_c['artifacts']['int8_size_bytes']:,} B | {'-':<9} {m_t['artifacts']['int8_size_bytes']:,} B")
print("=" * 90)

print("\n" + "=" * 90)
print(f"{'PER-CLASS ACCURACY ON HELD-OUT SLURP TEST SET (13 SUPPORTED CLASSES)':^90}")
print("=" * 90)
print(f"{'Supported Class':<25} | {'Support':<8} | {'Control INT8':<18} | {'Treatment INT8':<18} | {'Delta':<10}")
print("-" * 90)
c_slurp_pc = c_i8['slurp_per_class']
t_slurp_pc = t_i8['slurp_per_class']
for lbl in SUPPORTED_SLURP_CLASSES:
    if lbl in t_slurp_pc:
        supp = t_slurp_pc[lbl]['support']
        c_acc = c_slurp_pc.get(lbl, {}).get('accuracy', 0.0) * 100
        t_acc = t_slurp_pc[lbl]['accuracy'] * 100
        delta = t_acc - c_acc
        delta_str = f"{delta:+6.2f}%"
        note = " [LOW SUPPORT]" if supp < 10 else ""
        print(f"{lbl:<25} | {supp:<8} | {c_acc:6.2f}%            | {t_acc:6.2f}%            | {delta_str:<10}{note}")
    else:
        print(f"{lbl:<25} | {'0 (untested)':<8} | {'N/A':<18} | {'N/A':<18} | {'N/A':<10} [INCONCLUSIVE]")
print("=" * 90)
""")

# Cell 7: Markdown Discussion
cell_conclusion = nbf.v4.new_markdown_cell(r"""### Key Takeaways & Discussion for Codex Independent Audit

1. **Option B Command Retention (Regression Check)**:
   - Measures whether introducing diverse acoustic conditions and natural conversational phrasings from SLURP degraded performance on the narrow fixed Option B command set.
   - Evaluates retention of command accuracy and macro F1 on Option B held-out speakers.

2. **Cross-Corpus Real Speech Generalization**:
   - Assesses model accuracy and macro F1 on natural, continuous, and far-field speech from 16 unseen speakers in the SLURP held-out test partition.
   - Evaluates performance across high-support command categories (`PLAY_MUSIC`, `WEATHER`, `TIME`, `LIGHT_OFF`, `VOLUME_UP`).

3. **Inconclusive & Low-Support Classes**:
   - Identifies classes with limited test support in the SLURP partition:
     - `ALARM_9_00PM`: Only 5 total prompt recordings in the entire corpus (all partitioned into train); 0 held-out test support makes this class inconclusive on SLURP evaluation.
     - Classes with fewer than 10 test samples (e.g. `COLOR_GREEN`, `COLOR_RED`, `ALARM_8_00AM`) where sampling variance is elevated.

4. **Wake-Word Invariants & Discriminative Safety**:
   - Evaluates whether operating threshold $\theta^* = 0.45$ maintains 100% human wake recall (2/2 held-out groups) and zero false wake alarms on Option B commands.
   - Assesses false wake activations on natural conversational SLURP utterances to ensure open-domain speech does not trigger unintended wake activations.

5. **INT8 Edge Readiness**:
   - Validates that static INT8 quantization maintains high FP32/INT8 prediction parity and stays well within edge deployment budgets (~39.5 KB vs. 500 KB limit).
""")

nb.cells = [cell_header, cell_env, cell_manifest, cell_prep, cell_train_c, cell_train_t, cell_eval, cell_conclusion]

with NOTEBOOK_PATH.open('w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"[OK] Notebook written: {NOTEBOOK_PATH.name} ({len(nb.cells)} cells)")

if '--execute' in sys.argv:
    print("\n" + "=" * 80)
    print("  EXECUTING NOTEBOOK CELLS (TRAINING RUN C & RUN T)")
    print("=" * 80)
    ep = ExecutePreprocessor(timeout=3600, kernel_name='ai222-231')
    ep.preprocess(nb, {'metadata': {'path': str(NOTEBOOKS_DIR)}})
    with NOTEBOOK_PATH.open('w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"[OK] Executed notebook saved: {NOTEBOOK_PATH.name}")

