# ME2 Spoken Command Dataset — latest linked revision audit

**Date:** 2026-10-02 (Asia/Manila)
**Status:** Repulled and fully audited; an isolated scratch-trained candidate was trained, frozen, scored once, and deployed separately for manual Pi testing.

This report supersedes the dataset interpretation in `classagreementvcm-audit-20261002.md`. The earlier report treated a documented category of in-scope, unsupported slot values as malformed/OOS rows. The latest dataset README shows that category is intentional. The old snapshot and report are retained as historical evidence.

## Repull and integrity

- Hugging Face source: `airimonda/ai231-me2-voice-commands`, immutable revision `a90b8d106349b02c5570a1a258503386043f63b2`.
- Local snapshot: `data/ai231-me2-voice-commands-hf-a90b8d10/`. The prior revision remains untouched at `data/ai231-me2-voice-commands-hf-25111444/`.
- All 13 expected files were checked against the Hub revision and their remote content hashes; no missing, extra, or mismatched files.
- Full scan: 81,686 rows (train 10,682; test 4,418; holdout 196; numerals 66,390). All 81,686 embedded WAVs decoded as mono 16 kHz PCM16. There were zero row errors and zero cross-split audio leakage or cross-label collisions. One identical/normalized-PCM duplicate group occurs within a single split and label; it does not cross an evaluation boundary. The near-clipping amplitude heuristic flagged 1,795 waveforms for awareness; it did not treat them as decode failures.
- `variations.csv` is unchanged (93 rows; SHA-256 `6004b3b2b58137058c705b5769b0d094e05a5cff6ddcbed6de548b4a0105fcda`). The README changed. Train, test, and all six numeral shards have identical content hashes to the prior snapshot. The holdout Parquet changed.
- Holdout comparison by manifest filename found 59 new and 59 replaced/removed rows: 56 new in-scope `real_voice` clips and 3 new OOS Common Voice clips replaced 56 prior in-scope clips and 3 prior OOS clips. The holdout still contains 186 in-scope rows (two for each of 93 variations) and 10 OOS rows. Treat this exact revision as the frozen holdout for any later approved demo; do not use it for training, threshold selection, or repeated model selection.

Machine report: `runs/classagreementvcm-audit-20261002-repull/gold_dataset_audit.json`. The full scan used `scripts/classagreementvcm/audit_gold_dataset.py` with the new snapshot path/revision and recorded the owner's group confirmation about adding OOS data to test and holdout.

## Label and OOS interpretation

The dataset schema contains 19 top-level intents, 31 permitted intent/slot combinations, and 93 phrasing variations. All 93 schema rows map to the existing 31 command/slot labels. The linked benchmark repository at commit `3bd722173a040356cc71a4a902188af183540b43` scores the 19-intent and 93-command levels; a command result is represented as `(intent, slot)`. Its reject scoring accepts either an explicit OOS result or no response.

The 158 train rows previously reported as bad labels are explicitly documented by the current README as group recordings for schema intents with **other slot values**. They have `out_of_scope=0` and a bucket such as `TIMER (other slot value)`. Full-scan counts: TIMER 37, ALARM 23, TEMPERATURE 49, BRIGHTNESS 33, CREATE_REMINDER 16. Keep them as a separate unsupported-slot category; do not relabel them as OOS, force them to a nearest supported slot, or discard them. The latest train split also contains 201 true OOS records; test contains 47; holdout contains 10.

The dataset-owner message observed in the course group says OOS items were added to test and holdout and asks members to re-pull. The current benchmark README confirms the 196-row holdout contains 186 in-scope command clips and 10 OOS clips. Thus the holdout-presence confirmation is resolved. This does **not** decide how the candidate model should reject an utterance or an in-scope intent paired with an unsupported slot. That policy must be specified and calibrated on gold-train-derived validation only; frozen test and holdout stay untouched.

## Training and comparison outcome

The user's follow-up authorized a private coursework candidate after this dataset audit. The candidate was trained from random initialization on the 31 supported command/value labels; details and per-class scores are in `docs/evaluations/ME2_VCM_Dataset_Comparison_20261002.md`. A train-derived source/speaker-group validation partition was used for checkpoint and rejection-gate selection. The frozen 4,418-row test was evaluated once only after the model and scoring script were locked. The candidate slightly trails the current model on overall accuracy and macro F1, although it scores better on the smaller real/varied-source slices and has fewer false actions under its validation-selected gate. It has not replaced the active model.

The candidate is installed as `/home/dalmacio/Desktop/me2-dataset-candidate` and manually running on loopback port 7865, with GPIO disabled and boot autostart off. Pi ARM64 integrity/inference verification passed. The Pi has no microphone attached, so live speech evaluation is still pending. No dataset audio was packaged.

## Remaining gates and actions

- Source usage and consent evidence remains unresolved: the README's SLURP license statement conflicts with upstream audio terms; the exact Timers and Such license, Xela download/derivative tracking, classmate consent, and synthetic voice/reference permissions are not evidenced. The trained model is for this private coursework trial only pending rights/consent review.
- The candidate does not have a learned OOS output. It uses a validation-calibrated no-action gate and keeps the 158 unsupported-slot examples distinct from the 201/47/10 true OOS records.
- The benchmark accepts explicit OOS **or silence** as reject. The candidate selected a no-action gate using only train-derived validation. Its frozen test was opened once; do not use it for another model selection pass.
- The candidate shares the active wake model because the dataset has no wake-positive labels. The gold test contains 4,418 wake-negative clips; one false wake accept was observed at 0.95, but wake recall is unavailable.
- The Pi candidate is installed and verified separately. Its audio capture device is absent, so conduct the requested live A/B speech comparison only after attaching a microphone. Keep `/home/dalmacio/Desktop/dandan` as the default until then.

Benchmark source: [VCM benchmark README at the inspected commit](https://github.com/airimonda/vcm-benchmark/blob/3bd722173a040356cc71a4a902188af183540b43/README.md). Dataset source: [ME2 Spoken Command Dataset README at revision a90b8d1](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/blob/a90b8d106349b02c5570a1a258503386043f63b2/README.md).
