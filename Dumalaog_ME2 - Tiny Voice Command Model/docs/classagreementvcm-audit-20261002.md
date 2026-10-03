# ME2 Spoken Command Dataset — classagreementvcm audit

**Date:** 2026-10-02 (Asia/Manila)
**Status:** Baseline captured; dataset integrity audit complete; implementation/training paused at Phase 1 gates.

> **Superseded data interpretation:** See `classagreementvcm-repull-audit-20261002.md` for the latest linked revision and corrected interpretation. The current README documents the 158 train rows as intentional in-scope commands with non-schema slot values, not malformed labels or mislabeled OOS. The owner confirmation for OOS examples in holdout is now recorded. This report remains as the initial-revision audit history; its data-gate conclusions below must not be treated as current.

This audit follows `docs/classagreementvcm-implementation-plan.md`. It is a data-integrity and provenance check only. No model was trained, no gold test or holdout predictions were run, no active model or Pi files were changed, and no Pi connection was attempted.

## Snapshot and audio checks

The local snapshot is the public Hugging Face revision `25111444af3adff7588d86ab27c304bb080895cf`. The revision is confirmed by the Hub API; all 13 expected files (README, phrase sheet, attributes file, and 10 Parquet shards) match the remote revision by file hash or content hash. The local Parquet row totals also match the dataset README:

| Partition | Rows | Use in this candidate |
|---|---:|---|
| Train | 10,682 | Candidate fitting after data/rights gates pass; validation must be carved from this partition by speaker/synthetic voice. |
| Test | 4,418 | Frozen final comparison only after every model choice is locked. |
| Holdout | 196 | Live demonstration only; audit inventory is 186 allowed-command rows plus 10 OOS rows. Owner confirmation is still required. |
| Numerals | 66,390 | Excluded from command fitting and scoring. |

All 81,686 command and numeral payloads decoded as 16 kHz mono PCM16 WAV and matched their manifest durations within the audit tolerance. The audit found no empty waveforms, duplicate normalized audio, cross-label audio collisions, or speaker/synthetic identity overlap across train, test, and holdout. It recorded 1,795 waveforms reaching the near-clipping amplitude heuristic; this is a review statistic, not a decode failure.

The source phrase sheet contains 93 rows mapping to 19 command groups and 31 permitted command/value leaves. Those 31 leaves map to the active model's label set. The row-level manifest audit found a separate problem described below, so the data are not yet safe to fit.

## Initial interpretation of the 158 rows (superseded)

There are **158 training rows** that are marked `out_of_scope=0` but are in a `<command> (other slot value)` bucket and have blank `slot_value` and `variation`. They cannot map to any of the 31 permitted leaves. Their command groups are:

| Command group | Rows |
|---|---:|
| TIMER | 37 |
| ALARM | 23 |
| TEMPERATURE | 49 |
| BRIGHTNESS | 33 |
| CREATE_REMINDER | 16 |
| **Total** | **158** |

The first audit script emitted 632 row-level diagnostics across these records (four schema/scope checks per affected row). The newer README establishes that the rows are intentional and remain in-scope at the intent level; their unsupported slot values need a model policy, not label correction. The test and holdout partitions have no such rows. Do not change their flags or drop them. The later group-owner update and linked benchmark README confirm the 10 OOS holdout examples are intentional.

## Source/license audit

The dataset card has no single dataset-level license value, so source-specific terms govern. The pinned README and upstream primary sources do not fully agree:

- The pinned README lists SLURP as CC BY 4.0, while the upstream SLURP repository says its **audio** is CC BY-NC 4.0. Treat the audio as non-commercial and correct the provenance record only after the data owner reconciles this discrepancy. [Upstream SLURP terms](https://github.com/pswietojanski/slurp)
- Fluent.ai releases Fluent Speech Commands under CC BY-NC-ND 4.0 for academic research only and rules out commercial use. [Fluent.ai dataset terms](https://fluent.ai/fr/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/)
- The Sonos SNIPS terms permit internal model training and research only for non-commercial academic/research purposes; they also impose voice-data privacy responsibilities and require citation if published. The local project does not establish who accepted those terms or document compliance. [Sonos license](https://github.com/sonos/spoken-language-understanding-research-datasets/blob/master/LICENSE)
- Zenodo marks Timers and Such as `other-open`; its exact data license file is absent from the snapshot. [Timers and Such record](https://zenodo.org/records/4623772)
- KU Leuven RDR states CC BY 4.0 for the Multi-Sensor Voice Command data and requires tracking full and partial downloads/derivatives for GDPR compliance. The candidate audit found no evidence that this project's tracking obligation has been satisfied. [KU Leuven dataset terms](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi%3A10.48804%2FIEKKVZ)
- The README identifies group recordings as not shareable outside class without each speaker's consent. Consent for model training/use must still be evidenced. The synthetic group dataset does not identify the exact SilencioPH reference release/license or document permission to clone those voices.
- Google Speech Commands v2 is CC BY 4.0. [Google dataset announcement](https://research.google/blog/launching-the-speech-commands-dataset/)
- Mozilla Common Voice is CC0 unless a specific release says otherwise; Mozilla asks that copies not be mirrored or redistributed outside its designated distribution channel. [Mozilla data terms](https://commonvoice.mozilla.org/dav/terms)
- MLEnd and Google numeral-only rows are excluded from this command model, so their terms are not used to authorize candidate training.

These findings block a full-corpus training run until the dataset owner reconciles the SLURP conflict, supplies the Timers and Such license, confirms Xela tracking, and provides class recording/synthetic voice rights. The model and any release must remain non-commercial if it uses non-commercial sources.

## Baseline preserved

`runs/classagreementvcm-audit-20261002/baseline.json` records the existing dirty-worktree status hash, active pair, code/frontend hashes, Desktop launcher/model hashes, and local release receipts. The active pair remains `me2-vcm-20261001-wake-refresh`: wake SHA-256 `6c2e941580d8ea778ed664c929e893c226efb91e3e787e65837908c81153deac`; intent SHA-256 `12984401fe312758dbd7ecd53ae9358b0a41b27ffcd024a9ae263614e2e7b977`. Its wake threshold remains 0.95; its intent gates remain 0.68 confidence and 0.15 margin. The newest local release receipt is still pending, while the prior Pi receipt is historical; the Pi was not contacted.

No files in the active release or Desktop deployment were modified by this audit. The candidate audit outputs are under the ignored `runs/classagreementvcm-audit-20261002/` directory. The machine-readable report is `runs/classagreementvcm-audit-20261002/gold_dataset_audit.json`.

## Gates before training

1. Decide how the candidate handles supported intents with non-schema slot values, without relabeling them as true OOS.
2. Choose and validate an OOS/reject behavior. The benchmark accepts an explicit OOS result or no response; a separate 32nd output is a design choice, not a benchmark requirement.
3. Dataset owner resolves source rights/consent/attribution and Xela tracking records above.
4. Ratify acceptance gates before any frozen-test exposure. Specify minimum per-leaf support, OOS false-action bound, and latency percentile/measurement device.

After these gates pass, proceed with a dedicated loader, frozen group-disjoint validation split from train only, and an isolated from-scratch candidate run. Keep the current wake bytes and active intent/deployment unchanged; do not run the normal training-to-Pi sync path. Test scoring, live holdout demonstration, candidate Pi installation, and any later promotion each remain separate gates.
