# Real-Voice SLURP Comparison

**Date:** 2026-09-28 (Asia/Manila)
**Notebook:** `archive/notebooks/ME2_SLURP_Speech_Comparison.ipynb` (archived experiment)
**Paired metrics:** `runs/paired_slurp_comparison_metrics.json`

## Question and dataset

This experiment asks whether human command recordings from SLURP improve the
existing scratch-trained 32-class VCM. The Option B command corpus is already
human-recorded (17,986 rows from 100 reference speakers); the synthetic speech
in both matched arms is the existing wake-word data. It uses only SLURP's
`slurp_real` human audio archive; no SLURP synthetic audio is included. The
official archive is 3,918,185,662 bytes; its published Zenodo MD5 is
`9efc0f058ced47bf5131c7cb2cade513`, which matches the downloaded archive.
The local SHA-256 is
`7f8fdd5e58960d22d2d0557612a407dd57bfa6188776f2d4fd47926a68b18dd5`.

The strict mapper retains 2,574 prompts across 13 of the model's 31 command
labels. It maps only explicit clock-time queries to `TIME`, rejects calendar
and ambiguous slot cases, and deduplicates recordings by `(usrid, recid)`.
The resulting manifests contain 5,426 train recordings from 123 speakers,
818 validation recordings from 14 speakers, and 636 test recordings from 16
speakers. Speaker intersections are zero. One microphone variant is retained
per source recording. All manifest paths resolve. The training feature labels
and source IDs match the manifest, and the training cache sidecar matches the
manifest SHA-256. Validation cache labels match their manifest; SLURP validation
features are loaded by the trainer but are not used for training or checkpoint
selection. The paired trainer's legacy cache helper does not fingerprint cache
files, so a future run should use manifest-keyed caches. The notebook output
shows the SLURP test cache was freshly extracted from the current 636-row test
manifest after both models were fixed.

SLURP audio is licensed CC BY-NC 4.0. Keep the raw audio local, attribute the
dataset, and limit derived model use to noncommercial academic work unless a
different license is obtained. The annotations are separately licensed CC BY
4.0. See the [official SLURP repository](https://github.com/pswietojanski/slurp)
and [Zenodo archive](https://zenodo.org/records/4274930).

## Matched training setup

Both models use TinyDSCNN-48 with the same 32-class label map, seed 231, fresh
random initialization, initial parameter SHA-256
`76ea309101b22a3efdc80fdc9629e7d82a2cf45a5fad88c54796d522914442a6`, 35 epochs,
and the same wake/background examples. The wake positives include the existing
synthetic wake-word data in both arms; this experiment changes command-speech
training only.

- **Control:** Option B command training data plus the common wake/background
  data.
- **Treatment:** for SLURP-supported labels, use a 75:25 Option B/SLURP mix;
  25% of that class's Option B rows are replaced by SLURP rows. This preserves
  the per-class and total training-example budget. It is not a full additive
  run retaining every Option B row plus extra SLURP rows.

Both arms are selected on validation data before a single final dual evaluation
on the 1,798 Option B test commands and the 636 speaker-held-out SLURP test
commands. The saved model hashes match the current FP32 and INT8 ONNX files in
each run directory. The six notebook code cells have execution counts and no
stored errors. No test cell was rerun during this audit.

## Results

The following are saved INT8 ONNX results at the same 0.45 wake threshold:

| Held-out set / metric | Option B only | Option B + SLURP mix | Change |
|---|---:|---:|---:|
| Option B accuracy (1,798) | 94.49% | 94.10% | -0.39 pp |
| Option B macro-F1 | 94.48% | 91.20% | -3.28 pp |
| Option B false wake activations | 0 | 1 | +1 |
| SLURP accuracy (636) | 5.66% | 19.34% | +13.68 pp |
| SLURP macro-F1 | 2.52% | 7.34% | +4.82 pp |
| SLURP false wake activations | 0 | 0 | 0 |
| INT8 ONNX size | 39,494 B | 39,494 B | unchanged |

The largest class-level SLURP INT8 gains were `PLAY_MUSIC` (+23.47 percentage
points), `WEATHER` (+14.74), and `LIGHT_OFF` (+12.07). `VOLUME_UP` decreased by
7.89 points and `VOLUME_DOWN` by 3.85. `ALARM_9_00PM` had no SLURP test examples;
several other classes had fewer than 10, so those per-class estimates are
unstable.

## Interpretation

The SLURP mix improves cross-corpus recognition substantially, but the treatment
still recognizes only 19.34% of SLURP test commands and has 7.34% macro-F1.
It therefore does not yet demonstrate dependable natural-command operation.
Option B accuracy is nearly retained, but its macro-F1 falls 3.28 points and
one held-out command is falsely classified as a wake word. This is a useful
experimental direction, not evidence that the treatment is better overall or
ready to replace the current candidate.

This is one seed and one held-out split. SLURP has uneven mapped-class support,
and the 636-example test set has very low support for several labels. Wake
results are weak evidence as well: human wake recall is 2/2 and synthetic wake
recall is 35/35 for both models. Neither establishes robust wake-word behavior.

The next comparison should use an explicitly additive treatment retaining the
full Option B command set and adding real SLURP examples, with a compute-matched
control. Since the SLURP test split has now been used for this comparison, that
next version needs a new, untouched speaker-held-out evaluation set; the
existing SLURP test set must not be used to select its recipe. [Fluent Speech
Commands](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/)
is another relevant human-recorded smart-home corpus (30,043 utterances, 97
speakers, 16 kHz, 31 intents). Its CC BY-NC-ND 4.0 license is more restrictive;
review its terms with the instructor/provider before downloading or using it to
train a model.

The SLURP treatment artifact is not deployed to the Raspberry Pi. The existing
Pi service/model status is tracked separately in `HANDOFF_INDEX.md`.
