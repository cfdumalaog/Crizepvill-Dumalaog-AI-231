# ME2 - VCM on Raspberry Pi 5 — personalized model candidate

This folder stages the selected, fresh-trained binary wake and 31-class intent
INT8 ONNX files with their threshold and integrity metadata. It is a **candidate
pair for integration and hardware audit**, not an installed Pi release.

The total ONNX size is approximately 75 KiB. Input is mono 16 kHz audio, padded
or cropped by the current frontend to 2.5 seconds and converted to 40-band
log-mel features. `metadata.json` records model signatures, SHA-256 checksums,
threshold provenance, measured candidate metrics, and limits. The wake runtime
expects `export_summary.json` beside this file so it can load the calibrated
threshold.

The wake threshold is 0.9992183447. It yielded 4/5 held-out personal wake clips
and zero false accepts across 1,798 held-out Option B command clips plus 42
held-out personal non-wake clips. These small, one-speaker sets do not establish
robustness or false wakes per hour. The personalized intent model classified
24/28 strict-mapped personal command clips correctly (85.7%); personal classes
and counts are listed in the full evaluation report.

No model has been copied to the powered-down Raspberry Pi. Before promotion,
retest with the actual microphone, gather data from at least three speakers and
room conditions, then measure end-to-end accuracy, false accepts, latency, and
memory on the Pi. See `../../docs/PERSONAL_WAKE_INTENT_EVALUATION_20260928.md`.
