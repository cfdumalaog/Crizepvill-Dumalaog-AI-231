# ME2 - VCM on Raspberry Pi 5 — fresh two-model candidate

This versioned folder contains the selected scratch-trained binary wake detector and personalized 31-class INT8 intent model from `personalized-vcm-retrain-20260928-190911`. Both ONNX files passed a local CPU smoke test with finite outputs and expected signatures. Total ONNX size: 76,740 bytes.

The wake model uses the personal-hard-negative variant, with a validation-selected threshold of 0.9985431433. It detected 7/8 held-out wake clips and accepted 0/57 held-out personal non-wake clips and 0/1798 Option B command clips at that threshold. The active desktop operating threshold is a separate user-requested 0.90 override: a five-position replay detected 8/8 wake clips, accepted 1/57 personal non-wake clips (one `volume_down` clip), and accepted 0/1798 Option B command clips. These short utterance results do not estimate continuous false-wakes-per-hour.

The personalized intent INT8 model scored 88.1% on 42 held-out personal clips and 96.0% on the reused Option B test set. Canonical INT8 scored 45.2% and 96.3%, respectively.

See `metadata.json` for class ordering, validation and desktop operating thresholds, hashes, model shapes and evaluation limits; see the dated executed notebook and `final_evaluation.json` in `runs/personalized-vcm-retrain-20260928-190911` for full training metrics. **This pair is active in the local desktop VCM at 127.0.0.1:7863 and is not deployed to the Raspberry Pi.** Personal clip splits overlap speakers, the Option B test was used in prior experiments, and no Pi mic/latency/runtime test has been done.
