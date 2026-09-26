# Tiny VCM handoff — 2026-09-26 (Asia/Manila)

Read the workspace handoff and setup notes first. Use the single root `.venv`.
The [README](README.md) is the current operation and results guide. The old handoff
is archived under `docs/legacy/`; its synthetic-data and Pi-readiness claims were
not supported. Preserve historical source/assets; do not treat them as verified.

## Current state

- User requested from-scratch PyTorch training visible in Jupyter, always-listening
  PC demo with Hi/Hello Dandan wake phrase, Pi 5 4 GB preparation and course-repo
  organization. User explicitly has **no human recordings or Pi available yet**
  at task start; check `data/human/manifest.csv` for any subsequently saved takes.
- Assignment: collect smart-device speech across ten categories, train from scratch,
  design a collective benchmark, evaluate individually, demonstrate standalone
  on-device control on RPi4/5. No pretrained recognition, ASR, cloud model or LLM.
  Earlier supplied targets: 16 kHz WAV, at least three real people, multiple
  conditions, <500 KB INT8, <10 ms edge inference. Due Sep 26.
- Current source: `tinyvcm/`, `demo.py`, `record_dataset.py`, `deployment/`.
  Current executed notebook: `notebooks/ME2_From_Scratch_Verified.ipynb`.
  It was run visibly in Jupyter, saved with code counts 1–6 and no error outputs.
- Run `20260926-111536`: 100 fresh epochs, best epoch 84, TinyDSCNN48 with 14,282
  parameters. No pretrained model was loaded. Export is 44,640 bytes, static QDQ
  INT8; optimized PC graph has 9 QLinearConv and 1 QGemm operator.
- Synthetic split: 1,688 train / 826 val / 1,221 test. Voice Zira held out; remaining
  voices' +2-rate groups validation-only. Source groups and WAV hashes disjoint.
  TTS and generated noise are **not human or real-room recordings**.
- Validation 90.80%, FP32 test 81.08%, INT8 test 79.28%. Noise SNR20/10/0 accuracy
  73.76/59.49/18.40%. Continuous synthetic replay: 194/216 wakes and 77/216 correct
  sequences. This is development evidence; human recognition is unmeasured.
- Separate Windows inference process: p50/p95 model 0.501/0.679 ms,
  frontend+model 1.150/1.826 ms, RSS73.45 MB. Not Raspberry Pi measurements.
- Services launched: authenticated loopback Jupyter8890, assistant7861,
  recorder7862. Root launchers `Start-VCM-Notebook.ps1`, `Start-PC-Demo.ps1`,
  `Start-VCM-Recorder.ps1` reuse the shared environment. Check service availability
  before launching a duplicate. Never copy Jupyter authentication tokens to logs.
- Recorder workflow: select speaker/condition/class, record, listen, **Save this
  take**, wait for confirmation. Files/manifest under `data/human/`. Saving does
  not retrain. User was told to keep stable anonymous IDs, collect repeated takes,
  use at least three actual people and mute the assistant while recording.
- Lesson repo already existed; updated safely via fast-forward to `ec3c5be`,
  preserving `PDF_REVIEW.md` and `lecture_notes`. No duplicate checkout created.
- Shared environment additions: onnx1.23.0, onnxruntime1.30.0, ml_dtypes0.6.0,
  flatbuffers25.12.19; `pip check` passed. No additional PC environment created.

## Known limitations and cautions

- Recognition is not submission-ready: only 35.65% correct synthetic streaming
  sequences. No live human success rate, false wakes/hour or Pi measurement exists.
- Baseline 26 classes lacks unrelated-speech training. Human mode adds `_unknown_`.
  Human data audit requires all classes and >=2 conditions per speaker, >=3 people.
  Last sorted speaker=test, penultimate=val, others=train; more people recommended.
- Existing TTS clips may already be truncated. Human collector rejects >1.5-second
  speech; increase the shared window and retrain if natural commands need longer.
- Thresholds are provisional. Pause after wake for chime, then speak. Compound
  wake+command and arbitrary slot values are unsupported. No echo canceller.
- Light GPIO/DS18B20 are optional and untested on hardware; thermostat, phone and
  reminders are explicitly demonstration behavior. Clock/timer durations are real.
  Audio playback is a generated tune. Timers/alarms disappear on restart.
- The ZIP is an application bundle, not an OS image. Flash 64-bit Pi OS, copy,
  install dependencies, choose audio device and verify wiring before boot service.
- Older notebooks, simulation metrics, project spreadsheet and `rpi_deployment/`
  remain historical. They must not be cited as human or Raspberry Pi evidence.
- All ME2 work is local/uncommitted. No new push or visibility change was requested.

## Open items

1. Collect consented human WAVs/unknown speech/room noise; audit per-class and
   condition coverage. Record natural complete phrases, not rushed fragments.
2. Preserve baseline notebook, switch DATA_MODE to human and run a fresh training
   notebook. Calibrate/tune only on training/validation and report test separately.
3. Agree class benchmark using `docs/BENCHMARK.md`; measure real continuous false
   alarms and full command success. Improve model/window/thresholds based on validation.
4. Regenerate deployment from the human-trained run and verify portable inference.
5. Obtain Pi5 4GB, microphone, speaker and low-voltage LED/buzzer/sensor. Install,
   measure latency/RAM/sustained audio, wire and demonstrate actual outputs.
6. Student review and academic provenance before submission. Do not claim the
   task finished merely because size and PC inference meet their numerical goals.

## Verification and change log

- **2026-09-26:** Audited misleading legacy provenance/quantization/Pi claims;
  implemented and ran 100-epoch from-scratch DS-CNN notebook; static INT8 export
  and explicit held-out metrics; started native-mic PC dashboard and local human
  recorder; added endpointed wake lifecycle, real-duration timers, labeled device
  simulations and Pi installer/bundle. Twelve focused tests passed; notebook six
  code cells saved without errors; optimized graph uses integer conv/dense;
  standalone PC benchmark and 216-sequence replay recorded; native Realtek WDM-KS
  48 kHz microphone captured/resampled with zero short-check drops. Bash installer
  syntax and bundle checksum checks passed. Human/Pi evaluation remains open.
