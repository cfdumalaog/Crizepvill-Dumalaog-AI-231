# Tiny Voice Command Model — AI 231 ME2

**Status: a trained, running prototype; human recognition and Raspberry Pi validation are unfinished.**
The current implementation is `tinyvcm/` + `demo.py`, trained in
[`notebooks/ME2_From_Scratch_Verified.ipynb`](notebooks/ME2_From_Scratch_Verified.ipynb).
It uses random initialization and no pretrained recognition model, ASR, LLM or cloud inference.

## Open and use it on this PC

From the shared workspace root, use its existing `.venv`:

```powershell
.\Start-VCM-Notebook.ps1
.\Start-PC-Demo.ps1
.\Start-VCM-Recorder.ps1
```

The notebook opens in JupyterLab on port 8890. The assistant dashboard is at
<http://127.0.0.1:7861>; the human recording tool is at <http://127.0.0.1:7862>.
Only start a launcher if its service is not already running. Ctrl+C stops a
foreground service; the dashboard's mute button suspends recognition.

While unmuted, the native microphone continuously processes local audio, seeking
**Hi Dandan** or **Hello Dandan**. Wait for the wake chime/LISTENING indicator,
then say a short command. The command window expires after seven seconds.
A completed command returns to standby. This follows the wake-then-command idea
of a voice assistant, with a small fixed vocabulary. One-breath wake+command and
free-form conversation are not supported. Live audio is not saved or uploaded.

## Actual results, 2026-09-26

Authoritative run: `runs/20260926-111536`, 100 CUDA epochs from fresh random weights.
The best validation checkpoint was epoch 84. The network is **TinyDSCNN48**,
14,282 parameters, not the published BC-ResNet-1. Its common 16 kHz frontend
uses a 25 ms window, 10 ms hop and 40 mel bands over 1.5 seconds (40 × 151).

| Measurement | Observed result | Scope |
|---|---:|---|
| Serialized INT8 ONNX | 44,640 bytes | Nine convolutions and one dense layer have INT8 weights |
| Validation accuracy | 90.80% | Held-out speech-rate groups of training TTS voices |
| FP32 / INT8 test accuracy | 81.08% / 79.28% | 1,221 files including held-out Zira synthetic voice and generated noise |
| Continuous correct wake+command sequences | 77/216 (35.65%) | Exploratory synthetic file replay, not human microphone trials |
| Wakes in that replay | 194/216 (89.81%) | Does not measure false wakes/hour |
| Neural inference p50 / p95 | 0.501 / 0.679 ms | Windows PC, one CPU thread; not Raspberry Pi |
| Frontend + neural inference p50 / p95 | 1.150 / 1.826 ms | Windows PC; excludes audio context and endpoint waiting |
| Separate inference-process RSS | 73.45 MB | Windows PC; training-process RAM is much larger |
| Accuracy with synthetic noise at 20 / 10 / 0 dB SNR | 73.76 / 59.49 / 18.40% | Held-out synthetic speech mixed with generated noise |

**Continuous recognition is not reliable enough for submission.** The old
4,383-file collection was generated using Windows SAPI voices David/Hazel/Zira;
these are not three people. The current audit selects 3,735 attributable files
(1,688 train / 826 validation / 1,221 test), keeps source variants together, and
rejects duplicate hashes across splits. It does not establish human accuracy.
The synthetic baseline has 26 classes; human training adds an unrelated-speech
class. Three real speakers, a real Pi and physical actuators are still required.

## What to do after recording

1. Mute the assistant. Select an anonymous ID such as `person-01`, the actual
   condition and the correct class. Confirm the speaker's consent.
2. Record the suggested phrase naturally, stop and listen back. Click **Save this
   take** and wait for the saved confirmation. The saved audio can be played back.
3. Repeat, aiming for ten takes per class per condition. Change Class and record
   the next phrase. Alternate both wake variants for the wake class. Record
   unrelated speech, silence and background noise under their respective labels.
4. Cover at least two conditions for each person, preferably all three. Repeat
   with at least two other actual people using their own stable IDs. Five people
   are preferable because three-way speaker holdout leaves only one training
   person when there are exactly three.
5. Files save automatically under `data/human/`, with `manifest.csv`. Saving does
   not update the model. In the notebook set `DATA_MODE = "human"`, then
   **Kernel → Restart Kernel and Run All Cells**. The audit runs before training
   and explains missing speakers, conditions or classes. Keep the current
   synthetic notebook as the baseline evidence when starting a human run.
6. Validate on held-out people and continuous negatives. Tune using validation
   data only. Re-export, run the standalone benchmark and rebuild the release
   before deploying the new model. See [benchmark protocol](docs/BENCHMARK.md).

The 1.5-second limit is provisional. If natural phrases do not fit, increase the
shared window and retrain; do not rush speakers or cut off words. The current
collector rejects overlong speech and audible clipping. Synthetic speech must
never be presented as human collection. The held-out speaker is the last sorted
ID, validation the penultimate, and the other IDs form training.

## Ten command categories retained

| Category | Supported fixed vocabulary | Current action |
|---|---|---|
| Music | Play music | Local demonstration melody |
| Questions | What time is it; what is the weather | Actual local clock; weather unavailable offline |
| Lights | Turn on/off the lights | Virtual lights; optional wired RGB LED on Pi |
| Dimming | 25/50/75/100 percent | Virtual brightness; optional PWM LED |
| Timer | One/five/ten minutes | Real elapsed-time timer and chime |
| Alarm | Seven AM | Actual next local 7 AM and chime |
| Thermostat | Cooler/warmer/72 degrees | Simulated setpoint; optional DS18B20 readout |
| Media | Pause/resume/next; volume up/down | Controls the local demonstration melody |
| Reminders | Check my reminders | In-memory demonstration list |
| Calls | Call mom | Labeled simulation; no phone call is placed |

These are fixed intents, not arbitrary slot values or a general question-answering
system. Timers and alarms do not survive a restart. Runtime audio uses the native
mic sample rate and resamples to 16 kHz; the PC Realtek WDM-KS 48 kHz input was
successfully captured with zero dropped chunks in a short check.

## Raspberry Pi package

Use [`TinyVCM_RaspberryPi5.zip`](TinyVCM_RaspberryPi5.zip), which contains `vcm/`,
or copy `release/`. Read [installation and wiring](deployment/README.md).
This is an **application bundle, not a bootable SD image**. Flash Raspberry Pi OS
64-bit, boot it, copy the bundle and run its installer. One-time package downloads
are needed; inference then runs offline. The Windows environment cannot be copied
to ARM Linux. No Raspberry Pi was available for installation, latency, memory or
physical GPIO validation. The <10 ms Pi inference goal remains unverified.

## Organization and provenance

- `notebooks/ME2_From_Scratch_Verified.ipynb`: current executed six-code-cell report,
  epoch output, feature plot, curves and confusion matrix.
- `tinyvcm/`: shared audited data pipeline, PyTorch training, frontend, ONNX runtime,
  microphone worker, wake state and device implementation.
- `demo.py`, `record_dataset.py`: current local assistant and data collector.
- `runs/`: measured metrics, manifests, weights and plots; `latest.json` identifies
  the current model. `scripts/`: notebook builder, launcher, replay and packaging.
- `deployment/`: canonical installer/runtime requirements/wiring; `release/` and
  ZIP are generated deliverables. Checkpoints/datasets/releases stay out of Git.
- `src/`, `simulation/`, `rpi_deployment/`, old apps/notebook/spreadsheet/transcript:
  preserved historical work. Their earlier accuracy, real-speaker, fully-INT8,
  Pi performance and completion claims are superseded by this audit.
- `docs/legacy/`: original README/handoff snapshots, explicitly historical.
- Class lessons already exist at `../../AI 222/Deep-Learning-Experiments` and were
  fast-forward updated to `ec3c5be`. No lesson pretrained weights are used here.

Verification includes 12 focused pipeline/runtime tests, saved notebook execution
with no errors, actual integer ONNX operators on the PC, release hashes, Bash
syntax and native microphone capture. Software tests do not certify speech quality
or physical hardware. AI assistance: Codex implemented and executed the current
pipeline; student understanding and final review remain necessary.
