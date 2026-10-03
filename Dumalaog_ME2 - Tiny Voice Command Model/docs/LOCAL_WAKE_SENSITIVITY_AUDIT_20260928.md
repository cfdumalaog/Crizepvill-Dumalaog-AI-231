# Local binary wake sensitivity audit — 2026-09-28

## Decision

Keep the current scratch-trained, two-class INT8 ONNX weights for the local
microphone check. The saved wake recordings already score as wake; the current
likely failure point is the live audio path (input level/RMS gate), and the
validation threshold leaves little score margin. The model weights and the
validation-selected threshold in the run metadata have not been changed.

## Replay results

The selected artifact is
`runs/vcm-binary-wake-20260928-103823/models/binary_wake_int8.onnx`,
37,425 bytes, 13,106 parameters. Replaying each saved human recording at five
rolling-buffer alignments gave 16/16 hits at the existing 0.9751819 operating
threshold. The dataset has 16 unique takes from one speaker.

For a local sensitivity check, threshold 0.90 retained 3/3 validation wake
clips and yielded 0/1,818 false accepts on the command validation windows. It
also gave 16/16 hits on the saved wake clips, 0/1,798 on the previously used
Option B command holdout, and 0/90 on the short background excerpts. At 0.60,
the command windows produced 7 false accepts in both the 1,818 validation set
and the 1,798 previously used holdout. The 0.90 value is therefore a local
debug override, not the run's calibrated/release threshold; test/holdout
numbers are exploratory and must not be presented as a fresh test.

The fixed-rate desktop loop then exposed a timing issue that the five static
placements did not capture. Replaying each take as 150 ms microphone chunks
through the assistant's rolling buffer and RMS gate gave 14/16 wake detections
with the original 0.35 s minimum inference interval. Using a 0.12 s interval
(one chance per incoming chunk) gave 16/16 in the same simulation. Both
intervals produced 0/90 background-excerpt activations at threshold 0.90; the
maximum simulated noise score was 0.776740. This supports the local cadence
change, but it is still file replay and not a microphone test.

These are offline file replays, not real-time microphone tests. The background
clips are short and come from the augmentation corpus. They do not establish
false activations per hour, robustness to unseen near-miss speech, or support
for another speaker.

## Runtime diagnostic change

The assistant accepts `--vad-threshold` and `--inference-interval`, reports the
configured values, mic RMS, and inference count in `/api/state`, and displays
mic RMS, VAD gate, inference interval, binary wake probability, and threshold
in the Studio page. This makes the audio gate and model score independently
visible. Defaults remain VAD 0.012 and interval 0.35 s for the existing Pi
release path. The local candidate instance is configured with a 0.006 RMS gate,
0.90 wake threshold, and 0.12 s interval only for sensitivity testing; GPIO
remains off, and no Pi files or release bundle were changed.

Open `http://127.0.0.1:7863/studio` and say the wake phrase at the normal
distance. If the inference count does not rise while speaking, the mic source,
input level, or VAD gate is blocking the audio. If the count rises but the wake
probability remains below 90%, record new wake attempts through the actual
microphone and add those recordings to the next training set. Do not lower the
threshold further until non-wake room audio and near-miss phrases have been
recorded and scored.

## Verification and limitations

- Offline audit: 16/16 saved wake files detected at thresholds 0.975 and 0.90;
  chunk simulation 14/16 at 0.35 s vs 16/16 at 0.12 s; command validation
  false accepts 0/1,818 at 0.90; short background excerpts 0/90.
- Regression suites: 34 passed in 10.48 seconds after the final inference-
  interval change; Python compileall and `git diff --check` passed.
- Notebook audit cell executed and stored in `notebooks/ME2_Wake2_Binary.ipynb`;
  it replays the selected ONNX model and does not train or edit weights.
- Raspberry Pi was not accessed. No live spoken wake attempt occurred in this
  audit. The available 16 takes are from one speaker; obtain takes from at least
  three speakers plus room noise and near-miss phrases before calling the model
  robust or final.

## Live PC microphone follow-up

The initial local server had opened Windows WDM-KS input 36, `Microphone Array 1
(Realtek HD Audio Mic input with SST)`, but its `/api/state` RMS stayed at zero
and the inference count remained zero. A no-save PortAudio probe found the
Windows default Fifine MME input 1 and Fifine WASAPI input 25 could not be opened
from this process; WDM-KS Fifine inputs 40 and 41 did return samples. Input 41
was selected because it had the stronger measured short-window RMS. The Studio
page reads audio from the Python process, so browser microphone permission does
not govern this listener.

`tinyvcm/microphone.py` now prefers Windows' default input during automatic
selection, penalizes WDM-KS endpoints until preferred inputs fail, and accepts
an explicit device index or name fragment. `vcm_app.py` exposes
`--device` / `--list-input-devices`, reports stream state, capture block/chunk
counts and audio age, and displays the selected source. `scripts/Start-Local-VCM.ps1`
selects Fifine by name and starts the current two-model path on port 7863.

Live input works: API snapshots reported input 41, an active stream, recent
chunks, nonzero RMS, zero dropped blocks, and no runtime error. The prior
0.006-gate one-hit setup produced several unconfirmed voice-wake events. The
current launcher trial uses RMS gate 0.02, wake threshold 0.90, and two
consecutive positive windows within 0.45 seconds. A 30-second no-save sample
had three windows above the gate but no model score above 0.90. Saved one-speaker
WAV replay produced two-window wake hits on 15/16 takes at those settings. One
later wake event occurred while the application was open without a controlled
spoken test, so its cause is unknown; the current wake path is still not
validated. A longer room-negative observation, near-miss speech, and the user's
three deliberate wake attempts remain pending.

For local weather, `Devices()` stays offline by default. The PC launcher opts in
to Open-Meteo; the `/api/test_weather` route returned current Diliman values
without an API key. Voice weather runs on a separate thread so an HTTP request
does not block mic processing. Request failures return a truthful unavailable
message. The Pi and model files were not changed.
