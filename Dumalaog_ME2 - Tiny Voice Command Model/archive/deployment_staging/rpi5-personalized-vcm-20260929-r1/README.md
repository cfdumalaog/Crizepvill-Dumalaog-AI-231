# ME2 - VCM on Raspberry Pi 5 — personalized two-model release

This versioned application bundle contains the latest separate, scratch-trained
binary wake detector and personalized 31-class INT8 intent model. It is an
application bundle, not an SD-card image. The models occupy 76,740 bytes in
total. Model metadata and source metrics are in `candidate_metadata.json`.

## Manual launch

The VCM is intended to be started manually and does not install an enabled
systemd service. On the Pi, open a terminal and run:

```bash
cd ~/tinyvcm-rpi5-personalized-20260929-r1
./start-vcm.sh
```

Press **Ctrl+C** to stop it. The UI binds only to `127.0.0.1:7860`. From a PC,
open an SSH tunnel with `ssh -L 7860:127.0.0.1:7860 dalmacio@192.168.254.106`,
then browse to `http://127.0.0.1:7860/studio` or `/assistant`.

The microphone argument is not hard-coded. Automatic selection tries any
available Pi capture input. The UI's input dropdown lists devices attached to
the Pi; select one and press **Restart VCM** to apply it. If no capture device is
connected, the UI and model still start, the page reports that no microphone is
available, and the listener retries device discovery every two seconds. After
connecting a microphone, choose **Refresh audio devices** and select it.

The speaker dropdown controls browser music and page audio where the browser
supports output routing. It lists outputs on the computer running the browser;
when using the Pi's own browser, those are the Pi's browser outputs. Browser
speech synthesis uses that browser's default output.

## Models and operating settings

- Wake classifier: `NON_WAKE` / `WAKE_WORD`, INT8, personal-hard-negative run.
- Intent classifier: 31 Option B classes, INT8.
- Wake operating threshold: `0.90`, the previously requested setting. The
  validation-selected threshold remains `0.9985431433` in the candidate report.
- VAD RMS gate: `0.006`; inference interval: `0.12` seconds; post-wake timeout:
  `10` seconds.
- GPIO is disabled. Lights remain software-simulated.
- No cloud model, ASR, or network service is used for classification.

At threshold 0.90, saved-clip replay detected 8/8 held-out personal wake clips,
accepted 1/57 personal non-wake clips, and accepted 0/1,798 Option B command
clips. These small offline tests do not establish live-microphone accuracy or
continuous-room false wakes per hour. The user should test the wake and commands
with the actual Pi microphone.

## Verification

Use the existing Pi runtime at `~/.venvs/tinyvcm-rpi5`. Check bundle hashes,
labels, frontend shape, CPU ONNX inference, and p50/p95 latency for both models:

```bash
~/.venvs/tinyvcm-rpi5/bin/python verify_personalized_rpi.py
```

To additionally test the currently attached microphone:

```bash
~/.venvs/tinyvcm-rpi5/bin/python verify_personalized_rpi.py --audio
```

This release intentionally leaves startup under the user's control; do not
enable `tiny-vcm-antigrav.service` for this manual-launch setup.
