# Current Antigrav Raspberry Pi 5 bundle

This is the deployment path for the current TinyDSCNN-48 Antigrav model. The
older `deployment/setup_rpi.sh`, `release/`, and root-level
`TinyVCM_RaspberryPi5.zip` belong to the legacy synthetic 26-label runtime.
The release builder creates a separate, hash-verified package so those files
remain intact.

The package contains both current INT8 classifiers. The 31-label Option B
classifier was initialized and trained from scratch for the assignment. The
32-label model adds `WAKE_WORD` and was warm-started from the 31-label model;
it is an experimental single-speaker wake-word extension, not a scratch-trained
or multi-speaker-validated wake-word model. The assistant defaults to this
32-label model so wake gating is available, and `--model models/antigrav_optionb_int8.onnx`
selects the assignment classifier explicitly.

Inference and audio processing run locally. No ASR, LLM, cloud model, or network
connection is used for wake detection or command classification. Weather is a
local demo acknowledgement. Voice media commands operate on the bundled curated
playlist; searching YouTube is an optional UI action that needs internet.
The dashboard binds to loopback by default because its control endpoints have
no authentication. Use it directly on the Pi or through an SSH tunnel; expose it
to a trusted LAN only after considering that limitation.

## Build on the Windows training PC

From the shared workspace root, build with the canonical environment:

```powershell
& '.\.venv\Scripts\python.exe' `
  'AI 231\Dumalaog_ME2 - Tiny Voice Command Model\deployment\package_antigrav_rpi.py'
```

The builder refuses to overwrite an existing output. It writes an ignored
`deployment/dist/rpi5-antigrav-vcm-release/` folder and
`deployment/dist/TinyVCM_RPi5_Antigrav_RELEASE.zip`, checks each
payload SHA-256, and runs CPU ONNX/frontend smoke checks for both output shapes.
The ZIP is an application bundle, not an SD-card image.

## Install on Raspberry Pi OS 64-bit

Copy the ZIP to the Pi, then run:

```bash
unzip TinyVCM_RPi5_Antigrav_RELEASE.zip
cd ~/tinyvcm-rpi5
bash setup_rpi.sh
```

The setup script installs audio/ONNX runtime libraries in a Pi-local virtual
environment, verifies both model hashes and executes both ONNX models before
installing a loopback-only systemd unit. It does not copy the Windows `.venv`,
enable the service, or enable GPIO automatically.

Check a USB microphone before starting the assistant:

```bash
~/.venvs/tinyvcm-rpi5/bin/python -m tinyvcm.microphone
```

Start interactively first:

```bash
~/.venvs/tinyvcm-rpi5/bin/python antigrav_demo.py --host 127.0.0.1 --port 7860
```

The program continuously captures short audio blocks locally and checks the
rolling 2.5-second window. In standby it accepts only the `WAKE_WORD` output;
after wake it accepts commands for ten seconds. To check the web dashboard
from the PC without exposing it on the LAN, use an SSH tunnel:

```powershell
ssh -L 7860:127.0.0.1:7860 dalmacio@<pi-host>
```

Then open `http://127.0.0.1:7860/assistant` on the PC. Press Ctrl+C to stop the
foreground assistant. Once microphone, wake, and command behavior pass, start
on boot with `sudo systemctl enable --now tiny-vcm-antigrav.service`.

GPIO is opt-in (`--gpio`) and must only be enabled after connecting an
appropriate low-voltage LED/buzzer circuit and checking its pins. The default
demo changes simulated device state. Do not connect mains voltage to Pi GPIO.

## Verification limits

`python verify_release.py` proves file integrity, exact 31/32 label maps,
frontend shape/finiteness, and CPU ONNX execution; it prints measured model-only
latency for that machine. `python verify_release.py --audio` also checks the
default microphone. These checks do not establish wake-word false-accept rate,
multi-speaker robustness, GPIO wiring safety, or command accuracy on the Pi.
The wake extension needs more speakers and held-out wake/background recordings
before it can be described as robust or Alexa-equivalent.
