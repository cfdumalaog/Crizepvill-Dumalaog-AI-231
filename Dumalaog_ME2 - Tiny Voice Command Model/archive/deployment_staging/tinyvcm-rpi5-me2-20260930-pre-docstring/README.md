# ME2 - VCM on Raspberry Pi 5

This is the single current Pi application package. It contains two scratch-trained INT8 ONNX models: binary `NON_WAKE`/`WAKE_WORD` and 31-class intent. The models are 76,740 bytes combined. The dataset is the **ME2 Spoken Command Dataset**. This is an application ZIP, not an SD-card image. No Pi copy or live Pi test of this version has happened yet.

## Manual installation and launch

Copy the ZIP to the Pi, extract it in your home folder, then run:

```bash
cd ~/tinyvcm-rpi5-me2-20260930
~/.venvs/tinyvcm-rpi5/bin/python verify_personalized_rpi.py
./start-vcm.sh
```

If the existing Pi runtime is missing, install a Python virtual environment named `~/.venvs/tinyvcm-rpi5`, install PortAudio development/runtime support (`libportaudio2`, `portaudio19-dev`), and install `requirements-runtime.txt` into it. The launcher deliberately does not install or enable a boot service; stop it with Ctrl+C.

The app binds to `127.0.0.1:7860`. On the Pi, open `http://127.0.0.1:7860/studio`. From a PC on the same network, use an SSH tunnel, for example `ssh -L 7860:127.0.0.1:7860 dalmacio@<pi-ip>`, then open that URL on the PC. Do not treat this as a public internet address.

The microphone defaults to automatic discovery. The Studio/Assistant input dropdown switches the Pi capture device live; the listener retries after a disconnect without restarting the models. The speaker dropdown routes browser music/page audio where supported. Browser text-to-speech follows the browser's default system output. **Restart VCM** is available separately if needed.

## Behavior and evidence

The fixed demo wake threshold is 0.95; intent confidence uses its separate runtime decision path. A wake opens a 10-second inactivity window for commands. The RGB light, numeric volume, thermostat, calls and messages are UI/code simulations; music uses bundled local melodies. Weather is an online Open-Meteo request and needs internet, but no API key. Classification is fully local and uses no cloud model or ASR.

On a reused 1,798-clip reference test partition, the intent model scored 97.11% accuracy and 97.11% macro F1. Six classes remain below 95% F1 or recall; the weakest F1 is 94.12%. At wake threshold 0.95, a five-placement saved-clip replay hit 8/8 held-out personal wake takes, with 1/57 personal non-wake and 1/1,798 reference-command false accepts. These are small offline checks, not continuous-room or Pi-microphone measurements. `candidate_metadata.json` contains the full per-class report, hashes, and test limitations.

To test Pi audio capture and paired CPU inference:

```bash
~/.venvs/tinyvcm-rpi5/bin/python verify_personalized_rpi.py --audio
```

Then speak the wake phrase and a range of commands into the actual Pi microphone. Record false wakes per hour and end-to-end latency before claiming the hardware requirement is met. GPIO actuation is not enabled in this release.
