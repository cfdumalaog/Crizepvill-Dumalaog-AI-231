# ME2 - VCM on Raspberry Pi 5

This is the manually launched Pi application package. It contains binary `NON_WAKE`/`WAKE_WORD` and 31-class intent INT8 ONNX models. The dataset is the **ME2 Spoken Command Dataset**. This is an application package, not an SD-card image. Launch settings, including port, GPIO, wake gate, and intent action gates, are recorded in `deployment.json`; the launcher reads those values instead of assuming a port or enabling GPIO.

The Pi Desktop contains **ME2 - VCM on Raspberry Pi 5**. Double-click it to start the VCM if it is stopped and open the assistant. The Assistant and Studio pages include **Stop VCM**, which stops only this VCM process; it does not stop Kiko or any other service. Launch remains manual, with no boot autostart.

Accepted commands are logged as JSONL in `~/vcm_benchmark/` using the class
benchmark's 19-intent `intent`, optional `slot`, `infer_ms`, and `audio_ms`
fields. The class holdout run is still an in-room microphone/speaker test and
must be run separately with its benchmark script.

## Manual installation and launch

Extract the release into `~/Desktop/dandan`, then run:

```bash
cd ~/Desktop/dandan
~/.venvs/tinyvcm-rpi5/bin/python verify_personalized_rpi.py
./launch-vcm.sh
```

If the existing Pi runtime is missing, install a Python virtual environment named `~/.venvs/tinyvcm-rpi5`, install PortAudio development/runtime support (`libportaudio2`, `portaudio19-dev`), and install `requirements-runtime.txt` into it. The launcher deliberately does not install or enable a boot service; stop it with Ctrl+C.

The app binds to the loopback address and port in `deployment.json`. On the Pi, use that port with `/assistant` or `/studio`. From a PC, create an SSH tunnel with the same port at both ends, then open the matching local `/assistant` URL. Do not treat this as a public internet address.

The microphone defaults to automatic discovery. The Studio/Assistant input dropdown switches the Pi capture device live; the listener retries after a disconnect without restarting the models. The speaker dropdown routes browser music/page audio where supported. Playback volume controls the browser player and attempts to set the Pi system mixer. Voice feedback uses one cached browser voice for consistency. **Restart VCM** is available separately if needed.

## Behavior and evidence

The current wake gate is 0.95; intent confidence and margin are separate action gates. A wake opens a 10-second command window that expires even during continued background audio. When `gpio` is enabled, BCM 17/27/22 drive the RGB LED channels; when disabled, the Assistant and Studio light panel is a simulation. Buzzer output is disabled. Thermostat, calls and messages are UI/code simulations; music uses bundled local melodies. Weather is an online Open-Meteo request and needs internet, but no API key. Classification is fully local and uses no cloud model or ASR.

The model report and per-class results are in `candidate_metadata.json`. The reference command split has been reused in earlier iterations, and personal recordings are uneven across speakers and acoustic conditions. The small fixed-threshold replay is not a continuous-room false-activation-per-hour test. Check those limitations alongside the exact metrics before interpreting performance.

To test Pi audio capture and paired CPU inference:

```bash
~/.venvs/tinyvcm-rpi5/bin/python verify_personalized_rpi.py --audio
```

Then speak the wake phrase and a range of commands into the actual Pi microphone. Record false wakes per hour and end-to-end latency before claiming the hardware requirement is met. For the LED wiring and pin map, read `docs/RASPBERRY_PI_5_LED_WIRING.md`. Use a separate 330 Ω series resistor on every LED channel; never connect an LED directly to a GPIO pin. The Assistant's LED buttons are local demo controls and bypass speech classification; voice-classified light commands use the same device driver.

## Isolated dataset candidate

The evaluation candidate is installed separately at `~/Desktop/me2-dataset-candidate`, binds only to `127.0.0.1:7865`, and has `gpio: false`. It uses the same frozen wake model plus a scratch-trained TinyDSCNN-48 intent model with validation-selected action gates 0.76/0.0. It is not the default release: the paired supported-command test slightly favored the existing model, while the candidate had fewer false actions on the finite OOS test. Do not copy this directory over `~/Desktop/dandan`.
