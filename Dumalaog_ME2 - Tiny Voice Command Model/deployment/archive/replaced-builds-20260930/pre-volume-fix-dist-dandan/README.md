# ME2 - VCM on Raspberry Pi 5

This is the manually launched Pi application package. It contains two locally trained INT8 ONNX models: binary `NON_WAKE`/`WAKE_WORD` and a 31-class intent classifier. The dataset is the **ME2 Spoken Command Dataset**. This is an application package, not an SD-card image. The package uses BCM GPIO 17/27/22 for the red/green/blue LED channels and includes the wiring diagram in `docs/RASPBERRY_PI_5_LED_WIRING.md`.

## Manual installation and launch

Extract the release into `~/dandan`, then run:

```bash
cd ~/dandan
~/.venvs/tinyvcm-rpi5/bin/python verify_personalized_rpi.py
./launch-vcm.sh
```

If the existing Pi runtime is missing, install a Python virtual environment named `~/.venvs/tinyvcm-rpi5`, install PortAudio development/runtime support (`libportaudio2`, `portaudio19-dev`), and install `requirements-runtime.txt` into it. The launcher deliberately does not install or enable a boot service; stop it with Ctrl+C.

The app binds to `127.0.0.1:7860`. On the Pi, open `http://127.0.0.1:7860/assistant` or `/studio`. From a PC on the same network, use an SSH tunnel, for example `ssh -N -L 7860:127.0.0.1:7860 dalmacio@<pi-ip>`, then open `http://127.0.0.1:7860/assistant`. Do not treat this as a public internet address.

The microphone defaults to automatic discovery. The Studio/Assistant input dropdown switches the Pi capture device live; the listener retries after a disconnect without restarting the models. The speaker dropdown routes browser music/page audio where supported. Playback volume controls the browser player and attempts to set the Pi system mixer. Voice feedback uses one cached browser voice for consistency. **Restart VCM** is available separately if needed.

## Behavior and evidence

The fixed demo wake threshold is 0.95; intent confidence uses its separate runtime decision path. A wake opens a 10-second command window that expires even during continued background audio. The Pi launcher enables the low-voltage RGB LED channels; the Assistant and Studio pages display the same light state. The screen is a simulation when GPIO is disabled. Buzzer output is disabled. Thermostat, calls and messages are UI/code simulations; music uses bundled local melodies. Weather is an online Open-Meteo request and needs internet, but no API key. Classification is fully local and uses no cloud model or ASR.

The model report and per-class results are in `candidate_metadata.json`. The reference command split has been reused in earlier iterations, and personal recordings are uneven across speakers and acoustic conditions. The small fixed-threshold replay is not a continuous-room false-activation-per-hour test. Check those limitations alongside the exact metrics before interpreting performance.

To test Pi audio capture and paired CPU inference:

```bash
~/.venvs/tinyvcm-rpi5/bin/python verify_personalized_rpi.py --audio
```

Then speak the wake phrase and a range of commands into the actual Pi microphone. Record false wakes per hour and end-to-end latency before claiming the hardware requirement is met. For the LED wiring and pin map, read `docs/RASPBERRY_PI_5_LED_WIRING.md`. Use a separate 330 Ω series resistor on every LED channel; never connect an LED directly to a GPIO pin. The Assistant's LED buttons are local demo controls and bypass speech classification; voice-classified light commands use the same device driver.
