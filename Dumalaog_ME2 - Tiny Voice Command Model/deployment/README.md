# Tiny VCM on Raspberry Pi 5 (4 GB)

This is an **application bundle, not a bootable disk image**. Training was performed
from random weights on the Windows PC. The synthetic baseline does not yet satisfy
the three-human-speaker requirement. Human recognition, false alarms per hour,
ARM latency and physical wiring remain to be measured.

## Install once

1. Use [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to flash
   Raspberry Pi OS **64-bit**. Set your own username, Wi-Fi and SSH in Imager.
   Do not assume a default `pi` account. Flashing erases the selected SD card.
2. Boot the Pi with a suitable power supply and cooling. Attach a USB microphone.
   Pi 5 has no built-in analogue 3.5 mm audio jack; use USB or HDMI sound.
3. Transfer this entire folder to `~/vcm` on the Pi. From your PC, for example:
   `scp -r release YOUR_USER@raspberrypi.local:~/vcm`
4. On the Pi: `cd ~/vcm && bash setup_rpi.sh`.
   Internet is needed to install packages. Subsequent inference needs no network.
   The script uses one shared Pi environment at `~/.venv`; the Windows `.venv`
   cannot be copied to ARM Linux.
5. Test: `~/.venv/bin/python demo.py --check-seconds 10`.
   List audio devices with `~/.venv/bin/python -m sounddevice`.
   Select a microphone with `--device N` if needed.
6. Run `~/.venv/bin/python demo.py`, open `http://127.0.0.1:7861` on the Pi,
   say **Hi Dandan**, pause for the chime, then say **lights on**.
7. After testing: `sudo systemctl enable --now tiny-vcm` for boot startup.
   Stop: `sudo systemctl stop tiny-vcm`; logs: `journalctl -u tiny-vcm -f`.
   Change startup options in `vcm.env`, then restart the service.

For a PC dashboard connected through SSH, forward the private local port:
`ssh -L 7861:127.0.0.1:7861 YOUR_USER@raspberrypi.local`.
The dashboard deliberately binds to loopback and is not exposed to the network.

## Low-voltage hardware demonstration

| Part | BCM GPIO / physical pin | Connection |
|---|---|---|
| RGB LED red | GPIO17 / pin11 | Series 330 ohm resistor to red anode |
| RGB LED green | GPIO27 / pin13 | Series 330 ohm resistor to green anode |
| RGB LED blue | GPIO22 / pin15 | Series 330 ohm resistor to blue anode |
| Common cathode | GND / pin6 | LED common cathode to ground |
| Buzzer control | GPIO23 / pin16 | Use a transistor driver for powered buzzer; common ground |
| Optional DS18B20 | GPIO4 / pin7 | Data with 4.7k pull-up to 3.3V; enable 1-Wire in raspi-config |

Run with `--gpio` after wiring. The backend explicitly uses gpiozero's lgpio
driver for Pi 5. Never wire mains appliances directly to GPIO. LEDs demonstrate
light control; thermostat and phone intents remain labeled simulations. The
optional DS18B20 reading appears in `/api/state`; no sensor reading is invented.

Timers use minutes × 60 seconds. The 7 AM alarm uses the Pi's local clock.
Timers and alarms are in memory and do not survive a process restart.
Music is a locally generated demonstration tune. Wake activation reduces its
actual playback gain; speech playback/echo cancellation is not implemented.

## Acceptance measurements (run on the real Pi)

`~/.venv/bin/python -m tinyvcm.runtime --output pi_benchmark.json`

Record model p50/p95/p99 and frontend-plus-model latency, RSS, exact board/OS,
microphone, distance and noise condition. The <10 ms target refers to neural
inference, not the 1.5-second context window plus confirmation/response time.
The desktop measurements shipped with this bundle are **not Pi results**.
For long negative tests, record duration, false wakes and the actual negative
speech corpus. Short silence tests cannot establish zero false alarms per hour.

The compact ONNX contains INT8 convolution/dense weights and calibrated activation
quantizers; frontend and softmax are floating point. Full Python/runtime RAM is
larger than the model file and must be measured separately.
