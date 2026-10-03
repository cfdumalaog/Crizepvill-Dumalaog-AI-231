# ME2 - VCM on Raspberry Pi 5 — RGB LED wiring

The existing Pi VCM project was inspected read-only as the wiring reference. Its RGB driver maps `RGBLED(17, 27, 22)`. This release uses the same Broadcom (BCM) numbering and does not modify that project.

| LED channel | BCM GPIO used by code | Physical 40-pin header position |
|---|---:|---:|
| Red | GPIO 17 | Pin 11 |
| Green | GPIO 27 | Pin 13 |
| Blue | GPIO 22 | Pin 15 |
| Shared ground | GND | Pin 6 (or another GND pin) |

Wire each color channel from its GPIO through its own 330 Ω series resistor to the LED channel. Connect the common-cathode return (or each separate LED's cathode) to ground. Do not connect an LED directly to a GPIO pin. Use the correct common-cathode RGB LED; a common-anode part has different polarity and should not be connected using this diagram.

```text
Raspberry Pi physical header                 Common-cathode RGB LED

Pin 11 / BCM GPIO 17 ──[330 Ω]────────────── Red
Pin 13 / BCM GPIO 27 ──[330 Ω]────────────── Green
Pin 15 / BCM GPIO 22 ──[330 Ω]────────────── Blue
Pin  6 / GND ────────────────────────────── Common cathode
```

The project code and Pi line inventory confirm the intended GPIO numbers, but software cannot sense whether wires are physically attached or which color lead is connected. Confirm the wiring visually before powering the LED circuit. The VCM starts with brightness at 0%, so its initial light state is off. The buzzer pin used by an older version is not initialized or driven by this release.

The Assistant and Studio show the same live light color and brightness. The Assistant's buttons send one of the fixed light commands to the local VCM: `LIGHT_OFF`, `LIGHT_ON`, `COLOR_RED`, `COLOR_GREEN`, or `COLOR_BLUE`. Buttons provide a direct hardware demo when `--gpio` is enabled, and show the software simulation otherwise. They bypass the wake and intent classifiers; speaking a command still follows the two-model path.
