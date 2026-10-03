# Hardware Wiring & Physical Setup Guide for Raspberry Pi 4 / 5

This guide provides the exact pinouts, wiring schematics, and connection procedures to assemble the **Tiny Voice Command Model (VCM)** physical edge demo on a **Raspberry Pi 4 Model B** or **Raspberry Pi 5**.

---

## 1. Raspberry Pi 40-Pin GPIO Header Reference

```text
                        3.3V Power  [01] [02]  5V Power
     I2C1 SDA (GPIO 02) / Pin 03    [03] [04]  5V Power
     I2C1 SCL (GPIO 03) / Pin 05    [05] [06]  Ground (GND)
             GPIO 04    / Pin 07    [07] [08]  GPIO 14 (UART TX)
             Ground     / Pin 09    [09] [10]  GPIO 15 (UART RX)
    RED LED (GPIO 17)   / Pin 11    [11] [12]  GPIO 18 (PCM CLK)
  GREEN LED (GPIO 27)   / Pin 13    [13] [14]  Ground (GND)
   BLUE LED (GPIO 22)   / Pin 15    [15] [16]  GPIO 23 (BUZZER)
             3.3V Power / Pin 17    [17] [18]  GPIO 24
             GPIO 10    / Pin 19    [19] [20]  Ground (GND)
             GPIO 09    / Pin 21    [21] [22]  GPIO 25
             GPIO 11    / Pin 23    [23] [24]  GPIO 08
             Ground     / Pin 25    [25] [26]  GPIO 07
             GPIO 00    / Pin 27    [28] [28]  GPIO 01
             GPIO 05    / Pin 29    [29] [30]  Ground (GND)
             GPIO 06    / Pin 31    [31] [32]  GPIO 12
             GPIO 13    / Pin 33    [33] [34]  Ground (GND)
             GPIO 19    / Pin 35    [35] [36]  GPIO 16
             GPIO 26    / Pin 37    [37] [38]  GPIO 20
             Ground     / Pin 39    [39] [40]  GPIO 21
```

---

## 2. Complete Component Wiring Table

| Component | Component Pin | Raspberry Pi Pin Name | Physical Header Pin | Notes / In-Line Components |
| :--- | :--- | :--- | :--- | :--- |
| **Common-Cathode RGB LED** | Anode Red (R) | **GPIO 17** | **Pin 11** | In series with **330 Ω resistor** |
| | Anode Green (G) | **GPIO 27** | **Pin 13** | In series with **330 Ω resistor** |
| | Anode Blue (B) | **GPIO 22** | **Pin 15** | In series with **330 Ω resistor** |
| | Common Cathode (-) | **GND** | **Pin 06** or **Pin 14** | Connect directly to Ground rail |
| **Active Piezo Buzzer** | Positive (+) | **GPIO 23** | **Pin 16** | Direct connection |
| | Negative (-) | **GND** | **Pin 20** | Connect directly to Ground rail |
| **0.96" I2C OLED (SSD1306)** | VCC | **3.3V Power** | **Pin 01** | Use 3.3V (do not use 5V) |
| | GND | **GND** | **Pin 09** | Connect directly to Ground rail |
| | SDA | **GPIO 02 (I2C1 SDA)**| **Pin 03** | Hardware I2C Data line |
| | SCL | **GPIO 03 (I2C1 SCL)**| **Pin 05** | Hardware I2C Clock line |
| **USB Microphone** | USB Type-A | Any USB Port | USB Port 1 or 2 | Plug & Play Linux ALSA device |
| **Speaker** | USB / 3.5mm AUX | USB or Audio Jack | USB / 3.5mm | *RPi 5 requires USB speaker/DAC or HDMI monitor audio* |
| **HDMI Monitor** | Micro-HDMI | **HDMI 0** Port | Micro-HDMI Port 0 | Connect to external monitor (video + audio playback) |

---

## 3. MicroSD Card Installation & 1-Step Setup

### Step A: Flash Raspberry Pi OS
1. Download **Raspberry Pi Imager** on your PC.
2. Select OS: **Raspberry Pi OS (64-bit)** (Bookworm or Bullseye).
Store all Pi credentials outside this repository; do not put them in this guide.
4. Flash the MicroSD card and insert it into the Raspberry Pi.

### Step B: Copy Project Files to Raspberry Pi
Copy the `rpi_deployment` folder to the Raspberry Pi home directory (via SCP, USB drive, or Git):
```bash
# Example using SCP from PC:
scp -r "AI 231/Dumalaog_ME2 - Tiny Voice Command Model/rpi_deployment" pi@<RASPI_IP>:/home/pi/
```

### Step C: Run Automated Setup
SSH into your Raspberry Pi and execute:
```bash
cd /home/pi/rpi_deployment
chmod +x setup_rpi.sh
./setup_rpi.sh
```
This script will:
- Enable hardware I2C bus (`/dev/i2c-1`)
- Install all necessary audio drivers and ALSA libraries
- Create a virtual environment `.venv_rpi` and install dependencies
- Register and configure `vcm.service` as a background systemd service

---

## 4. Testing Your Hardware Connections

### 1. Verify I2C OLED Display
Run the I2C detection tool:
```bash
i2cdetect -y 1
```
You should see device address `3c` appear in the grid:
```text
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         -- -- -- -- -- -- -- --
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
30: -- -- -- -- -- -- -- -- -- -- -- -- 3c -- -- --
```

### 2. Verify USB Microphone Input
List connected recording devices:
```bash
arecord -l
```
Test recording 3 seconds of audio:
```bash
arecord -D default -d 3 -f cd -t wav test_mic.wav
```

### 3. Verify Audio Output (Speaker)
List playback devices:
```bash
aplay -l
```
Test playback:
```bash
aplay -D default assets/sample_music.wav
```

---

## 5. Running the Voice Assistant
 
### Option 1: Native HDMI Monitor Screen (Voiced + Kiosk Display)
Run the dedicated monitor display on your external HDMI monitor:
```bash
source .venv_rpi/bin/activate
python monitor_display.py
```
*Press **F11** anytime to toggle full-screen kiosk mode.* Displays glowing animated assistant halo, large across-the-room spoken subtitles, simulated OLED mirror, and hardware telemetry.

### Option 2: Zero-Button Hands-Free Always-Listening (Console)
```bash
source .venv_rpi/bin/activate
python live_listen.py
```
*Continuously samples microphone; triggers on "Hi Dandan" / "Hello Dandan" and speaks back aloud.*

### Option 3: Interactive Web Dashboard
```bash
source .venv_rpi/bin/activate
python app.py
```
*Access from local network browser at `http://<RASPI_IP>:7860`.*

### Option 4: Run Automatically on Boot as an Appliance:
```bash
sudo systemctl enable --now vcm.service
```
To inspect live logs from the background service:
```bash
journalctl -u vcm.service -f
```

---

## 6. Real-World Commands Verification Checklist

Speak naturally ~1 to 2 feet from the microphone:

| Number | Voice Command | Voiced Spoken Output | Observed Hardware Action on Raspberry Pi |
| :---: | :--- | :--- | :--- |
| **Wake**| *"Hi Dandan"* | *"Hi Dandan! I'm listening. What can I do for you?"* | Cyan LED pulses, OLED shows *"ASSISTANT AWAKE"*, double chime |
| **#1** | *"play music"* | *"Playing chill beats playlist now."* | Cyan LED lights up, OLED shows *"NOW PLAYING"*, music plays through speaker |
| **#2** | *"what's the weather"* | *"The weather in Diliman is 29°C and partly cloudy..."* | Chime beeps, OLED displays *"WEATHER REPORT: 29°C"* |
| **#2** | *"what time is it"* | *"The current time is [time]."* | Chime beeps, OLED displays current system time and date |
| **#3** | *"turn on lights"* | *"Turning on the lights."* | RGB LED illuminates bright white (100% duty cycle) |
| **#3** | *"turn off lights"* | *"Turning off the lights."* | RGB LED turns off completely |
| **#4** | *"dim lights to 50 percent"* | *"Dimming lights to fifty percent."* | RGB LED dims to 50% PWM brightness in warm white |
| **#5** | *"set a timer for 5 minutes"* | *"Five minute timer started."* | OLED shows *"TIMER STARTED: 5 min"*; buzzer alarms upon expiration |
| **#6** | *"set an alarm"* | *"Alarm set for seven AM."* | Yellow LED illuminates, OLED displays *"ALARM SET: 07:00 AM"* |
| **#7** | *"make it cooler"* | *"Decreasing thermostat temperature to [X] degrees."* | Blue LED illuminates, thermostat decreases by 1°F on OLED |
| **#7** | *"make it warmer"* | *"Increasing thermostat temperature to [X] degrees."* | Red LED illuminates, thermostat increases by 1°F on OLED |
| **#8** | *"pause"* / *"resume"* | *"Music playback paused."* / *"resumed."* | Pauses/resumes audio playback, OLED updates media state |
| **#8** | *"volume up"* / *"volume down"*| *"Increasing volume to [X] percent."* | Adjusts speaker output volume in 10% steps |
| **#9** | *"what are my reminders"* | *"You have three active reminders. First: Submit AI 231 ME2..."* | Chime beeps, OLED scrolls through active reminder items |
| **#10** | *"call mom"* | *"Initiating voice call to Mom."* | Green LED illuminates, phone dial tone plays, OLED shows *"Calling Mom..."* |

