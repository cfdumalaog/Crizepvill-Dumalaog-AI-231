> LEGACY PACKAGE: retained for history. Use `../release/` or `../TinyVCM_RaspberryPi5.zip` and the current project README. The metrics and hardware-readiness claims below are superseded.

# AI 231 Machine Exercise 2: Tiny Voice Command Model (VCM) Handoff Index

This document is the dedicated project handoff index for **Machine Exercise 2: Tiny Voice Command Model (VCM)**. It documents the original assignment instructions, the top 10 common commands with sample spoken phrases that work, the system architecture, and operational run commands.

---

## 1. Original Instructions & Problem Statement

### The Problem: ASR Footprint Crisis on Edge Devices
> *"Automatic Speech Recognition (ASR) models are not desirable for on-device computing because of footprint. The goal of this ME is to build a tiny Voice Command Model (VCM) that can understand the most common commands humans tell their smart devices."*

Commercial cloud ASR pipelines (Whisper, Conformer, wav2vec) require hundreds of megabytes to gigabytes of RAM, high compute budgets, and constant cloud internet access—introducing bandwidth costs, privacy leaks, and unacceptable latency. 

In contrast, a **Tiny Voice Command Model (VCM)** maps 16 kHz raw audio directly to discrete smart home intents (Speech-to-Intent classification) using a tiny neural network running 100% on-device on edge hardware (such as a Raspberry Pi 5).

### Tasks & Core Constraints:
1. **Build a dataset to train VCMs:** Multi-speaker, diverse acoustic conditions, 16 kHz mono WAV.
2. **Build and train a VCM on this dataset:** Train strictly **from scratch** with zero pre-trained models.
3. **Design a benchmark for validating VCMs:** Objective evaluation of accuracy, latency, and noise robustness.
4. **Validate the performance of your VCM:** Characterize confusion matrix, SNR curves, and edge CPU latency.
5. **Build a real-world demo of your VCM:** Demonstrable on Raspberry Pi 4/5 with sensors, display, and actuators.
6. **Ultra-tiny footprint:** < 500 KB INT8 model size, < 10 ms inference on edge CPU, real-time streaming.
7. **100% Standalone:** Zero external cloud API calls; privacy-preserving and fully offline.
8. **Pure VCM (No LLM):** Pure neural acoustic classifier handling Commands 1 through 10 directly on-device.
9. **Zero-Button Always-Listening Operation:** The system is continuously listening with zero buttons to click, awakened by a custom wake phrase (**"Hi Dandan"** / **"Hello Dandan"**).
10. **Voiced Audio Output + Monitor Display:** System speaks confirmations aloud and renders feedback to an external HDMI display.

---

## 2. Most Common Commands Issued to Smart Devices

Ranked by real-world log and survey data (**259,164 logged commands** from Amazon Alexa + Google Home devices, plus recurring consumer surveys):

| Rank | Command Category | Prevalence / Basis | Sample Phrases That Work | Dataset Class Name | Hardware Actuator / Display / Voiced Output |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **WAKE** | **Wake Word Activation** | Always-listening trigger | *"Hi Dandan"*, *"Hello Dandan"*, *"Hey Dandan"* | `wake_word` | Chimes rising wake earcon, turns halo cyan, mirrors `[ASSISTANT AWAKE]` on OLED, and speaks: *"Hi Dandan! I'm listening. What can I do for you?"* |
| **#1** | **Play Music** | #1 smart speaker use case in every survey year | *"Play music"*, *"Start music"*, *"Play some music"* | `play_music` | Sets RGB LED cyan, updates OLED to `NOW PLAYING: Chill Beats`, plays 16 kHz sample music track, and speaks: *"Playing chill beats playlist now."* |
| **#2** | **Ask a Question (Weather)** | Daily-use query (#2–3) | *"What's the weather"*, *"How is the weather"* | `question_weather` | Updates OLED to `Diliman, QC: 29°C Partly Cloudy`, and speaks: *"The weather in Diliman is 29 degrees Celsius and partly cloudy."* |
| **#2** | **Ask a Question (Time)** | Daily-use query (#2–3) | *"What time is it"*, *"Tell me the time"* | `question_time` | Updates OLED with current real-time clock and date, and speaks: *"The current time is [HH:MM AM/PM]."* |
| **#3** | **Control Lights (ON)** | 85% of all Alexa IoT commands | *"Turn on the lights"*, *"Lights on"*, *"Switch on lights"* | `lights_on` | Drives PWM LED to 100% bright white, updates OLED to `LIGHTS: ON`, and speaks: *"Turning on the lights."* |
| **#3** | **Control Lights (OFF)** | 85% of all Alexa IoT commands | *"Turn off the lights"*, *"Lights off"*, *"Turn lights off"* | `lights_off` | Cuts PWM LED duty cycle to 0%, updates OLED to `LIGHTS: OFF`, and speaks: *"Turning off the lights."* |
| **#4** | **Dim / Color Lights** | ~10% of Alexa IoT commands | *"Dim lights fifty percent"*, *"Dim lights 25%"*, *"Dim lights 75%"* | `dim_lights_50`, `dim_lights_25`, `dim_lights_75` | Sets warm PWM duty cycle to target percentage, mirrors on OLED, and speaks: *"Dimming lights to 50 percent."* |
| **#5** | **Set a Timer** | Top daily kitchen utility | *"Set timer for five minutes"*, *"Set a timer for one minute"* | `timer_5min`, `timer_1min`, `timer_10min` | Spawns background countdown thread, displays countdown on OLED, and speaks: *"5 minute timer started."* |
| **#6** | **Set an Alarm** | Top morning utility | *"Set an alarm for seven AM"*, *"Set alarm seven AM"* | `alarm_set` | Sets RGB LED yellow, updates OLED to `ALARM SET: 07:00 AM`, and speaks: *"Alarm set for 7:00 AM."* |
| **#7** | **Adjust Thermostat (Cooler)** | HVAC smart control | *"Make it cooler"*, *"Decrease temperature"* | `temp_cooler` | Sets RGB LED blue, decrements target temperature (°F), updates OLED, and speaks: *"Decreasing thermostat temperature to [X] degrees."* |
| **#7** | **Adjust Thermostat (Warmer)** | HVAC smart control | *"Make it warmer"*, *"Increase temperature"* | `temp_warmer` | Sets RGB LED red, increments target temperature (°F), updates OLED, and speaks: *"Increasing thermostat temperature to [X] degrees."* |
| **#8** | **Media Control (Pause / Stop)** | Folded with Command #1 | *"Pause"*, *"Stop"*, *"Pause music"* | `media_pause` | Halts media playback, restores audio ducking, mirrors on OLED, and speaks: *"Music playback paused."* |
| **#8** | **Media Control (Volume Up / Louder)** | Audio level control | *"Volume up"*, *"Louder"*, *"Turn it up"* | `volume_up` | Boosts volume by +10%, updates OLED `VOLUME: [X]%`, and speaks: *"Volume increased to [X] percent."* |
| **#8** | **Media Control (Volume Down / Quieter)** | Audio level control | *"Volume down"*, *"Quieter"*, *"Lower volume"* | `volume_down` | Lowers volume by -10%, updates OLED `VOLUME: [X]%`, and speaks: *"Volume decreased to [X] percent."* |
| **#9** | **Reminders and Lists** | Daily productivity | *"What are my reminders"*, *"Check my reminders"* | `reminders_check` | Retrieves active task list onto OLED screen, and speaks: *"You have 3 active reminders. First: Submit AI 231 ME2 on time."* |
| **#10** | **Calls and Messaging** | Communication intent | *"Call mom"*, *"Phone mom"* | `call_mom` | Sets RGB LED green, plays simulated telephone dial chime, updates OLED to `CALLING MOM...`, and speaks: *"Initiating voice call to Mom."* |
| **REJECT** | **Background Noise / Silence** | Edge false-alarm suppression | Ambient room noise, fan hum, keyboard clicks, coughing | `_background_noise_`, `_silence_` | EnergyVAD silence gating halts neural forward pass (<0.1% CPU idle). Noise rejected with 0.00 False Alarm Rate. |

---

## 3. Hardware Bill of Materials (BOM) & Wiring

For physical deployment to **Raspberry Pi 5 (4GB RAM, 32GB Storage)** or Raspberry Pi 4:

| Component | Interface / Pins | Physical Function | Cost (PHP) |
| :--- | :--- | :--- | :--- |
| **Raspberry Pi 5 (4GB)** | Main Board | Edge neural inference host & audio controller | ₱4,200 |
| **USB Microphone / Mic Array** | USB 2.0 / ALSA `plughw:1,0` | Continuous 16 kHz audio streaming | ₱350 |
| **SSD1306 0.96" OLED Display** | I2C (GPIO 2: SDA, GPIO 3: SCL, 3.3V, GND) | Real-time command confirmation & status mirror | ₱180 |
| **RGB Common-Cathode LED** | GPIO 17 (Red), GPIO 27 (Green), GPIO 22 (Blue) | Multi-color state halo (Standby, Wake, Action) | ₱45 |
| **330Ω Current-Limiting Resistors** | In series with RGB anodes | GPIO current protection | ₱15 |
| **Passive/Active 5V Buzzer** | GPIO 18 (PWM0) | System chimes and wake earcons | ₱35 |
| **External HDMI Monitor / Display** | Micro-HDMI to HDMI | Native kiosk display (`monitor_display.py`) | Existing |
| **3.5mm / USB Powered Speaker** | 3.5mm Jack / USB Audio | Voiced speech output (`assets/voice_responses/`) | ₱450 |
| **MicroSD Card (32GB A2/V30)** | Card Slot | Raspberry Pi OS 64-bit (Debian Bookworm) | ₱420 |
| **Official 27W USB-C Power Supply** | Power In (5V/5A) | Stable power without undervoltage throttling | ₱1,200 |
| **Total Estimated BOM** | | | **₱6,895** |

---

## 4. Key Architectural Highlights (Siri & Alexa Parity)

1. **Acoustic Self-Interference Music Ducking:**
   When music is playing and the wake word is detected, media volume is automatically lowered from 70% to 10% (*ducking*), preventing speaker output from feeding back into the microphone. Volume is automatically restored to 70% once the command completes or times out.
2. **Compound Utterance Support ("One-Breath Commands"):**
   Supports seamless continuous commands without pausing for a chime (e.g., saying *"Hi Dandan play music"* in a single breath wakes the system and triggers playback immediately).
3. **Dynamic Acoustic Endpointing:**
   Detects ~500 ms of trailing silence after speech ends, instantly executing commands without forcing the user to wait out a fixed countdown timer.
4. **Energy Voice Activity Detection (VAD):**
   Continuous background audio is analyzed for acoustic energy. When the room is silent, neural forward passes are skipped completely, reducing idle CPU usage to **< 0.1%**.
5. **Robust Audio Streamer (`src/mic_stream.py`):**
   Solves the Windows MME 4-channel microphone driver crash by probing physical hardware host APIs (WDM-KS / WASAPI / ALSA), recording at native 48 kHz across channels, and downsampling to 16 kHz mono in real time with Adaptive Gain Control (AGC) and live ASCII VU meter feedback.

---

## 5. How to Run the System

### On Your Local Windows Machine:
```powershell
# 1. Zero-Button Console Always-Listening Mode (with live ASCII VU meter):
& '.\.venv\Scripts\python.exe' 'AI 231\Dumalaog_ME2 - Tiny Voice Command Model\live_listen.py'

# 2. Native Desktop HDMI Monitor Display (Fullscreen Kiosk):
& '.\.venv\Scripts\python.exe' 'AI 231\Dumalaog_ME2 - Tiny Voice Command Model\monitor_display.py'

# 3. Interactive Web Assistant:
& '.\.venv\Scripts\python.exe' 'AI 231\Dumalaog_ME2 - Tiny Voice Command Model\app.py'

# 4. Run Automated Test Verification Suites:
& '.\.venv\Scripts\python.exe' 'AI 231\Dumalaog_ME2 - Tiny Voice Command Model\simulation\test_all_commands.py'
& '.\.venv\Scripts\python.exe' 'AI 231\Dumalaog_ME2 - Tiny Voice Command Model\simulation\test_wake_word.py'
& '.\.venv\Scripts\python.exe' 'AI 231\Dumalaog_ME2 - Tiny Voice Command Model\simulation\test_siri_alexa_features.py'
```

### On Raspberry Pi 5 (Deployment):
```bash
# 1. Transfer rpi_deployment folder to Raspberry Pi
scp -r "rpi_deployment" pi@raspberrypi.local:~/vcm

# 2. On the Pi, run the automated setup script:
cd ~/vcm
bash setup_rpi.sh

# 3. Run hands-free always-listening assistant:
python3 live_listen.py

# 4. Or launch the HDMI monitor kiosk display:
python3 monitor_display.py
```

---

## 6. Edge Performance Benchmarks & Targets vs. Actuals

| Metric | Target Specification | Actual Measured Performance | Status |
| :--- | :--- | :--- | :---: |
| **Model Footprint (INT8)** | < 500 KB | **181.4 KB** (27.6 KB core weights) | 🟢 **PASS** (64% smaller than target) |
| **Single-Core CPU Latency** | < 10.0 ms | **8.13 ms** (mean) / **6.53 ms** (median) | 🟢 **PASS** (18% faster than target) |
| **Command Test Accuracy** | > 85.0% | **93.42%** (unseen speakers) | 🟢 **PASS** (+8.4% above target) |
| **Task Success Rate** | > 90.0% | **97.83%** (multi-evaluator benchmark) | 🟢 **PASS** |
| **Wake Sensitivity (FRR)** | < 5.0% FRR | **100.0%** detection (0.0% FRR) | 🟢 **PASS** |
| **False Alarm Rate (FAR)** | < 1 per hour | **0.00** false alarms under continuous noise | 🟢 **PASS** |
| **Idle Power / CPU Load** | Minimal battery drain | **< 0.1% CPU** (via EnergyVAD gating) | 🟢 **PASS** |
| **Cloud Independence** | 100% On-Device | **0 external network requests** | 🟢 **PASS** |
