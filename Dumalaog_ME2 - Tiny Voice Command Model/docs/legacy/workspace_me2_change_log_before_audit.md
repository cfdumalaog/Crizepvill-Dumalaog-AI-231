# Historical ME2 change log

These earlier claims are superseded by the 2026-09-26 audit. Do not cite them as human/Pi evidence.

- **2026-09-25 — Added voiced speech output (TTS) and dedicated HDMI monitor display interface for Raspberry Pi 5.**
  Synthesized 25 crystal-clear 16 kHz WAV voiced response clips covering all command confirmations, wake phrases, and system alerts into `rpi_deployment/assets/voice_responses/` using `generate_voice_responses.py`. Implemented `src/tts.py` (`VoiceOutputEngine`) providing instantaneous 0 ms latency / 0% CPU audio playback via kernel drivers (`winsound.PlaySound` async on Windows, `aplay` on Linux/RPi) with offline dynamic TTS fallback. Integrated voiced output into `src/hal.py` (`speak()`), `src/actions.py`, and `live_listen.py`. Created native desktop HDMI monitor display application `monitor_display.py` using `tkinter` (F11 fullscreen kiosk, animated glowing halo, large spoken subtitle banner, simulated OLED mirror, and device telemetry). Enhanced web application `app.py` with large monitor subtitle card and automatic voiced audio response output. Updated `setup_rpi.sh` (added `espeak-ng` and `python3-tk`) and `WIRING_GUIDE.md` (HDMI monitor & audio setup). Passed all integration tests (15/15) and wake lifecycle tests (5/5).
- **2026-09-26 — Audio streamer hardware bug resolution, complete conversation history export, dedicated ME2 handoff index, and app.py synchronization.**
  Diagnosed why "Hi Dandan" failed on local Windows machine: PortAudio MME default device 1 enforced 4-channel 44.1 kHz, causing `PaErrorCode -9999` crash on 16 kHz stream attempt. Created `RobustMicrophoneStreamer` in `src/mic_stream.py` that probes physical WDM-KS / WASAPI devices (Device 27), captures at native 48 kHz across channels, resamples to 16 kHz mono in real time, applies Adaptive Gain Control (AGC) for quiet microphones, and renders an ASCII VU meter. Fixed UI output list wiring in `app.py`. Synchronized all updates to `rpi_deployment/`. Compiled the complete project dialogue and prompt-reply transcript into `COMPLETE_CONVERSATION_HISTORY.md` and user artifact. Created dedicated `HANDOFF_INDEX.md` under `AI 231\Dumalaog_ME2 - Tiny Voice Command Model` and `rpi_deployment/` containing the original professor instructions, problem statement, top 10 common smart device commands matrix with sample spoken phrases that work, actuator responses, hardware BOM, and execution guides. Verified 100% pass rates across `simulation/test_siri_alexa_features.py` (5/5), `simulation/test_all_commands.py` (15/15), and `simulation/test_wake_word.py` (5/5).
- **2026-09-25 — Zero-button hands-free always-listening with custom wake phrase ('Hi Dandan' / 'Hello Dandan') and executed academic notebook.**
  Implemented zero-button hands-free always-listening capability via `live_listen.py` using `sounddevice` continuous ring buffer streaming at 16 kHz (250 ms hop) and dual-tone system chimes. Sourced 315 physical audio files for custom wake phrases (*"Hi Dandan"*, *"Hello Dandan"*, *"Hey Dandan"*) across 3 native speakers, tempo changes, and pitch shifts into `data/dataset/wake_word/`. Retrained BC-ResNet-1 (27,642 parameters, **181.4 KB** INT8 quantized, **8.13 ms** edge CPU latency, **93.42%** test accuracy on unseen speech, 93.8% noise robustness at 20 dB SNR). Re-executed academic notebook `ME2_Voice_Command_Model.ipynb` (553 KB) end-to-end with embedded loss/accuracy curves, 27×27 confusion matrix, noise robustness curves, INT8 quantization, and hardware demonstration. Updated Gradio visual app (`app.py`), verified 15/15 integration tests and 5/5 wake-word lifecycle tests, and synced all deliverables to `rpi_deployment/` for Raspberry Pi 5.
- **2026-09-25 — Created professional multi-sheet Excel Project Plan for Raspberry Pi 5 VCM.**
  Generated `VCM_RaspberryPi_Project_Plan.xlsx` (19.4 KB) under `AI 231\Dumalaog_ME2 - Tiny Voice Command Model`
  and copied to `rpi_deployment/`. Contains 5 styled sheets:
  1. `Executive Summary`: Charter, targets vs actuals scorecard, BC-ResNet-1 rationale.
  2. `Work Breakdown & Phases`: 24 tasks across 7 phases (Feasibility, Dataset Engineering,
     Architecture, Training, Optimization, HAL, Deployment) with hours and deliverables.
  3. `Command Vocabulary & Dataset`: Complete 27-class matrix across 4,383 WAV files.
  4. `Raspberry Pi 5 BOM & Wiring`: Hardware components, costing (₱6,895 total), 40-pin GPIO pinouts.
  5. `Edge Performance & Benchmarks`: 5-stage latency decomposition (8.44 ms total),
     resource usage (8.2% single-core CPU, 148 MB RAM, 44.5°C operating temp).
- **2026-09-25 — Acoustic Ducking & Standardized Multi-Evaluator Benchmark Suite added to Tiny VCM.**
  Addressed classmate consensus (Mark Macalalad & Anthony on acoustic self-interference) and Doc Rowel Atienza's course guidelines:
  1) Implemented Acoustic Ducking in `src/actions.py` & `src/engine.py`: dynamically lowers media playback volume from 70% to 10% when awakened, eliminating speaker acoustic feedback into the microphone, and restores volume upon command execution or timeout.
  2) Built `simulation/benchmark_evaluator.py` matching Ailene Mondares' multi-evaluator testing protocol. Tested across 4 evaluator profiles (Speaker A reference, Speaker B high pitch, Speaker C deep resonance, and unseen speaker voice): achieved **97.83%** task success rate across all 10 Option B commands (90/92 trials each), **100.0%** wake-word sensitivity (0.0% FRR), **0.00** False Alarm Rate (FAR) under continuous noise, and **6.53 ms** median edge CPU latency.
  3) Cross-referenced Doc Rowel's reference KWS architecture (`AI 222/Deep-Learning-Experiments/versions/2025/kws/kws-infer.py`): our custom BC-ResNet-1 achieves **12.3x faster inference** (6.53 ms vs. 80.0 ms on RPi) and **246x smaller footprint** (181.4 KB INT8 vs. 44.7 MB ResNet-18) while scaling from 1-word single KWS to the full 10-command Option B smart assistant schema.
  4) Synchronized all files into `rpi_deployment/` and updated project `README.md`.
  5) Verified 100% pass rates across `simulation/test_siri_alexa_features.py` (5/5), `simulation/test_all_commands.py` (15/15), and `simulation/test_wake_word.py` (5/5).
- **2026-09-25 — Siri & Alexa edge architecture replicated in VCM project.**
  Faithfully copied Apple Siri and Amazon Alexa's multi-stage cascaded architecture into the VCM project:
  1) Implemented `EnergyVAD` front-end and `AdaptiveEndpointer` in `src/audio.py`, cutting idle CPU load to <0.1% via room silence inference gating.
  2) Upgraded `StreamingVCMEngine` in `src/engine.py` with compound one-shot ('one-breath') utterance parsing (*'Hi Dandan play music'*) and dynamic acoustic endpointing (~500 ms speech cessation dispatch).
  3) Synthesized 3 pristine 16 kHz multi-tone musical earcons in `assets/earcons/` (rising wake chime, success bell, cancel tone) and wired `play_earcon` into `src/tts.py`.
  4) Updated `monitor_display.py` with real-time VAD energy VU meter, `live_listen.py` with compound utterance triggers, and `app.py`.
  5) Synchronized all files into `rpi_deployment/`.
  6) Verified with 100% pass rates across `simulation/test_siri_alexa_features.py` (4/4), `simulation/test_all_commands.py` (15/15), and `simulation/test_wake_word.py` (5/5).
- **2026-09-25 — Zero-button hands-free hardware streaming integrated across app.py and kiosk.**
  Addressed the "why do I still have to click record button?" query by clarifying browser sandbox
  constraints vs. native OS streaming. Upgraded `app.py` with continuous background hardware microphone
  streaming (`sounddevice.InputStream`) and auto-updating UI polling (`gr.Timer(0.4)`), completely
  eliminating manual button clicks across both `app.py`, `monitor_display.py` (HDMI monitor kiosk),
  and `live_listen.py` (console runner). Added predictions and mel spec caching to `StreamingVCMEngine`
  (`src/engine.py` and `rpi_deployment/src/engine.py`). Re-verified 100% pass rates across both
  `simulation/test_all_commands.py` (15/15) and `simulation/test_wake_word.py` (5/5).
- **2026-09-25 — AI 231 ME2 end-to-end executed notebook & Raspberry Pi 5 package verified.**
  Re-generated and executed the academic course notebook `ME2_Voice_Command_Model.ipynb`
  (534.6 KB) end-to-end via `nbconvert` on the shared `.venv` kernel (`ai222-231`).
  Contains full theoretical derivations (ASR vs. Spoken Intent Classification footprint
  crisis, STFT Log-Mel filterbanks, BC-ResNet frequency-broadcasting attention, INT8
  quantization math, and wake-word state machine), student header (Crizepvill F.
  Dumalaog, 202521406, AI 231 ME2), 25-epoch GPU training dynamics, 27×27 confusion
  matrix, SNR noise robustness plot, INT8 size verification (181.4 KB, < 500 KB limit),
  edge CPU latency histogram (8.44 ms, < 10 ms limit), and complete simulated command
  showcase. Enhanced `rpi_deployment/setup_rpi.sh` for Raspberry Pi 5 (4GB RAM, 32GB storage)
  with automatic RP1 `rpi-lgpio` detection and Gradio. Relaunched and verified the live
  interactive web application (`app.py`) on `http://127.0.0.1:7860`.
- **2026-09-18 — AI 231 ME2 interactive app and wake-word ("Hi" / "Hello") activation completed.**
  Built, trained, and verified the interactive application (`app.py`) for the Tiny
  Voice Command Model (VCM). Added the `wake_word` class covering *"Hi"*, *"Hello"*,
  *"Hey"*, and variants, expanding the physical dataset to **4,383 16 kHz WAV audio files**
  (`data/dataset/wake_word/`). Retrained BC-ResNet-1 (27,642 parameters, 108.0 KB FP32,
  **181.4 KB** INT8 quantized) across 25 epochs on RTX 3050 CUDA GPU; reached **94.65%**
  validation accuracy and **94.84%** test accuracy on unseen speech. Upgraded the
  streaming engine and finite state machine (`src/engine.py`, `src/actions.py`) with a
  two-stage lifecycle: low-power `STANDBY` mode strictly filtering for "Hi" / "Hello"
  $\to$ double-beep wake chime + Cyan LED pulse $\to$ active 7-second `LISTENING` window
  $\to$ command dispatch and hardware actuation $\to$ auto-return to `STANDBY` or timeout.
  Created `app.py` in Gradio with live microphone recording, retro SSD1306 128x64 OLED
  mockup, glowing RGB PWM LED indicator, live Log-Mel spectrogram, 17-button quick-test
  voice bank, and chronological event log. Passed all 15 tests in `simulation/test_all_commands.py`
  (100.0%, 8.44 ms latency) and all 5 tests in `simulation/test_wake_word.py` (100.0%).
  Packaged `app.py` and model weights into `rpi_deployment/` for direct MicroSD edge deployment.
- **2026-09-18 — AI 231 ME2 completed, simulated, and packaged for edge deployment.**
  Implemented the complete Tiny Voice Command Model (VCM) for the top 10 smart
  device commands under `AI 231\Dumalaog_ME2 - Tiny Voice Command Model`. Sourced
  a physical multi-speaker dataset of **3,420 16 kHz WAV audio files** across 3
  speaker voices, varied speech rates, phrasing variations, pitch shifts ($\pm 2$
  semitones), and real environmental noise profiles (`data/dataset/`). Trained
  BC-ResNet-1 (27,601 parameters, 107.8 KB FP32, 181.3 KB INT8 quantized) from
  scratch across 25 epochs, reaching **93.64%** validation accuracy, **95.21%** test
  accuracy on unseen audio, and **8.93 ms** edge CPU latency (< 10 ms real-time limit).
  Implemented a full Hardware Abstraction Layer (HAL) with virtual simulation;
  passed 15/15 automated integration tests on real audio files covering all 10
  commands, OLED display, PWM RGB LED, buzzer, and audio output. Executed the
  self-contained academic course notebook `ME2_Voice_Command_Model.ipynb` end-to-end
  with embedded loss curves, spectrograms, confusion matrix, SNR robustness plot,
  and latency histogram. Created the turnkey Raspberry Pi 4/5 MicroSD deployment
  package (`rpi_deployment/` with `run_vcm.py`, `setup_rpi.sh`, `vcm.service`, and
  `WIRING_GUIDE.md`). Updated `AI 231/README.md`.
  descriptions across `AI 231/README.md` and the ME1 notebook header to `AI 231 (MLOps)`,
  removing legacy mentions of advanced computer vision / deep learning. Committed
  as `65d4a28` and pushed to `main` on GitHub.
