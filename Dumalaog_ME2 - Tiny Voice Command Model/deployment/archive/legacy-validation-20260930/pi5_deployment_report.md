# Raspberry Pi 5 Edge Deployment & Verification Report

**Project:** Tiny Voice Command Model (AI 231 ME2) — TinyDSCNN-48 Antigrav  
**Deployment Date:** 2026-09-27 (Asia/Manila)  
**Target Hardware:** Raspberry Pi 5 4GB (`cfdfnjrpi5`, IP `192.168.254.106`)  
**Deployment Mode:** Application bundle over SSH (`~/tinyvcm-rpi5`)  
**Status:** **VERIFIED** — Release bundle installed, models verified on ARM64 CPU, USB microphone capturing live audio, real wake-to-command interaction succeeded, systemd service enabled and loopback-bound.

---

## 1. Release Bundle & Artifact Integrity

| Property | Value | Evidence Source |
| :--- | :--- | :--- |
| **Bundle Archive** | `TinyVCM_RPi5_Antigrav_RELEASE.zip` | `deployment/dist/` |
| **Archive File Size** | **73,973 bytes** | Filesystem stat |
| **PC SHA-256 Checksum** | `57fb5c8442cb1ec0ed8d241cc3c3d5556912c216fe0b13c76a3c56091a0f7b4f` | `Get-FileHash` |
| **Remote Pi SHA-256** | `57fb5c8442cb1ec0ed8d241cc3c3d5556912c216fe0b13c76a3c56091a0f7b4f` | `sha256sum` on Pi (Exact match) |
| **Extraction Path** | `/home/dalmacio/tinyvcm-rpi5/` | Pi local directory |
| **Manifest Verified Files** | **16 files** (including `tinyvcm/config.py` and model metadata) | `verify_release.py` output |

---

## 2. Target System Environment

- **Hostname:** `cfdfnjrpi5`
- **Operating System:** Debian GNU/Linux 13 (trixie) 64-bit
- **Architecture:** `aarch64` (ARMv8.2-A / Cortex-A76 quad-core)
- **Kernel:** Linux 6.18
- **System Python:** `Python 3.13.5`
- **Isolated Virtualenv:** `/home/dalmacio/.venvs/tinyvcm-rpi5/`
- **System Memory:** 4.0 GiB total, 2.8 GiB available
- **Storage Space:** 117 GB total, 7.7 GB used, 105 GB available (7% utilization)
- **User & Groups:** User `dalmacio` confirmed in groups `audio` and `gpio`

---

## 3. Bundled Model Provenance & Checksums

| Model | Provenance / Role | Classes | Parameters | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :---: | :---: | ---: | :--- |
| **`antigrav_optionb_int8.onnx`** | **Course Deliverable:** Trained strictly from scratch (seed 231, 0 pretrained weights) | 31 | 14,527 | 39,315 | `cf3c393b99bcbe991baf6c529b18d47c044a3e8c898fa41348762f7be8d24a0a` |
| **`antigrav_wake32_int8.onnx`** | **Experimental Extension:** Warm-started / fine-tuned from `best_model.pt` | 32 | 14,576 | 39,384 | `551836d42f33a34c9a6d13e37e963bcdf82470ba9ae7d71f080941306e351bf3` |

Both models use static INT8 QDQ quantization with per-channel symmetric weights and unsigned activations, executing via ONNX Runtime's `CPUExecutionProvider` using ARM NEON vector extensions.

---

## 4. Hardware Audio Capture Verification

- **Selected Capture Device:** `fifine Microphone: USB Audio (hw:2,0) / ALSA / 44100 Hz / 1 ch`
- **ALSA Hardware ID:** Card 2, Device 0 (`plughw:2,0`)
- **Resampling Pipeline:** Captures at hardware native 44,100 Hz and applies polyphase sinc filtering (`scipy.signal.resample_poly`) to standardize to 16,000 Hz.
- **Microphone Test Outcome:**
  - Resampled samples captured: **50,400 samples** over 3.0 seconds.
  - Observed peak amplitude: **0.0366** (active non-zero room acoustic capture).
  - Dropped audio blocks: **0 dropped blocks**.

---

## 5. Latency & Resource Utilization Benchmarks

Measured on the Raspberry Pi 5 physical hardware (100 timed iterations each, after 10 warm-up iterations):

| Model Artifact | Model-Only $p_{50}$ | Model-Only $p_{95}$ | Frontend + Model $p_{50}$ | Frontend + Model $p_{95}$ | Edge Target ($< 10	ext{ ms}$) |
| :--- | ---: | ---: | ---: | ---: | :---: |
| **`antigrav_optionb_int8.onnx` (31 classes)** | **1.476 ms** | **1.804 ms** | **7.059 ms** | **8.003 ms** | **PASS** |
| **`antigrav_wake32_int8.onnx` (32 classes)** | **1.430 ms** | **1.700 ms** | **6.900 ms** | **8.007 ms** | **PASS** |

### Additional Edge Metrics
- **CPU Throttling (`vcgencmd get_throttled`):** `throttled=0x0` (No under-voltage, no frequency capping, no thermal throttling).
- **Process Memory Footprint:** VmRSS **70,400 kB (~68.8 MB)** (easily fitting in Pi 5 memory).

---

## 6. Live Speech Interaction & Gating Validation

The assistant was launched in the foreground and tested through the physical Fifine USB microphone with live user interaction:

| Test Step | Stimulus / Action | Observed Response | Measured Confidence / Latency | Outcome |
| :--- | :--- | :--- | :--- | :---: |
| **1. Standby Gate** | Ambient noise / room talking | Suppressed; remains in `STANDBY` | Confidence < 0.60 | **PASS** |
| **2. Wake Detection** | Spoken phrase: *"Hi Dandan"* | State transitioned `STANDBY` $	o$ `LISTENING` | Wake probability > 0.60 | **PASS** |
| **3. Command 1** | Spoken phrase: *"Play music"* | Dispatched melody: *"Playing 'Lofi Chill Beats' by Lofi Girl / ChilledCow"* | Command accepted | **PASS** |
| **4. Command 2** | Spoken phrase: *"Pause music"* | Dispatched action: *"Music paused."* | Command accepted | **PASS** |
| **5. Live Classification** | Spoken phrase: *"Turn on the lights"* | Recognized `LIGHT_ON` (top-1) | Latency: 7.51 ms | **PASS** |
| **6. Inactivity Timeout** | 5.0 seconds silence | Automatically returned `LISTENING` $	o$ `STANDBY` | Deadline timer expired | **PASS** |

---

## 7. Systemd Service & Security Configuration

The background service was enabled and verified using systemd:

- **Unit File:** `/etc/systemd/system/tiny-vcm-antigrav.service`
- **Service State (`systemctl is-enabled`):** `enabled`
- **Runtime State (`systemctl is-active`):** `active`
- **Main PID:** 3433
- **Network Security Binding:** Bound strictly to **`127.0.0.1:7860`** (`0.0.0.0` disabled). Unauthenticated controls are not exposed to the local network.
- **Remote Access Method:** Verified via OpenSSH port-forwarding tunnel (`ssh -L 7860:127.0.0.1:7860 dalmacio@192.168.254.106`), allowing the user to access `http://127.0.0.1:7860/assistant` securely from the PC browser.
- **Hardware GPIO:** Disabled by default (`--gpio` flag omitted).

---

## 8. Verified Facts vs. Unverified Items

### Verified Facts
1. The 31-class Option B model is trained strictly from scratch without pretrained weights; the 32-class wake model is an experimental warm-started extension.
2. The release package was built with overwrite protection, transferred via SCP, and verified by SHA-256 hash match.
3. Both models execute on ARM Cortex-A76 CPU with $p_{95}$ latency under 2.0 ms (model-only) and under 8.1 ms (full frontend + model).
4. Physical Fifine USB microphone captures non-zero audio samples without dropped blocks.
5. Live wake phrase "Hi Dandan" transitions the assistant from `STANDBY` to `LISTENING`, and inactivity timeout returns to `STANDBY` after 5 seconds.
6. The service is persistently enabled and securely restricted to loopback.

### Remaining Limitations & Unverified Items
1. **Multi-Speaker Wake Robustness:** The 32-class wake word model was fine-tuned on user `person-01` takes and is **not** validated across diverse human speakers; it must not be claimed as Alexa-equivalent or universally robust.
2. **High-Noise Acoustic Environments:** Continuous speech recognition was not tested in extreme acoustic noise (>70 dB SPL).
3. **Physical GPIO Actuators:** No physical LEDs or buzzers were wired to the Pi 40-pin header; device actions execute in labeled simulation mode.
4. **Cloud / Network Independence:** All acoustic processing is local; no cloud ASR, LLM, or external APIs are used.

## 2026-09-28 Task C1 Deployment Addendum

This addendum records the timeout-only release deployed after the historical 2026-09-27 report above. It supersedes that report's current-timeout value; it does not reinterpret its earlier spoken-wake/command claims as evidence for the current wake model.

- Release: TinyVCM_RPi5_Antigrav_RELEASE.zip (74,029 bytes), SHA-256 b03b448f609e546beb449e3739dc23b018cd4f7daa9faa9d428058014c548b49. The copy transferred to the Pi matched. The app is installed at /home/dalmacio/tinyvcm-rpi5; rollback copy is /home/dalmacio/tinyvcm-rpi5-backup-c1-20260928.
- On-device verification: verify_release.py passed all 16 manifest files using the existing Pi environment. Option B model-only latency was p50/p95 1.303/1.507 ms; wake32 was 1.501/1.647 ms. Deployed model hashes are unchanged: Option B cf3c393b99bcbe991baf6c529b18d47c044a3e8c898fa41348762f7be8d24a0a; wake32 551836d42f33a34c9a6d13e37e963bcdf82470ba9ae7d71f080941306e351bf3.
- Service/UI/API: tiny-vcm-antigrav.service remained enabled and is active, bound to 127.0.0.1:7860. /assistant and /studio returned HTTP 200. /api/state reported STANDBY, timeout_sec=10, classes_count=32, the expected 32-class model label, Fifine USB mic, zero dropped chunks, and no error.
- Manual inactivity timer check: POST /api/wake at 09:35:08 entered LISTENING. The model logged accepted TEMPERATURE_22 events at 09:35:10 and 09:35:12, each of which reset the deadline. The service logged the inactivity timeout and returned to STANDBY at 09:35:22, ten seconds after the last accepted event. These log entries do not establish that the user intentionally spoke those commands. Treat this as evidence that accepted events reset the configured timer and that timeout returns to STANDBY, not as a voice-wake or command-accuracy pass.
- Scope: No wake model was trained or replaced. GPIO remained disabled/simulated. No new spoken wake-word test was completed during this deployment check; the user-reported wake failure and the weak saved-clip result (8/16 over threshold) remain open.

## 2026-09-29 Personalized R2 Manual Deployment Addendum

This addendum supersedes earlier statements above about the current model pair, active service, and boot behavior. Earlier spoken-interaction results describe the historical release only; they are not evidence for the R2 binary-wake model.

- **Release:** `TinyVCM_RPi5_Personalized_20260929-r2.zip`, 86,747 bytes, SHA-256 `850820df3c833c0138635ed91f22c2f8a84ef752d2c1ce8a324a13fda00356ea`. The Pi archive hash matched the PC. Installed at `/home/dalmacio/tinyvcm-rpi5-personalized-20260929-r2`; previous `/home/dalmacio/tinyvcm-rpi5` was left intact for rollback. The archive verifier passed all 20 packaged file checks.
- **Models:** Binary wake `antigrav_binary_wake_int8.onnx`, 37,425 bytes, SHA-256 `f7d1c3f7826bc16426eac02c10c05e077502f28bbc764c81510c7af08fbfb002`, outputs `NON_WAKE`/`WAKE_WORD`; intent `antigrav_personalized_intent_int8.onnx`, 39,315 bytes, SHA-256 `c5d995fb96809220013197246177ef8be9ad183ba0aab73ea058faac24c0c0a3`, 31 classes. Combined size is 76,740 bytes (74.9 KiB). ONNX CPU inference and exact label maps passed on ARM64.
- **Latency:** After warmup, 100 timed iterations on the Pi measured wake model-only p50/p95 1.306/1.334 ms and frontend+model p50/p95 5.489/5.529 ms. Intent model-only p50/p95 1.328/1.360 ms and frontend+model p50/p95 5.147/5.270 ms. These meet the 10 ms frontend+model target in this benchmark.
- **Manual launch / no boot start:** Executable launcher is `/home/dalmacio/tinyvcm-rpi5-personalized-20260929-r2/start-vcm.sh`. The app accepts manual foreground launch and Ctrl+C shutdown. System-level `tiny-vcm-antigrav.service` and `vcm-antigrav.service`, plus user-level `vcm-antigrav.service`, all report `disabled` and `inactive`; checks of user crontab, `/etc/rc.local`, and `~/.config/autostart` found no VCM entry. `ss` showed no listener on port 7860 after smoke testing. It will not autostart on boot.
- **UI/API smoke:** Running `./start-vcm.sh` produced HTTP 200 Studio/API responses and a valid STANDBY state with the expected two-model identity, 31 intent classes, 0.90 operating wake threshold, 10-second timeout, and no runtime error. The microphone status was “No microphone connected”; `/api/audio-inputs` returned an empty list.
- **Hardware limit:** `arecord -l` reports no capture hardware and sounddevice enumerates no input devices. Consequently no Pi live wake/intent or microphone-selection trial was possible. The UI can stay running without a mic; after connecting one, refresh the Pi input-device list and choose it. The browser output selector routes page/browser media on the computer running the browser; browser spoken feedback uses that browser's default output. GPIO remains disabled. No audio was captured or saved, and no public tunnel was started.
