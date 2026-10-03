## AI 231 ME2 TinyDSCNN-48 Antigrav Engineering & Fine-Tuning Guide

This guide is written specifically for the student and future coding agents to understand the architecture, data structures, deployment flow, and fine-tuning procedure for `TinyDSCNN-48 Antigrav`.

### 1. Architectural & Frontend Specifications
- **Audio Capture**: 16 kHz Mono linear PCM ($f_s = 16,000\text{ Hz}$), 16-bit.
- **Temporal Window**: Fixed 2.5-second buffer ($L = 40,000\text{ samples}$), accommodating 89% of full spoken command sentences.
- **Short-Time Fourier Transform (STFT)**:
  - Periodic Hann window $w[n]$ of length $N = 512$ ($32\text{ ms}$).
  - Hop size $H = 160$ ($10\text{ ms}$).
  - Number of time frames: $T = \lfloor (40,000 - 512) / 160 \rfloor + 1 = 251\text{ frames}$.
  - FFT size: 512 points, yielding 257 unique positive frequency bins.
- **Mel Filterbank Projection**:
  - $M = 40$ triangular Mel filterbanks spanning $20\text{ Hz} \to 8,000\text{ Hz}$ ($f_s/2$).
  - Perceptual conversion: $m(f) = 2595 \log_{10}(1 + f/700)$.
- **Dynamic Compression & Standardization**:
  - Log energy: $S_{\text{log-mel}} = \ln(S_{\text{mel}} + 10^{-6})$.
  - Z-score normalization: $\hat{S} = (S_{\text{log-mel}} - \mu) / (\sigma + 10^{-5})$.
  - Output tensor shape: $\mathbf{(1, 40, 251)}$ ($1\text{ channel} \times 40\text{ Mel bins} \times 251\text{ time steps}$).
- **Neural Network Topology (`TinyDSCNN-48`)**:
  - **Stem**: Conv2D ($5\times 5$, stride $(2,2)$, 48 ch, no bias) + BatchNorm2d + ReLU $\to (48, 20, 126)$ ($1,296$ params).
  - **Block 1**: Depthwise ($3\times 3$, stride 1) + Pointwise ($1\times 1$, 48 ch) with BN+ReLU $\to (48, 20, 126)$ ($2,928$ params).
  - **Block 2**: Depthwise ($3\times 3$, stride $(1,2)$, time $\downarrow$) + Pointwise ($1\times 1$, 48 ch) with BN+ReLU $\to (48, 20, 63)$ ($2,928$ params).
  - **Block 3**: Depthwise ($3\times 3$, stride 1) + Pointwise ($1\times 1$, 48 ch) with BN+ReLU $\to (48, 20, 63)$ ($2,928$ params).
  - **Block 4**: Depthwise ($3\times 3$, stride 1) + Pointwise ($1\times 1$, 48 ch) with BN+ReLU $\to (48, 20, 63)$ ($2,928$ params).
  - **Classifier Head**: AdaptiveAvgPool2d(1) $\to (48, 1, 1)$, Dropout($p=0.15$):
    - ME2 Spoken Command Dataset deliverable: Linear($48 \to 31$ classes) $\to 31$ logits ($1,519$ params). Total: **14,527 parameters** ($39,315\text{ B}$ static INT8 ONNX).
    - Wake extension: Linear($48 \to 32$ classes) $\to 32$ logits ($1,568$ params). Total: **14,576 parameters** ($39,384\text{ B}$ static INT8 ONNX).
- **Depthwise Separable Efficiency Proof**:
  - Computational reduction: $\frac{\text{FLOPs}_{\text{DS}}}{\text{FLOPs}_{\text{std}}} = \frac{1}{C_{\text{out}}} + \frac{1}{D_k^2} = \frac{1}{48} + \frac{1}{9} \approx 0.1319 \approx \mathbf{\frac{1}{7.58}}$ ($7.58\times$ fewer FLOPs, $8.58\times$ fewer parameters than standard convolution).

### 2. Dataset Partitioning & Precomputed Cache
- **ME2 Spoken Command Dataset path**: `AI 231\Dumalaog_ME2 - Tiny Voice Command Model\data\option_b\`
- **Manifest**: `manifest.csv` (17,986 audio clips across 100 human speakers and 31 command classes).
- **Disjoint Splits (Zero Speaker Overlap)**:
  - `train`: 80 speakers (14,370 clips)
  - `val`: 10 held-out speakers `s81`–`s90` (1,818 clips)
  - `test`: 10 held-out speakers `s91`–`s100` (1,798 clips)
- **Features Cache**: `data/option_b/features_cache.npz` (1.75 GB). Contains precomputed Log-Mel spectrograms for all 17,986 audio files. Eliminates repeated on-the-fly STFT computation during multi-epoch training.

### 3. Training & Validation Provenance
- **Scratch Training Run**: `runs/antigrav-20260927-115145/`
- **Initial Weight State**: Pure random initialization recorded via SHA-256 hash `f1cfaafbb84cfc58...` (zero pretrained weights).
- **Hyperparameters**:
  - Epochs: 35
  - Batch size: 64
  - Optimizer: AdamW (initial $\text{lr} = 3 \times 10^{-3}$, weight decay $= 10^{-4}$)
  - Scheduler: Cosine Annealing decay down to $\eta_{\min} = 3 \times 10^{-5}$
  - Augmentations: SpecAugment (time masking, frequency masking, temporal rolling) + Label Smoothing (0.03) + Class-balanced cross-entropy weights
- **Performance**:
  - Best Validation Accuracy: **94.11%** (Epoch 35)
  - Final Held-Out Test Accuracy on Unseen Speakers: **96.05%** (Loss: 0.1654)
- **Checkpoints**:
  - Best PyTorch weights: `runs/antigrav-20260927-115145/best_model.pt`
  - Training history & metrics: `runs/antigrav-20260927-115145/history.json` and `metrics.json`
  - Classification report: `runs/antigrav-20260927-115145/classification_report.txt`

### 4. ONNX Quantization & Benchmarks
- **FP32 ONNX Export**:
  - Path: `models/antigrav_optionb_fp32.onnx`
  - File size: 60,466 bytes (~59.0 KB)
  - Test accuracy: **96.11%**
  - Export parameter: `dynamo=False` in `torch.onnx.export` to bypass optional `onnxscript` requirement in PyTorch 2.13.
- **Static INT8 QDQ Quantization**:
  - Path: `models/antigrav_optionb_int8.onnx`
  - File size: **39,315 bytes (~38.4 KB)**
  - Test accuracy: **96.27%**
  - Prediction parity with FP32: **99.05%**
  - Quantization strategy: `CalibrationDataReader` over 256 validation Log-Mel samples, symmetric per-channel weights (`QInt8`), unsigned activations (`QUInt8`), `QuantFormat.QDQ`.
- **Latency Benchmarks**:
  - The saved PC CPU benchmark reports p50 = 0.202 ms and p95 = 0.657 ms.
  - An earlier note recorded Pi 5 p50 = 1.272 ms and p95 = 1.313 ms, but this
    audit did not reproduce that result with the current model. Treat it as
    historical until measured again on the target Pi.

### 5. Raspberry Pi 5 Deployment Status
- The 2026-09-30 Assistant RGB V2 application bundle is installed at `/home/dalmacio/tinyvcm-rpi5-me2-20260930-assistant-lights-v2`. It is an application ZIP, not an SD-card image.
- The VCM is currently manually running on Pi loopback port 7860. It is not configured to autostart. From Windows, the temporary SSH tunnel maps local port 7864 to Pi port 7860; browse to `http://127.0.0.1:7864/assistant` or `/studio`.
- ARM64 ONNX checks and frontend-plus-model inference p95 passed at 5.52 ms wake and 5.27 ms intent. The fixed LED API issued on/off and red/green/blue states; GPIO 17/22/27 were outputs and final brightness was 0%. The Pi has no capture microphone, so live speech is not verified. Existing `/home/dalmacio/vcm` source was read-only as the pin-map reference and was not modified.
- The tracked release ONNX file and root-level TinyVCM_RaspberryPi5.zip remain legacy artifacts. The separate rpi_deployment directory is an older BC-ResNet project.

### 6. Device Action Behavior & Network Limits
- Device actions are deterministic code paths; Windows uses simulated lights,
  while the Pi release drives RGB channels on BCM GPIO 17/27/22. Both Assistant
  and Studio display the same light state. Assistant demo buttons use a
  loopback-only allowlist and bypass the classifiers; voice actions continue
  through wake and intent inference. The test suite covers device state and
  fixed-command validation.
- Weather uses Open-Meteo when enabled and needs a network; classification
  remains local.
- Call and message labels produce demo acknowledgments; they do not place calls
  or send messages. GPIO API and line states passed on Pi; no camera or light
  sensor was available for independent optical verification.

### 7. Historical Notes and Resolved Implementation Details
- The earlier synthetic SAPI baseline and BC-ResNet implementation are retained
  for comparison. The current assignment artifact is the 31-class ME2 Spoken
  Command Dataset model; its listed 100 speakers come from the reference corpus and should not
  be described as 100 student-collected speakers.
- Previous notes described Pi Connect, a Pi microphone, and remote deployment.
  Those details were not independently reproduced in this audit and do not
  establish that the current model can be deployed from the existing files.
- The current ONNX export uses the classic exporter (`dynamo=False`). The
  current notebook frontend uses a 40,000-sample target window.

### 8. Reproducible Checks and Model Selection
- From the ME2 project directory, run the shared-environment command:
  ..\..\.venv\Scripts\python.exe -m pytest -q
  The project pytest.ini restricts default discovery to active tests/ and skips
  archived prototypes. The latest active suite passed 28 tests.
- Saved notebook execution records are 7/7 code cells for the canonical intent
  notebook, 4/4 for the binary wake notebook, and 5/5 for the dated personalized
  run notebook; none contains saved error output. This audit did not rerun their
  training cells.
- The desktop launcher selects the paired models under
  `deployment/current_vcm/`: binary wake `NON_WAKE`/`WAKE_WORD` followed by the
  31-class intent model. It validates both SHA-256 hashes and runs with the
  local operating threshold 0.95; model-validation metadata remains separate.
- The 32-class wake extension is a historical, warm-started experiment. The
  earlier installed Pi release is also historical and is not the current
  desktop candidate. See the exercise HANDOFF_INDEX.md for candidate hashes,
  label-order evidence, limitations and open work.

