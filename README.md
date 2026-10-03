# Crizepvill Dumalaog — AI 231

This is the single canonical repository for Crizepvill Dumalaog's AI 231 machine
exercises and coursework at the University of the Philippines Diliman.

## Student Information
- **Name:** Crizepvill F. Dumalaog
- **Student Number:** 202521406
- **Course:** AI 231 (MLOps)

## Machine Exercises

| Exercise | Topic | Location | Status |
|---|---|---|---|
| **ME1** | Three-layer MNIST CNN using PyTorch tensors, Einops, and `einsum` | [`Dumalaog_ME1 - CNN using Einops/ME1_MNIST_Manual_CNN.ipynb`](<Dumalaog_ME1 - CNN using Einops/ME1_MNIST_Manual_CNN.ipynb>) | Complete — **98.60%** test accuracy after 5 epochs; fully self-contained notebook. |
| **ME2** | Tiny Voice Command Model for smart devices | [Current two-model notebook](<Dumalaog_ME2 - Tiny Voice Command Model/notebooks/ME2_Tiny_VCM_Training.ipynb>) | Refreshed ME2 intent + retained wake pair on Pi Desktop/dandan; Gold candidate removed, not promoted. Reference 96.89% accuracy / 96.88% macro F1; personal clip holdout 94.04% / 95.12%. Gold live benchmark has not been scored; its two-slide runtime brief is in the [updated deck](<Dumalaog_ME2 - Tiny Voice Command Model/docs/slides/ME2_VCM_Gold_Live_Benchmark_20261003.pptx>). See the [project README](<Dumalaog_ME2 - Tiny Voice Command Model/README.md>). |

## Repository Policy

- All AI 231 coursework belongs in this repository, organized with one directory per machine exercise or project.
- The shared Python virtual environment is located at the workspace root (`..\.venv`).
- ME1 retains its self-contained notebook. ME2 has a legacy synthetic `tinyvcm/` baseline and a separate `tinyvcm_model/` ME2 Spoken Command Dataset pipeline; their models and preprocessing are not interchangeable. Use the exercise handoff to identify the current model.
- The SLURP experiment uses only real human recordings, but its audio is CC BY-NC 4.0. It is suitable only for noncommercial academic work unless a different license is obtained. The treatment does not replace the current candidate.
- Generated datasets (`data/`), checkpoints (`*.pt`), and temporary files are excluded via `.gitignore`.
- API keys, VPN configurations, credentials, and secrets must never be committed.

## Agent Provenance & Traceability

This repository structure, initial setup, and the ME1 self-contained notebook implementation were generated and executed by an AI coding agent per the course instructions (*"agent initialized and committed"*). The student has inspected, validated, and understands all theoretical formulations, manual tensor operations, window sliding mechanics, Einops transformations, Einsum contractions, and experimental results.

### Agent Publication & Execution Audit Log
- **Agent Environment / Model:** Google Antigravity / OnIt (`Qwen/Qwen3.8-27B` & Advanced Coding Agent)
- **Execution Target:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.0, PyTorch 2.13.0+cu130)
- **Primary Task:** 3-layer CNN for MNIST using raw leaf tensors, Einops, and Einsum with zero `torch.nn` layer modules
- **Training Duration:** 5 full epochs (60,000 training images)
- **Benchmark Metric:** **98.60%** test accuracy on the official 10,000-image MNIST test split
- **Repository Remote:** [`https://github.com/cfdumalaog/Crizepvill-Dumalaog-AI-231`](https://github.com/cfdumalaog/Crizepvill-Dumalaog-AI-231)


## ME2 historical execution provenance (synthetic baseline, 2026-09-26)

This entry describes the earlier synthetic SAPI baseline only. The present canonical model uses the 17,986-row ME2 Spoken Command Dataset manifest and was trained from scratch; later binary-wake candidates are recorded separately in the exercise handoff. The current human manifest has 851 saved entries and 614 unique PCM groups from three speaker IDs. Personal splits are clip-held-out with speaker overlap, so they do not establish unseen-speaker accuracy. The Pi application is deployed and manually running, but live voice testing is blocked until a microphone is connected. The SLURP treatment is an offline experiment and has not been deployed. See the exercise README and HANDOFF_INDEX.md for the current audit and open work. Earlier ME1 provenance above is preserved unchanged.
