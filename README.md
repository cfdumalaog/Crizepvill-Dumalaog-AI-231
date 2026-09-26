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
| **ME2** | Tiny Voice Command Model for smart devices | [Current executed notebook](<Dumalaog_ME2 - Tiny Voice Command Model/notebooks/ME2_From_Scratch_Verified.ipynb>) | Provisional — 44.64 KB INT8, 79.28% isolated synthetic accuracy; 35.65% continuous synthetic task success. Human dataset, reliable recognition and physical Pi validation pending. |

## Repository Policy

- All AI 231 coursework belongs in this repository, organized with one directory per machine exercise or project.
- The shared Python virtual environment is located at the workspace root (`..\.venv`).
- ME1 retains its self-contained notebook. ME2 uses an executed report notebook with shared `tinyvcm/` helpers so training and deployed audio preprocessing stay identical.
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


## ME2 execution provenance (2026-09-26)

Codex audited the earlier ME2 claims and implemented/executed the current from-scratch DS-CNN pipeline. The dataset is synthetic, not recordings of three people. No Raspberry Pi was available. See the exercise README for actual measurements and remaining assignment work; student review is still required. Earlier ME1 provenance above is preserved unchanged.
