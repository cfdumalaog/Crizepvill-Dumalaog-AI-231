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

## Repository Policy

- All AI 231 coursework belongs in this repository, organized with one directory per machine exercise or project.
- The shared Python virtual environment is located at the workspace root (`..\.venv`).
- Each machine exercise is completely self-contained within its Jupyter notebook, containing all data processing, mathematical layer implementations, training loops, evaluation, and visual reports.
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

