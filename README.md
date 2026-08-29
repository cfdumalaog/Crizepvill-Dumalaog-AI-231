# Crizepvill Dumalaog — AI 231

This is the single repository for Crizepvill Dumalaog's AI 231 machine
exercises and future course work.

## Machine exercises

| Exercise | Topic | Status |
|---|---|---|
| [ME1](ME1_CNN_Einops/) | Three-layer MNIST CNN implemented with PyTorch tensors, Einops, and `einsum` | Complete — **98.50%** official test accuracy after exactly five epochs. |

## Repository policy

- All AI 231 work belongs in this repository, with one directory per machine
  exercise or project.
- The shared Python environment remains one level above this repository at
  `..\.venv`; a second environment must not be created inside the repository.
- Generated datasets, caches, and model checkpoints are ignored. Compact
  verified metrics and report figures may be committed with each exercise.
- API keys, VPN configurations, credentials, and other secrets must never be
  committed.

## Shared environment

From PowerShell:

```powershell
Set-Location 'C:\Users\danda\Desktop\MEng AI Notebooks\AI 222 231'
& '.\.venv\Scripts\Activate.ps1'
Set-Location '.\AI 231'
```

Install/reconcile declared dependencies only if required:

```powershell
python -m pip install -r requirements.txt
```

## Agent provenance and student responsibility

The ME1 repository scaffolding, implementation, notebook execution, automated
checks, Git initialization, and initial commit were performed by a coding
agent in response to the assignment. The student must still inspect, run, and
understand the implementation, tensor shapes, training behavior, predictions,
and limitations before submission.

The existing `pytorch_iris_example.py` is retained as an earlier introductory
example. It is separate from ME1 and is not part of ME1's no-CNN-library
compliance boundary.
