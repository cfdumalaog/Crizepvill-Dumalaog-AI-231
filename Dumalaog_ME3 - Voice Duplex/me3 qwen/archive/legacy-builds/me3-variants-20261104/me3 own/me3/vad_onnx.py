"""Torch-free Silero VAD adapter used by the ME3 runtime.

Pipecat ships the Silero ONNX model and an ONNX Runtime wrapper. Reusing that
asset keeps the Pi deployment CPU-only: the voice loop does not need the
large PyTorch/CUDA dependency tree merely to classify 512-sample frames.
"""
from __future__ import annotations

from importlib import resources

import numpy as np


def load_vad_model():
    """Return Pipecat's bundled CPU Silero ONNX model wrapper."""
    from pipecat.audio.vad.silero import SileroOnnxModel

    model_path = resources.files("pipecat.audio.vad.data").joinpath("silero_vad.onnx")
    return SileroOnnxModel(str(model_path), force_onnx_cpu=True)


def reset_vad_model(model) -> None:
    """Reset recurrent ONNX state when a new stream or turn begins."""
    reset = getattr(model, "reset_states", None)
    if callable(reset):
        reset()


def frame_probability(model, frame: np.ndarray, sample_rate: int = 16000) -> float:
    """Score one PCM16-normalized frame with the bundled ONNX model."""
    audio = np.asarray(frame, dtype=np.float32).reshape(-1)
    result = model(audio, sample_rate)
    return float(np.asarray(result).reshape(-1)[0])
