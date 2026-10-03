"""
Audio processing, feature extraction, and augmentation pipeline for VCM.
Implements Log-Mel Spectrogram extraction matching the edge inference runtime,
SpecAugment, environmental noise mixing, and a real-time sliding ring buffer.
"""

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import soundfile as sf
from typing import Optional, Tuple

from .config import (
    SAMPLE_RATE,
    NUM_SAMPLES,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    N_MELS,
    F_MIN,
    F_MAX
)


class LogMelFrontend(nn.Module):
    """
    Differentiable and exportable Log-Mel Spectrogram frontend.
    Converts raw 16kHz audio waveforms into normalized 2D acoustic features.
    """
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        win_length: int = WIN_LENGTH,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
        eps: float = 1e-6
    ):
        super().__init__()
        self.eps = eps
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            center=True,
            norm="slaney",
            mel_scale="slaney"
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape (batch, num_samples) or (batch, 1, num_samples)
        Returns:
            log_mel: Tensor of shape (batch, 1, n_mels, time_frames)
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        
        # Squeeze channel dim for MelSpectrogram: (batch, num_samples)
        x = waveform.squeeze(1)
        mel_spec = self.mel_transform(x)  # (batch, n_mels, time_frames)
        log_mel = torch.log(mel_spec + self.eps)
        
        # Per-instance standardization (mean=0, std=1 across time and freq)
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True) + self.eps
        normalized_mel = (log_mel - mean) / std
        
        # Add channel dimension: (batch, 1, n_mels, time_frames)
        return normalized_mel.unsqueeze(1)


def apply_spec_augment(
    mel_spec: torch.Tensor,
    freq_mask_max: int = 6,
    time_mask_max: int = 16,
    num_freq_masks: int = 1,
    num_time_masks: int = 2
) -> torch.Tensor:
    """
    SpecAugment: Zero-out random frequency and time stripes to improve generalization.
    Args:
        mel_spec: Tensor of shape (batch, 1, n_mels, time_frames)
    """
    augmented = mel_spec.clone()
    batch_size, _, n_mels, time_frames = augmented.shape

    for b in range(batch_size):
        # Frequency masking
        for _ in range(num_freq_masks):
            f_len = torch.randint(0, freq_mask_max + 1, (1,)).item()
            if f_len > 0 and f_len < n_mels:
                f0 = torch.randint(0, n_mels - f_len, (1,)).item()
                augmented[b, :, f0:f0 + f_len, :] = 0.0

        # Time masking
        for _ in range(num_time_masks):
            t_len = torch.randint(0, time_mask_max + 1, (1,)).item()
            if t_len > 0 and t_len < time_frames:
                t0 = torch.randint(0, time_frames - t_len, (1,)).item()
                augmented[b, :, :, t0:t0 + t_len] = 0.0

    return augmented


def mix_noise_at_snr(
    clean_audio: np.ndarray,
    noise_audio: np.ndarray,
    snr_db: float
) -> np.ndarray:
    """
    Mix clean speech with background noise at a target Signal-to-Noise Ratio (dB).
    """
    clean_len = len(clean_audio)
    if len(noise_audio) < clean_len:
        # Repeat noise to match audio length
        reps = int(np.ceil(clean_len / len(noise_audio)))
        noise_audio = np.tile(noise_audio, reps)
    
    # Randomly slice noise to match clean length
    max_start = len(noise_audio) - clean_len
    start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
    noise_slice = noise_audio[start_idx:start_idx + clean_len]

    # Calculate powers
    clean_power = np.mean(clean_audio ** 2) + 1e-12
    noise_power = np.mean(noise_slice ** 2) + 1e-12

    # Target noise power based on SNR
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    
    mixed = clean_audio + scale * noise_slice
    
    # Normalize peak to avoid clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(np.float32)


class AudioRingBuffer:
    """
    Thread-safe FIFO circular buffer for continuous streaming inference.
    Stores the most recent N_SAMPLES of audio at 16kHz (e.g. 1.5 seconds).
    """
    def __init__(self, capacity_samples: int = NUM_SAMPLES):
        self.capacity = capacity_samples
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.samples_pushed = 0

    @property
    def is_filled(self) -> bool:
        """Only True once at least capacity_samples have been pushed."""
        return self.samples_pushed >= self.capacity

    def push(self, chunk: np.ndarray):
        """Append a newly captured audio chunk and discard the oldest samples."""
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        chunk_len = len(chunk)
        self.samples_pushed += chunk_len
        if chunk_len >= self.capacity:
            self.buffer[:] = chunk[-self.capacity:]
        else:
            self.buffer = np.roll(self.buffer, -chunk_len)
            self.buffer[-chunk_len:] = chunk

    def get_window(self) -> np.ndarray:
        """Returns the current 1.5-second audio window for model evaluation."""
        return self.buffer.copy()

    def get_recent(self, num_samples: int) -> np.ndarray:
        """Returns the most recent N samples from the circular buffer."""
        n = min(num_samples, self.capacity)
        return self.buffer[-n:].copy()

    def get_slice(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Returns a specific sample slice within the circular buffer."""
        return self.buffer[start_idx:end_idx].copy()

    def reset(self):
        """Zero-out the ring buffer and reset sample count."""
        self.buffer.fill(0.0)
        self.samples_pushed = 0

    clear = reset


class EnergyVAD:
    """
    Lightweight, low-power Voice Activity Detector (VAD) modeled after
    Apple Siri Always-On Processor (AOP) acoustic front-end.
    Tracks short-time Root-Mean-Square (RMS) energy and an adaptive noise floor.
    Bypasses heavy neural forward passes when ambient room silence is detected.
    """
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        snr_threshold: float = 1.8,
        min_energy_threshold: float = 0.002,
        noise_adaptation_rate: float = 0.05
    ):
        self.sample_rate = sample_rate
        self.snr_threshold = snr_threshold
        self.min_energy_threshold = min_energy_threshold
        self.noise_adaptation_rate = noise_adaptation_rate
        self.noise_floor = min_energy_threshold
        self.last_energy = 0.0
        self.is_speech_active = False

    def compute_energy(self, audio_chunk: np.ndarray) -> float:
        """Computes Root-Mean-Square (RMS) energy of an audio chunk."""
        if len(audio_chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_chunk ** 2) + 1e-12))

    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, float, float]:
        """
        Evaluates an audio chunk.
        Returns:
            (is_speech, energy, snr_ratio)
        """
        energy = self.compute_energy(audio_chunk)
        self.last_energy = energy

        # Dynamic speech threshold
        speech_thresh = max(self.min_energy_threshold, self.noise_floor * self.snr_threshold)
        is_speech = energy >= speech_thresh

        if not is_speech:
            # Adapt noise floor on quiet frames
            self.noise_floor = (
                (1.0 - self.noise_adaptation_rate) * self.noise_floor
                + self.noise_adaptation_rate * energy
            )
            self.noise_floor = max(1e-5, self.noise_floor)

        self.is_speech_active = is_speech
        snr = energy / (self.noise_floor + 1e-12)
        return is_speech, energy, snr


class AdaptiveEndpointer:
    """
    Real-time speech boundary and endpoint detector modeled after Amazon Alexa
    and Apple Siri dynamic endpointing.
    Detects:
    1. Speech Onset: Transition from silence to sustained user speech.
    2. Speech Endpoint: Sustained silence (~500ms) after speech activity,
       triggering instant query dispatch without waiting for arbitrary fixed timeouts.
    """
    STATE_IDLE = "IDLE"
    STATE_SPEECH_ACTIVE = "SPEECH_ACTIVE"
    STATE_ENDPOINT_REACHED = "ENDPOINT_REACHED"

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        silence_duration_sec: float = 0.45,
        min_speech_duration_sec: float = 0.20
    ):
        self.sample_rate = sample_rate
        self.silence_duration_sec = silence_duration_sec
        self.min_speech_duration_sec = min_speech_duration_sec

        self.state = self.STATE_IDLE
        self.speech_samples = 0
        self.silence_samples = 0

    def reset(self):
        """Reset the endpointer state machine."""
        self.state = self.STATE_IDLE
        self.speech_samples = 0
        self.silence_samples = 0

    def process(self, chunk_len: int, is_speech: bool) -> str:
        """
        Updates endpointer state with a newly captured frame.
        Returns current state: IDLE, SPEECH_ACTIVE, or ENDPOINT_REACHED.
        """
        if self.state == self.STATE_IDLE:
            if is_speech:
                self.speech_samples += chunk_len
                min_samples = int(self.min_speech_duration_sec * self.sample_rate)
                if self.speech_samples >= min_samples:
                    self.state = self.STATE_SPEECH_ACTIVE
                    self.silence_samples = 0
            else:
                self.speech_samples = max(0, self.speech_samples - chunk_len)

        elif self.state == self.STATE_SPEECH_ACTIVE:
            if is_speech:
                self.speech_samples += chunk_len
                self.silence_samples = 0
            else:
                self.silence_samples += chunk_len
                silence_limit = int(self.silence_duration_sec * self.sample_rate)
                if self.silence_samples >= silence_limit:
                    self.state = self.STATE_ENDPOINT_REACHED

        return self.state
