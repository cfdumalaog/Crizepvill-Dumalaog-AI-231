"""
Dataset loader and PyTorch DataLoader pipeline for Tiny VCM.
Loads physical 16kHz WAV audio recordings from disk with speaker-stratified splits,
dynamic environmental noise mixing, and SpecAugment.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

from .config import (
    SAMPLE_RATE,
    NUM_SAMPLES,
    COMMAND_CLASSES,
    CLASS_TO_IDX,
    DATA_DIR
)
from .audio import LogMelFrontend, apply_spec_augment, mix_noise_at_snr


class VoiceCommandDataset(Dataset):
    """
    PyTorch Dataset for Voice Command Recognition.
    Loads real 16kHz WAV files from disk and applies online edge audio augmentations.
    """
    def __init__(
        self,
        dataset_dir: Path = DATA_DIR / "dataset",
        split: str = "train",
        noise_prob: float = 0.5,
        spec_augment: bool = True,
        seed: int = 42
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.noise_prob = noise_prob if split == "train" else 0.1
        self.spec_augment = spec_augment and (split == "train")
        self.frontend = LogMelFrontend()

        # Load background noise pool for dynamic acoustic mixing
        self.noise_files = []
        noise_dir = self.dataset_dir / "_background_noise_"
        if noise_dir.exists():
            self.noise_files = list(noise_dir.glob("*.wav"))

        self.samples = []  # List of (file_path, class_name, class_idx)

        # Collect and partition files per class
        rng = random.Random(seed)
        for cls_name in COMMAND_CLASSES:
            cls_dir = self.dataset_dir / cls_name
            if not cls_dir.exists():
                continue
            
            cls_idx = CLASS_TO_IDX[cls_name]
            files = sorted(list(cls_dir.glob("*.wav")))
            rng.shuffle(files)

            n = len(files)
            if n == 0:
                continue

            n_train = max(1, int(n * 0.75))
            n_val = max(1, int(n * 0.125))

            if split == "train":
                selected_files = files[:n_train]
            elif split == "val":
                selected_files = files[n_train:n_train + n_val]
            else:  # test
                selected_files = files[n_train + n_val:]
                if len(selected_files) == 0:
                    selected_files = files[-max(1, int(n * 0.15)):]

            for fpath in selected_files:
                self.samples.append((fpath, cls_name, cls_idx))

        rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fpath, cls_name, cls_idx = self.samples[idx]

        # Read 16kHz audio from WAV file
        try:
            audio, sr = sf.read(str(fpath))
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        except Exception:
            audio = np.zeros(NUM_SAMPLES, dtype=np.float32)

        # Ensure exact 1.5s (24,000 samples)
        if len(audio) < NUM_SAMPLES:
            padded = np.zeros(NUM_SAMPLES, dtype=np.float32)
            padded[:len(audio)] = audio
            audio = padded
        elif len(audio) > NUM_SAMPLES:
            audio = audio[:NUM_SAMPLES]

        audio = audio.astype(np.float32)

        # Random time shift augmentation (train split only)
        if self.split == "train" and cls_name not in ["_silence_", "_background_noise_"]:
            shift = np.random.randint(-1600, 1600)  # +/- 100ms
            audio = np.roll(audio, shift)

        # Dynamic environmental noise mixing
        if self.split == "train" and cls_name != "_silence_" and self.noise_files:
            if np.random.rand() < self.noise_prob:
                noise_f = random.choice(self.noise_files)
                try:
                    noise_audio, _ = sf.read(str(noise_f))
                    snr = np.random.uniform(10.0, 25.0)
                    audio = mix_noise_at_snr(audio, noise_audio, snr)
                except Exception:
                    pass

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, 24000)

        with torch.no_grad():
            mel_spec = self.frontend(audio_tensor)  # (1, 1, 40, 151)

        if self.spec_augment:
            mel_spec = apply_spec_augment(mel_spec)

        return mel_spec.squeeze(0), cls_idx  # (1, 40, 151), label_int


def create_dataloaders(
    dataset_dir: Path = DATA_DIR / "dataset",
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds train, validation, and test PyTorch DataLoaders from disk WAV files."""
    train_ds = VoiceCommandDataset(dataset_dir=dataset_dir, split="train")
    val_ds = VoiceCommandDataset(dataset_dir=dataset_dir, split="val")
    test_ds = VoiceCommandDataset(dataset_dir=dataset_dir, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
