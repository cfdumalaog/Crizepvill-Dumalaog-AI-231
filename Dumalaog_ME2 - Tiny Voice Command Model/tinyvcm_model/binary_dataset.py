"""Binary wake-word dataset preparation and temporal jitter augmentation.

Provides:
- Deduplication of 18 human wake recordings into 16 unique PCM waveforms.
- Fixed 10 / 3 / 3 speaker-source human wake partitions (both val and test include quiet-far takes).
- Grouped 7 / 1 / 1 synthetic wake partitions (by TTS voice/rate prompt, 35 files per group).
- Option B speaker-disjoint negative partitioning (80 train / 10 val / 10 test).
- Streaming buffer emulation via randomized temporal shift/rolling (0 to 1.5s offset).
- Training-only audio augmentation and stratified INT8 calibration extraction.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .config import SAMPLES, SR, FEATURE_SHAPE
from .frontend import Frontend, fit_audio


def compute_pcm_sha256(audio_path: Path) -> Tuple[str, np.ndarray, int]:
    """Load audio and compute SHA-256 of raw decoded PCM float32 bytes."""
    wav, sr = sf.read(str(audio_path), dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    pcm_bytes = wav.tobytes()
    pcm_hash = hashlib.sha256(pcm_bytes).hexdigest()
    return pcm_hash, wav, sr


def place_audio_with_jitter(wav: np.ndarray, target_samples: int = SAMPLES, rng: np.random.Generator = None) -> np.ndarray:
    """Place audio into target_samples with random temporal offset to emulate rolling streaming buffer.
    
    Instead of strictly centering, places the utterance anywhere from the beginning to the trailing edge.
    """
    if len(wav) >= target_samples:
        return fit_audio(wav, target_samples=target_samples)
    
    pad_total = target_samples - len(wav)
    if rng is None:
        pad_left = pad_total // 2
    else:
        # Randomize offset: 0 to pad_total
        pad_left = int(rng.integers(0, pad_total + 1))
    
    out = np.zeros(target_samples, dtype=np.float32)
    out[pad_left:pad_left + len(wav)] = wav
    return out


def augment_waveform_binary(wav: np.ndarray, rng: np.random.Generator, noises: List[np.ndarray]) -> np.ndarray:
    """Augment audio with gain scaling, temporal jitter, and additive background noise."""
    w = wav.copy()
    
    # 1. Random Gain (0.70 to 1.30)
    gain = rng.uniform(0.70, 1.30)
    w = w * gain

    # 2. Place in 2.5s window with random temporal offset
    w_jittered = place_audio_with_jitter(w, target_samples=SAMPLES, rng=rng)

    # 3. Additive noise (SNR 20 to 35 dB)
    if noises and rng.random() < 0.70:
        noise = noises[rng.integers(0, len(noises))]
        if len(noise) >= len(w_jittered):
            start = rng.integers(0, len(noise) - len(w_jittered) + 1)
            n_slice = noise[start:start + len(w_jittered)]
        else:
            n_slice = np.tile(noise, int(np.ceil(len(w_jittered) / len(noise))))[:len(w_jittered)]
        
        sig_pow = np.mean(w_jittered ** 2) + 1e-9
        noi_pow = np.mean(n_slice ** 2) + 1e-9
        target_snr_db = rng.uniform(20.0, 35.0)
        target_noi_pow = sig_pow / (10 ** (target_snr_db / 10.0))
        scale = np.sqrt(target_noi_pow / noi_pow)
        w_jittered = w_jittered + n_slice * scale

    return fit_audio(w_jittered, target_samples=SAMPLES)


def spec_augment(spec: np.ndarray, rng: np.random.Generator, max_t: int = 25, max_f: int = 6) -> np.ndarray:
    """Time and frequency masking on log-mel spectrogram (1, 40, 251)."""
    s = spec.copy()
    # Freq mask
    f = int(rng.integers(0, max_f + 1))
    if f > 0:
        f0 = int(rng.integers(0, 40 - f))
        s[:, f0:f0 + f, :] = 0.0
    # Time mask
    t = int(rng.integers(0, max_t + 1))
    if t > 0:
        t0 = int(rng.integers(0, 251 - t))
        s[:, :, t0:t0 + t] = 0.0
    return s


class BinaryWakeDataset(Dataset):
    """PyTorch Dataset for binary wake-word training/evaluation."""
    def __init__(self, features: np.ndarray, labels: np.ndarray, augment: bool = False, rng_seed: int = 231):
        assert len(features) == len(labels), "Features and labels length mismatch"
        self.features = features
        self.labels = labels
        self.augment = augment
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features[idx]
        if self.augment:
            feat = spec_augment(feat, self.rng)
        return torch.from_numpy(feat).float(), torch.tensor(self.labels[idx], dtype=torch.long)


def prepare_binary_data(project_root: Path, rng_seed: int = 231) -> Dict:
    """Load and partition audio into strict train, val, and test binary sets."""
    frontend = Frontend()
    rng = np.random.default_rng(rng_seed)

    # 1. Background Noise clips for augmentation
    noise_dir = project_root / 'data' / 'dataset' / '_background_noise_'
    noises = []
    if noise_dir.exists():
        for nf in sorted(noise_dir.glob('*.wav')):
            nw, _ = sf.read(str(nf), dtype='float32')
            if nw.ndim > 1:
                nw = nw.mean(axis=1)
            noises.append(nw)

    # 2. Human Wake Deduplication & Partitioning
    human_manifest = project_root / 'data' / 'human' / 'manifest.csv'
    with human_manifest.open('r', encoding='utf-8') as f:
        human_rows = list(csv.DictReader(f))

    wake_rows = [r for r in human_rows if r['label'] == 'wake_word']
    pcm_groups: Dict[str, List[dict]] = {}
    for r in wake_rows:
        wav_path = project_root / r['path']
        if not wav_path.exists():
            continue
        h, wav_data, _ = compute_pcm_sha256(wav_path)
        pcm_groups.setdefault(h, []).append({
            'row': r,
            'path': str(r['path']),
            'condition': r['condition'],
            'wav': wav_data
        })

    # Sort groups deterministically
    unique_hashes = sorted(pcm_groups.keys())
    assert len(unique_hashes) == 16, f"Expected 16 unique PCM waveforms, found {len(unique_hashes)}"
    
    # Identify near vs far
    near_hashes = [h for h in unique_hashes if pcm_groups[h][0]['condition'] == 'quiet-near']
    far_hashes = [h for h in unique_hashes if pcm_groups[h][0]['condition'] == 'quiet-far']
    assert len(near_hashes) == 14 and len(far_hashes) == 2, f"Near: {len(near_hashes)}, Far: {len(far_hashes)}"

    # 10 train (all near), 3 val (2 near + 1 far), 3 test (2 near + 1 far)
    train_hw_hashes = near_hashes[:10]
    val_hw_hashes = near_hashes[10:12] + [far_hashes[0]]
    test_hw_hashes = near_hashes[12:14] + [far_hashes[1]]

    train_hw_items = [it for h in train_hw_hashes for it in pcm_groups[h]]
    val_hw_items = [it for h in val_hw_hashes for it in pcm_groups[h]]
    test_hw_items = [it for h in test_hw_hashes for it in pcm_groups[h]]

    # 3. Synthetic Wake Partitioning (9 groups across 315 files)
    synth_wake_dir = project_root / 'data' / 'dataset' / 'wake_word'
    synth_files = sorted(synth_wake_dir.glob('*.wav'))
    synth_groups: Dict[str, List[Path]] = {}
    for sf_path in synth_files:
        grp = re.sub(r'_\d+_[^_]+$', '', sf_path.stem)
        synth_groups.setdefault(grp, []).append(sf_path)
    
    unique_synth_groups = sorted(synth_groups.keys())
    assert len(unique_synth_groups) == 9, f"Expected 9 synthetic groups, found {len(unique_synth_groups)}"

    train_synth_groups = unique_synth_groups[:7]
    val_synth_groups = unique_synth_groups[7:8]
    test_synth_groups = unique_synth_groups[8:]

    train_synth_files = [f for g in train_synth_groups for f in synth_groups[g]]
    val_synth_files = [f for g in val_synth_groups for f in synth_groups[g]]
    test_synth_files = [f for g in test_synth_groups for f in synth_groups[g]]

    # 4. Option B Command Negatives from precomputed cache
    cache_path = project_root / 'data' / 'option_b' / 'features_cache.npz'
    cache = np.load(cache_path)
    optb_train_x = cache['train_x'].astype(np.float32)  # 14,370 clips
    optb_val_x = cache['val_x'].astype(np.float32)      # 1,818 clips
    optb_test_x = cache['test_x'].astype(np.float32)    # 1,798 clips

    # 5. Build Training Positives with Temporal Jitter
    train_pos_features = []
    # A. Human wake: 75 augmented samples per item (12 items -> 900 samples)
    for it in train_hw_items:
        base_wav = it['wav']
        # Clean centered
        c_fitted = fit_audio(base_wav, target_samples=SAMPLES)
        train_pos_features.append(frontend(c_fitted))
        # 74 augmented versions with jitter
        for _ in range(74):
            aug_wav = augment_waveform_binary(base_wav, rng, noises)
            train_pos_features.append(frontend(aug_wav))

    # B. Synthetic wake: 2 augmented samples per file (245 files -> 490 samples)
    for sf_path in train_synth_files:
        sw, _ = sf.read(str(sf_path), dtype='float32')
        if sw.ndim > 1:
            sw = sw.mean(axis=1)
        # Clean centered
        c_fitted = fit_audio(sw, target_samples=SAMPLES)
        train_pos_features.append(frontend(c_fitted))
        # 1 jittered version
        aug_sw = augment_waveform_binary(sw, rng, noises)
        train_pos_features.append(frontend(aug_sw))

    train_pos_x = np.stack(train_pos_features, axis=0)
    train_pos_y = np.ones(len(train_pos_x), dtype=np.int64)

    # 6. Build Training Negatives (Option B commands + noise/silence)
    # Subsample 3,500 training negatives to maintain a balanced ~2.5:1 ratio
    neg_indices = rng.choice(len(optb_train_x), size=min(3500, len(optb_train_x)), replace=False)
    train_neg_x = optb_train_x[neg_indices]
    train_neg_y = np.zeros(len(train_neg_x), dtype=np.int64)

    # Combine training partition
    train_x = np.concatenate([train_neg_x, train_pos_x], axis=0)
    train_y = np.concatenate([train_neg_y, train_pos_y], axis=0)

    # Shuffle training set
    perm = rng.permutation(len(train_y))
    train_x = train_x[perm]
    train_y = train_y[perm]

    # 7. Build Validation Set
    val_pos_features = []
    # Human wake val (3 groups)
    for it in val_hw_items:
        w_fit = fit_audio(it['wav'], target_samples=SAMPLES)
        val_pos_features.append(frontend(w_fit))
    # Synthetic wake val (35 files)
    for sf_path in val_synth_files:
        sw, _ = sf.read(str(sf_path), dtype='float32')
        if sw.ndim > 1:
            sw = sw.mean(axis=1)
        val_pos_features.append(frontend(fit_audio(sw, target_samples=SAMPLES)))
    
    val_pos_x = np.stack(val_pos_features, axis=0)
    val_pos_y = np.ones(len(val_pos_x), dtype=np.int64)

    # Validation negatives: all 1,818 Option B val clips
    val_neg_x = optb_val_x
    val_neg_y = np.zeros(len(val_neg_x), dtype=np.int64)

    val_x = np.concatenate([val_neg_x, val_pos_x], axis=0)
    val_y = np.concatenate([val_neg_y, val_pos_y], axis=0)

    # 8. Calibration Set for Static INT8 (256 samples: 128 positive, 128 negative, strictly training)
    calib_pos_idx = rng.choice(len(train_pos_x), size=128, replace=False)
    calib_neg_idx = rng.choice(len(train_neg_x), size=128, replace=False)
    calib_x = np.concatenate([train_pos_x[calib_pos_idx], train_neg_x[calib_neg_idx]], axis=0)
    rng.shuffle(calib_x)

    return {
        'train_x': train_x,
        'train_y': train_y,
        'val_x': val_x,
        'val_y': val_y,
        'calib_x': calib_x,
        'val_hw_items': val_hw_items,
        'val_synth_files': val_synth_files,
        'test_hw_items': test_hw_items,
        'test_synth_files': test_synth_files,
        'optb_test_x': optb_test_x,
        'splits_info': {
            'train_samples': len(train_y),
            'train_positives': int((train_y == 1).sum()),
            'train_negatives': int((train_y == 0).sum()),
            'val_samples': len(val_y),
            'val_positives': int((val_y == 1).sum()),
            'val_negatives': int((val_y == 0).sum()),
            'calib_samples': len(calib_x),
            'train_hw_groups': len(train_hw_hashes),
            'val_hw_groups': len(val_hw_hashes),
            'test_hw_groups': len(test_hw_hashes),
            'train_synth_groups': len(train_synth_groups),
            'val_synth_groups': len(val_synth_groups),
            'test_synth_groups': len(test_synth_groups),
        }
    }
