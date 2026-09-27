"""Dataset loader, audit, and parallel feature extraction for Option B."""
import csv
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path
import numpy as np
import scipy.signal
import soundfile as sf

from .config import LABELS, MANIFEST_PATH, OPTION_B_DATA, ROOT, SAMPLES, SR, TIME_STEPS, MELS
from .frontend import Frontend, fit_audio


def audit_dataset(output_dir=None):
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f'Manifest not found at {MANIFEST_PATH}')

    with MANIFEST_PATH.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r['full_path'] = str(OPTION_B_DATA / r['path'])

    speakers = sorted(list(set(r['speaker'] for r in rows)))
    train_speakers = sorted(list(set(r['speaker'] for r in rows if r['split'] == 'train')))
    val_speakers = sorted(list(set(r['speaker'] for r in rows if r['split'] == 'val')))
    test_speakers = sorted(list(set(r['speaker'] for r in rows if r['split'] == 'test')))

    # Verify disjoint speaker holdouts
    assert not (set(train_speakers) & set(val_speakers)), "Train and Val speakers overlap!"
    assert not (set(train_speakers) & set(test_speakers)), "Train and Test speakers overlap!"
    assert not (set(val_speakers) & set(test_speakers)), "Val and Test speakers overlap!"

    split_counts = dict(Counter(r['split'] for r in rows))
    class_counts = dict(Counter(r['label'] for r in rows))

    report = {
        "dataset_name": "Option B Spoken Command Dataset (Mark Macalalad AI 231 MEX2)",
        "total_active_clips": len(rows),
        "total_speakers": len(speakers),
        "speaker_splits": {
            "train_speakers_count": len(train_speakers),
            "val_speakers_count": len(val_speakers),
            "test_speakers_count": len(test_speakers),
            "val_speakers": val_speakers,
            "test_speakers": test_speakers
        },
        "split_counts": split_counts,
        "classes_count": len(LABELS),
        "classes": LABELS,
        "class_distribution": class_counts,
        "disjoint_speakers_verified": True
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / 'dataset_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    return rows, LABELS, report


def _process_single(row, labels, frontend):
    try:
        data, sr = sf.read(row['full_path'], dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=-1)
        if sr != SR:
            g = gcd(int(SR), int(sr))
            data = scipy.signal.resample_poly(data, SR // g, sr // g).astype(np.float32)
        fitted = fit_audio(data, SAMPLES)
        feat = frontend(fitted)
        target = labels.index(row['label'])
        return feat, target, fitted
    except Exception as e:
        raise RuntimeError(f"Failed processing {row['full_path']}: {e}")


def build_features(rows, labels, cache_path=None):
    if cache_path is None:
        cache_path = OPTION_B_DATA / 'features_cache.npz'
    cache_file = Path(cache_path)
    if cache_file.exists():
        print(f"Loading precomputed features from {cache_file.name}...", flush=True)
        data = np.load(cache_file)
        splits = {
            'train': (data['train_x'], data['train_y'], data['train_w']),
            'val': (data['val_x'], data['val_y'], data['val_w']),
            'test': (data['test_x'], data['test_y'], data['test_w']),
        }
        print(f"Features loaded instantly: train {splits['train'][0].shape}, val {splits['val'][0].shape}, test {splits['test'][0].shape}", flush=True)
        return splits

    frontend = Frontend()
    splits = {}

    for split in ('train', 'val', 'test'):
        selected = [r for r in rows if r['split'] == split]
        n = len(selected)
        print(f"Extracting features for {split} ({n} audio clips)...", flush=True)
        t0 = time.perf_counter()

        features = np.zeros((n, 1, MELS, TIME_STEPS), dtype=np.float32)
        targets = np.zeros(n, dtype=np.int64)
        waves = np.zeros((n, SAMPLES), dtype=np.float32)

        def worker(idx):
            feat, tgt, wav = _process_single(selected[idx], labels, frontend)
            features[idx] = feat
            targets[idx] = tgt
            waves[idx] = wav

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(worker, range(n)))

        dt = time.perf_counter() - t0
        rate = n / dt
        print(f"{split} complete: {n} clips in {dt:.2f}s ({rate:.0f} clips/s). Feature shape: {features.shape}", flush=True)
        splits[split] = (features, targets, waves)

    try:
        np.savez_compressed(
            cache_file,
            train_x=splits['train'][0], train_y=splits['train'][1], train_w=splits['train'][2],
            val_x=splits['val'][0], val_y=splits['val'][1], val_w=splits['val'][2],
            test_x=splits['test'][0], test_y=splits['test'][1], test_w=splits['test'][2]
        )
        print(f"Saved feature cache to {cache_file.name} ({cache_file.stat().st_size / 1e6:.1f} MB)", flush=True)
    except Exception as e:
        print(f"Warning: could not save feature cache: {e}", flush=True)

    return splits
