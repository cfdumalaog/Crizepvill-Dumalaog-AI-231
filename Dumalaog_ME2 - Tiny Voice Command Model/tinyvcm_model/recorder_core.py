"""Shared recording validation and manifest writing for web and desktop UIs."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import re
import tempfile
import uuid

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import SAMPLES, SR
from .frontend import fit_audio
from .recording_labels import (
    BACKGROUND_LABEL,
    RECORDER_LABELS,
    RECORDER_PHRASES,
    RECORDER_VARIATIONS,
    RECORDER_VARIANT_IDS,
    SILENCE_LABEL,
)


def ensure_manifest_columns(manifest: Path) -> list[str]:
    """Add prompt provenance columns without altering saved historical values."""
    manifest = Path(manifest)
    fields = ['path', 'speaker', 'condition', 'label', 'source_id', 'suggested_phrase', 'phrase_variant']
    if not manifest.exists() or manifest.stat().st_size == 0:
        return fields
    with manifest.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        existing = list(reader.fieldnames or [])
        rows = list(reader)
    if 'suggested_phrase' in existing and 'phrase_variant' in existing:
        return existing
    required = {'path', 'speaker', 'condition', 'label', 'source_id'}
    missing = required - set(existing)
    if missing:
        raise ValueError(f'Cannot safely update manifest columns; missing {sorted(missing)}.')
    for field in ('suggested_phrase', 'phrase_variant'):
        if field not in existing:
            existing.append(field)
    handle, temporary_name = tempfile.mkstemp(prefix=manifest.name + '.', suffix='.tmp', dir=manifest.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=existing)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return existing


def save_recording_data(project_root: Path, audio, speaker: str, condition: str,
                        label: str, consent: bool, phrase: str | None = None):
    """Validate and save a recording; raise ValueError for UI-specific display."""
    if not consent:
        raise ValueError('Confirm that the real speaker agrees to these local course-project recordings.')
    if not re.fullmatch(r'[A-Za-z0-9-]{2,30}', speaker or ''):
        raise ValueError('Use an anonymous speaker ID, for example person-01.')
    if condition not in ('quiet-near', 'fan-near', 'quiet-far'):
        raise ValueError('Choose a supported recording condition.')
    if label not in RECORDER_LABELS:
        raise ValueError('Choose a supported recording label.')
    if phrase is None:
        phrase = RECORDER_PHRASES[label]
    if label in RECORDER_VARIATIONS and phrase not in RECORDER_VARIATIONS[label]:
        raise ValueError('Choose one of the approved phrases for this intent.')
    if label not in RECORDER_VARIATIONS:
        phrase = RECORDER_PHRASES[label]
    if audio is None:
        raise ValueError('Record an utterance first.')
    rate, wave = audio
    rate = int(rate)
    if rate <= 0:
        raise ValueError('The microphone returned an invalid sample rate.')
    wave = np.asarray(wave)
    if np.issubdtype(wave.dtype, np.integer):
        wave = wave.astype(np.float32) / np.iinfo(wave.dtype).max
    else:
        wave = wave.astype(np.float32, copy=False)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('The recording is empty or invalid.')
    if np.mean(np.abs(wave) > .99) > .001:
        raise ValueError('The recording is clipping; move back or reduce microphone gain and try again.')
    divisor = math.gcd(rate, SR)
    wave = resample_poly(wave, SR // divisor, rate // divisor)
    if label not in (BACKGROUND_LABEL, SILENCE_LABEL):
        threshold = max(.004, float(np.max(np.abs(wave))) * .04)
        active = np.flatnonzero(np.abs(wave) > threshold)
        if not len(active):
            raise ValueError('No audible speech was found.')
        wave = wave[max(0, active[0] - 800):min(len(wave), active[-1] + 801)]
        if len(wave) > SAMPLES:
            raise ValueError('Speech is longer than the 2.5-second model window. Use a shorter phrase; words will not be cut off.')
    elif len(wave) < SAMPLES:
        raise ValueError('Record at least 2.5 seconds for background or silence.')
    else:
        wave = wave[:SAMPLES]
    wave = fit_audio(wave)

    project_root = Path(project_root)
    source = uuid.uuid4().hex
    directory = project_root / 'data' / 'human' / speaker / condition / label
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{source}.wav'
    sf.write(path, wave, SR, subtype='PCM_16')
    manifest = project_root / 'data' / 'human' / 'manifest.csv'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ensure_manifest_columns(manifest)
    with manifest.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if manifest.stat().st_size == 0:
            writer.writeheader()
        variant = RECORDER_VARIANT_IDS[label].get(phrase, 'canonical') if label in RECORDER_VARIANT_IDS else ''
        writer.writerow({
            'path': path.relative_to(project_root).as_posix(), 'speaker': speaker,
            'condition': condition, 'label': label, 'source_id': source,
            'suggested_phrase': phrase, 'phrase_variant': variant,
        })
    with manifest.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    count = sum(row['speaker'] == speaker and row['condition'] == condition and row['label'] == label for row in rows)
    status = f'Saved {count} take(s) for {speaker} / {condition} / {label}. Total dataset: {len(rows)} recordings.'
    return status, (SR, wave)
