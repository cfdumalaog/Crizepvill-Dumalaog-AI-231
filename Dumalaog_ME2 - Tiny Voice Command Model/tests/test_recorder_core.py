from __future__ import annotations

import csv

import numpy as np
import pytest
import soundfile as sf

from tinyvcm_model.config import SR, SAMPLES
from tinyvcm_model.recording_labels import RECORDER_PHRASES
from tinyvcm_model.recorder_core import save_recording_data


def test_desktop_recorder_core_saves_wav_and_exact_prompt(tmp_path):
    wave = (.15 * np.sin(2 * np.pi * 440 * np.arange(SR // 2) / SR)).astype(np.float32)
    phrase = RECORDER_PHRASES['LIGHT_OFF']

    status, (rate, saved) = save_recording_data(
        tmp_path, (SR, wave), 'person-11', 'quiet-near', 'LIGHT_OFF', True, phrase,
    )

    assert rate == SR
    assert saved.shape == (SAMPLES,)
    assert 'person-11' in status and 'LIGHT_OFF' in status
    manifest = tmp_path / 'data' / 'human' / 'manifest.csv'
    with manifest.open(newline='', encoding='utf-8') as stream:
        row = next(csv.DictReader(stream))
    assert row['suggested_phrase'] == phrase
    assert row['phrase_variant'] in {'canonical', 'v1', 'v2', 'v3'}
    loaded, saved_rate = sf.read(tmp_path / row['path'])
    assert saved_rate == SR
    assert loaded.shape == (SAMPLES,)


def test_desktop_recorder_core_rejects_missing_consent(tmp_path):
    with pytest.raises(ValueError, match='agrees'):
        save_recording_data(tmp_path, (SR, np.zeros(SR, dtype=np.float32)),
                            'person-11', 'quiet-near', 'LIGHT_OFF', False)
