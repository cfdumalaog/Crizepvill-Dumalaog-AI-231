import numpy as np
import soundfile as sf

from tinyvcm_model.config import LABELS
from tinyvcm_model.data import _process_single
from tinyvcm_model.frontend import Frontend


def test_option_b_loader_resamples_twelve_khz_source_to_model_rate(tmp_path):
    source = tmp_path / "optionb_12khz.wav"
    time = np.arange(12_000, dtype=np.float32) / 12_000
    audio = 0.1 * np.sin(2 * np.pi * 440 * time)
    sf.write(source, audio, 12_000, subtype="PCM_16")

    features, target, fitted_audio = _process_single(
        {"full_path": str(source), "label": LABELS[0]}, LABELS, Frontend()
    )

    assert fitted_audio.shape == (40_000,)
    assert features.shape == (1, 40, 251)
    assert target == 0
