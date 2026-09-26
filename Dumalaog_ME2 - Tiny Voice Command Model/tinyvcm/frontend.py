"""One NumPy frontend shared by training, Windows and ARM deployment."""
import numpy as np
from .config import SR, SAMPLES, HOP, NFFT, WINDOW, MELS


def _hz_mel(hz):
    hz = np.asarray(hz, dtype=np.float64)
    return np.where(hz < 1000, hz / (200 / 3), 15 + np.log(np.maximum(hz, 1) / 1000) / (np.log(6.4) / 27))


def _mel_hz(mel):
    return np.where(mel < 15, mel * (200 / 3), 1000 * np.exp((mel - 15) * (np.log(6.4) / 27)))


class Frontend:
    def __init__(self):
        edges = _mel_hz(np.linspace(_hz_mel(20), _hz_mel(8000), MELS + 2))
        fft_hz = np.linspace(0, SR / 2, NFFT // 2 + 1)
        ramps = edges[:, None] - fft_hz[None, :]
        widths = np.diff(edges)
        fb = np.maximum(0, np.minimum(-ramps[:-2] / widths[:-1, None], ramps[2:] / widths[1:, None]))
        self.fb = (fb * (2 / (edges[2:] - edges[:-2]))[:, None]).astype(np.float32)
        self.window = np.pad(np.hanning(WINDOW + 1)[:-1], ((NFFT-WINDOW)//2, (NFFT-WINDOW)//2)).astype(np.float32)

    def __call__(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) != SAMPLES:
            raise ValueError(f'Expected {SAMPLES} mono samples at {SR} Hz, got {len(audio)}')
        padded = np.pad(audio, (NFFT//2, NFFT//2), mode='reflect')
        frames = np.lib.stride_tricks.sliding_window_view(padded, NFFT)[::HOP]
        power = np.abs(np.fft.rfft(frames * self.window, axis=-1)) ** 2
        mel = np.log(self.fb @ power.T + 1e-6).astype(np.float32)
        if float(np.ptp(mel)) < 1e-5:
            return np.zeros((1, MELS, SAMPLES // HOP + 1), dtype=np.float32)
        mel = (mel - mel.mean()) / (mel.std(ddof=1) + 1e-6)
        return mel[None, :, :].astype(np.float32)


def fit_audio(audio):
    """Center-pad short utterances; reject long utterances rather than truncating words."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) > SAMPLES:
        raise ValueError('Utterance exceeds 1.5 seconds; use a shorter phrase or retrain with a longer window.')
    pad = SAMPLES - len(audio)
    return np.pad(audio, (pad//2, pad-pad//2))
