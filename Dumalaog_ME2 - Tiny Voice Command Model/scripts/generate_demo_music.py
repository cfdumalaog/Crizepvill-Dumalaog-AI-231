"""Generate two original, short, loopable local tunes for offline media demos."""
from pathlib import Path
import numpy as np
import soundfile as sf

OUT = Path(__file__).resolve().parents[1] / 'assets'
SR = 16000


def tune(notes, path):
    parts = []
    for freq in notes:
        n = SR // 2
        t = np.arange(n, dtype=np.float32) / SR
        if freq:
            tone = (np.sin(2 * np.pi * freq * t) + 0.27 * np.sin(4 * np.pi * freq * t))
            envelope = np.minimum(1.0, t / 0.025) * np.minimum(1.0, (0.5 - t) / 0.05)
            parts.append((0.12 * tone * envelope).astype(np.float32))
        else:
            parts.append(np.zeros(n, dtype=np.float32))
    sf.write(path, np.concatenate(parts), SR, subtype='PCM_16')


def main():
    OUT.mkdir(exist_ok=True)
    tune([261.63, 329.63, 392.00, 523.25, 392.00, 329.63, 293.66, 261.63,
          261.63, 329.63, 392.00, 523.25, 587.33, 523.25, 392.00, 329.63], OUT / 'demo_melody_1.wav')
    tune([220.00, 277.18, 329.63, 440.00, 329.63, 277.18, 246.94, 220.00,
          246.94, 293.66, 369.99, 493.88, 369.99, 293.66, 277.18, 246.94], OUT / 'demo_melody_2.wav')
    for path in sorted(OUT.glob('demo_melody_*.wav')):
        print(f'{path.name}: {path.stat().st_size} bytes')


if __name__ == '__main__':
    main()
