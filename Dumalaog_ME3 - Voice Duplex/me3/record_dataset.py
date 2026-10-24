"""Record the 8 canonical benchmark stimuli to data/bench/*.wav."""
from __future__ import annotations

import time
from pathlib import Path

from .config import load_config

DATA = Path(__file__).resolve().parent.parent / "data" / "bench"

PROMPTS = [
    ("hello", "Say: Hello"),
    ("what_time_is_it", "Say: What time is it?"),
    ("set_a_timer_5_minutes", "Say: Set a timer for five minutes"),
    ("tell_me_system_status", "Say: Tell me the system status"),
    ("stop", "Say: Stop"),
    ("repeat_that", "Say: Repeat that"),
    ("play_the_news", "Say: Play the news"),
    ("what_is_2_plus_2", "Say: What is two plus two?"),
]


def record_one(name: str, prompt: str, sr: int = 16000, timeout: float = 6.0):
    import sounddevice as sd
    import numpy as np
    print(f"\n[{name}] {prompt} ... 3s countdown")
    time.sleep(3)
    rec = sd.rec(int(sr * timeout), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    # trim trailing silence crudely
    arr = rec[:, 0]
    nonzero = np.nonzero(np.abs(arr) > 300)[0]
    if len(nonzero):
        arr = arr[max(0, nonzero[0] - 480):nonzero[-1] + 480]
    DATA.mkdir(parents=True, exist_ok=True)
    import wave
    with wave.open(str(DATA / f"{name}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(arr.tobytes())
    print(f"  saved {name}.wav ({len(arr)/sr:.2f}s)")


def main():
    cfg = load_config()
    sr = int(cfg["audio"]["sample_rate"])
    for name, prompt in PROMPTS:
        record_one(name, prompt, sr)
    print(f"\nDone. Files in {DATA}")


if __name__ == "__main__":
    main()
