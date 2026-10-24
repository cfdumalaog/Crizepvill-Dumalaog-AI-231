"""Verify the benchmark dataset meets the standard (sr/channels/bit/duration)."""
from __future__ import annotations

import wave
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "bench"
EXPECTED = ["hello", "what_time_is_it", "set_a_timer_5_minutes",
            "tell_me_system_status", "stop", "repeat_that",
            "play_the_news", "what_is_2_plus_2"]


def main():
    ok = True
    print(f"Checking {DATA}\n")
    for name in EXPECTED:
        p = DATA / f"{name}.wav"
        if not p.exists():
            print(f"  MISSING  {name}.wav"); ok = False; continue
        with wave.open(str(p), "rb") as w:
            sr = w.getframerate(); ch = w.getnchannels()
            sw = w.getsampwidth(); n = w.getnframes()
        dur = n / sr
        good = (sr == 16000 and ch == 1 and sw == 2 and 0.3 <= dur <= 8.0)
        mark = "OK " if good else "BAD"
        if not good:
            ok = False
        print(f"  [{mark}] {name}.wav  sr={sr} ch={ch} bits={sw*8} dur={dur:.2f}s")
    print("\nDataset standard:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
