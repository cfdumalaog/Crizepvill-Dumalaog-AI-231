# -*- coding: utf-8 -*-
"""Curate a human-voice stimulus set for the ME3 FDB benchmark.

Reads ME2 recorded human WAVs (READ-ONLY), copies selected short
utterances into ME3 data/bench/, and writes data/bench/human_manifest.json
with full provenance. Never modifies ME2 originals.
"""
import csv
import hashlib
import io
import json
import shutil
import sys
import wave
from pathlib import Path

ME3 = Path(__file__).resolve().parents[1]
ME2 = ME3.parent / "Dumalaog_ME2 - Tiny Voice Command Model"
MANIFEST = ME2 / "data" / "human" / "manifest.csv"
OUT_DIR = ME3 / "data" / "bench"
OUT_MAN = OUT_DIR / "human_manifest.json"

LATENCY_LABELS = [
    "question_time", "question_weather", "TIME", "WEATHER",
    "LIGHT_ON", "play_music", "TIMER_10s", "volume_up",
    "media_pause", "CALL",
]
N_LATENCY = 12
N_STOP = 4
SR_WANT = 16000


def wav_dur(p: Path) -> float:
    with wave.open(str(p), "rb") as w:
        return w.getnframes() / float(w.getframerate()), w.getframerate(), w.getnchannels(), w.getsampwidth()


def main() -> int:
    if not MANIFEST.exists():
        print(f"missing ME2 manifest: {MANIFEST}")
        return 1
    rows = list(csv.DictReader(io.open(MANIFEST, encoding="utf-8-sig")))
    print(f"ME2 human manifest rows: {len(rows)}")

    picked_lat: dict[str, list[dict]] = {lb: [] for lb in LATENCY_LABELS}
    picked_stop_all: list[dict] = []

    for r in rows:
        if r.get("condition") != "quiet-near":
            continue
        src = ME2 / r["path"]
        if not src.exists():
            continue
        try:
            dur, sr, ch, sw = wav_dur(src)
        except Exception as exc:
            print(f"skip {r['path']}: {exc}")
            continue
        if sr != SR_WANT or ch != 1 or sw != 2:
            print(f"skip (format {sr}/{ch}/{sw}): {r['path']}")
            continue
        lb = r["label"]
        if lb == "STOP":
            if 0.8 <= dur <= 3.0:
                picked_stop_all.append({"row": r, "src": src, "dur": dur})
        elif lb in LATENCY_LABELS:
            if 1.0 <= dur <= 4.0:
                picked_lat[lb].append({"row": r, "src": src, "dur": dur})

    # Speaker-diverse pick: seeded shuffle, 1 per speaker first, then
    # allow a 2nd take from the same speaker, then fill regardless.
    import random
    rng = random.Random(20261020)

    def pick(cands, n):
        cands = sorted(cands, key=lambda c: (c["row"]["speaker"],
                                             c["src"].name))
        rng.shuffle(cands)
        used: dict[str, int] = {}
        out: list[dict] = []
        for c in cands:
            if len(out) >= n:
                break
            if used.get(c["row"]["speaker"], 0) == 0:
                out.append(c)
                used[c["row"]["speaker"]] = 1
        for c in cands:
            if len(out) >= n:
                break
            if c not in out and used.get(c["row"]["speaker"], 0) < 2:
                out.append(c)
                used[c["row"]["speaker"]] = used.get(c["row"]["speaker"], 0) + 1
        seen = {c["src"] for c in out}
        for c in cands:
            if len(out) >= n:
                break
            if c["src"] not in seen:
                out.append(c)
                seen.add(c["src"])
        return out

    all_lat = [c for lb in LATENCY_LABELS for c in picked_lat[lb]]
    rng.shuffle(all_lat)
    latency_picks = pick(all_lat, N_LATENCY)
    stop_picks = pick(picked_stop_all, N_STOP)
    print(f"latency picks: {len(latency_picks)}, stop picks: {len(stop_picks)}")
    entries = []
    # remove stale hum_*.wav from previous curation runs
    for old_wav in OUT_DIR.glob("hum_*.wav"):
        old_wav.unlink()
    for kind, picks in (("latency", latency_picks), ("stop", stop_picks)):
        for i, p in enumerate(picks):
            r, src, dur = p["row"], p["src"], p["dur"]
            h = hashlib.sha256(src.read_bytes()).hexdigest()
            slug = "".join(c if c.isalnum() else "_" for c in r["label"])[:12]
            name = f"hum_{kind}_{slug}_{r['speaker']}_{h[:6]}"
            dst = OUT_DIR / f"{name}.wav"
            shutil.copy2(src, dst)
            entries.append({
                "name": name,
                "wav": f"data/bench/{name}.wav",
                "kind": kind,
                "label": r["label"],
                "speaker": r["speaker"],
                "condition": r["condition"],
                "duration_s": round(dur, 3),
                "source_me2_path": r["path"],
                "source_sha256": h,
                "provenance": "ME2 human recordings (user-consented, local dev)",
            })
            print(f"  {kind:8s} {name}.wav  ({r['speaker']}, {r['label']}, {dur:.2f}s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with io.open(OUT_MAN, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"source": "ME2 human recordings", "entries": entries},
                  f, indent=2, ensure_ascii=False)
    print(f"wrote {OUT_MAN} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
