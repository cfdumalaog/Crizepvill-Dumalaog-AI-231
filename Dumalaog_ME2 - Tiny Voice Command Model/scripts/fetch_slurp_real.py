"""Fetch, verify, and extract official SLURP real audio from Zenodo.

Audio source: https://github.com/pswietojanski/slurp (DOI 10.18653/v1/2020.emnlp-main.588)
Zenodo record: https://zenodo.org/records/4274930/files/slurp_real.tar.gz
License: CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)

This script implements:
1. Multi-threaded parallel chunk download with resume capability.
2. File integrity check (expected size 3,918,185,662 bytes).
3. Extraction to data/external/slurp/audio/slurp_real/ without tracking raw audio in Git.
"""
from __future__ import annotations

import concurrent.futures
from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLURP_DIR = PROJECT_ROOT / 'data' / 'external' / 'slurp'
AUDIO_DIR = SLURP_DIR / 'audio'
TAR_PATH = SLURP_DIR / 'slurp_real.tar.gz'
CHUNKS_DIR = SLURP_DIR / '.chunks'
ZENODO_URL = 'https://zenodo.org/records/4274930/files/slurp_real.tar.gz'
EXPECTED_SIZE = 3918185662
CHUNK_SIZE = 32 * 1024 * 1024  # 32 MB per chunk (~117 chunks total)
MAX_WORKERS = 12


def get_remote_file_size(url: str) -> int:
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get('Content-Length', 0))


def download_chunk(url: str, chunk_idx: int, start_byte: int, end_byte: int, chunk_file: Path) -> int:
    if chunk_file.exists() and chunk_file.stat().st_size == (end_byte - start_byte + 1):
        return chunk_file.stat().st_size

    temp_file = chunk_file.with_suffix('.tmp')
    headers = {'Range': f'bytes={start_byte}-{end_byte}'}
    req = urllib.request.Request(url, headers=headers)
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()
                if len(data) != (end_byte - start_byte + 1):
                    raise IOError(f"Incomplete read: got {len(data)}, expected {end_byte - start_byte + 1}")
                with temp_file.open('wb') as f:
                    f.write(data)
                temp_file.replace(chunk_file)
                return len(data)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return 0


def fetch_slurp_archive():
    SLURP_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if TAR_PATH.exists() and TAR_PATH.stat().st_size == EXPECTED_SIZE:
        print(f"[OK] Archive already exists and matches expected size: {TAR_PATH} ({TAR_PATH.stat().st_size:,} bytes)")
        return

    print("=" * 80)
    print("  FETCHING OFFICIAL SLURP REAL AUDIO ARCHIVE (CC BY-NC 4.0)")
    print(f"  URL: {ZENODO_URL}")
    print(f"  Target: {TAR_PATH} (Expected: {EXPECTED_SIZE:,} bytes / ~3.65 GB)")
    print(f"  Workers: {MAX_WORKERS} parallel threads | Chunk size: {CHUNK_SIZE // (1024*1024)} MB")
    print("=" * 80)

    # 1. Plan chunks
    total_size = EXPECTED_SIZE
    num_chunks = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunks = []
    for idx in range(num_chunks):
        start = idx * CHUNK_SIZE
        end = min((idx + 1) * CHUNK_SIZE - 1, total_size - 1)
        chunk_file = CHUNKS_DIR / f"chunk_{idx:04d}_{start}_{end}.part"
        chunks.append((idx, start, end, chunk_file))

    # Check already downloaded chunks
    downloaded_bytes = sum(c[3].stat().st_size for c in chunks if c[3].exists())
    print(f"Existing progress: {downloaded_bytes / (1024*1024):,.1f} MB / {total_size / (1024*1024):,.1f} MB ({downloaded_bytes / total_size * 100:.1f}%)")

    t0 = time.time()
    last_print = t0
    completed_chunks = sum(1 for c in chunks if c[3].exists() and c[3].stat().st_size == (c[2] - c[1] + 1))

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(download_chunk, ZENODO_URL, idx, start, end, chunk_file): idx
            for idx, start, end, chunk_file in chunks
        }
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            try:
                future.result()
                completed_chunks += 1
                curr_time = time.time()
                if curr_time - last_print >= 5.0 or completed_chunks == num_chunks:
                    current_bytes = sum(c[3].stat().st_size for c in chunks if c[3].exists())
                    elapsed = curr_time - t0
                    speed = (current_bytes - downloaded_bytes) / elapsed if elapsed > 0 else 0
                    pct = current_bytes / total_size * 100
                    rem_bytes = total_size - current_bytes
                    eta = rem_bytes / speed if speed > 0 else 0
                    print(f"  Progress: {current_bytes/(1024*1024):,.1f}/{total_size/(1024*1024):,.1f} MB ({pct:5.1f}%) | "
                          f"Chunks: {completed_chunks}/{num_chunks} | "
                          f"Speed: {speed/(1024*1024):.2f} MB/s | ETA: {eta/60:.1f} min")
                    last_print = curr_time
            except Exception as e:
                print(f"[ERROR] Chunk {idx} failed: {e}", file=sys.stderr)
                raise

    print("\nAll chunks downloaded successfully. Assembling target archive...")
    with TAR_PATH.open('wb') as out_f:
        for idx, start, end, chunk_file in chunks:
            with chunk_file.open('rb') as in_f:
                shutil.copyfileobj(in_f, out_f, length=16*1024*1024)

    assert TAR_PATH.stat().st_size == EXPECTED_SIZE, (
        f"Size mismatch: got {TAR_PATH.stat().st_size}, expected {EXPECTED_SIZE}"
    )
    print(f"[OK] Assembled archive: {TAR_PATH} ({TAR_PATH.stat().st_size:,} bytes)")

    # Clean up chunks
    shutil.rmtree(CHUNKS_DIR, ignore_errors=True)
    print("[OK] Cleaned up temporary download chunks.")


def extract_slurp_archive():
    target_extract = AUDIO_DIR / 'slurp_real'
    if target_extract.exists() and any(target_extract.glob('*.flac')):
        existing_flac = len(list(target_extract.glob('*.flac')))
        print(f"[OK] SLURP real audio already extracted: {existing_flac:,} FLAC files in {target_extract}")
        return

    print("=" * 80)
    print(f"  EXTRACTING SLURP REAL AUDIO TO: {AUDIO_DIR}")
    print("=" * 80)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with tarfile.open(TAR_PATH, 'r:gz') as tar:
        tar.extractall(path=AUDIO_DIR)
    dt = time.time() - t0
    flac_count = len(list((AUDIO_DIR / 'slurp_real').glob('*.flac')))
    print(f"[OK] Extracted {flac_count:,} FLAC files in {dt:.1f}s to {AUDIO_DIR / 'slurp_real'}")


def main():
    fetch_slurp_archive()
    extract_slurp_archive()


if __name__ == '__main__':
    main()
