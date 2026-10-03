"""Offline integrity, label-map, ONNX, and frontend smoke checks for Pi releases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import time

import numpy as np
import onnxruntime as ort

from tinyvcm_antigrav.config import FEATURE_SHAPE, LABELS, SAMPLES, SR, WAKE_LABELS
from tinyvcm_antigrav.frontend import Frontend


ROOT = Path(__file__).resolve().parent


def verify_hashes() -> dict:
    manifest_path = ROOT / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("sha256")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Bundle manifest has no SHA-256 file list")
    for relative, expected in entries.items():
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe manifest path: {relative}")
        path = (ROOT / Path(*rel.parts)).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Missing or out-of-bundle file: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {relative}")
    return manifest


def run_smoke(model_name: str, labels: list[str], frontend: Frontend, samples: np.ndarray) -> dict:
    path = ROOT / "models" / model_name
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    if "CPUExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"CPU provider unavailable for {model_name}")
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if len(input_meta.shape) != 4 or input_meta.shape[1:] != list(FEATURE_SHAPE):
        raise ValueError(f"{model_name} input shape {input_meta.shape} does not match [batch, {FEATURE_SHAPE}]")
    if len(output_meta.shape) != 2 or output_meta.shape[-1] != len(labels):
        raise ValueError(f"{model_name} output shape {output_meta.shape} does not end in {len(labels)} classes")

    waveforms = [np.zeros(SAMPLES, dtype=np.float32), samples]
    feature_batches = [frontend(wave)[None] for wave in waveforms]
    if any(x.shape != (1, *FEATURE_SHAPE) or not np.isfinite(x).all() for x in feature_batches):
        raise ValueError("Frontend produced malformed or non-finite features")

    for features in feature_batches:
        logits = session.run(None, {input_meta.name: features})[0]
        if logits.shape != (1, len(labels)) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid model output from {model_name}: {logits.shape}")

    # Measure the model-only CPU path for evidence; this does not include audio capture.
    features = feature_batches[-1]
    for _ in range(5):
        session.run(None, {input_meta.name: features})
    timings = []
    for _ in range(25):
        start = time.perf_counter()
        session.run(None, {input_meta.name: features})
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "model": model_name,
        "bytes": path.stat().st_size,
        "classes": len(labels),
        "wake_index": labels.index("WAKE_WORD") if "WAKE_WORD" in labels else None,
        "p50_cpu_ms": round(statistics.median(timings), 3),
        "p95_cpu_ms": round(float(np.percentile(timings, 95)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", action="store_true", help="Also open the default microphone for a short capture check")
    args = parser.parse_args()

    manifest = verify_hashes()
    metadata = json.loads((ROOT / "model_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("wake_model_labels") != WAKE_LABELS or metadata.get("scratch_model_labels") != LABELS:
        raise ValueError("Model metadata label order does not match the bundled runtime")

    frontend = Frontend()
    time_axis = np.arange(SAMPLES, dtype=np.float32) / SR
    tone = (0.1 * np.sin(2 * np.pi * 440 * time_axis)).astype(np.float32)
    results = [
        run_smoke("antigrav_optionb_int8.onnx", LABELS, frontend, tone),
        run_smoke("antigrav_wake32_int8.onnx", WAKE_LABELS, frontend, tone),
    ]
    if args.audio:
        from tinyvcm.microphone import Microphone

        microphone = Microphone(hop=0.15)
        try:
            microphone.start()
            chunk = microphone.read(timeout=2.0)
            if chunk is None or not len(chunk) or not np.isfinite(chunk).all():
                raise RuntimeError("Microphone opened but returned no valid audio")
            print(f"Microphone capture OK: {microphone.name}; {len(chunk)} samples; peak={np.max(np.abs(chunk)):.4f}")
        finally:
            microphone.close()

    print(json.dumps({"integrity": "passed", "verified_files": len(manifest["sha256"]), "models": results}, indent=2))


if __name__ == "__main__":
    main()
