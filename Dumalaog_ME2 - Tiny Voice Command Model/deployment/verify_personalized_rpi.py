"""Verify the personalized Pi bundle and benchmark its paired INT8 models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import time

import numpy as np
import onnxruntime as ort

from tinyvcm_model.config import (
    FEATURE_SHAPE, INTENT_CONFIDENCE_THRESHOLD, INTENT_MARGIN_THRESHOLD,
    LABELS, SAMPLES, SR,
)
from tinyvcm_model.frontend import Frontend


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest() -> dict:
    manifest = json.loads((ROOT / "bundle_manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("sha256")
    if manifest.get("schema_version") != 1 or not isinstance(files, dict) or not files:
        raise ValueError("Missing or unsupported bundle manifest")
    for name, expected in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path: {name}")
        path = (ROOT / Path(*relative.parts)).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Missing or out-of-bundle file: {name}")
        if sha256(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
    return manifest


def run_model(name: str, expected_classes: int, features: np.ndarray, frontend: Frontend, tone: np.ndarray) -> dict:
    path = ROOT / name
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if "CPUExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"CPU provider unavailable for {name}")
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if list(input_meta.shape[1:]) != list(FEATURE_SHAPE):
        raise ValueError(f"Unexpected input tensor for {name}: {input_meta.shape}")
    if output_meta.shape[-1] != expected_classes:
        raise ValueError(f"Unexpected output tensor for {name}: {output_meta.shape}")
    for batch in (features, frontend(tone)[None]):
        logits = session.run(None, {input_meta.name: batch})[0]
        if logits.shape != (1, expected_classes) or not np.isfinite(logits).all():
            raise ValueError(f"Non-finite or incorrectly shaped output from {name}: {logits.shape}")
    sample = frontend(tone)[None]
    for _ in range(10):
        session.run(None, {input_meta.name: sample})
    model_only = []
    end_to_end = []
    for _ in range(100):
        started = time.perf_counter()
        session.run(None, {input_meta.name: sample})
        model_only.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        current = frontend(tone)[None]
        session.run(None, {input_meta.name: current})
        end_to_end.append((time.perf_counter() - started) * 1000)
    percentile = lambda values, q: round(float(np.percentile(values, q)), 3)
    return {
        "file": name,
        "bytes": path.stat().st_size,
        "classes": expected_classes,
        "providers": session.get_providers(),
        "model_only_ms": {"p50": percentile(model_only, 50), "p95": percentile(model_only, 95)},
        "frontend_plus_model_ms": {"p50": percentile(end_to_end, 50), "p95": percentile(end_to_end, 95)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", action="store_true", help="Also verify a short capture from an attached microphone")
    args = parser.parse_args()
    manifest = verify_manifest()
    metadata = json.loads((ROOT / "candidate_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "export_summary.json").read_text(encoding="utf-8"))
    deployment = json.loads((ROOT / "deployment.json").read_text(encoding="utf-8"))
    if metadata["intent"]["classes"] != LABELS:
        raise ValueError("Packaged 31-class intent label order does not match runtime config")
    if metadata["wake"]["labels"] != ["NON_WAKE", "WAKE_WORD"]:
        raise ValueError("Wake model is not the expected binary classifier")
    if metadata["wake"]["model"]["sha256"] != sha256(ROOT / deployment["wake_model"]):
        raise ValueError("Packaged wake model does not match candidate metadata")
    if metadata["intent"]["model"]["sha256"] != sha256(ROOT / deployment["intent_model"]):
        raise ValueError("Packaged intent model does not match candidate metadata")
    if float(deployment["intent_confidence_threshold"]) != INTENT_CONFIDENCE_THRESHOLD:
        raise ValueError("Packaged intent confidence gate differs from deployment.json")
    if float(deployment["intent_margin_threshold"]) != INTENT_MARGIN_THRESHOLD:
        raise ValueError("Packaged intent margin gate differs from deployment.json")
    if float(summary["local_operating_threshold"]) != float(deployment["wake_threshold"]):
        raise ValueError("Wake operating threshold mismatch")

    frontend = Frontend()
    samples = np.zeros(SAMPLES, dtype=np.float32)
    tone = (0.08 * np.sin(2 * np.pi * 440 * np.arange(SAMPLES, dtype=np.float32) / SR)).astype(np.float32)
    features = frontend(samples)[None]
    if features.shape != (1, *FEATURE_SHAPE) or not np.isfinite(features).all():
        raise ValueError(f"Invalid frontend result: {features.shape}")
    results = [
        run_model(deployment["wake_model"], 2, features, frontend, tone),
        run_model(deployment["intent_model"], len(LABELS), features, frontend, tone),
    ]
    if args.audio:
        from tinyvcm.microphone import Microphone

        mic = Microphone()
        try:
            mic.start()
            chunk = mic.read(timeout=2.0)
            if chunk is None or not len(chunk) or not np.isfinite(chunk).all():
                raise RuntimeError("Microphone returned no valid audio samples")
            print(f"Microphone capture passed: {mic.name}; {len(chunk)} resampled samples")
        finally:
            mic.close()
    print(json.dumps({
        "integrity": "passed",
        "verified_files": len(manifest["sha256"]),
        "wake_threshold": deployment["wake_threshold"],
        "intent_labels": len(LABELS),
        "model_results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
