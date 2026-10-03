"""Build a separate, integrity-checked Pi bundle for current Antigrav models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = DEPLOY / "dist" / "rpi5-antigrav-vcm-release"
DEFAULT_ARCHIVE = DEPLOY / "dist" / "TinyVCM_RPi5_Antigrav_RELEASE.zip"
SOURCES = {
    "antigrav_demo.py": ROOT / "antigrav_demo.py",
    "tinyvcm/__init__.py": ROOT / "tinyvcm" / "__init__.py",
    "tinyvcm/config.py": ROOT / "tinyvcm" / "config.py",
    "tinyvcm/devices.py": ROOT / "tinyvcm" / "devices.py",
    "tinyvcm/microphone.py": ROOT / "tinyvcm" / "microphone.py",
    "tinyvcm_antigrav/__init__.py": ROOT / "tinyvcm_antigrav" / "__init__.py",
    "tinyvcm_antigrav/config.py": ROOT / "tinyvcm_antigrav" / "config.py",
    "tinyvcm_antigrav/frontend.py": ROOT / "tinyvcm_antigrav" / "frontend.py",
    "tinyvcm_antigrav/media_engine.py": ROOT / "tinyvcm_antigrav" / "media_engine.py",
    "models/antigrav_optionb_int8.onnx": ROOT / "models" / "antigrav_optionb_int8.onnx",
    "models/antigrav_wake32_int8.onnx": ROOT / "models" / "antigrav_wake32_int8.onnx",
    "README.md": DEPLOY / "README-antigrav-rpi.md",
    "requirements-runtime.txt": DEPLOY / "requirements-antigrav-rpi.txt",
    "setup_rpi.sh": DEPLOY / "setup_antigrav_rpi5.sh",
    "verify_release.py": DEPLOY / "verify_antigrav_release.py",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def bundle_metadata() -> dict:
    from tinyvcm_antigrav.config import LABELS, WAKE_LABELS

    base = ROOT / "runs" / "antigrav-20260927-115145"
    info = json.loads((base / "run_info.json").read_text(encoding="utf-8"))
    wake_hash = digest(ROOT / "models" / "antigrav_wake32_int8.onnx")
    wake = None
    for candidate in sorted((ROOT / "runs").glob("antigrav-wake-32class-*"), reverse=True):
        for filename in ("antigrav_wake32_int8.onnx", "antigrav_optionb_int8.onnx"):
            model_file = candidate / "models" / filename
            if model_file.is_file() and digest(model_file) == wake_hash:
                wake = candidate
                break
        if wake:
            break
    wake_info_path = wake / "run_info.json" if wake else None
    wake_summary_path = wake / "models" / "export_summary.json" if wake else None
    wake_info = json.loads(wake_info_path.read_text(encoding="utf-8")) if wake_info_path and wake_info_path.exists() else {}
    wake_summary = json.loads(wake_summary_path.read_text(encoding="utf-8")) if wake_summary_path and wake_summary_path.exists() else {}
    return {
        "bundle": "TinyVCM Raspberry Pi 5 Antigrav",
        "schema_version": 1,
        "sample_rate_hz": 16000,
        "input_feature_shape": [1, 1, 40, 251],
        "default_model": "antigrav_wake32_int8.onnx",
        "scratch_model": "antigrav_optionb_int8.onnx",
        "scratch_model_labels": LABELS,
        "wake_model_labels": WAKE_LABELS,
        "scratch_training": {
            "run": base.name,
            "seed": info.get("seed", 231),
            "pretrained_weights_used": info.get("pretrained_weights_used", False),
            "model_sha256": digest(ROOT / "models" / "antigrav_optionb_int8.onnx"),
        },
        "wake_extension": {
            "run": wake.name if wake else None,
            "initialization": "warm-started from the scratch 31-class checkpoint",
            "speaker_validation": "single-speaker supplemental recordings; multi-speaker performance unverified",
            "wake_label": "WAKE_WORD",
            "model_sha256": wake_hash,
            "training_metadata_present": bool(wake_info),
            "reported_test_accuracy_int8": wake_summary.get("test_accuracy_int8_onnx"),
            "reported_test_split_provenance": "Not independently reconstructed; treat the saved score as experimental.",
            "reproducible_training_verified": False,
        },
        "limitations": [
            "The 32-class wake extension is experimental and not multi-speaker validated.",
            "The 31-class scratch model has no wake-word output.",
            "File/model smoke tests do not establish spoken-command or GPIO performance.",
        ],
    }


def safe_zip(archive_path: Path, files: list[str], stage: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted([*files, "model_metadata.json", "bundle_manifest.json"]):
            rel = PurePosixPath(relative)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe package path: {relative}")
            source = stage / Path(*rel.parts)
            archive.write(source, f"tinyvcm-rpi5/{relative}")
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        names = archive.namelist()
        if bad:
            raise ValueError(f"Corrupt ZIP member: {bad}")
        if len(names) != len(set(names)) or any(".." in PurePosixPath(n).parts for n in names):
            raise ValueError("ZIP contains duplicate or unsafe paths")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory and archive")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    default_archive = DEFAULT_ARCHIVE if output == DEFAULT_OUTPUT.resolve() else output.with_suffix(".zip")
    archive_path = (args.archive or default_archive).resolve()
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if archive_path.exists() and args.overwrite:
        archive_path.unlink()
    if output.exists() or archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite release output: {output} or {archive_path}")
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required release inputs missing:\n" + "\n".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="antigrav-rpi-stage-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        for relative, source in SOURCES.items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (stage / "model_metadata.json").write_text(
            json.dumps(bundle_metadata(), indent=2) + "\n", encoding="utf-8"
        )
        try:
            (stage / "bundle_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sha256": {
                            relative: digest(stage / relative)
                            for relative in sorted([*SOURCES.keys(), "model_metadata.json"])
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            subprocess.run([sys.executable, "verify_release.py"], cwd=stage, check=True)
            stage.rename(output)
            try:
                safe_zip(archive_path, list(SOURCES), output)
            except Exception:
                shutil.rmtree(output)
                raise
        except Exception:
            if archive_path.exists():
                archive_path.unlink()
            raise
    print(f"Built current Pi bundle: {output}")
    print(f"Built verified ZIP: {archive_path} ({archive_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
