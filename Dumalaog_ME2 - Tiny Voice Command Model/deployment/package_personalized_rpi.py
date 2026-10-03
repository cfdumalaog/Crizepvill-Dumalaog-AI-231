"""Build a versioned Raspberry Pi bundle for the personalized two-model VCM.

The desktop demo launcher is installed executable by the Pi installer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = Path(__file__).resolve().parent
CANDIDATE = DEPLOY / "current_vcm"
DEFAULT_OUTPUT = DEPLOY / "dist" / "dandan"
DEFAULT_ARCHIVE = DEPLOY / "dist" / "dandan-vcm-20260930.zip"

SOURCES = {
    "vcm_app.py": ROOT / "vcm_app.py",
    "assets/demo_melody_1.wav": ROOT / "assets" / "demo_melody_1.wav",
    "assets/demo_melody_2.wav": ROOT / "assets" / "demo_melody_2.wav",
    "scripts/restart_local_app.py": ROOT / "scripts" / "restart_local_app.py",
    "tinyvcm/__init__.py": ROOT / "tinyvcm" / "__init__.py",
    "tinyvcm/config.py": ROOT / "tinyvcm" / "config.py",
    "tinyvcm/audio_output.py": ROOT / "tinyvcm" / "audio_output.py",
    "tinyvcm/devices.py": ROOT / "tinyvcm" / "devices.py",
    "tinyvcm/microphone.py": ROOT / "tinyvcm" / "microphone.py",
    "tinyvcm/restart.py": ROOT / "tinyvcm" / "restart.py",
    "tinyvcm_runtime/__init__.py": ROOT / "tinyvcm_model" / "__init__.py",
    "tinyvcm_runtime/config.py": ROOT / "tinyvcm_model" / "config.py",
    "tinyvcm_runtime/frontend.py": ROOT / "tinyvcm_model" / "frontend.py",
    "tinyvcm_runtime/media_engine.py": ROOT / "tinyvcm_model" / "media_engine.py",
    "models/wake_binary_int8.onnx": CANDIDATE / "models" / "binary_wake_int8.onnx",
    "models/intent_31_int8.onnx": CANDIDATE / "models" / "intent_int8.onnx",
    "candidate_metadata.json": CANDIDATE / "metadata.json",
    "export_summary.json": CANDIDATE / "export_summary.json",
    "requirements-runtime.txt": DEPLOY / "requirements-runtime.txt",
    "README.md": DEPLOY / "README-personalized-rpi.md",
    "docs/RASPBERRY_PI_5_LED_WIRING.md": ROOT / "docs" / "RASPBERRY_PI_5_LED_WIRING.md",
    "verify_personalized_rpi.py": DEPLOY / "verify_personalized_rpi.py",
    "launch-vcm.sh": DEPLOY / "launch-vcm.sh",
    "start-vcm-demo.sh": DEPLOY / "start-vcm-demo.sh",
}


def sources_for_candidate(candidate: Path) -> dict[str, Path]:
    sources = dict(SOURCES)
    sources.update({
        "models/wake_binary_int8.onnx": candidate / "models" / "binary_wake_int8.onnx",
        "models/intent_31_int8.onnx": candidate / "models" / "intent_int8.onnx",
        "candidate_metadata.json": candidate / "metadata.json",
        "export_summary.json": candidate / "export_summary.json",
    })
    return sources

DEPLOYMENT = {
    "title": "ME2 - VCM on Raspberry Pi 5",
    "kind": "personalized_binary_wake_plus_31_intent",
    "port": 7860,
    "host": "127.0.0.1",
    "wake_threshold": 0.95,
    "wake_threshold_note": "User-requested operating threshold; validation-selected threshold remains in candidate_metadata.json.",
    "vad_threshold": 0.006,
    "inference_interval_sec": 0.12,
    "post_wake_timeout_sec": 10.0,
    "microphone_default": "automatic; no device index is hard-coded",
    "autostart": False,
    "gpio": True,
    "intent_confidence_threshold": 0.68,
    "intent_margin_threshold": 0.15,
    "live_weather": True,
    "wake_confirmations": 1,
    "gpio_led_channels_bcm": {"red": 17, "green": 27, "blue": 22},
    "gpio_led_header_pins_physical": {"red": 11, "green": 13, "blue": 15, "ground": 6},
    "gpio_series_resistor_ohms_per_channel": 330,
    "gpio_buzzer": False,
    "wake_model": "models/wake_binary_int8.onnx",
    "intent_model": "models/intent_31_int8.onnx",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def patch_intent_gate_config(payload: bytes, confidence: float, margin: float) -> bytes:
    """Apply release-specific action gates to the packaged runtime config only."""
    text = payload.decode("utf-8")
    for name, value in (("INTENT_CONFIDENCE_THRESHOLD", confidence),
                        ("INTENT_MARGIN_THRESHOLD", margin)):
        pattern = rf"(?m)^{name}\s*=\s*[^\r\n#]+"
        text, replacements = re.subn(pattern, f"{name} = {value:.8g}", text)
        if replacements != 1:
            raise ValueError(f"Expected exactly one {name} assignment in packaged config; found {replacements}")
    return text.encode("utf-8")


def safe_zip(archive_path: Path, stage: Path, members: list[str], prefix: str) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted(members):
            rel = PurePosixPath(relative)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe package path: {relative}")
            source = stage / Path(*rel.parts)
            info = zipfile.ZipInfo(f"{prefix}/{relative}")
            info.create_system = 3  # Preserve the Pi launcher's executable bit.
            info.external_attr = (0o100755 if relative in {"launch-vcm.sh", "start-vcm-demo.sh"} else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.read_bytes())
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Personalized Pi ZIP failed its CRC check")
        names = archive.namelist()
        if len(names) != len(set(names)) or any(".." in PurePosixPath(name).parts for name in names):
            raise ValueError("ZIP contains duplicate or unsafe paths")
        launcher = archive.getinfo(f"{prefix}/launch-vcm.sh")
        if launcher.create_system != 3 or (launcher.external_attr >> 16) & 0o777 != 0o755:
            raise ValueError("Pi launcher ZIP entry must preserve executable Unix permissions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--candidate-dir", type=Path, default=CANDIDATE)
    parser.add_argument("--port", type=int, default=DEPLOYMENT["port"])
    parser.add_argument("--gpio", action=argparse.BooleanOptionalAction, default=DEPLOYMENT["gpio"])
    parser.add_argument("--intent-confidence-threshold", type=float, default=DEPLOYMENT["intent_confidence_threshold"])
    parser.add_argument("--intent-margin-threshold", type=float, default=DEPLOYMENT["intent_margin_threshold"])
    parser.add_argument("--live-weather", action=argparse.BooleanOptionalAction, default=DEPLOYMENT["live_weather"])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    archive = args.archive.resolve()
    sources = sources_for_candidate(args.candidate_dir.resolve())
    if output.exists() or archive.exists():
        raise FileExistsError(f"Refusing to overwrite an existing release: {output} or {archive}")
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required personalized release input missing:\n" + "\n".join(missing))
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    for name, value in (("intent confidence", args.intent_confidence_threshold),
                        ("intent margin", args.intent_margin_threshold)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    release_config = dict(DEPLOYMENT)
    release_config.update({
        "port": args.port,
        "gpio": args.gpio,
        "intent_confidence_threshold": args.intent_confidence_threshold,
        "intent_margin_threshold": args.intent_margin_threshold,
        "live_weather": args.live_weather,
    })

    output.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="personalized-rpi-stage-", dir=output.parent) as temp_dir:
        stage = Path(temp_dir) / "bundle"
        stage.mkdir()
        for relative, source in sources.items():
            target = stage / Path(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = source.read_bytes()
            if relative in {"vcm_app.py", "verify_personalized_rpi.py", "tinyvcm/devices.py"}:
                payload = payload.replace(b"tinyvcm_model", b"tinyvcm_runtime")
            if relative == "tinyvcm_runtime/config.py":
                payload = (payload.replace(b"OPTION_B_DATA", b"COMMAND_DATA")
                           .replace(b"option_b", b"spoken_commands"))
            if relative == "tinyvcm/config.py":
                payload = payload.replace(b"tinyvcm_model", b"tinyvcm_runtime")
            if relative == "tinyvcm_runtime/__init__.py":
                payload = b'"""ME2 VCM runtime package."""\n'
            if relative == "tinyvcm_runtime/config.py":
                payload = patch_intent_gate_config(
                    payload, args.intent_confidence_threshold, args.intent_margin_threshold
                )
            target.write_bytes(payload)
            shutil.copystat(source, target)
        (stage / "deployment.json").write_text(json.dumps(release_config, indent=2) + "\n", encoding="utf-8")
        members = [*sources.keys(), "deployment.json"]
        (stage / "bundle_manifest.json").write_text(
            json.dumps({"schema_version": 1, "sha256": {name: digest(stage / Path(*PurePosixPath(name).parts)) for name in sorted(members)}}, indent=2) + "\n",
            encoding="utf-8",
        )
        members.append("bundle_manifest.json")
        output.mkdir(parents=True)
        for relative in members:
            source = stage / Path(*PurePosixPath(relative).parts)
            target = output / Path(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        try:
            verify_env = dict(__import__("os").environ)
            verify_env["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run([sys.executable, str(output / "verify_personalized_rpi.py")], cwd=output,
                           check=True, env=verify_env)
            safe_zip(archive, stage, members, output.name)
        except Exception:
            shutil.rmtree(output)
            if archive.exists():
                archive.unlink()
            raise
    print(f"Built release directory: {output}")
    print(f"Built ZIP: {archive} ({archive.stat().st_size:,} bytes, SHA-256 {digest(archive)})")


if __name__ == "__main__":
    main()
