"""Capture a candidate-only baseline without contacting or changing the Pi."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parents[1]
DESKTOP_COPY = Path.home() / "Desktop" / "dandan"
OUTPUT = PROJECT / "runs" / "classagreementvcm-audit-20261002" / "baseline.json"
sys.path.insert(0, str(PROJECT))
from tinyvcm_model import config  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def selected_hashes(paths: list[Path], base: Path) -> dict:
    return {
        str(path.relative_to(base)).replace("\\", "/"): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(paths)
        if path.is_file()
    }


def capture() -> dict:
    deployment = PROJECT / "deployment"
    active = deployment / "current_vcm"
    metadata = json_file(active / "metadata.json")
    intent_path = active / "models" / "intent_int8.onnx"
    wake_path = active / "models" / "binary_wake_int8.onnx"
    pi_receipt = json_file(deployment / "last_pi_deployment.json")
    pending_release = json_file(deployment / "latest_release.json")
    code_paths = [
        PROJECT / "vcm_app.py",
        PROJECT / "tinyvcm_model" / "config.py",
        PROJECT / "tinyvcm_model" / "model.py",
        PROJECT / "tinyvcm_model" / "frontend.py",
    ]
    active_paths = [path for path in active.rglob("*") if path.is_file()]

    desktop_paths = []
    if DESKTOP_COPY.is_dir():
        safe_names = {"Start-VCM.bat", "Start-VCM.ps1", "Sync-VCM.ps1", "Update-VCM.bat", "vcm_app.py", "candidate_metadata.json", "bundle_manifest.json", "deployment.json", "README.md"}
        desktop_paths = [path for path in DESKTOP_COPY.rglob("*") if path.is_file() and (path.name in safe_names or path.suffix.lower() == ".onnx")]

    status_result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    status_lines = [line for line in status_result.stdout.splitlines() if line.strip()]
    preexisting = [line for line in status_lines if "scripts/classagreementvcm/" not in line.replace("\\", "/") and "runs/classagreementvcm-audit-20261002/" not in line.replace("\\", "/")]
    status_digest = hashlib.sha256("\n".join(preexisting).encode("utf-8")).hexdigest()

    release_fields = ("status", "source_run", "zip_sha256", "intent_sha256", "wake_sha256")
    pi_fields = ("status", "source_run", "zip_sha256", "intent_sha256", "wake_sha256", "deployed_at")
    return {
        "captured_at_asia_manila": datetime.now(ZoneInfo("Asia/Manila")).isoformat(),
        "scope": "read-only candidate baseline; all active and Desktop deployment files preserved; Pi not contacted",
        "git": {
            "preexisting_dirty_entry_count": len(preexisting),
            "preexisting_status_sha256": status_digest,
            "candidate_audit_files_excluded_from_preexisting_status": True,
        },
        "active_pair": {
            "source_run": metadata.get("source_run"),
            "dataset_name": metadata.get("dataset_name"),
            "intent_classes": metadata.get("intent", {}).get("classes"),
            "wake_labels": metadata.get("wake", {}).get("labels"),
            "wake_operating_threshold": metadata.get("wake", {}).get("local_operating_threshold"),
            "intent_confidence_threshold": config.INTENT_CONFIDENCE_THRESHOLD,
            "intent_margin_threshold": config.INTENT_MARGIN_THRESHOLD,
            "frontend": {"sample_rate_hz": config.SR, "window_seconds": config.DURATION, "feature_shape": list(config.FEATURE_SHAPE)},
            "metadata_sha256": sha256(active / "metadata.json"),
            "intent_onnx_sha256": sha256(intent_path),
            "wake_onnx_sha256": sha256(wake_path),
            "all_current_vcm_files": selected_hashes(active_paths, PROJECT),
        },
        "active_code_sha256": selected_hashes(code_paths, PROJECT),
        "windows_desktop_copy": {
            "path": str(DESKTOP_COPY),
            "present": DESKTOP_COPY.is_dir(),
            "safe_launcher_and_model_file_hashes": selected_hashes(desktop_paths, DESKTOP_COPY) if desktop_paths else {},
            "secret_configuration_files_read_or_hashed": False,
        },
        "pi_last_recorded_receipt": {field: pi_receipt.get(field) for field in pi_fields if field in pi_receipt},
        "latest_local_release_state": {field: pending_release.get(field) for field in release_fields if field in pending_release},
        "pi_live_state": "NOT CHECKED; receipt is historical evidence only",
    }


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result = capture()
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "active_run": result["active_pair"]["source_run"],
        "active_wake_sha256": result["active_pair"]["wake_onnx_sha256"],
        "active_intent_sha256": result["active_pair"]["intent_onnx_sha256"],
        "preexisting_dirty_entries": result["git"]["preexisting_dirty_entry_count"],
        "desktop_copy_present": result["windows_desktop_copy"]["present"],
        "latest_release_state": result["latest_local_release_state"].get("status"),
        "pi_live_state": result["pi_live_state"],
    }, indent=2))
