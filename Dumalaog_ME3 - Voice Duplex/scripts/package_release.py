"""ME3 release packager (mirrors ME2's package_personalized_rpi.py).

Builds a self-contained, versioned release of the full-duplex agent from the
workspace source tree, writes a sha256 bundle manifest, zips it, and records
`deployment/latest_release.json` with status="prepared".

"Prepared" means: packaged and integrity-checked LOCALLY. It is NOT proof the
Pi received it - that is deployment/last_pi_deployment.json (written by
deploy_to_pi.py). This separation is the core of the ME2 local-dev workflow.

Usage:
    python scripts/package_release.py [--force]

--force re-packages even if the source fingerprint is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DIST = PROJECT / "dist"
DEPLOY = PROJECT / "deployment"
RELEASE_NAME = "me3_voice_duplex"

# Files/dirs that make up a deployable release (relative to PROJECT).
# Everything the Pi needs to run: code, config, installer, systemd units,
# dataset spec, and the plan. Excludes dev-only artifacts and heavy caches.
INCLUDE = [
    "me3",                 # all MCP servers + agent + benchmark + config loader
    "config",              # config/config.yaml (single source of truth)
    "web",                 # tracked presenter surface for local/Pi browser access
    "data/DATASET.md",     # class-group shared benchmark dataset spec
    "install.sh",          # idempotent Pi installer
    "deploy",              # systemd units
    "requirements.txt",    # pinned deps (if present)
    "plan",                # 10-page plan + generator (for the class record)
    "docs/PAPER_ALIGNMENT.md",
    "docs/PHILIPPINE_CALL_CENTER_PILOT.md",
    "docs/ELEVENLABS_INTEGRATION.md",
    "docs/ME3_MANUAL_ELI5_AND_DEEP.md",
    "docs/CHANNEL_INTEGRATIONS.md",
    "docs/DEMO_RUNBOOK.md",
    "docs/WORKSTATION_FIRST_RELEASE.md",
    "scripts/fdb_v15_eval.py",
    "scripts/check_paper_alignment.py",
    "scripts/pi_audio_preflight.py",
    "scripts/injection_eval.py",
    "scripts/workstation_acceptance.py",
    "Building AI Voice Agents - Summary.pdf",
    "FULL-DUPLEX-BENCH V1.5 - EVALUATING OVERLAP.pdf",
    "refs/FDB_paper_2503.04721.pdf",
    "refs/full-duplex-bench/v1_v1.5/evaluation/instruction/behavior.txt",
    "README.md",
]

# Never ship these.
EXCLUDE_SUFFIXES = {".pyc", ".log", ".tmp"}
EXCLUDE_DIRS = {"__pycache__", ".venv", "archive", "dist", "runs", ".git"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_release_files() -> list[Path]:
    files = []
    for rel in INCLUDE:
        base = PROJECT / rel
        if base.is_file():
            files.append(base)
        elif base.is_dir():
            for p in sorted(base.rglob("*")):
                if not p.is_file():
                    continue
                if p.suffix in EXCLUDE_SUFFIXES:
                    continue
                if any(part in EXCLUDE_DIRS for part in p.relative_to(base).parts):
                    continue
                files.append(p)
    # de-dup + stable order
    seen, out = set(), []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def source_fingerprint(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(PROJECT).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256(p).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def package(force: bool = False) -> dict:
    files = iter_release_files()
    if not files:
        raise FileNotFoundError("No release files found - is the ME3 tree intact?")

    fp = source_fingerprint(files)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    release_id = f"me3-{stamp}-{fp[:8]}"

    # Reuse an existing prepared release if the fingerprint is unchanged.
    latest_path = DEPLOY / "latest_release.json"
    if not force and latest_path.is_file():
        prev = json.loads(latest_path.read_text(encoding="utf-8"))
        if prev.get("source_fingerprint") == fp and prev.get("status") == "prepared":
            print(f"Source unchanged (fingerprint {fp[:12]}...) - reusing {prev['release_id']}")
            return prev

    stage = DIST / release_id / RELEASE_NAME
    stage.mkdir(parents=True, exist_ok=True)

    manifest = {"release_id": release_id, "created_utc": stamp, "files": {}}
    for p in files:
        rel = p.relative_to(PROJECT).as_posix()
        dest = stage / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        manifest["files"][rel] = sha256(dest)

    (stage / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Zip the release for transfer.
    zip_path = DIST / f"{release_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(DIST / release_id))

    release = {
        "release_id": release_id,
        "status": "prepared",
        "created_utc": stamp,
        "source_fingerprint": fp,
        "file_count": len(files),
        "archive": f"dist/{release_id}/{RELEASE_NAME}",
        "zip": f"dist/{release_id}.zip",
        "zip_sha256": sha256(zip_path),
        "target": "rpi",
        "note": "Packaged + integrity-checked locally. NOT yet on the Pi.",
    }
    DEPLOY.mkdir(exist_ok=True)
    write_json(latest_path, release)
    print(f"Prepared release {release_id} ({len(files)} files) -> {zip_path.name}")
    return release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-package even if the source fingerprint is unchanged")
    args = parser.parse_args()
    try:
        release = package(force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(release, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
