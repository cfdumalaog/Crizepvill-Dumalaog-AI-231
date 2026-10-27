"""Run the ME3 workstation-first acceptance gate.

This is the required promotion step before a release may be sent to the
Raspberry Pi.  It runs locally on the Windows development machine and does
not contact the Pi or place external side effects.  The result is recorded in
``deployment/workstation_acceptance.json`` and is bound to the exact release
ID and source fingerprint produced by ``package_release.py``.

The gate intentionally separates:

* local software/demo readiness: tested here and eligible for promotion; and
* Pi hardware readiness: checked later with ``pi_audio_preflight.py``.

Usage from the ME3 project directory::

    python scripts/workstation_acceptance.py
    python scripts/workstation_acceptance.py --skip-audio

The default gate checks the shared environment, all ME3 tests, Python and
Bash syntax, MCP/audio smoke, paper alignment, repository whitespace, and a
fresh integrity-checked release package.  ``--skip-audio`` is for a headless
machine; it does not make a physical audio claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEPLOY = PROJECT / "deployment"
REPORT = DEPLOY / "workstation_acceptance.json"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|auth[_-]?token|password|secret)\s*[:=]\s*\S+"
)


def _safe_tail(value: str, limit: int = 800) -> str:
    """Keep evidence useful without persisting credential-like values."""
    value = _SECRET_RE.sub(r"\1=[REDACTED]", value)
    return value[-limit:]


def _run(name: str, command: list[str], *, cwd: Path | None = None,
         timeout: int = 900) -> dict:
    started = time.perf_counter()
    rendered = " ".join(command)
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd or PROJECT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=os.environ.copy(),
        )
        status = "pass" if result.returncode == 0 else "fail"
        detail = {
            "name": name,
            "status": status,
            "returncode": result.returncode,
            "duration_s": round(time.perf_counter() - started, 3),
            "command": rendered,
            "stdout_tail": _safe_tail(result.stdout),
            "stderr_tail": _safe_tail(result.stderr),
        }
    except FileNotFoundError as exc:
        detail = {
            "name": name,
            "status": "fail",
            "returncode": None,
            "duration_s": round(time.perf_counter() - started, 3),
            "command": rendered,
            "stdout_tail": "",
            "stderr_tail": f"{type(exc).__name__}: {exc}",
        }
    except subprocess.TimeoutExpired as exc:
        detail = {
            "name": name,
            "status": "fail",
            "returncode": None,
            "duration_s": round(time.perf_counter() - started, 3),
            "command": rendered,
            "stdout_tail": _safe_tail(str(exc.stdout or "")),
            "stderr_tail": f"timeout after {timeout}s",
        }
    print(f"  [{detail['status'].upper():4}] {name} ({detail['duration_s']}s)")
    if detail["status"] != "pass":
        tail = detail["stderr_tail"] or detail["stdout_tail"]
        if tail:
            print(f"        {tail.replace(chr(10), ' ')[:500]}")
    return detail


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package() -> dict:
    """Package the already-tested source and return its verified metadata."""
    from scripts.package_release import package

    release = package(force=True)
    zip_path = PROJECT / release["zip"]
    if not zip_path.is_file():
        raise FileNotFoundError(f"release archive missing: {zip_path}")
    actual = _sha256(zip_path)
    if actual != release.get("zip_sha256"):
        raise ValueError("release archive hash changed after packaging")
    return release


def _required_presentable_files() -> list[str]:
    return [
        "README.md",
        "web/index.html",
        "docs/CHANNEL_INTEGRATIONS.md",
        "docs/DEMO_RUNBOOK.md",
        "docs/ME3_MANUAL_ELI5_AND_DEEP.md",
        "docs/PHILIPPINE_CALL_CENTER_PILOT.md",
        "docs/acm/ME3_Voice_Duplex_ACM.tex",
        "docs/acm/ME3_Voice_Duplex_ACM.pdf",
        "docs/presentation/ME3_Voice_Duplex_Presentation_v8.pptx",
    ]


def validate_presentable_files() -> dict:
    missing = [rel for rel in _required_presentable_files()
               if not (PROJECT / rel).is_file()]
    return {
        "name": "presentable_artifacts",
        "status": "pass" if not missing else "fail",
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="skip the PortAudio device query; use only on a headless machine",
    )
    args = parser.parse_args()

    py = sys.executable
    checks: list[dict] = []
    print("ME3 workstation-first acceptance gate")
    print(f"  project: {PROJECT}")
    print("  policy : only this accepted release may be promoted to the Pi")

    checks.append(_run(
        "shared_environment",
        [py, "scripts/dev_env_check.py"],
    ))
    checks.append(_run("pytest", [py, "-m", "pytest", "tests", "-q"],
                      timeout=1200))
    checks.append(_run(
        "python_compile",
        [py, "-m", "compileall", "-q", "me3", "scripts"],
    ))
    checks.append(_run("bash_syntax", ["bash", "-n", "install.sh"]))
    smoke = [py, "scripts/local_smoke_test.py"]
    if not args.skip_audio:
        smoke.append("--with-audio")
    checks.append(_run("mcp_and_audio_smoke", smoke))
    checks.append(_run(
        "paper_alignment",
        [py, "scripts/check_paper_alignment.py"],
    ))
    checks.append(_run(
        "git_whitespace",
        ["git", "diff", "--check", "--", "Dumalaog_ME3 - Voice Duplex"],
        cwd=PROJECT.parent,
    ))

    artifact_check = validate_presentable_files()
    checks.append(artifact_check)
    print(f"  [{artifact_check['status'].upper():4}] presentable_artifacts")
    if artifact_check.get("missing"):
        print(f"        missing: {', '.join(artifact_check['missing'])}")

    failed = [c["name"] for c in checks if c.get("status") != "pass"]
    release = None
    package_error = None
    if not failed:
        try:
            release = _package()
            print(f"  [PASS] package_integrity ({release['release_id']})")
        except Exception as exc:  # noqa: BLE001 - report gate failure
            package_error = f"{type(exc).__name__}: {exc}"
            print(f"  [FAIL] package_integrity ({package_error})")
            failed.append("package_integrity")
    else:
        print("  [SKIP] package_integrity (earlier checks failed)")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report = {
        "schema": 1,
        "acceptance_id": f"workstation-{stamp}",
        "status": "pass" if not failed else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": os.environ.get("COMPUTERNAME", "unknown"),
        "python": sys.version.split()[0],
        "mode": "windows_workstation_only",
        "external_side_effects": False,
        "pi_contacted": False,
        "promotion_policy": "workstation_acceptance_required",
        "release_id": release.get("release_id") if release else None,
        "source_fingerprint": release.get("source_fingerprint") if release else None,
        "zip_sha256": release.get("zip_sha256") if release else None,
        "checks": checks,
        "failed_checks": failed,
        "package_error": package_error,
        "pi_hardware_note": (
            "This gate proves workstation software/demo readiness only. "
            "Run Pi ALSA/PortAudio preflight and a human hardware turn "
            "before enabling the Pi service."
        ),
        "pi_deploy_allowed": not failed and release is not None,
    }
    DEPLOY.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Acceptance report: {REPORT}")
    if failed:
        print(f"RESULT: FAIL - {', '.join(failed)}")
        return 1
    print("RESULT: PASS - release is eligible for the Pi promotion step.")
    print("Next: run scripts/deploy_to_pi.py --dry-run, review it, then deploy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
