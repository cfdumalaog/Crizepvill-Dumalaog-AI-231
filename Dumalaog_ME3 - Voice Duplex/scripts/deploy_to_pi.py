"""ME3 - Voice Duplex: one-command deploy to the Raspberry Pi.

Flow (mirrors ME2's deploy_latest_vcm.py):
  1. Read deployment/latest_release.json (must exist; build with
     scripts/package_release.py first if missing).
  2. Verify the zip sha256 against the manifest.
  3. scp the zip to the Pi, unzip into rpi_path, run install.sh there.
  4. Write deployment/last_pi_deployment.json (status=deployed | pending_sync).

Usage:
  python scripts/deploy_to_pi.py                 # deploy latest prepared release
  python scripts/deploy_to_pi.py --host 192.168.1.70 --user dalmacio
  python scripts/deploy_to_pi.py --dry-run       # verify only, no transfer
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parent.parent
DEPLOY = PROJECT / "deployment"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def _die(msg: str, state: dict | None = None) -> None:
    if state:
        (DEPLOY / "last_pi_deployment.json").write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=None, help="Pi hostname/IP (default: config)")
    ap.add_argument("--user", default=None, help="Pi user (default: config)")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify release + connectivity only")
    args = ap.parse_args()

    # 1. latest release
    lr_path = DEPLOY / "latest_release.json"
    if not lr_path.is_file():
        _die("no prepared release. Run scripts/package_release.py first.")
    lr = json.loads(lr_path.read_text(encoding="utf-8"))
    if lr.get("status") != "prepared":
        _die(f"latest release status is {lr.get('status')!r}, expected 'prepared'")
    zip_path = PROJECT / lr["zip"]
    if not zip_path.is_file():
        _die(f"release zip missing: {zip_path}")

    # 2. integrity
    actual = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if actual != lr.get("zip_sha256"):
        _die("zip sha256 mismatch - release archive is corrupt")
    print(f"[1/4] release OK: {lr['release_id']} ({len(actual)//2} bytes verified)")

    # 3. target
    from me3.config import load_config
    cfg = load_config()
    dep = cfg.get("deployment", {})
    host = args.host or dep.get("rpi_host", "")
    user = args.user or dep.get("rpi_user", "dalmacio")
    rpi_path = dep.get("rpi_path",
                       "/home/dalmacio/Desktop/kiko/me3_voice_duplex")
    if not host:
        _die("no rpi_host configured (config.yaml deployment.rpi_host)")
    print(f"[2/4] target: {user}@{host}:{rpi_path}")

    if args.dry_run:
        print("[dry-run] stopping before transfer")
        return

    # 4. connectivity + transfer
    remote = f"{user}@{host}"
    probe = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
         remote, "echo me3-ok"],
        capture_output=True, text=True, timeout=30)
    if probe.returncode != 0 or "me3-ok" not in probe.stdout:
        state = {
            "status": "pending_sync", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                          time.gmtime()),
            "release_id": lr["release_id"], "host": host, "path": rpi_path,
            "zip_sha256": actual,
            "reason": "Pi unreachable (ssh failed). Package preserved locally.",
        }
        _die(f"Pi unreachable at {host} - release preserved as pending_sync", state)
    print("[3/4] ssh OK")

    r = subprocess.run(
        ["scp", "-o", "BatchMode=yes", str(zip_path),
         f"{remote}:/tmp/me3-release.zip"],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        _die(f"scp failed: {r.stderr[:300]}",
             {"status": "pending_sync", "at": time.strftime(
                 "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "release_id": lr["release_id"], "host": host, "path": rpi_path,
              "zip_sha256": actual, "reason": f"scp failed: {r.stderr[:200]}"})

    parent_dir = str(PurePosixPath(rpi_path).parent)
    stage_dir = f"/tmp/me3-stage-{lr['release_id']}"
    remote_cmd = (
        f"set -e; "
        f"mkdir -p {shlex.quote(parent_dir)} {shlex.quote(rpi_path)}; "
        f"rm -rf {shlex.quote(stage_dir)}; "
        f"unzip -q /tmp/me3-release.zip -d {shlex.quote(stage_dir)}; "
        f"test -d {shlex.quote(stage_dir + '/me3_voice_duplex')}; "
        f"cp -a {shlex.quote(stage_dir + '/me3_voice_duplex')}/. {shlex.quote(rpi_path)}/; "
        f"find {shlex.quote(rpi_path)} -type f \\( -name '*.sh' -o -name '*.service' \\) -exec sed -i 's/\\r$//' {{}} +; "
        f"rm -rf {shlex.quote(stage_dir)} /tmp/me3-release.zip; "
        f"cd {shlex.quote(rpi_path)} && bash install.sh"
    )
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", remote, remote_cmd],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        _die(f"remote install failed (rc={r.returncode}): {r.stderr[-500:]}",
             {"status": "pending_sync", "at": time.strftime(
                 "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "release_id": lr["release_id"], "host": host, "path": rpi_path,
              "zip_sha256": actual, "reason": "remote install.sh failed"})
    print("[4/4] installed on Pi")

    state = {
        "status": "deployed",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "release_id": lr["release_id"], "host": host, "path": rpi_path,
        "zip_sha256": actual, "reason": None,
    }
    (DEPLOY / "last_pi_deployment.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"DEPLOYED {lr['release_id']} -> {remote}:{rpi_path}")
    print("Next: on the Pi run  systemctl --user enable --now me3-agent")


if __name__ == "__main__":
    main()
