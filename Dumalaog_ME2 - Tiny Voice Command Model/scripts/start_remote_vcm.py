"""Start a password-protected, temporary public microphone-evaluation tunnel.

The evaluator binds only to 127.0.0.1. Cloudflare Tunnel provides the public
HTTPS endpoint; the remote process never saves audio and never runs actions.
"""
from __future__ import annotations

import getpass
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading


PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "runs" / "personalized-vcm-20260928-155109"
WAKE = RUN / "wake_broad_negatives" / "models" / "wake_personalized_int8.onnx"
INTENT = RUN / "intent_personalized" / "models" / "intent_personalized_int8.onnx"
THRESHOLD = "0.9992183446884155"
PBKDF2_ITERATIONS = 200_000


def _cloudflared() -> str:
    configured = os.environ.get("CLOUDFLARED_BIN")
    executable = shutil.which(configured) if configured else shutil.which("cloudflared")
    if not executable and configured and Path(configured).is_file():
        executable = str(Path(configured).resolve())
    if not executable:
        raise RuntimeError(
            "cloudflared is not installed or not on PATH. Install the official "
            "Cloudflare cloudflared client, open a new terminal, and rerun this script."
        )
    return executable


def _prompt_credentials() -> tuple[str, bytes, bytes]:
    username = input("Remote evaluator username [vcm-evaluator]: ").strip() or "vcm-evaluator"
    if ":" in username or not username or len(username) > 128:
        raise ValueError("Username must be 1-128 characters and cannot contain a colon")
    password = getpass.getpass("Remote evaluator password (minimum 16 characters): ")
    if len(password) < 16:
        raise ValueError("Use a unique password with at least 16 characters")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise ValueError("Passwords did not match")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS, dklen=32)
    del password, confirmation
    return username, salt, digest


def main() -> int:
    try:
        cloudflared = _cloudflared()
        if not WAKE.is_file() or not INTENT.is_file():
            raise RuntimeError(f"Candidate ONNX pair is incomplete under {RUN}")
        username, salt, digest = _prompt_credentials()
    except (RuntimeError, ValueError) as exc:
        print(f"Cannot start remote VCM: {exc}", file=sys.stderr)
        return 2

    auth_env = {
        "VCM_REMOTE_USER": username,
        "VCM_REMOTE_SALT": salt.hex(),
        "VCM_REMOTE_PASSWORD_HASH": digest.hex(),
        "VCM_REMOTE_WAKE_MODEL": str(WAKE),
        "VCM_REMOTE_INTENT_MODEL": str(INTENT),
        "VCM_REMOTE_WAKE_THRESHOLD": THRESHOLD,
    }
    local_server = tunnel = None
    local_thread = None
    try:
        sys.path.insert(0, str(PROJECT))
        import remote_vcm_eval

        os.environ.update(auth_env)
        try:
            local_server = remote_vcm_eval.create_server(host="127.0.0.1", port=7870)
        finally:
            for key in auth_env:
                os.environ.pop(key, None)
        local_thread = threading.Thread(
            target=lambda: local_server.serve_forever(poll_interval=0.2),
            name="authenticated-vcm-local-server",
            daemon=True,
        )
        local_thread.start()
        print("Local evaluator ready at http://127.0.0.1:7870/ (loopback only).", flush=True)
        tunnel_env = os.environ.copy()
        for key in auth_env:
            tunnel_env.pop(key, None)
        tunnel = subprocess.Popen(
            [cloudflared, "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:7870"],
            cwd=PROJECT,
            env=tunnel_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert tunnel.stdout is not None
        def forward_tunnel_log() -> None:
            assert tunnel is not None and tunnel.stdout is not None
            for line in tunnel.stdout:
                match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if match:
                    print(f"\nTemporary public evaluator URL: {match.group(0)}", flush=True)
                    print(f"Sign in as: {username}", flush=True)
                    print("The password is the one entered above; this URL is temporary. Press Ctrl+C to stop.\n", flush=True)
                else:
                    print(f"[cloudflared] {line.rstrip()}", flush=True)

        reader = threading.Thread(target=forward_tunnel_log, name="cloudflared-output", daemon=True)
        reader.start()
        print("Starting password-protected evaluation tunnel. Audio is not saved; no actions are executed.", flush=True)
        while tunnel.poll() is None:
            if local_thread is not None and not local_thread.is_alive():
                raise RuntimeError("Local evaluator thread stopped unexpectedly")
            threading.Event().wait(0.5)
        raise RuntimeError(f"cloudflared exited with status {tunnel.returncode}")
    except KeyboardInterrupt:
        print("Stopping evaluator and public tunnel…", flush=True)
    except (OSError, RuntimeError) as exc:
        print(f"Remote evaluator stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait(timeout=2)
        if tunnel is not None and tunnel.stdout is not None:
            tunnel.stdout.close()
        if local_server is not None:
            local_server.shutdown()
            local_server.server_close()
        if local_thread is not None:
            local_thread.join(timeout=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
