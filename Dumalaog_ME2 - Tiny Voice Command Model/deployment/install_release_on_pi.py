"""Verify and replace only this user's Desktop/dandan VCM, with rollback.

This file is transferred over authenticated SSH and executed with system Python.
It never installs a boot service or scans other Desktop projects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import time
import urllib.request
import zipfile

HOME = Path('/home/dalmacio')
ACTIVE = HOME / 'Desktop/dandan'
ARCHIVES = HOME / 'archive'
PYTHON = HOME / '.venvs/tinyvcm-rpi5/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_bundle(archive, stage):
    """Reject traversal, duplicate entries and symlinks before writing files."""
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if not names or len(names) != len(set(names)):
            raise ValueError('Empty ZIP or duplicate paths')
        prefix = PurePosixPath(names[0]).parts[0]
        for info in bundle.infolist():
            path = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (path.is_absolute() or '..' in path.parts or '\\' in info.filename
                    or path.parts[0] != prefix or len(path.parts) < 2
                    or (mode & 0o170000) == 0o120000):
                raise ValueError(f'Unsafe ZIP member: {info.filename}')
            target = stage.joinpath(*path.parts[1:])
            if not target.resolve().is_relative_to(stage.resolve()):
                raise ValueError('ZIP member escapes stage')
        stage.mkdir()
        for info in bundle.infolist():
            target = stage.joinpath(*PurePosixPath(info.filename).parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(info))
            target.chmod(0o755 if target.name in {'launch-vcm.sh', 'start-vcm-demo.sh'} else 0o644)


def app_pids():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if str(ACTIVE / 'vcm_app.py').encode() in args:
                found.append(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found


def stop_app():
    pids = app_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while app_pids() and time.monotonic() < deadline:
        time.sleep(.2)
    for pid in app_pids():
        os.kill(pid, signal.SIGKILL)


def state():
    with urllib.request.urlopen('http://127.0.0.1:7860/api/state', timeout=2) as response:
        return json.load(response)


def verify_state(metadata):
    current = state()
    expected = {
        'model_source_run': metadata['source_run'],
        'wake_model_sha256': metadata['wake']['model']['sha256'],
        'intent_model_sha256': metadata['intent']['model']['sha256'],
        'classes_count': 31, 'binary_wake_enabled': True,
        'wake_threshold': .95, 'timeout_sec': 10.0,
    }
    for key, value in expected.items():
        if current.get(key) != value:
            raise ValueError(f'Live API mismatch for {key}: {current.get(key)!r}')
    if len(app_pids()) != 1:
        raise RuntimeError('Expected exactly one Desktop/dandan VCM process')
    return {key: current.get(key) for key in (
        *expected, 'state', 'microphone', 'microphone_stream_active',
        'audio_age_sec', 'dropped', 'error',
    )}


def start_app():
    with (ACTIVE / 'vcm.log').open('ab') as log:
        subprocess.Popen([str(ACTIVE / 'launch-vcm.sh')], cwd=ACTIVE,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         start_new_session=True)


def install_desktop_launcher():
    """Install one Pi Desktop icon for this VCM only; leave neighboring folders untouched."""
    desktop = HOME / 'Desktop'
    launcher = desktop / 'ME2 - VCM on Raspberry Pi 5.desktop'
    content = (
        '[Desktop Entry]\n'
        'Version=1.0\n'
        'Type=Application\n'
        'Name=ME2 - VCM on Raspberry Pi 5\n'
        'Comment=Start the ME2 voice command model and open its assistant\n'
        'Exec=/home/dalmacio/Desktop/dandan/start-vcm-demo.sh\n'
        'Path=/home/dalmacio/Desktop/dandan\n'
        'Terminal=true\n'
        'Categories=Utility;\n'
    )
    temporary = launcher.with_suffix('.desktop.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.chmod(0o755)
    temporary.replace(launcher)


def wait_ready(metadata):
    deadline = time.monotonic() + 20
    last = None
    while time.monotonic() < deadline:
        try:
            return verify_state(metadata)
        except Exception as exc:
            last = exc
            time.sleep(.5)
    raise RuntimeError(f'VCM did not become healthy: {last}')


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if Path.home() != HOME or ACTIVE.resolve() != HOME / 'Desktop/dandan':
        raise RuntimeError('This installer is scoped only to dalmacio/Desktop/dandan')
    if digest(args.archive) != args.sha256:
        raise ValueError('Transferred ZIP hash does not match Windows')
    cache = HOME / '.cache/me2-vcm-deploy'
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'install.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        tag = time.strftime('%Y%m%d-%H%M%S') + '-' + args.sha256[:10]
        stage = HOME / 'Desktop' / ('.dandan-stage-' + tag)
        backup = ARCHIVES / ('dandan-predeploy-' + tag)
        failed = ARCHIVES / ('dandan-failed-' + tag)
        if any(p.exists() for p in (stage, backup, failed)):
            raise FileExistsError('Release stage or backup already exists')
        extract_bundle(args.archive, stage)
        verification = subprocess.run([str(PYTHON), str(stage / 'verify_personalized_rpi.py')],
                                      cwd=stage, capture_output=True, text=True, check=True)
        checked = json.loads(verification.stdout)
        metadata = json.loads((stage / 'candidate_metadata.json').read_text())
        # Validate all new files on ARM64 before interrupting the old app.
        ARCHIVES.mkdir(exist_ok=True)
        try:
            state()
            if not app_pids():
                raise RuntimeError('Port 7860 belongs to another application; refusing to stop it')
        except OSError:
            pass
        had_active = ACTIVE.exists()
        if ACTIVE.is_symlink():
            raise RuntimeError('Active deployment must not be a symlink')
        stop_app()
        if had_active:
            ACTIVE.rename(backup)
        try:
            stage.rename(ACTIVE)
            start_app()
            health = wait_ready(metadata)
            install_desktop_launcher()
            for route in ('assistant', 'studio'):
                with urllib.request.urlopen('http://127.0.0.1:7860/' + route, timeout=3) as response:
                    if response.status != 200:
                        raise RuntimeError(f'{route} health check failed')
        except Exception:
            stop_app()
            if ACTIVE.exists():
                ACTIVE.rename(failed)
            if had_active:
                backup.rename(ACTIVE)
                start_app()
            raise
        receipt = {
            'status': 'deployed', 'hostname': os.uname().nodename,
            'architecture': os.uname().machine, 'installed_at': str(ACTIVE),
            'source_run': metadata['source_run'], 'zip_sha256': args.sha256,
            'backup': str(backup) if had_active else None,
            'autostart': False, 'verification': checked, 'live_api': health,
        }
        (cache / 'last_deployment.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
