"""Stage, package and synchronize the latest completed two-model run to the Pi.

Training calls this by default. An unreachable Pi is recorded as pending and
returns a failure; the Desktop launcher retries synchronization before opening
the app. No scheduler or boot service is installed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
DEPLOY = PROJECT / 'deployment'
DESKTOP = Path.home() / 'Desktop/dandan'
sys.path.insert(0, str(DEPLOY))
from package_personalized_rpi import sources_for_candidate  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def latest_completed_run(run_name=None):
    runs = PROJECT / 'runs'
    required = ('final_evaluation.json', 'wake_binary/models/export_summary.json',
                'wake_binary/models/wake_personalized_int8.onnx',
                'intent_personalized/models/export_summary.json',
                'intent_personalized/models/intent_personalized_int8.onnx')
    candidates = []
    for run in runs.iterdir():
        if not run.is_dir() or run.is_symlink() or not re.fullmatch(r'[A-Za-z0-9_-]+', run.name):
            continue
        if run_name is not None and run.name != run_name:
            continue
        if all((run / name).is_file() for name in required):
            report = json.loads((run / required[0]).read_text(encoding='utf-8'))
            if ('binary' in report.get('wake_model_comparison', {})
                    and 'personalized_int8' in report.get('intent_model_comparison', {})):
                candidates.append(run)
    if not candidates:
        raise FileNotFoundError('No completed binary-wake + intent run found')
    return max(candidates, key=lambda p: (p / required[0]).stat().st_mtime_ns)


def source_fingerprint(candidate):
    sources = sources_for_candidate(candidate)
    sources['package_builder.py'] = DEPLOY / 'package_personalized_rpi.py'
    hasher = hashlib.sha256()
    for name, path in sorted(sources.items()):
        hasher.update(name.encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def run(command, **kwargs):
    result = subprocess.run(command, text=True, encoding='utf-8', errors='replace', **kwargs)
    if result.returncode:
        message = ((result.stderr or '') + (result.stdout or '')).strip()
        raise RuntimeError(f'Command failed ({result.returncode}): {message[-3000:]}')
    return result


def stage_run(source):
    current = DEPLOY / 'current_vcm'
    metadata = current / 'metadata.json'
    if metadata.is_file():
        previous = json.loads(metadata.read_text(encoding='utf-8'))
        if (previous['source_run'] == source.name
                and previous['wake']['model']['sha256'] == digest(source / 'wake_binary/models/wake_personalized_int8.onnx')
                and previous['intent']['model']['sha256'] == digest(source / 'intent_personalized/models/intent_personalized_int8.onnx')
                and digest(current / previous['wake']['model']['path']) == previous['wake']['model']['sha256']
                and digest(current / previous['intent']['model']['path']) == previous['intent']['model']['sha256']):
            return current
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    stage = DEPLOY / ('stage-' + stamp)
    run([sys.executable, str(PROJECT / 'scripts/stage_current_vcm.py'), '--run', source.name,
         '--threshold', '0.95', '--destination', str(stage)], cwd=PROJECT, check=False)
    if current.exists():
        if current.is_symlink() or current.resolve().parent != DEPLOY.resolve():
            raise ValueError('Unsafe active candidate path')
        archive = DEPLOY / 'archive' / ('current_vcm-' + stamp)
        archive.parent.mkdir(exist_ok=True)
        current.rename(archive)
    stage.rename(current)
    return current


def prepare_release(source):
    current = stage_run(source)
    fingerprint = source_fingerprint(current)
    latest_path = DEPLOY / 'latest_release.json'
    if latest_path.is_file():
        cached = json.loads(latest_path.read_text(encoding='utf-8'))
        archive = PROJECT / cached['archive']
        if (cached.get('source_fingerprint') == fingerprint and archive.is_file()
                and digest(archive) == cached.get('zip_sha256')):
            return cached
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    release_id = source.name + '-' + stamp
    output = DEPLOY / 'dist' / release_id / 'dandan'
    archive = DEPLOY / 'dist' / ('dandan-vcm-' + release_id + '.zip')
    run([sys.executable, str(DEPLOY / 'package_personalized_rpi.py'), '--candidate-dir', str(current),
         '--output-dir', str(output), '--archive', str(archive)], cwd=PROJECT)
    metadata = json.loads((current / 'metadata.json').read_text(encoding='utf-8'))
    release = {
        'status': 'prepared', 'source_run': source.name, 'source_fingerprint': fingerprint,
        'archive': archive.relative_to(PROJECT).as_posix(), 'zip_sha256': digest(archive),
        'bundle_manifest_sha256': digest(output / 'bundle_manifest.json'),
        'wake_sha256': metadata['wake']['model']['sha256'],
        'intent_sha256': metadata['intent']['model']['sha256'],
        'wake_threshold': .95, 'installed_at': '/home/dalmacio/Desktop/dandan',
        'created_at_local': datetime.now().isoformat(),
    }
    write_json(latest_path, release)
    return release


def update_desktop(release):
    DESKTOP.mkdir(exist_ok=True)
    if DESKTOP.is_symlink() or DESKTOP.resolve().parent != (Path.home() / 'Desktop').resolve():
        raise ValueError('Unsafe Desktop deployment destination')
    source = PROJECT / release['archive']
    target = DESKTOP / source.name
    shutil.copy2(source, target)
    for previous in DESKTOP.glob('dandan-vcm-*.zip'):
        if previous != target and previous.is_file():
            archive = DESKTOP / 'archive' / previous.name
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                previous.rename(archive)
    for name in ('Start-VCM.bat', 'Start-VCM.ps1', 'Sync-VCM.ps1', 'Update-VCM.bat'):
        shutil.copy2(DEPLOY / name, DESKTOP / name)
    shortcut_path = Path.home() / 'Desktop' / 'ME2 - VCM on Raspberry Pi 5.lnk'
    try:
        import win32com.client
        shortcut = win32com.client.Dispatch('WScript.Shell').CreateShortcut(str(shortcut_path))
        shortcut.TargetPath = str(DESKTOP / 'Start-VCM.bat')
        shortcut.WorkingDirectory = str(DESKTOP)
        shortcut.Description = 'Start and open the latest ME2 VCM on Raspberry Pi 5'
        shortcut.IconLocation = r'%SystemRoot%\System32\SHELL32.dll, 137'
        shortcut.Save()
    except ImportError:
        pass
    write_json(DESKTOP / 'sync-config.json', {
        'python': str(PROJECT.parents[1] / '.venv/Scripts/python.exe'),
        'script': str(PROJECT / 'scripts/deploy_latest_vcm.py'),
    })
    write_json(DESKTOP / 'latest_release.json', release)


def synchronize(run_name=None, pi_host='cfdfnjrpi5.local', pi_user='dalmacio', host_key_alias='192.168.254.106'):
    if pi_user != 'dalmacio' or not re.fullmatch(r'[A-Za-z0-9_.:-]+', pi_host):
        raise ValueError('Expected dalmacio and a plain hostname or IP')
    lock = DEPLOY / 'sync.lock'
    # Fail visibly rather than race another training/deployment process.
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    release = None
    try:
        source = latest_completed_run(run_name)
        print('Synchronizing latest completed run:', source.name, flush=True)
        release = prepare_release(source)
        update_desktop(release)
        key = Path.home() / '.ssh/id_ed25519'
        options = ['-i', str(key), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                   '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=5',
                   '-o', 'ServerAliveCountMax=2', '-o', 'HostKeyAlias=' + host_key_alias]
        target = pi_user + '@' + pi_host
        check = """import hashlib,json,urllib.request
from pathlib import Path
p=Path('/home/dalmacio/Desktop/dandan')
s=json.load(urllib.request.urlopen('http://127.0.0.1:7860/api/state',timeout=2))
manifest=json.loads((p/'bundle_manifest.json').read_text())
for name,expected in manifest['sha256'].items():
    f=(p/name).resolve()
    assert f.is_relative_to(p) and hashlib.sha256(f.read_bytes()).hexdigest()==expected
print(json.dumps({'state':s,'manifest':hashlib.sha256((p/'bundle_manifest.json').read_bytes()).hexdigest()}))
"""
        probe = subprocess.run(['ssh', *options, target, 'python3 -'], input=check, text=True,
                               capture_output=True, timeout=15)
        ready = False
        if probe.returncode == 0:
            result = json.loads(probe.stdout)
            s = result['state']
            ready = (result['manifest'] == release['bundle_manifest_sha256']
                     and s.get('wake_model_sha256') == release['wake_sha256']
                     and s.get('intent_model_sha256') == release['intent_sha256']
                     and s.get('model_source_run') == source.name
                     and s.get('wake_threshold') == .95 and s.get('timeout_sec') == 15.0)
        if not ready:
            remote_dir = '/home/dalmacio/.cache/me2-vcm-deploy/' + release['zip_sha256'][:16]
            run(['ssh', *options, target, 'mkdir -p ' + shlex.quote(remote_dir)], capture_output=True, timeout=15)
            archive = PROJECT / release['archive']
            run(['scp', *options, str(archive), str(DEPLOY / 'install_release_on_pi.py'),
                 target + ':' + remote_dir + '/'], capture_output=True, timeout=30)
            command = ('python3 ' + shlex.quote(remote_dir + '/install_release_on_pi.py')
                       + ' --archive ' + shlex.quote(remote_dir + '/' + archive.name)
                       + ' --sha256 ' + shlex.quote(release['zip_sha256']))
            installed = run(['ssh', *options, target, command], capture_output=True, timeout=90)
            receipt = json.loads(installed.stdout)
            print(json.dumps(receipt, indent=2), flush=True)
            write_json(DEPLOY / 'last_pi_deployment.json', receipt)
            shutil.copy2(DEPLOY / 'last_pi_deployment.json', DESKTOP / 'last_pi_deployment.json')
        else:
            print('Pi already runs the latest verified pair; no restart needed.', flush=True)
        release.update(status='deployed', pi_host=pi_host, last_verified_at_local=datetime.now().isoformat())
        release.pop('last_error', None)
        write_json(DEPLOY / 'latest_release.json', release)
        write_json(DESKTOP / 'latest_release.json', release)
        return release
    except Exception as exc:
        if release is not None:
            release.update(status='pending', last_error=str(exc), pi_host=pi_host)
            write_json(DEPLOY / 'latest_release.json', release)
            write_json(DESKTOP / 'latest_release.json', release)
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=None, help='Otherwise select the newest completed paired run')
    parser.add_argument('--pi-host', default='cfdfnjrpi5.local')
    parser.add_argument('--pi-user', default='dalmacio')
    parser.add_argument('--host-key-alias', default='192.168.254.106', help='Reuse the previously trusted Pi host key')
    args = parser.parse_args()
    synchronize(args.run, args.pi_host, args.pi_user, args.host_key_alias)


if __name__ == '__main__':
    main()
