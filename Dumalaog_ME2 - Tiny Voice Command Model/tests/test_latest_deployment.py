"""Exercise release selection, unsafe archives, and stale running-model checks."""
import importlib.util
from pathlib import Path
import zipfile

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load('pi_installer', PROJECT / 'deployment/install_release_on_pi.py')
sync = load('vcm_sync', PROJECT / 'scripts/deploy_latest_vcm.py')
package = load('pi_package_builder', PROJECT / 'deployment/package_personalized_rpi.py')


@pytest.mark.parametrize('member', ['dandan/../../outside', '/outside', 'dandan/..\\outside'])
def test_rejects_unsafe_zip_before_writing(tmp_path, member):
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr(member, b'bad')
    stage = tmp_path / 'stage'
    with pytest.raises(ValueError):
        installer.extract_bundle(archive, stage)
    assert not stage.exists()


def test_rejects_model_files_changed_under_old_process(monkeypatch):
    metadata = {'source_run': 'latest', 'wake': {'model': {'sha256': 'new-wake'}},
                'intent': {'model': {'sha256': 'new-intent'}}}
    monkeypatch.setattr(installer, 'state', lambda: {'model_source_run': 'old'})
    with pytest.raises(ValueError, match='model_source_run'):
        installer.verify_state(metadata)


def test_pi_release_uses_fifteen_second_post_wake_window():
    assert package.DEPLOYMENT['post_wake_timeout_sec'] == 15.0


def test_installer_verifies_fifteen_second_post_wake_window(monkeypatch):
    current = {
        'model_source_run': 'active',
        'wake_model_sha256': 'wake-sha',
        'intent_model_sha256': 'intent-sha',
        'classes_count': 31,
        'binary_wake_enabled': True,
        'wake_threshold': 0.95,
        'timeout_sec': 15.0,
    }
    metadata = {
        'source_run': 'active',
        'wake': {'model': {'sha256': 'wake-sha'}},
        'intent': {'model': {'sha256': 'intent-sha'}},
    }
    monkeypatch.setattr(installer, 'state', lambda: current)
    monkeypatch.setattr(installer, 'app_pids', lambda: [123])

    result = installer.verify_state(metadata)

    assert result['timeout_sec'] == 15.0


def test_incomplete_new_run_cannot_replace_completed_pair(tmp_path, monkeypatch):
    import json
    runs = tmp_path / 'runs'
    completed = runs / 'complete'
    incomplete = runs / 'new-but-incomplete'
    incomplete.mkdir(parents=True)
    (incomplete / 'final_evaluation.json').write_text('{}')
    for name in ('wake_binary/models/export_summary.json', 'wake_binary/models/wake_personalized_int8.onnx',
                 'intent_personalized/models/export_summary.json', 'intent_personalized/models/intent_personalized_int8.onnx'):
        path = completed / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'placeholder')
    (completed / 'final_evaluation.json').write_text(json.dumps({
        'wake_model_comparison': {'binary': {}}, 'intent_model_comparison': {'personalized_int8': {}}}))
    monkeypatch.setattr(sync, 'PROJECT', tmp_path)
    assert sync.latest_completed_run() == completed


def test_release_extraction_keeps_executable_launcher(tmp_path):
    archive = tmp_path / 'valid.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('dandan/launch-vcm.sh', b'#!/bin/sh\n')
        bundle.writestr('dandan/models/wake.onnx', b'model')
    stage = tmp_path / 'stage'
    installer.extract_bundle(archive, stage)
    assert (stage / 'models/wake.onnx').read_bytes() == b'model'


def test_unreachable_pi_records_pending_instead_of_success(tmp_path, monkeypatch):
    import json
    import subprocess
    deploy = tmp_path / 'deployment'
    desktop = tmp_path / 'desktop'
    deploy.mkdir()
    desktop.mkdir()
    monkeypatch.setattr(sync, 'DEPLOY', deploy)
    monkeypatch.setattr(sync, 'DESKTOP', desktop)
    monkeypatch.setattr(sync, 'latest_completed_run', lambda _name: Path('complete'))
    monkeypatch.setattr(sync, 'prepare_release', lambda _source: {'source_run': 'complete'})
    monkeypatch.setattr(sync, 'update_desktop', lambda _release: None)
    def unavailable(*_args, **_kwargs):
        raise subprocess.TimeoutExpired('ssh', 15)
    monkeypatch.setattr(sync.subprocess, 'run', unavailable)
    with pytest.raises(subprocess.TimeoutExpired):
        sync.synchronize()
    assert json.loads((deploy / 'latest_release.json').read_text())['status'] == 'pending'
    assert not (deploy / 'last_pi_deployment.json').exists()
    assert not (deploy / 'sync.lock').exists()
