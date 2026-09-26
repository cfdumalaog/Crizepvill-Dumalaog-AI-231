"""Check bundle integrity before startup; works without training dependencies."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
manifest=json.loads((root/'bundle_manifest.json').read_text())
for name,expected in manifest['sha256'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root): raise ValueError('Path outside bundle')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError(f'Checksum mismatch: {name}')
print(f'Integrity passed: {len(manifest["sha256"])} files. Training run {manifest["training_run"]}.')
