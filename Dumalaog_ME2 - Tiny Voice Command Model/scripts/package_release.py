"""Build the inference-only Pi bundle from a measured training run."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import subprocess
import zipfile

root=Path(__file__).resolve().parents[1]
latest=json.loads((root/'runs/latest.json').read_text())['run']
run=root/'runs'/latest
out=root/'release'
out.mkdir(exist_ok=True)
copied=[]
def copy(source,destination):
    shutil.copy2(source,destination)
    copied.append(destination.relative_to(out).as_posix())
for directory in ['tinyvcm','models']: (out/directory).mkdir(exist_ok=True)
runtime=['__init__.py','config.py','frontend.py','runtime.py','microphone.py','devices.py']
for name in runtime: copy(root/'tinyvcm'/name,out/'tinyvcm'/name)
copy(root/'demo.py',out/'demo.py')
for p in (root/'deployment').iterdir():
    if p.is_file(): copy(p,out/p.name)
for name in ['model_int8.onnx','metadata.json','metrics.json','dataset_audit.json','stream_replay.json']:
    copy(run/name,out/'models'/name)
# Measure this exact model in a fresh inference-only process, not the trainer.
subprocess.run([sys.executable,'-m','tinyvcm.runtime','--output',str(run/'release_runtime_benchmark.json')],cwd=root,check=True,stdout=subprocess.DEVNULL)
copy(run/'release_runtime_benchmark.json',out/'models/runtime_benchmark.json')
# A trained export must exist and pass the byte limit before packaging.
assert (out/'models/model_int8.onnx').stat().st_size<500000
files={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in sorted(copied)}
(out/'bundle_manifest.json').write_text(json.dumps({'training_run':latest,'sha256':files},indent=2),encoding='utf-8')
archive=root/'TinyVCM_RaspberryPi5.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name in [*files,'bundle_manifest.json']: z.write(out/name,'vcm/'+name)
print(f'Packaged {len(files)} files from {latest}: {archive.name} ({archive.stat().st_size:,} bytes).')
