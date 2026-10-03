"""Refresh data/human/manifest.xlsx from the recorder's canonical phrase source."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tinyvcm_model.manifest_export import workbook_payload


def export_manifest() -> Path:
    runtime = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
    node = runtime / 'bin/node.exe'
    if not node.is_file():
        raise RuntimeError('The bundled spreadsheet runtime is unavailable; CSV and phrase files remain authoritative.')
    output = PROJECT / 'outputs/manifest-export'
    output.mkdir(parents=True, exist_ok=True)
    junction = output / 'node_modules'
    if not junction.exists():
        target = runtime / 'node_modules'
        command = "New-Item -ItemType Junction -Path '%s' -Target '%s' | Out-Null" % (str(junction).replace("'", "''"), str(target).replace("'", "''"))
        subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                        command], check=True)
    payload = output / 'manifest_payload.json'
    payload.write_text(json.dumps(workbook_payload(PROJECT), indent=2), encoding='utf-8')
    builder = output / 'build_manifest.mjs'
    shutil.copyfile(PROJECT / 'scripts/build_manifest_workbook.mjs', builder)
    subprocess.run([str(node), str(builder), str(payload), str(output)], check=True, cwd=PROJECT)
    destination = PROJECT / 'data/human/manifest.xlsx'
    temporary = destination.with_suffix('.tmp.xlsx')
    shutil.copyfile(output / 'manifest.xlsx', temporary)
    os.replace(temporary, destination)
    return destination


if __name__ == '__main__':
    print(export_manifest())
