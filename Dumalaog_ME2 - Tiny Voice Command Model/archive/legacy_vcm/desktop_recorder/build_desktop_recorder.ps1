$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspace = (Resolve-Path (Join-Path $project '..\..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "The shared workspace Python environment is missing: $python"
}
$output = Join-Path $project 'launchers\ME2-VCM-Recorder'
$work = Join-Path $project 'outputs\pyinstaller-work'
$spec = Join-Path $project 'outputs\pyinstaller-spec'
$groundTruth = Join-Path $project 'docs\ground_truth_phrases'
$addData = "$groundTruth;docs/ground_truth_phrases"
Push-Location $project
try {
    & $python -m PyInstaller `
        --noconfirm --clean --onedir --windowed `
        --name 'ME2 - VCM Recorder' `
        --distpath $output --workpath $work --specpath $spec `
        --add-data $addData `
        --collect-all _sounddevice_data --collect-all _soundfile_data `
        --exclude-module torch --exclude-module torchvision --exclude-module torchaudio `
        --exclude-module matplotlib --exclude-module gradio `
        --hidden-import tinyvcm_model.recording_labels `
        --hidden-import tinyvcm_model.recorder_core `
        --hidden-import tinyvcm_model.manifest_export `
        desktop_recorder.py
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
$executable = Join-Path (Join-Path $output 'ME2 - VCM Recorder') 'ME2 - VCM Recorder.exe'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Build completed without the expected application: $executable"
}
Write-Host "Standalone recorder built: $executable"
