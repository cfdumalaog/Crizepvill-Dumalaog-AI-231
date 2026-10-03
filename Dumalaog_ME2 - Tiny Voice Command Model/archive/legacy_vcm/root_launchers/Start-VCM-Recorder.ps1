$ErrorActionPreference = 'Stop'
Write-Host 'Recorder: http://127.0.0.1:7862. Stop or mute the assistant while recording.'
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'AI 231\Dumalaog_ME2 - Tiny Voice Command Model\record_dataset.py')
