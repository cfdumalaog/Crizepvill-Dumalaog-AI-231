param(
    [string]$Device = 'auto',
    [int]$Port = 7863,
    [ValidateRange(0.0, 1.0)]
    [double]$WakeThreshold = 0.95
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspace = (Resolve-Path (Join-Path $project '..\..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$candidate = Join-Path $project 'deployment\current_vcm'
$intent = Join-Path $candidate 'models\intent_int8.onnx'
$wake = Join-Path $candidate 'models\binary_wake_int8.onnx'
$metadataPath = Join-Path $candidate 'metadata.json'
$wakeSummaryPath = Join-Path $candidate 'export_summary.json'

foreach ($required in @($python, $intent, $wake, $metadataPath, $wakeSummaryPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required local VCM file is missing: $required"
    }
}

$metadataHashesJson = & $python -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print(json.dumps([d['intent']['model']['sha256'],d['wake']['model']['sha256']]))" $metadataPath
if ($LASTEXITCODE -ne 0) {
    throw "Could not read the model hashes from $metadataPath."
}
$metadataHashes = ($metadataHashesJson -join "`n") | ConvertFrom-Json
$wakeSummary = Get-Content -LiteralPath $wakeSummaryPath -Raw | ConvertFrom-Json
$actualIntentHash = (Get-FileHash -LiteralPath $intent -Algorithm SHA256).Hash.ToLowerInvariant()
$expectedIntentHash = [string]$metadataHashes[0]
if ($actualIntentHash -ne $expectedIntentHash.ToLowerInvariant()) {
    throw "Candidate ONNX hash mismatch for $intent (metadata=$expectedIntentHash, actual=$actualIntentHash)."
}
$actualWakeHash = (Get-FileHash -LiteralPath $wake -Algorithm SHA256).Hash.ToLowerInvariant()
$expectedWakeHash = [string]$metadataHashes[1]
if ($actualWakeHash -ne $expectedWakeHash.ToLowerInvariant()) {
    throw "Candidate ONNX hash mismatch for $wake (metadata=$expectedWakeHash, actual=$actualWakeHash)."
}

$validationThreshold = [double]$wakeSummary.validation_selected_threshold
if ($validationThreshold -lt 0 -or $validationThreshold -gt 1) {
    throw "Candidate validation wake threshold is invalid: $validationThreshold"
}
$threshold = [double]$WakeThreshold

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue
    $ownerName = if ($owner) { $owner.Name } else { 'unknown process' }
    throw "Port $Port is already listening (PID $($listener.OwningProcess), $ownerName). Close the existing VCM before starting another microphone listener, or choose a free port with -Port."
}

Write-Host 'ME2 - VCM on Raspberry Pi 5'
Write-Host "Studio: http://127.0.0.1:$Port/studio"
if ($Device -eq 'auto') {
    Write-Host 'Audio device selection: Automatic (any available input)'
} else {
    Write-Host "Audio device selection: $Device"
}
Write-Host "Wake model: $wake"
Write-Host "Intent model: $intent"
Write-Host "Wake threshold: $threshold (local operating override; model validation threshold: $validationThreshold)"

Push-Location $project
try {
    $arguments = @(
        '--host', '127.0.0.1', '--port', [string]$Port,
        '--model', $intent, '--binary-wake-model', $wake,
        '--wake-threshold', [string]$threshold,
        '--vad-threshold', '0.006', '--inference-interval', '0.12', '--wake-confirmations', '1',
        '--live-weather'
    )
    if ($Device -ne 'auto') {
        $arguments += @('--device', $Device)
    }
    & $python (Join-Path $project 'vcm_app.py') @arguments
}
finally {
    Pop-Location
}
