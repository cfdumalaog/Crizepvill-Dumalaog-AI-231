# ME3 Class-group Voice Recorder launcher.
# Mirrors Start-Local-ME3.ps1 structure. No LLM/brain logic; just serves the
# Gradio recorder on 0.0.0.0:7871 (ME2 owns 7861-7865 - do not touch).
param(
    [int]$Port = 7871
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspace = (Resolve-Path (Join-Path $project '..\..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Shared workspace .venv python not found at $python"
}

# Detect a LAN IP to print (best effort; never prints credentials).
$ip = '127.0.0.1'
try {
    $s = [System.Net.Sockets.Socket]::new('InterNetwork','Dgram','Udp')
    $s.Connect('8.8.8.8', 80)
    $ip = $s.LocalAddress.IPAddressToString
    $s.Close()
} catch { }

Write-Host 'ME3 - Class-group Voice Recorder (browser tool)'
Write-Host "Project : $project"
Write-Host "Python  : $python"
Write-Host "URL     : http://${ip}:${Port}   (lab LAN only - no public tunnel)"
Write-Host 'Port    : ' + $Port + '  (ME2 owns 7861-7865; not touched)'
Write-Host 'Stop    : Ctrl+C in this window (or close the window)'
Write-Host ''
Write-Host 'Students: open the URL above on a laptop or phone on the same LAN.'
Write-Host 'Sentences: see data\DATASET.md section 2 (8 canonical stimuli).'
Write-Host 'Files   : data\bench\speakers\<speaker_label>\<scenario_id>.wav'
Write-Host 'Manifest: data\bench\manifest.jsonl'
Write-Host ''

Push-Location $project
try {
    & $python -m me3.web_recorder
}
finally {
    Pop-Location
}
# Propagate python's exit code so the .bat can `pause` on failure.
exit $LASTEXITCODE
