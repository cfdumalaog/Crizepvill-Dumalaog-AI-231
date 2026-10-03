param([string]$PiHost = 'cfdfnjrpi5.local', [string]$PiUser = 'dalmacio')
$ErrorActionPreference = 'Stop'
$configPath = Join-Path $PSScriptRoot 'sync-config.json'
if (Test-Path -LiteralPath $configPath) {
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $python = $config.python
    $script = $config.script
} else {
    $project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    $workspace = (Resolve-Path (Join-Path $project '..\..')).Path
    $python = Join-Path $workspace '.venv\Scripts\python.exe'
    $script = Join-Path $project 'scripts\deploy_latest_vcm.py'
}
& $python -u $script --pi-host $PiHost --pi-user $PiUser
if ($LASTEXITCODE -ne 0) { throw 'Latest-model synchronization failed. See latest_release.json for pending status.' }
