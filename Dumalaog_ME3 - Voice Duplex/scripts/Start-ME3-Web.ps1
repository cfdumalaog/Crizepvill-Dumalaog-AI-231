[CmdletBinding()]
param(
    [string]$Path = 'web',
    [ValidateRange(1024, 65535)]
    [int]$Port = 8123,
    [switch]$OpenBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspace = (Resolve-Path (Join-Path $project '..\..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Shared workspace .venv python not found at $python"
}

$candidate = [IO.Path]::GetFullPath((Join-Path $project $Path))
$projectRoot = [IO.Path]::GetFullPath($project.TrimEnd('\') + '\')
if (-not $candidate.StartsWith($projectRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Web path must stay inside the ME3 project: $Path"
}
if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
    throw "Web directory not found: $candidate. Ask ME3 to build the web app first."
}

$localUrl = "http://127.0.0.1:$Port/"
$lanUrls = @()
try {
    $lanUrls = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
        Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
        Select-Object -ExpandProperty IPAddress |
        ForEach-Object { "http://$_`:$Port/" })
} catch { }

Write-Host 'ME3 local web server'
Write-Host "Directory: $candidate"
Write-Host "Local URL: $localUrl"
if ($lanUrls.Count -gt 0) {
    Write-Host ('LAN URL  : ' + ($lanUrls -join ', '))
} else {
    Write-Host 'LAN URL  : not detected; use the local URL or the machine IP.'
}
Write-Host 'Binding  : 0.0.0.0 (local machine and LAN, subject to Windows Firewall)'
Write-Host 'Stop    : Ctrl+C in this window'

if ($DryRun) {
    Write-Host '[dry-run] no server started and no browser opened.'
    exit 0
}

if ($OpenBrowser) {
    Start-Process $localUrl
}

Push-Location $candidate
try {
    & $python -m http.server $Port --bind 0.0.0.0 --directory $candidate
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
