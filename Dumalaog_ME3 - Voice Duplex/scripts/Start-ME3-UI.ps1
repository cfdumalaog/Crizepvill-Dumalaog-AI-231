[CmdletBinding()]
param(
    [ValidateSet('auto', 'qwen', 'stub')]
    [string]$Brain = 'auto',
    [ValidateRange(1024, 65535)]
    [int]$Port = 8124,
    [switch]$OpenBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspace = (Resolve-Path (Join-Path $project '..\..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Shared workspace .venv python not found at $python" }

# Resolve the protected Qwen credential into this process only.  The value is
# never printed, persisted, or passed as a command-line argument.
$keyValue = $null
foreach ($scope in 'Process', 'User', 'Machine') {
    $v = [Environment]::GetEnvironmentVariable('VLLM_API_KEY', $scope)
    if (-not [string]::IsNullOrWhiteSpace($v)) { $keyValue = $v; break }
}
if ($null -eq $keyValue) {
    $helper = Join-Path $env:USERPROFILE '.codex\Get-VllmApiKey.ps1'
    if (Test-Path -LiteralPath $helper) {
        try {
            $h = & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $helper
            if (-not [string]::IsNullOrWhiteSpace($h)) { $keyValue = $h.Trim() }
        } catch { }
    }
}
if ($null -ne $keyValue -and [string]::IsNullOrWhiteSpace($env:VLLM_API_KEY)) { $env:VLLM_API_KEY = $keyValue }
$haveKey = $null -ne $keyValue

Write-Host 'ME3 local browser UI'
Write-Host "URL     : http://127.0.0.1:$Port/"
Write-Host "Brain   : $Brain (Qwen key $(if ($haveKey) { 'configured' } else { 'not configured' }))"
Write-Host 'Policy  : Twilio calls, SMS, and Resend email hard-disabled in this UI'
Write-Host 'Binding : 127.0.0.1 only'
Write-Host 'Stop    : Ctrl+C in this window'
if ($DryRun) { Write-Host '[dry-run] no server started and no browser opened.'; exit 0 }
if ($OpenBrowser) { Start-Process "http://127.0.0.1:$Port/" }
Push-Location $project
try {
    & $python scripts\local_ui_server.py --host 127.0.0.1 --port $Port
    exit $LASTEXITCODE
} finally { Pop-Location }
