param(
    [string]$Device = 'auto',
    [ValidateSet('auto', 'remote', 'stub')]
    [string]$Brain = 'auto'
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspace = (Resolve-Path (Join-Path $project '..\..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Shared workspace .venv python not found at $python"
}

# Brain resolution: auto = remote vLLM brain when VLLM_API_KEY is set in any
# scope, else stub. The value is only probed, never printed. NOTE: do NOT use
# `a -or b` to coalesce here - PowerShell -or returns a boolean, so the
# old check read "False" as a non-empty string and always claimed the key
# was configured. When the key lives in a User/Machine scope, copy it into
# this process env so the python child inherits it (in memory only).
$keyValue = $null
foreach ($scope in 'Process', 'User', 'Machine') {
    $v = [Environment]::GetEnvironmentVariable('VLLM_API_KEY', $scope)
    if (-not [string]::IsNullOrWhiteSpace($v)) { $keyValue = $v; break }
}

# Fallback: the Codex vLLM profile DPAPI credential helper. The key is
# decrypted into memory only, never printed or persisted anywhere else.
if ($null -eq $keyValue) {
    $helper = Join-Path $env:USERPROFILE '.codex\Get-VllmApiKey.ps1'
    if (Test-Path -LiteralPath $helper) {
        try {
            $h = & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $helper
            if (-not [string]::IsNullOrWhiteSpace($h)) { $keyValue = $h.Trim() }
        } catch { }
    }
}
if ($null -ne $keyValue -and [string]::IsNullOrWhiteSpace($env:VLLM_API_KEY)) {
    $env:VLLM_API_KEY = $keyValue
}
$haveKey = $null -ne $keyValue
$brainMode = $Brain
if ($brainMode -eq 'auto') { $brainMode = if ($haveKey) { 'remote' } else { 'stub' } }

Write-Host 'ME3 - Voice Duplex (local full-duplex loop)'
Write-Host "Project : $project"
Write-Host "Python  : $python"
Write-Host "LLM     : $brainMode (VLLM_API_KEY $(if ($haveKey) { 'configured' } else { 'not set' }))"
if ($Device -eq 'auto') { Write-Host 'Mic     : system default input' } else { Write-Host "Mic     : $Device" }
Write-Host 'Models  : silero VAD + whisper small.en + piper voice (download on first run if missing)'
Write-Host 'Stop    : Ctrl+C in this window (or close the window)'
Write-Host ''
Write-Host 'Full duplex: the mic stays live while the agent speaks.'
Write-Host 'Barge in : start talking to interrupt the agent mid-sentence.'
Write-Host 'Tip     : use headphones so the mic does not pick up the agent.'
Write-Host ''

$arguments = @('-m', 'me3.agent', '--brain', $brainMode)
if ($Device -ne 'auto') { $arguments += @('--device', $Device) }

Push-Location $project
try {
    & $python @arguments
}
finally {
    Pop-Location
}
# Propagate python's exit code so the .bat can `pause` on failure and the
# error message stays visible in the window.
exit $LASTEXITCODE

