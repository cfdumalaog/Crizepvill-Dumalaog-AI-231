param(
    [string]$Device = 'auto',
    [ValidateSet('auto', 'remote', 'qwen', 'luna', 'stub')]
    [string]$Brain = 'auto',
    [ValidateSet('current', 'qwen_instruct')]
    [string]$Variant = 'qwen_instruct',
    [string]$QwenInstructSecretFile = '',
    [switch]$Web
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspace = (Resolve-Path (Join-Path $project '..\..\..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Shared workspace .venv python not found at $python"
}

$env:ME3_VARIANT = $Variant
$env:ME3_CODE_VARIANT = 'qwen'
$env:PYTHONPATH = "$project$([IO.Path]::PathSeparator)$($env:PYTHONPATH)"

# Resolve the protected DGX3 handoff only in this process. Explicit runtime
# environment values take precedence, so the same launcher works with a Pi
# secret manager later. Credentials and service addresses are never printed.
$qwenAudioKey = $null
if ($Variant -eq 'qwen_instruct') {
    if ([string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_ASR_BASE_URL) -or
        [string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_TTS_BASE_URL) -or
        [string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_API_KEY)) {
        if ([string]::IsNullOrWhiteSpace($QwenInstructSecretFile)) {
            $QwenInstructSecretFile = Join-Path $workspace 'Secrets New ME3 Model Qwen Instruct.txt'
        }
        if (Test-Path -LiteralPath $QwenInstructSecretFile -PathType Leaf) {
            $protectedLines = [IO.File]::ReadAllLines($QwenInstructSecretFile)
            $dgxIp = $null
            foreach ($line in $protectedLines) {
                if ($line -match '(?i)IP\s*Address.*?((?:\d{1,3}\.){3}\d{1,3})') {
                    $dgxIp = $Matches[1]
                    break
                }
            }
            foreach ($line in $protectedLines) {
                if ($line -match '(?i)API\s*Key.*?\bis\s+(.+?)\s*$') {
                    $qwenAudioKey = $Matches[1].Trim().Trim('`"''')
                    break
                }
            }
            if ($qwenAudioKey -and [string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_API_KEY)) {
                $env:QWEN_INSTRUCT_API_KEY = $qwenAudioKey
            }
            if ($dgxIp) {
                if ([string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_ASR_BASE_URL)) {
                    $env:QWEN_INSTRUCT_ASR_BASE_URL = "http://$dgxIp`:8000/v1"
                }
                if ([string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_TTS_BASE_URL)) {
                    $env:QWEN_INSTRUCT_TTS_BASE_URL = "http://$dgxIp`:8001/v1"
                }
            }
        }
    }
}
$qwenAudioKeyConfigured = -not [string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_API_KEY)
$qwenAsrEndpointConfigured = -not [string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_ASR_BASE_URL)
$qwenTtsEndpointConfigured = -not [string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_TTS_BASE_URL)

# Brain resolution: auto = Qwen when VLLM_API_KEY is set, then GPT-5.6 Luna
# when OPENAI_API_KEY is set, else stub. Values are only probed, never printed.
# NOTE: do NOT use
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
$openaiKeyValue = $null
foreach ($scope in 'Process', 'User', 'Machine') {
    $v = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY', $scope)
    if (-not [string]::IsNullOrWhiteSpace($v)) { $openaiKeyValue = $v; break }
}
if ($null -ne $openaiKeyValue -and [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    $env:OPENAI_API_KEY = $openaiKeyValue
}
$haveOpenAIKey = $null -ne $openaiKeyValue
$brainMode = $Brain
if ($brainMode -eq 'auto') {
    $brainMode = if ($haveKey) { 'remote' } elseif ($haveOpenAIKey) { 'luna' } else { 'stub' }
}
if ($brainMode -eq 'qwen') { $brainMode = 'remote' }

$title = if ($Web) { 'ME3 - Voice Duplex (local browser console)' } else { 'ME3 - Voice Duplex (local full-duplex loop)' }
Write-Host $title
Write-Host "Project : $project"
Write-Host "Python  : $python"
Write-Host "LLM     : $brainMode (Qwen key $(if ($haveKey) { 'configured' } else { 'not set' }); Luna key $(if ($haveOpenAIKey) { 'configured' } else { 'not set' }))"
Write-Host "Audio   : $Variant (Qwen ASR endpoint $(if ($qwenAsrEndpointConfigured) { 'configured' } else { 'not set' }); TTS endpoint $(if ($qwenTtsEndpointConfigured) { 'configured' } else { 'not set' }); shared key $(if ($qwenAudioKeyConfigured) { 'configured' } else { 'not set' }))"
if ($Device -eq 'auto') { Write-Host 'Mic     : system default input' } else { Write-Host "Mic     : $Device" }
Write-Host 'Models  : local Silero VAD + Qwen Instruct remote ASR/TTS (active profile)'
Write-Host 'Stop    : Ctrl+C in this window (or close the window)'
Write-Host 'Assistant : http://127.0.0.1:8126/  (live voice UI - orb, state, transcript, tools)'
Write-Host 'Roadmap   : http://127.0.0.1:8126/roadmap  (what is left to do)'
Write-Host ''
Write-Host 'Full duplex: the mic stays live while the agent speaks.'
Write-Host 'Controls: protected playback; say "stop processing", "skip", or "steer".'
Write-Host 'Echo    : speaker-reference cancellation is enabled; headphones still give the cleanest barge-in.'
Write-Host ''

$arguments = if ($Web) {
    @('scripts\me3_pi_server.py', '--host', '127.0.0.1', '--port', '8126',
      '--audio-port', '8127', '--brain', $brainMode, '--variant', $Variant,
      '--output', 'browser')
} else {
    @('scripts\run_me3_module.py', 'me3.agent', '--brain', $brainMode,
      '--variant', $Variant)
}
if ($Device -ne 'auto') {
    $arguments += if ($Web) { @('--input-device', $Device) } else { @('--device', $Device) }
}

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

