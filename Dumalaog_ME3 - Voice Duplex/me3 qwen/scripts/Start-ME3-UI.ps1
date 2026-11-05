[CmdletBinding()]
param(
    [ValidateSet('auto', 'qwen', 'luna', 'stub')]
    [string]$Brain = 'auto',
    [ValidateSet('current', 'qwen_instruct')]
    [string]$Variant = 'qwen_instruct',
    [string]$QwenInstructSecretFile = '',
    [ValidateRange(1024, 65535)]
    [int]$Port = 8124,
    [switch]$OpenBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspace = (Resolve-Path (Join-Path $project '..\..\..')).Path
$python = Join-Path $workspace '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Shared workspace .venv python not found at $python" }

$env:ME3_VARIANT = $Variant
$env:ME3_CODE_VARIANT = 'qwen'
$env:PYTHONPATH = "$project$([IO.Path]::PathSeparator)$($env:PYTHONPATH)"
$qwenAudioKey = $null
if ($Variant -eq 'qwen_instruct') {
    if ([string]::IsNullOrWhiteSpace($QwenInstructSecretFile)) {
        $QwenInstructSecretFile = Join-Path $workspace 'Secrets New ME3 Model Qwen Instruct.txt'
    }
    if (Test-Path -LiteralPath $QwenInstructSecretFile -PathType Leaf) {
        $protectedLines = [IO.File]::ReadAllLines($QwenInstructSecretFile)
        $dgxIp = $null
        foreach ($line in $protectedLines) {
            if ($line -match '(?i)IP\s*Address.*?((?:\d{1,3}\.){3}\d{1,3})') { $dgxIp = $Matches[1]; break }
        }
        foreach ($line in $protectedLines) {
            if ($line -match '(?i)API\s*Key.*?\bis\s+(.+?)\s*$') { $qwenAudioKey = $Matches[1].Trim().Trim('`"'''); break }
        }
        if ($qwenAudioKey -and [string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_API_KEY)) { $env:QWEN_INSTRUCT_API_KEY = $qwenAudioKey }
        if ($dgxIp) {
            if ([string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_ASR_BASE_URL)) { $env:QWEN_INSTRUCT_ASR_BASE_URL = "http://$dgxIp`:8000/v1" }
            if ([string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_TTS_BASE_URL)) { $env:QWEN_INSTRUCT_TTS_BASE_URL = "http://$dgxIp`:8001/v1" }
        }
    }
}
$qwenAudioKeyConfigured = -not [string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_API_KEY)
$qwenAsrEndpointConfigured = -not [string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_ASR_BASE_URL)
$qwenTtsEndpointConfigured = -not [string]::IsNullOrWhiteSpace($env:QWEN_INSTRUCT_TTS_BASE_URL)

# Resolve protected inference credentials into this process only. Values are
# never printed, persisted, or passed as command-line arguments.
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
$openaiKeyValue = $null
foreach ($scope in 'Process', 'User', 'Machine') {
    $v = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY', $scope)
    if (-not [string]::IsNullOrWhiteSpace($v)) { $openaiKeyValue = $v; break }
}
if ($null -ne $openaiKeyValue -and [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    $env:OPENAI_API_KEY = $openaiKeyValue
}
$haveOpenAIKey = $null -ne $openaiKeyValue

Write-Host 'ME3 local browser UI'
Write-Host "URL     : http://127.0.0.1:$Port/"
Write-Host "Brain   : $Brain (Qwen key $(if ($haveKey) { 'configured' } else { 'not configured' }); Luna key $(if ($haveOpenAIKey) { 'configured' } else { 'not configured' }))"
Write-Host "Audio   : $Variant (ASR endpoint $(if ($qwenAsrEndpointConfigured) { 'configured' } else { 'not configured' }); TTS endpoint $(if ($qwenTtsEndpointConfigured) { 'configured' } else { 'not configured' }); shared key $(if ($qwenAudioKeyConfigured) { 'configured' } else { 'not configured' }))"
Write-Host 'Policy  : Twilio calls, SMS, and Resend email hard-disabled in this UI'
Write-Host 'Binding : 127.0.0.1 only'
Write-Host 'Stop    : Ctrl+C in this window'
if ($DryRun) { Write-Host '[dry-run] no server started and no browser opened.'; exit 0 }
if ($OpenBrowser) { Start-Process "http://127.0.0.1:$Port/" }
Push-Location $project
try {
    & $python scripts\local_ui_server.py --host 127.0.0.1 --port $Port --default-mode $Brain --variant $Variant
    exit $LASTEXITCODE
} finally { Pop-Location }
