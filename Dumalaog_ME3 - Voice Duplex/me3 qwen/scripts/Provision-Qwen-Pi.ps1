[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9_.@:-]+$')]
    [string]$SshTarget = 'dalmacio@cfdfnjrpi5.local',
    [ValidatePattern('^[A-Za-z0-9_.-]+$')]
    [string]$HostKeyAlias = '192.168.254.105',
    [ValidatePattern('^(?:\d{1,3}\.){3}\d{1,3}$')]
    [string]$DgxIp = '10.153.60.233',
    [switch]$UsePiBrainCredential,
    [switch]$UseReverseTunnel,
    [string]$SecretFile = ''
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspace = (Resolve-Path (Join-Path $project '..\..\..')).Path
if ($UsePiBrainCredential -and $UseReverseTunnel) {
    throw 'Choose only one of -UsePiBrainCredential or -UseReverseTunnel.'
}
if ($UsePiBrainCredential) {
    # The routed Pi endpoint accepts the credential already present in the
    # protected Pi ME3 environment, not the workstation-only DGX handoff key.
    $envText = ''
} else {
    if ([string]::IsNullOrWhiteSpace($SecretFile)) {
        $SecretFile = Join-Path $workspace 'Secrets New ME3 Model Qwen Instruct.txt'
    }
    if (-not (Test-Path -LiteralPath $SecretFile -PathType Leaf)) {
        throw "Protected Qwen handoff file is missing: $SecretFile"
    }

    # Read the protected handoff only in this process. The key is never
    # printed, persisted on Windows, or passed as an SSH command-line argument.
    $protectedLines = [IO.File]::ReadAllLines($SecretFile)
    $qwenKey = $null
    foreach ($line in $protectedLines) {
        if ($line -match '(?i)API\s*Key.*?\bis\s+(.+?)\s*$') {
            $qwenKey = $Matches[1].Trim().Trim('`"''')
            break
        }
    }
    if ([string]::IsNullOrWhiteSpace($qwenKey)) {
        throw 'Protected Qwen handoff is missing the API key.'
    }

    $asrEndpoint = if ($UseReverseTunnel) { 'http://127.0.0.1:18000/v1' } else { "http://$DgxIp`:8000/v1" }
    $ttsEndpoint = if ($UseReverseTunnel) { 'http://127.0.0.1:18001/v1' } else { "http://$DgxIp`:8001/v1" }
    $envText = @(
        "QWEN_INSTRUCT_ASR_BASE_URL=$asrEndpoint"
        "QWEN_INSTRUCT_TTS_BASE_URL=$ttsEndpoint"
        "QWEN_INSTRUCT_API_KEY=$qwenKey"
        "ME3_ASR_FALLBACK=faster_whisper"
        "ME3_ASR_FALLBACK_MODEL=base"
    ) -join "`n"
    $envText += "`n"
}

# Keep the three Qwen audio values in a dedicated mode-600 file. The ordinary
# installer may rewrite me3.env during upgrades; this file is intentionally
# outside that generated file and is loaded only by the Qwen launcher.
$remoteWrite = if ($UsePiBrainCredential) {
    'set -e; umask 077; set -a; . "$HOME/.config/me3/me3.env"; set +a; test -n "$VLLM_API_KEY"; mkdir -p "$HOME/.config/me3"; { printf "%s\\n" "QWEN_INSTRUCT_ASR_BASE_URL=http://10.153.60.233:8000/v1"; printf "%s\\n" "QWEN_INSTRUCT_TTS_BASE_URL=http://10.153.60.233:8001/v1"; printf "%s\\n" "QWEN_INSTRUCT_API_KEY=$VLLM_API_KEY"; } > "$HOME/.config/me3/qwen-instruct.env"; chmod 600 "$HOME/.config/me3/qwen-instruct.env"; echo QWEN_ENV_PROVISIONED'
} else {
    'set -e; umask 077; mkdir -p "$HOME/.config/me3"; cat > "$HOME/.config/me3/qwen-instruct.env"; chmod 600 "$HOME/.config/me3/qwen-instruct.env"; echo QWEN_ENV_PROVISIONED'
}
$psi = [Diagnostics.ProcessStartInfo]::new()
$psi.FileName = 'ssh'
$null = $psi.ArgumentList.Add('-T')
$null = $psi.ArgumentList.Add('-o')
$null = $psi.ArgumentList.Add('BatchMode=yes')
$null = $psi.ArgumentList.Add('-o')
$null = $psi.ArgumentList.Add('ConnectTimeout=10')
$null = $psi.ArgumentList.Add('-o')
$null = $psi.ArgumentList.Add("HostKeyAlias=$HostKeyAlias")
$null = $psi.ArgumentList.Add($SshTarget)
$null = $psi.ArgumentList.Add($remoteWrite)
$psi.UseShellExecute = $false
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$process = [Diagnostics.Process]::Start($psi)
$process.StandardInput.Write($envText)
$process.StandardInput.Close()
$stdout = $process.StandardOutput.ReadToEnd()
$stderr = $process.StandardError.ReadToEnd()
$process.WaitForExit()
if ($process.ExitCode -ne 0 -or $stdout.Trim() -ne 'QWEN_ENV_PROVISIONED') {
    if ($stderr) { Write-Error $stderr.Trim() }
    throw "Qwen environment provisioning failed (exit $($process.ExitCode))."
}

$verify = 'for n in QWEN_INSTRUCT_ASR_BASE_URL QWEN_INSTRUCT_TTS_BASE_URL QWEN_INSTRUCT_API_KEY; do if grep -q "^${n}=." "$HOME/.config/me3/qwen-instruct.env"; then printf "%s=set\\n" "$n"; else printf "%s=missing\\n" "$n"; fi; done; stat -c "MODE=%a" "$HOME/.config/me3/qwen-instruct.env"'
& ssh -T -o BatchMode=yes -o ConnectTimeout=10 $SshTarget $verify
if ($LASTEXITCODE -ne 0) { throw "Qwen environment verification failed (exit $LASTEXITCODE)." }
Write-Host 'Qwen audio environment provisioned on the Pi; values were not printed.'
