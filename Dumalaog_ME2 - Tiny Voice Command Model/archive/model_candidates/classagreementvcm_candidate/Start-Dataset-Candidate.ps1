param(
    [string]$PiHost = 'cfdfnjrpi5.local'
)

$ErrorActionPreference = 'Stop'
$piTarget = "dalmacio@$PiHost"
$expectedIntent = '4af9e1a0a6fb8a3a6c98eb02ed49d02dab6b92b0f18c47adbadaffb77ced35d1'
$expectedSource = 'classagreementvcm-20261002-a90b8d10-r3'
$remoteStart = 'if ! pgrep -af "[m]e2-dataset-candidate/vcm_app.py" >/dev/null; then cd /home/dalmacio/Desktop/me2-dataset-candidate && nohup ./launch-vcm.sh >/dev/null 2>&1 </dev/null & fi'

& ssh.exe -o BatchMode=yes -o ConnectTimeout=10 $piTarget $remoteStart
if ($LASTEXITCODE -ne 0) { throw "Cannot reach Pi or start the isolated candidate (ssh exit $LASTEXITCODE)." }

$existing = Get-NetTCPConnection -LocalPort 7865 -State Listen -ErrorAction SilentlyContinue
if (-not $existing) {
    $sshPath = (Get-Command ssh.exe).Source
    Start-Process -FilePath $sshPath -ArgumentList @(
        '-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
        '-L', '127.0.0.1:7865:127.0.0.1:7865', $piTarget
    ) -WindowStyle Hidden
    Start-Sleep -Seconds 2
}

$state = Invoke-RestMethod -Uri 'http://127.0.0.1:7865/api/state' -TimeoutSec 8
if ($state.model_source_run -ne $expectedSource -or $state.intent_model_sha256 -ne $expectedIntent) {
    throw 'Port 7865 is not serving the frozen dataset candidate; refusing to open it.'
}
Write-Host 'Dataset candidate is ready:'
Write-Host '  http://127.0.0.1:7865/assistant'
Write-Host '  http://127.0.0.1:7865/studio'
Write-Host "  Microphone: $($state.microphone)"
Start-Process 'http://127.0.0.1:7865/assistant'
