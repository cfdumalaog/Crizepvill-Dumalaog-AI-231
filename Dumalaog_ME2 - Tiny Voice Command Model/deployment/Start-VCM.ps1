param(
    [string]$PiHost = "cfdfnjrpi5.local",
    [string]$PiUser = "dalmacio",
    [int]$LocalPort = 7864,
    [int]$RemotePort = 7860
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot 'Sync-VCM.ps1') -PiHost $PiHost -PiUser $PiUser
$keyPath = Join-Path $env:USERPROFILE ".ssh\id_ed25519"
$target = "$PiUser@$PiHost"
$ssh = (Get-Command ssh.exe -ErrorAction Stop).Source

if (-not (Test-Path -LiteralPath $keyPath)) {
    throw "SSH key not found at $keyPath. Set up key-based access to the Pi first."
}

$remoteStart = 'if ! curl -fsS http://127.0.0.1:7860/api/state >/dev/null 2>&1; then cd /home/dalmacio/Desktop/dandan && nohup ./launch-vcm.sh >./vcm.log 2>&1 </dev/null & fi; for n in 1 2 3 4 5 6 7 8 9 10; do curl -fsS http://127.0.0.1:7860/api/state >/dev/null 2>&1 && exit 0; sleep 1; done; exit 1'
$remoteOutput = & $ssh -i $keyPath -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=192.168.254.106 -o ConnectTimeout=8 $target $remoteStart 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Could not start or reach the VCM on the Pi. SSH/API check returned: $remoteOutput"
}

$localUrl = "http://127.0.0.1:$LocalPort/api/state"
$localReady = $false
try {
    $null = Invoke-RestMethod -Uri $localUrl -TimeoutSec 2
    $localReady = $true
} catch { }

if (-not $localReady) {
    $occupied = Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue
    if ($occupied) {
        throw "Local port $LocalPort is already in use by a service that did not answer the VCM health check. Close that process or choose another LocalPort."
    }
    $tunnelArgs = @(
        "-N", "-L", "$LocalPort`:127.0.0.1:$RemotePort",
        "-i", $keyPath,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "HostKeyAlias=192.168.254.106",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=2",
        "-o", "ExitOnForwardFailure=yes",
        $target
    )
    $quotedArgs = $tunnelArgs | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }
    Start-Process -FilePath $ssh -ArgumentList ($quotedArgs -join " ") -WindowStyle Hidden | Out-Null

    $ready = $false
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $null = Invoke-RestMethod -Uri $localUrl -TimeoutSec 2
            $ready = $true
            break
        } catch { }
    }
    if (-not $ready) { throw "The SSH tunnel did not become ready on local port $LocalPort." }
}

Start-Process "http://127.0.0.1:$LocalPort/assistant"
Write-Host "ME2 - VCM on Raspberry Pi 5 is open at http://127.0.0.1:$LocalPort/assistant"
