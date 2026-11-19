[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9_.@:-]+$')]
    [string]$SshTarget = 'dalmacio@cfdfnjrpi5.local',
    [ValidatePattern('^[A-Za-z0-9_.-]+$')]
    [string]$HostKeyAlias = '192.168.254.105',
    [ValidatePattern('^/[A-Za-z0-9_./-]+$')]
    [string]$RemoteProject = '/home/dalmacio/Desktop/dandan-qwen',
    [ValidateRange(1024, 65535)]
    [int]$LocalPort = 8100,
    [ValidateRange(1024, 65535)]
    [int]$LocalAudioPort = 8101,
    [ValidateRange(1024, 65535)]
    [int]$RemotePort = 8126,
    [ValidateRange(1024, 65535)]
    [int]$RemoteAudioPort = 8127,
    [ValidateSet('auto', 'remote', 'qwen', 'openai', 'luna', 'stub')]
    [string]$Brain = 'auto',
    [ValidateSet('current', 'qwen_instruct')]
    [string]$Variant = 'qwen_instruct',
    [switch]$StartServer,
    [switch]$StopServer,
    [switch]$UseReverseTunnel,
    [switch]$OpenBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
if ($StartServer -and $StopServer) { throw 'Choose only one of -StartServer or -StopServer.' }

# The Pi hostname is the persistent connection identity. Its Wi-Fi address is
# intentionally DHCP, because a demo network can change. If Windows mDNS is
# blocked by a VPN client, use the Pi's stable Wi-Fi MAC from the neighbor
# table; this discovers the current DHCP address without hard-coding it.
$directHost = $null
if (-not $DryRun) {
    $targetParts = $SshTarget -split '@', 2
    if ($targetParts.Count -eq 2) {
        try {
            Resolve-DnsName -Name $targetParts[1] -ErrorAction Stop | Out-Null
            $directHost = $targetParts[1]
        } catch {
            $piMac = '2C-CF-67-70-8D-47'
            $neighbor = Get-NetNeighbor -AddressFamily IPv4 -ErrorAction SilentlyContinue |
                Where-Object { $_.LinkLayerAddress -eq $piMac -and
                    $_.State -notin @('Unreachable', 'Incomplete') } |
                Select-Object -First 1
            if ($neighbor) {
                $SshTarget = "$($targetParts[0])@$($neighbor.IPAddress)"
                $directHost = $neighbor.IPAddress
                Write-Host "Pi hostname did not resolve; using current hotspot lease $($neighbor.IPAddress)."
            }
        }
    }
}

# Qwen Instruct is the sole active ME3 deployment. The protected Kiko tree is
# user-owned and is intentionally not touched by this launcher.
$pidFile = "agent_workspace/.servers/me3-pi-server-$RemotePort.pid"
$logFile = "agent_workspace/.servers/me3-pi-server-$RemotePort.log"
$localUrl = "http://127.0.0.1:$LocalPort/"

Write-Host 'ME3 Pi web + browser audio (ME2-style SSH tunnel)'
Write-Host "Target       : $SshTarget"
Write-Host "Host key ID  : $HostKeyAlias"
Write-Host "Remote path  : $RemoteProject"
Write-Host "Remote UI    : 0.0.0.0:$RemotePort"
Write-Host "Remote audio : 0.0.0.0:$RemoteAudioPort"
if ($directHost) {
    Write-Host ("Direct URL   : http://{0}:{1}/ (current LAN lease)" -f $directHost, $RemotePort)
} else {
    Write-Host "Direct URL   : http://cfdfnjrpi5.local:$RemotePort/ (when mDNS resolves)"
}
Write-Host "Local URL    : $localUrl"
Write-Host 'Audio        : Pi microphone -> ME3 -> connected browser speaker by default'
Write-Host 'Scope        : SSH loopback tunnel; not a public Internet endpoint'
if ($Variant -eq 'qwen_instruct' -and $UseReverseTunnel) { Write-Host 'Qwen models : laptop VPN bridge for text inference, ASR, and TTS' }
if ($Variant -eq 'qwen_instruct' -and -not $UseReverseTunnel) { Write-Host 'Qwen audio  : direct Pi SHARC VPN route to DGX 10.153.60.233:8000/8001' }

$envBootstrap = 'if [ -f "$HOME/.config/me3/me3.env" ]; then set -a; . "$HOME/.config/me3/me3.env"; set +a; fi; '
$qwenEnvBootstrap = if ($Variant -eq 'qwen_instruct') { 'if [ -f "$HOME/.config/me3/qwen-instruct.env" ]; then set -a; . "$HOME/.config/me3/qwen-instruct.env"; set +a; fi; ' } else { '' }
$variantBootstrap = if ($Variant -eq 'qwen_instruct') { 'export ME3_TTS_ENGINE=qwen_remote; ' } else { '' }
$checkRemote = 'curl -fsS --max-time 3 http://127.0.0.1:' + $RemotePort + '/api/health >/dev/null && echo ME3_WEB_HTTP_200'
$modelRoute = if ($Variant -eq 'qwen_instruct' -and $UseReverseTunnel) {
    'systemctl --user set-environment ME3_QWEN_LLM_BASE_URL=http://127.0.0.1:18002/v1 ' +
    'ME3_QWEN_ASR_BASE_URL=http://127.0.0.1:18000/v1 ' +
    'ME3_QWEN_TTS_BASE_URL=http://127.0.0.1:18001/v1; '
} else {
    'systemctl --user unset-environment ME3_QWEN_LLM_BASE_URL ME3_QWEN_ASR_BASE_URL ME3_QWEN_TTS_BASE_URL; '
}
$legacyStart = 'mkdir -p agent_workspace/.servers; ' +
    'if ' + $checkRemote + ' >/dev/null 2>&1; then echo ME3_WEB_ALREADY_RUNNING; ' +
    'elif [ -f ' + $pidFile + ' ] && kill -0 $(cat ' + $pidFile + ') 2>/dev/null; then echo ME3_WEB_ALREADY_RUNNING; ' +
    'else nohup .venv/bin/python scripts/me3_pi_server.py --host 0.0.0.0 --port ' + $RemotePort +
    ' --audio-port ' + $RemoteAudioPort + ' --brain ' + $Brain + ' --variant ' + $Variant +
    ' >' + $logFile + ' 2>&1 </dev/null & echo $! > ' + $pidFile + '; fi; '
$startRemote = 'cd ' + $RemoteProject + '; ' + $envBootstrap + $qwenEnvBootstrap + $variantBootstrap +
    'if systemctl --user cat me3-agent.service >/dev/null 2>&1; then ' + $modelRoute +
    'systemctl --user restart me3-agent.service; echo ME3_SYSTEMD_RESTARTED; else ' + $legacyStart + 'fi; ' +
    'i=0; while [ $i -lt 20 ]; do if ' + $checkRemote + ' >/dev/null; then ' +
    $checkRemote + '; exit 0; fi; i=$((i+1)); sleep 1; done; ' + $checkRemote
$stopRemote = 'cd ' + $RemoteProject + '; ' +
    'if systemctl --user cat me3-agent.service >/dev/null 2>&1; then systemctl --user stop me3-agent.service; echo ME3_SYSTEMD_STOPPED; ' +
    'elif [ -f ' + $pidFile + ' ]; then kill $(cat ' + $pidFile + ') 2>/dev/null || true; rm -f ' +
    $pidFile + '; echo ME3_WEB_STOPPED; else echo ME3_WEB_NOT_STARTED_BY_LAUNCHER; fi'

if ($DryRun) {
    Write-Host '[dry-run] no SSH command executed.'
    if ($StartServer) { Write-Host "[dry-run] start: $startRemote" }
    if ($StopServer) { Write-Host "[dry-run] stop : $stopRemote" }
    $reverse = if ($Variant -eq 'qwen_instruct' -and $UseReverseTunnel) { ' -R 18000:10.153.60.233:8000 -R 18001:10.153.60.233:8001 -R 18002:202.92.159.240:8001' } else { '' }
    Write-Host "[dry-run] tunnel: ssh -N -L $LocalPort`:127.0.0.1`:$RemotePort -L $LocalAudioPort`:127.0.0.1`:$RemoteAudioPort$reverse $SshTarget"
    exit 0
}

if ($StopServer) {
    & ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o "HostKeyAlias=$HostKeyAlias" $SshTarget $stopRemote
    exit $LASTEXITCODE
}

if ($StartServer) {
    & ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o "HostKeyAlias=$HostKeyAlias" $SshTarget $startRemote
    if ($LASTEXITCODE -ne 0) { throw "Pi ME3 web/audio server did not start (exit $LASTEXITCODE)." }
}

if ($OpenBrowser) { Start-Process $localUrl }
Write-Host 'Stop       : Ctrl+C closes the tunnel; use -StopServer to stop the Pi process.'
$sshArgs = @(
    '-T', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
    '-o', 'ServerAliveInterval=30', '-o', 'ConnectTimeout=10',
    '-o', "HostKeyAlias=$HostKeyAlias", '-N',
    '-L', "$LocalPort`:127.0.0.1`:$RemotePort",
    '-L', "$LocalAudioPort`:127.0.0.1`:$RemoteAudioPort"
)
if ($Variant -eq 'qwen_instruct' -and $UseReverseTunnel) {
    $sshArgs += @('-R', '18000:10.153.60.233:8000', '-R', '18001:10.153.60.233:8001',
                  '-R', '18002:202.92.159.240:8001')
}
$sshArgs += $SshTarget
& ssh @sshArgs
exit $LASTEXITCODE
