[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9_.@:-]+$')]
    [string]$SshTarget = 'dalmacio@cfdfnjrpi5.local',
    [ValidatePattern('^/[A-Za-z0-9_./-]+$')]
    [string]$RemoteProject = '/home/dalmacio/Desktop/dandan-qwen',
    [ValidateRange(1024, 65535)]
    [int]$LocalPort = 8090,
    [ValidateRange(1024, 65535)]
    [int]$LocalAudioPort = 8091,
    [ValidateRange(1024, 65535)]
    [int]$RemotePort = 8126,
    [ValidateRange(1024, 65535)]
    [int]$RemoteAudioPort = 8127,
    [ValidateSet('auto', 'remote', 'qwen', 'luna', 'stub')]
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

# This is the selected ME3 deployment. The protected Kiko tree is user-owned
# and is intentionally not touched by the ME3 launcher. The Own and Qwen
# launchers pass different directories so they can run side by side.
$pidFile = "agent_workspace/.servers/me3-pi-server-$RemotePort.pid"
$logFile = "agent_workspace/.servers/me3-pi-server-$RemotePort.log"
$localUrl = "http://127.0.0.1:$LocalPort/"

Write-Host 'ME3 Pi web + browser audio (ME2-style SSH tunnel)'
Write-Host "Target       : $SshTarget"
Write-Host "Remote path  : $RemoteProject"
Write-Host "Remote UI    : 0.0.0.0:$RemotePort"
Write-Host "Remote audio : 0.0.0.0:$RemoteAudioPort"
Write-Host "Local URL    : $localUrl"
Write-Host 'Audio        : Pi microphone -> ME3 -> connected browser speaker by default'
Write-Host 'Scope        : SSH loopback tunnel; not a public Internet endpoint'
if ($Variant -eq 'qwen_instruct' -and $UseReverseTunnel) { Write-Host 'Qwen audio  : legacy SSH reverse tunnel to workstation VPN DGX 10.153.60.233:8000/8001' }
if ($Variant -eq 'qwen_instruct' -and -not $UseReverseTunnel) { Write-Host 'Qwen audio  : direct Pi SHARC VPN route to DGX 10.153.60.233:8000/8001' }

$envBootstrap = 'if [ -f "$HOME/.config/me3/me3.env" ]; then set -a; . "$HOME/.config/me3/me3.env"; set +a; fi; '
$qwenEnvBootstrap = if ($Variant -eq 'qwen_instruct') { 'if [ -f "$HOME/.config/me3/qwen-instruct.env" ]; then set -a; . "$HOME/.config/me3/qwen-instruct.env"; set +a; fi; ' } else { '' }
$variantBootstrap = if ($Variant -eq 'qwen_instruct') { 'export ME3_TTS_ENGINE=qwen_remote; ' } else { '' }
$checkRemote = 'curl -fsS --max-time 3 http://127.0.0.1:' + $RemotePort + '/api/health >/dev/null && echo ME3_WEB_HTTP_200'
$startRemote = 'cd ' + $RemoteProject + '; ' + $envBootstrap + $qwenEnvBootstrap + $variantBootstrap + 'mkdir -p agent_workspace/.servers; ' +
    'if [ -f ' + $pidFile + ' ] && kill -0 $(cat ' + $pidFile + ') 2>/dev/null; then ' +
    'echo ME3_WEB_ALREADY_RUNNING; ' +
    'else nohup .venv/bin/python scripts/me3_pi_server.py --host 0.0.0.0 --port ' + $RemotePort +
    ' --audio-port ' + $RemoteAudioPort + ' --brain ' + $Brain + ' --variant ' + $Variant +
    ' >' + $logFile + ' 2>&1 </dev/null & echo $! > ' + $pidFile + '; fi; ' +
    'i=0; while [ $i -lt 15 ]; do if ' + $checkRemote + ' >/dev/null; then ' +
    $checkRemote + '; exit 0; fi; i=$((i+1)); sleep 1; done; ' + $checkRemote
$stopRemote = 'cd ' + $RemoteProject + '; ' +
    'if [ -f ' + $pidFile + ' ]; then kill $(cat ' + $pidFile + ') 2>/dev/null || true; rm -f ' +
    $pidFile + '; echo ME3_WEB_STOPPED; else echo ME3_WEB_NOT_STARTED_BY_LAUNCHER; fi'

if ($DryRun) {
    Write-Host '[dry-run] no SSH command executed.'
    if ($StartServer) { Write-Host "[dry-run] start: $startRemote" }
    if ($StopServer) { Write-Host "[dry-run] stop : $stopRemote" }
    $reverse = if ($Variant -eq 'qwen_instruct' -and $UseReverseTunnel) { ' -R 18000:10.153.60.233:8000 -R 18001:10.153.60.233:8001' } else { '' }
    Write-Host "[dry-run] tunnel: ssh -N -L $LocalPort`:127.0.0.1`:$RemotePort -L $LocalAudioPort`:127.0.0.1`:$RemoteAudioPort$reverse $SshTarget"
    exit 0
}

if ($StopServer) {
    & ssh -T -o BatchMode=yes -o ConnectTimeout=10 $SshTarget $stopRemote
    exit $LASTEXITCODE
}

if ($StartServer) {
    & ssh -T -o BatchMode=yes -o ConnectTimeout=10 $SshTarget $startRemote
    if ($LASTEXITCODE -ne 0) { throw "Pi ME3 web/audio server did not start (exit $LASTEXITCODE)." }
}

if ($OpenBrowser) { Start-Process $localUrl }
Write-Host 'Stop       : Ctrl+C closes the tunnel; use -StopServer to stop the Pi process.'
$sshArgs = @(
    '-T', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
    '-o', 'ServerAliveInterval=30', '-o', 'ConnectTimeout=10', '-N',
    '-L', "$LocalPort`:127.0.0.1`:$RemotePort",
    '-L', "$LocalAudioPort`:127.0.0.1`:$RemoteAudioPort"
)
if ($Variant -eq 'qwen_instruct' -and $UseReverseTunnel) {
    $sshArgs += @('-R', '18000:10.153.60.233:8000', '-R', '18001:10.153.60.233:8001')
}
$sshArgs += $SshTarget
& ssh @sshArgs
exit $LASTEXITCODE
