[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9_.@:-]+$')]
    [string]$SshTarget = 'dalmacio@cfdfnjrpi5.local',
    [ValidateRange(1024, 65535)]
    [int]$LocalPort = 8090,
    [ValidateRange(1024, 65535)]
    [int]$RemotePort = 8090,
    [switch]$StartServer,
    [switch]$StopServer,
    [switch]$OpenBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
if ($StartServer -and $StopServer) {
    throw 'Choose only one of -StartServer or -StopServer.'
}

$remoteProject = '/home/dalmacio/Desktop/kiko/me3_voice_duplex'
$remoteWeb = 'web'
$pidFile = "agent_workspace/.servers/pi-http-$RemotePort.pid"
$logFile = "agent_workspace/.servers/pi-http-$RemotePort.log"
$localUrl = "http://127.0.0.1:$LocalPort/"

Write-Host 'ME3 Pi web access (SSH tunnel, ME2-style)'
Write-Host "Target   : $SshTarget"
Write-Host "Remote   : ${remoteWeb}:$RemotePort"
Write-Host "Local URL: $localUrl"
Write-Host 'Scope    : SSH loopback tunnel; not a public Internet endpoint'
Write-Host 'Stop     : Ctrl+C closes the tunnel'

$checkRemote = 'curl -fsS --max-time 3 http://127.0.0.1:' + $RemotePort +
    '/ >/dev/null && echo ME3_WEB_HTTP_200'
$startRemote = 'cd ' + $remoteProject + '; mkdir -p agent_workspace/.servers; ' +
    'if [ ! -d ' + $remoteWeb + ' ]; then echo ME3_WEB_DIRECTORY_MISSING; exit 2; fi; ' +
    'if [ -f ' + $pidFile + ' ] && kill -0 $(cat ' + $pidFile + ') 2>/dev/null; then ' +
    'echo ME3_WEB_ALREADY_RUNNING; ' +
    'else nohup .venv/bin/python -m http.server ' + $RemotePort +
    ' --bind 127.0.0.1 --directory ' + $remoteWeb +
    ' >' + $logFile + ' 2>&1 </dev/null & echo $! > ' + $pidFile + '; ' +
    'sleep 1; fi; ' + $checkRemote

$stopRemote = 'cd ' + $remoteProject + '; ' +
    'if [ -f ' + $pidFile + ' ]; then kill $(cat ' + $pidFile + ') 2>/dev/null || true; rm -f ' +
    $pidFile + '; echo ME3_WEB_STOPPED; else echo ME3_WEB_NOT_STARTED_BY_LAUNCHER; fi'

if ($DryRun) {
    Write-Host '[dry-run] no SSH command executed.'
    if ($StartServer) { Write-Host "[dry-run] start: $startRemote" }
    if ($StopServer) { Write-Host "[dry-run] stop : $stopRemote" }
    Write-Host "[dry-run] tunnel: ssh -N -L $LocalPort`:127.0.0.1`:$RemotePort $SshTarget"
    exit 0
}

if ($StopServer) {
    & ssh -T -o BatchMode=yes -o ConnectTimeout=10 $SshTarget $stopRemote
    exit $LASTEXITCODE
}

if ($StartServer) {
    & ssh -T -o BatchMode=yes -o ConnectTimeout=10 $SshTarget $startRemote
    if ($LASTEXITCODE -ne 0) {
        throw "Pi web server did not start or return HTTP 200 (exit $LASTEXITCODE)."
    }
}

if ($OpenBrowser) {
    Start-Process $localUrl
}

& ssh -T -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 `
    -o ConnectTimeout=10 -N -L "$LocalPort`:127.0.0.1`:$RemotePort" $SshTarget
exit $LASTEXITCODE
