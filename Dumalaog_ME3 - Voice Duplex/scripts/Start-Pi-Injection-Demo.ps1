[CmdletBinding()]
param(
    [ValidateSet('Status', 'WebApp')]
    [string]$Scenario = 'Status',
    [string]$SshTarget = 'dalmacio@cfdfnjrpi5.local',
    [string]$WebUrl = 'http://192.168.254.106:8090/',
    [switch]$OpenBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# These are deliberately fixed, reviewed fixtures. Keeping the scenario
# selection closed prevents an accidental arbitrary command/path from being
# interpolated into the SSH command used for a classroom demo.
$fixture = switch ($Scenario) {
    'Status' { 'tell_me_system_status.wav' }
    'WebApp' { 'build_webapp.wav' }
}
$interrupt = if ($Scenario -eq 'Status') {
    '--interrupt-wav runs/injection_eval_inputs/what_time_is_it.wav'
} else {
    ''
}
$remoteProject = '/home/dalmacio/Desktop/kiko/me3_voice_duplex'
$remoteEnv = '/home/dalmacio/.config/me3/me3.env'
$remoteCommand = @(
    "cd $remoteProject"
    '. .venv/bin/activate'
    "set -a && . $remoteEnv && set +a"
    "timeout 150s python scripts/injection_eval.py --input-wav runs/injection_eval_inputs/$fixture $interrupt --brain remote --max-seconds 60"
) -join ' && '

Write-Host "ME3 Pi injection demo: $Scenario"
Write-Host "Target: $SshTarget"
Write-Host "Fixture: $fixture"
Write-Host 'Mode: remote Qwen/tool loop with virtual audio sink'
Write-Host 'Credentials: loaded on the Pi only; never printed by this launcher'

if ($DryRun) {
    Write-Host '[dry-run] SSH command prepared; no remote command executed.'
    Write-Host "[dry-run] $remoteCommand"
    exit 0
}

& ssh -T -o BatchMode=yes -o ConnectTimeout=10 $SshTarget $remoteCommand
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0 -and $OpenBrowser) {
    Write-Host "Opening $WebUrl"
    Start-Process $WebUrl
}

exit $exitCode
