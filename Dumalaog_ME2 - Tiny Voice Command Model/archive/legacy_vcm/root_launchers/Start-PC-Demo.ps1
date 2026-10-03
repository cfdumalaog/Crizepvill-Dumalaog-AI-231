param([int]$Device = -1, [switch]$NoSound)
$ErrorActionPreference = 'Stop'
$vcmProject = Join-Path $PSScriptRoot 'AI 231\Dumalaog_ME2 - Tiny Voice Command Model'
$selection = if ($Device -ge 0) { [string]$Device } else { 'auto' }
if ($NoSound) { Write-Warning 'Mute voice feedback in the Assistant audio controls.' }
& (Join-Path $vcmProject 'scripts\Start-Local-VCM.ps1') -Device $selection
