# ME3 workstation-first development and Pi promotion

ME3 is finalized on the Windows workstation first. The Raspberry Pi is a
deployment and hardware-validation target, not the primary development
environment. A Pi deployment is permitted only for the exact release that
passes the local acceptance gate.

## The release rule

```text
edit on Windows
  -> local tests and MCP/audio smoke
  -> inspect the presentation-ready artifacts
  -> package and hash the exact source tree
  -> write workstation acceptance record
  -> deploy only that accepted release to the Pi
  -> run Pi audio preflight and hardware acceptance separately
```

The local gate does not contact the Pi and does not place calls, SMS messages,
emails, or other external side effects. It records its result in
`deployment/workstation_acceptance.json`, bound to the release ID, source
fingerprint, and archive SHA-256. `scripts/deploy_to_pi.py` refuses to transfer
a package if that record is missing, failed, stale, or mismatched.

## Windows acceptance command

From the ME3 project directory, use the shared workspace environment:

```powershell
$py = '..\..\.venv\Scripts\python.exe'
& $py scripts\workstation_acceptance.py
```

The gate runs:

- shared dependency verification;
- all ME3 unit tests;
- Python compilation;
- `bash -n install.sh`;
- four-MCP and Windows PortAudio smoke checks;
- paper/source alignment;
- ME3-scoped Git whitespace validation;
- presence of the README, manual, API guide, ACM source/PDF, runbook, pilot
  guide, and presentation;
- fresh release packaging and archive integrity verification.

For a headless machine, skip only the PortAudio query:

```powershell
& $py scripts\workstation_acceptance.py --skip-audio
```

That option does not establish a physical microphone or speaker claim.

## Review before promotion

Inspect `deployment/workstation_acceptance.json`. It must contain:

```text
status: pass
pi_deploy_allowed: true
pi_contacted: false
external_side_effects: false
```

The `release_id`, `source_fingerprint`, and `zip_sha256` in the acceptance
record must match `deployment/latest_release.json`.

Preview the transfer without contacting or modifying the Pi:

```powershell
& $py scripts\deploy_to_pi.py --dry-run
```

Only after reviewing that output should the operator transfer the accepted
bundle:

```powershell
& $py scripts\deploy_to_pi.py
```

The transfer installs the exact hashed archive. It does not copy the Windows
`.env`, Qwen credentials, ElevenLabs credentials, Twilio credentials, Resend
credentials, local run logs, model caches, or unrelated course files.

## Pi acceptance after promotion

Deployment success is not the same as hardware success. On the Pi, run:

```bash
cd /home/dalmacio/Desktop/kiko/me3_voice_duplex
source .venv/bin/activate
python scripts/pi_audio_preflight.py --json
python -m me3.agent --check --load-models
systemctl --user is-enabled me3-agent
systemctl --user is-active me3-agent
```

Enable the service only when the capture device has a positive input-channel
count and a known speaker route is available. Then perform one human spoken
turn and record the hardware evidence. Do not use a workstation acceptance
record as proof of Pi audio, Pi latency, or paper-native FULL-DUPLEX-BENCH v1.5
results.

## What the workstation gate proves

It proves that the presentable software bundle is internally consistent and
eligible for deployment. It can cover Qwen/tool tests, injected audio, local
MCP servers, Windows microphone/speaker tests, ElevenLabs playback, local web
apps, and fake-request channel tests.

It does not by itself prove:

- a live Twilio call, SMS, or Resend email;
- a production HTTPS Twilio media-stream bridge;
- a Filipino ElevenLabs voice until the approved voice ID is tested;
- Raspberry Pi microphone or speaker operation;
- a paper-native 499-trial overlap result.
