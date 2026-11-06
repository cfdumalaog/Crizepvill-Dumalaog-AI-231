# SHARC VPN setup for the ME3 Qwen Raspberry Pi

This document records the deployment procedure used for the active Pi at
`/home/dalmacio/Desktop/dandan-qwen`. It intentionally contains no VPN
username, password, API key, or `.ovpn` profile body. Keep those values outside
the repository.

## Why the VPN is required

The Qwen audio services are protected on the SHARC/DGX network:

- ASR: `Qwen/Qwen2-Audio-7B-Instruct` at `10.153.60.233:8000`
- TTS: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` at `10.153.60.233:8001`

The earlier laptop-only arrangement worked because the laptop had the SHARC
OpenVPN route and the Pi reached the services through an SSH reverse tunnel.
The current arrangement installs the SHARC VPN on the Pi itself, so the Pi
can call the DGX services directly while retaining the ME2-style browser
forward for audio output.

## Profile and protected files

The operator-provided profile is installed on the Pi as:

```text
/etc/openvpn/client/sharchpc.conf
```

The two-line OpenVPN login file is installed separately as:

```text
/etc/openvpn/client/sharchpc.auth
```

Both files are owned by `root:root` with mode `600`. The repository must never
contain either file or a copy of its contents.

The profile also contains `auth-nocache`, which prevents OpenVPN from retaining
the login credentials after authentication. The credential file is referenced
by path from the profile rather than placed on an OpenVPN command line.

## Installation performed on the Pi

The Pi is Debian 13 ARM64. The package was installed with:

```bash
sudo apt-get update
sudo apt-get install -y openvpn
```

The supplied `.ovpn` profile was copied to the client directory, its
`auth-user-pass` line was changed to reference the protected auth file, and the
client was enabled:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now openvpn-client@sharchpc.service
```

`enable` is the important boot-persistence step: systemd creates the
`multi-user.target` dependency for the named client instance, so the SHARC
profile is started again after a Pi restart. Verify it with:

```bash
systemctl is-enabled openvpn-client@sharchpc.service
systemctl list-dependencies multi-user.target --plain | \
  grep openvpn-client@sharchpc.service
```

The named client was restarted after hardening on 2026-11-07 and restored
`tun0`, the DGX route, authenticated HTTP 200 probes to both Qwen services, and
ME3 health `loop:true`. This was a service-restart validation rather than a
full Pi reboot, so no remote session was unnecessarily interrupted.

Do not place credentials on the command line. Supply them through the
root-only auth file or an equivalent protected secret-management mechanism.

## Safe verification

These checks reveal status and routes only; they do not print credentials:

```bash
systemctl is-active openvpn-client@sharchpc.service
ip -br addr show dev tun0
ip route get 10.153.60.233
curl -sS -o /dev/null -w '%{http_code}\n' \\
  http://10.153.60.233:8000/v1/models
curl -sS -o /dev/null -w '%{http_code}\n' \\
  http://10.153.60.233:8001/v1/models
```

Unauthenticated model probes normally return `401`; that still proves that the
Pi reached the protected DGX service. ME3's authenticated probes returned HTTP
200 after the SHARC tunnel was established.

## ME3 Qwen environment

The active Pi environment is:

```text
/home/dalmacio/.config/me3/qwen-instruct.env
```

It uses the direct VPN URLs:

```text
QWEN_INSTRUCT_ASR_BASE_URL=http://10.153.60.233:8000/v1
QWEN_INSTRUCT_TTS_BASE_URL=http://10.153.60.233:8001/v1
```

The API key remains in the protected environment file and is not documented
here. The cached faster-whisper fallback remains configured as a recovery path,
but the active verified backend after VPN installation is `qwen_remote`.

Restart ME3 after changing the environment:

```bash
systemctl --user restart me3-agent.service
curl -fsS http://127.0.0.1:8126/api/health
```

## Windows launcher behavior

`scripts/Start-ME3-Pi-Web.ps1` now assumes that the Pi has the SHARC route and
does not create a model reverse tunnel by default. It still creates the normal
SSH `-L` forwards for the Pi UI and browser audio. The old laptop-VPN reverse
tunnel is available only when explicitly requested:

```powershell
& '.\scripts\Start-ME3-Pi-Web.ps1' -Variant qwen_instruct -UseReverseTunnel
```

Use that legacy mode only when the Pi's SHARC VPN is intentionally offline.

## Evidence from this installation

On 2026-11-07 (Asia/Manila):

- The Pi received `tun0` with a SHARC address and a `10.0.0.0/8` route.
- Authenticated Pi probes returned HTTP 200 for both DGX ports.
- Direct Qwen ASR returned a non-empty transcript with backend
  `qwen_remote` in approximately 366 ms on the repository WAV fixture.
- The ME3 injection harness passed with one turn and zero errors using the
  Qwen Instruct variant.
- The Pi service remained active with `/api/health` reporting `loop:true`.

The injected test is model/agent evidence, not a physical-room microphone,
speaker, or echo-gate benchmark. Those remain separate hardware checks.
