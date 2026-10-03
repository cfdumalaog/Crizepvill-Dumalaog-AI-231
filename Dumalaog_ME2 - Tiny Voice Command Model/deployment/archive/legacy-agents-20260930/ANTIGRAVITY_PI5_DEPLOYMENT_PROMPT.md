# Antigravity task: deploy and verify Tiny VCM on Raspberry Pi 5

Work from the shared workspace `C:\Users\danda\Desktop\MEng AI Notebooks\AI 222 231`. Read the root `AGENTS.md`, `HANDOFF_INDEX.md`, `SETUP_NOTES.md`, the AI 231 ME2 `HANDOFF_INDEX.md` and README, and `deployment/README-antigrav-rpi.md` before changing anything. Preserve the existing working-tree changes. Use the shared `.venv` for PC-side checks; do not create another PC environment.

## Objective

Install the existing current Antigrav Raspberry Pi 5 application bundle on the user's Pi, verify it on the actual device, and leave a concise evidence-based deployment report and updated handoffs. This is an application deployment over SSH; the ZIP is not an SD-card image and does not need to be flashed.

## Verified access

The user installed their SSH public key using a PowerShell OpenSSH prompt. A passwordless BatchMode SSH check, both in the user's terminal and independently from this workspace, returned hostname `cfdfnjrpi5`. The target is user `dalmacio` at the previously observed address `192.168.254.106`. The identity file is `%USERPROFILE%\.ssh\id_ed25519`.

Use the existing key through OpenSSH. Never print, read, copy, or transmit the private-key contents. Do not ask for or put a password in a command, environment variable, file, log, chat, or source. If authentication or a sudo prompt needs user input, use a user-visible interactive prompt and let the user type it directly; do not work around a prompt by storing a secret.

## Correct artifacts and boundaries

- Current bundle: `AI 231\Dumalaog_ME2 - Tiny Voice Command Model\deployment\dist\TinyVCM_RPi5_Antigrav_RELEASE.zip`.
- Release documentation: `AI 231\Dumalaog_ME2 - Tiny Voice Command Model\deployment\README-antigrav-rpi.md`.
- Current deployment installer: `setup_rpi.sh` inside the ZIP (built from `deployment/setup_antigrav_rpi5.sh`).
- Use the ZIP's `tinyvcm-rpi5/` top-level directory and install under `~/tinyvcm-rpi5` as documented.
- Do not use `archive/legacy_vcm/TinyVCM_RaspberryPi5.zip`, `archive/legacy_vcm/release/`, the legacy `setup_rpi.sh`, or `archive/legacy_vcm/rpi_deployment/`; those are 26-class/BC-ResNet historical artifacts.
- Do not retrain or replace either model. The 31-class Option B classifier is the scratch-trained course model. The 32-class wake-word extension is warm-started and experimental; report those facts separately and do not claim multi-speaker wake robustness.
- Do not enable GPIO or connect/drive hardware pins. No physical wiring has been confirmed. Keep dashboard bound to `127.0.0.1`; do not expose its unauthenticated controls on the LAN.
- Voice classification and responses must remain offline/local. Do not add ASR, an LLM, cloud calls, or online services.

## Work sequence

1. Record the starting Git status and inspect the actual bundle, release manifest, setup script, and runtime instructions. Confirm the ZIP exists; record its byte size and SHA-256. If the bundle is missing or stale, rebuild only with the documented release builder, whose overwrite protection must remain in force, then rerun its checks.
2. Connect with OpenSSH key authentication and confirm `hostname` is `cfdfnjrpi5`. Before installing anything, collect `uname -m`, `/etc/os-release`, Python version, RAM, free disk, and available capture devices. Confirm 64-bit Raspberry Pi OS (`aarch64`) and sufficient free space; stop and report if the target differs or is not the expected Pi.
3. Transfer the exact ZIP to the Pi with `scp`, compare its remote SHA-256 with the PC hash, and only then extract it to `~/tinyvcm-rpi5`. Do not overwrite an unrelated directory or package.
4. Read the bundled installer, then run it as the regular `dalmacio` user. The script uses `sudo` for apt packages, builds a Pi-local venv, installs binary runtime requirements, verifies the manifest and both ONNX models on CPU, and installs—but does not enable—the loopback systemd unit. Do not use the Windows `.venv` on the Pi or install Python packages globally. Capture any failure and stop before proceeding on a broken install.
5. Run `verify_release.py` and `verify_release.py --audio`; also run the microphone module check in the release README. Verify both 31- and 32-output models, finite frontend features, CPU ONNX execution, model-only p50/p95 latency, and actual non-empty microphone capture. Record the selected audio device. Do not count a synthetic tone or a successful web response as a speech test.
6. Start the assistant in the foreground with the documented loopback-only command. Check startup logs and the local `/assistant` page/API through loopback or an SSH tunnel. Speak a wake phrase and then multiple representative commands through the Pi microphone; record observed wake/command outcomes and confidence values. If other consenting speakers are available, test them and background noise; otherwise mark unseen-speaker/noise validation as not done. Do not save raw audio unless the user explicitly agrees.
7. Measure warmed-up model-only and full frontend-plus-model CPU latency on the Pi (at least 100 timed iterations each, report p50/p95 and method). Keep the model-only result distinct from capture/window latency. Check memory use and throttling. Compare inference p95 against the 10 ms target, and report failure plainly if it misses; do not infer Pi latency from PC measurements.
8. Only after installer, microphone, wake, and command checks pass, enable the provided systemd service with the documented command. Verify `systemctl is-enabled`, `systemctl is-active`, service logs, and that the dashboard remains loopback-only. Do not reboot the Pi without the user present/asking; do not enable GPIO.
9. Write `deployment/validation/pi5_deployment_report.md` with the date, exact bundle SHA-256, target OS/architecture/RAM, installer/runtime results, model hashes and shapes, mic device, live speech test table, p50/p95 measurements, service state, failures, and remaining limitations. Separate verified facts from untested items; do not call the VCM perfect or Alexa-equivalent.
10. Update both the workspace root and ME2 handoff indexes (`Current state`, limitations/cautions, open items, and newest-first change log). Refresh the root handoff generated machine-state block using `scripts\Update-HandoffIndex.ps1` as required by `AGENTS.md`. Add only checks that actually ran. Do not commit or push unless the user later requests it.

## Acceptance criteria

Deployment is verified only if the Pi is positively identified, the transferred ZIP hash matches, the bundled verifier passes both model CPU checks on the Pi, the actual microphone produces valid samples, at least one real wake-to-command interaction succeeds, the service state is checked, and measured Pi latency/RAM are recorded. If any prerequisite fails, stop at that boundary and document exact evidence and next action. Wake-word robustness across speakers and noise remains unverified unless those tests are actually performed.
