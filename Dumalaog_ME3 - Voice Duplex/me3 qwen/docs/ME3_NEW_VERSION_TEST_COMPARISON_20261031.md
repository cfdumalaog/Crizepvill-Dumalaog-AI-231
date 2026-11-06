# ME3 new-version test and comparison — 2026-10-31 (Asia/Manila)

Scope: test the new ME3 version (working tree; this morning's release
`me3-20261031-093214-2856e842`, source fingerprint `2856e842...`), compare it
against the previous release (`me3-20261027-050639-4bf5e009`, the last release
deployed to the Pi), and verify every capability that can be verified on this
workstation **without the Raspberry Pi**.

During the test a bounded-retry fix was applied to `me3/brain.py` (see
Section 6). The accepted release after that fix is
**`me3-20261031-115057-b31a77fe`** (54 files, source fingerprint
`b31a77fedba1066b33973b31c7397063da4c94a56c11fd6b39bfd45e3ffe22ad`,
zip SHA-256 `2eda283e9f1e22e0e129f26dc05083695961f13ddc7238e9d23bdfcd14d573f2`).
All live services were restarted on that code.

## 1. What changed (old release on Pi vs new version)

Package contents: old = 50 files, new = 54 files.

**Added (5):**
- `me3/qwen_audio.py` — `qwen_instruct` audio profile: remote
  Qwen2-Audio-7B-Instruct ASR + Qwen3-TTS-12Hz-1.7B-CustomVoice TTS through
  OpenAI-compatible vLLM services (chat-completions multimodal ASR protocol
  with legacy `/audio/transcriptions` fallback; PCM TTS with sample-rate
  reporting).
- `me3/vocalis.py` — Vocalis-Pi duplex protocol: five external states,
  input/output event channels, JSON-RPC harness dispatch, barge-in ordering.
- `docs/VOCALIS_PROTOCOL.md` — protocol documentation.
- `scripts/compare_audio_variants.py` — audio-variant comparison harness.
- `web/ui.html` — local evaluation UI with automatic browser speech input,
  mode selection (stub/Qwen/Luna), tool-event reporting, channel readiness.

**Changed (20):** `me3/agent.py` (+5.4 KB: Vocalis events, sanitized tool
progress, Qwen Instruct wiring), `me3/audio_io.py` (+8 KB: playback
reference, speaker-feedback cancellation, room-reflection tail,
digital-loopback muting), `me3/brain.py` (+3 KB: Luna provider with
`reasoning_effort`, auto provider preference; further +1.3 KB bounded-retry
fix from Section 6), `me3/config.py`, `me3/mcp_asr.py`
(multilingual Whisper), `me3/mcp_tts.py`, `me3/bench_driver.py`,
`scripts/injection_eval.py`, `scripts/workstation_acceptance.py`,
`config/config.yaml` (new `variants.qwen_instruct` block), `install.sh`
(preserves protected Qwen/Telegram/Resend env vars across upgrades),
`web/index.html`, `README.md`, and seven docs updates.

**Identical (30):** benchmark/scoring modules, channels, tools, dataset docs,
deploy units, FDB reference materials.

## 2. Test-suite growth

| | Old era (commit `5edb0b5`) | New version (working tree) |
|---|---|---|
| Test files | 9 | 13 |
| Tests | 32 | 48 |
| New test files | — | `test_audio_io.py` (4), `test_qwen_audio.py` (4), `test_variants.py` (2), `test_vocalis_protocol.py` (4) |

## 3. Fresh verification on this workstation (no Pi), 2026-10-31

All commands ran in the shared `.venv` (Python 3.12.10). Timestamps are
Asia/Manila.

| # | Check | Result |
|---|---|---|
| 1 | `pytest tests` (new version, incl. retry fix) | **48/48 passed** in 2.81 s |
| 2 | `scripts/workstation_acceptance.py` (env, pytest, py/bash syntax, MCP+audio smoke, paper alignment, whitespace, package integrity) | **PASS**, all 8 sections; release `me3-20261031-115057-b31a77fe` (54 files) prepared with the post-fix fingerprint `b31a77fe...` |
| 3 | `scripts/local_smoke_test.py --with-audio` | **PASS**: 4 MCP servers healthy (ASR faster-whisper small int8 1.91 s, TTS Piper en_GB-alba-medium 22.05 kHz, VAD silero, tasks 25 tools), 5 benchmark scenarios, PortAudio 44 devices (default in=1, out=4) |
| 4 | Qwen text endpoint `202.92.159.240:8001/v1` (VPN) | `/v1/models` HTTP 200 (71 ms), model `Qwen/Qwen3.8-27B`; short chat HTTP 200 (638 ms, reply "OK"); 600-token generation HTTP 200 (10.4 s). Note: small `max_tokens` can be fully consumed by the model's reasoning field, leaving empty content — a client-side budgeting fact, not an endpoint fault |
| 5 | UI `current` @ 8124 (restarted on post-fix code) | `/api/health` ok: Qwen `Qwen/Qwen3.8-27B` configured, local Whisper+Piper, 23 tools, external channels disabled; live Qwen chat turn 11.4 s ok=true with `run_command` tool events, Filipino reply (Philippines profile active) |
| 6 | UI `qwen_instruct` @ 8125 (restarted on post-fix code) | `/api/health` ok: ASR `qwen_remote/Qwen/Qwen2-Audio-7B-Instruct` configured, TTS `qwen_remote/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` configured, shared key configured; live turns: 51.3 s tool-loop turn ok=true (20 tool events) and a tool-free turn 0.97 s replying exactly "pong" |
| 7 | Injection eval, `current` variant (Whisper ASR + Piper TTS + Qwen brain), virtual speaker | `runs/injection_eval/20261031-105244` **PASS**: exact transcript, tools `write_file` + `start_web_server`, 1 turn, 0 errors, 237.6 s. (First attempt failed only on a transient 180 s VPN read timeout — see Section 6.) |
| 8 | Injection eval, `qwen_instruct` variant (Qwen2-Audio ASR + Qwen3-TTS + Qwen brain) | `runs/injection_eval/20261031-110705` **PASS**: transcript via Qwen2-Audio, tools `list_files`, `read_file`, `start_web_server`, `web_servers`, `write_file`, 1 turn, 0 errors, 281.4 s, 35.2 s of TTS audio into the virtual sink |
| 9 | Barge-in, real Qwen brain, post-fix code (interrupt `stop.wav` at +3 s after reply speech starts) | `runs/injection_eval/20261031-121708` **PASS, `barge_in_detected: true`**: transcript exact, 6 tools (`fetch_url`, `list_files`, `read_file`, `start_web_server`, `web_servers`, `write_file`), 66.3 s reply played into the virtual sink, injected "stop" cut the reply short, second reply "It looks like your message got cut off. What would you like me to do?", 248.1 s, 0 errors. First real-brain barge-in pass (all earlier real-brain runs ended before the interrupt could be scheduled on a stalled/late turn) |
| 10 | Barge-in, real Qwen brain, same command as the earlier failed attempt (interrupt at +215 s) | `runs/injection_eval/20261031-120246` **PASS, 487.5 s** — the previously recurring 180 s VPN stall happened again and the new bounded retry recovered it (log: "chat request stalled (no response within 180 s); retrying once"). The interrupt was not scheduled in this run because the reply speech started so late that +215 s fell past the eval deadline; superseded by check 9 |
| 11 | `LocalVoiceLoop` (real mic + FxSound speaker, restarted on post-fix code) | reached **LISTENING** (`runs/local-run-20261031T120651Z.log`): models loaded vad=239 ms, asr=1832 ms (Whisper small), tts=1578 ms (Piper @ 22050 Hz), capture started 30 ms frames, "microphone signal OK (startup peak 481)"; ambient pickup handled correctly ("speech start" → "no valid speech; discarding"). A real barge-in was also observed on the pre-fix live instance: 19:34:13 "BARGE-IN: interrupting agent (state was awaiting)" + playback stopped + pending reply discarded |
| 12 | Presenter surface `web/index.html` @ 8123 | HTTP 200, 6,197 bytes, title "ME3 Voice Duplex" |
| 13 | Voice-built demo web app (artifact of checks 7–9) | the model-created `agent_workspace/voice-injection-demo/` was served live at `http://127.0.0.1:8010/` (HTTP 200, title "Voice Injection Demo") after check 7 and at `:8090` after check 9; demo servers stopped after verification |

Carried over from this morning's session (recorded in the handoff, not
re-run today): Stereo Mix speaker-feedback guard (5.82 s capture, 182 echo
frames suppressed, zero VAD/ASR turns), and the 02:08 qwen_instruct
injection pass. Stub-brain barge-in pass from 20261025-095934 (17.7 s,
`barge_in_detected: true`) remains the harness baseline for barge-in
ordering.

## 4. Capability comparison, old vs new (verified today unless noted)

| Capability | Old (Pi release 20261027) | New (workstation 20261031) | Evidence |
|---|---|---|---|
| Qwen3.8-27B tool brain (text + voice) | yes (Pi) | yes (workstation, live) | checks 5, 7, 8, 9 |
| Qwen Instruct ASR/TTS profile | no | **yes (live over VPN)** | checks 6, 8 |
| Vocalis-Pi duplex protocol (states/channels/JSON-RPC/barge-in ordering) | no | yes (protocol implemented, 4/4 tests) | check 2, tests |
| Browser evaluation UI with automatic speech input | no (UI added in between; not in Pi release) | **yes (live)** | checks 5, 6 |
| Multilingual Whisper ASR (Fil/EN/Taglish detection) | Whisper small.en-era config | yes (multilingual small) | checks 3, 7, 11 |
| Speaker-feedback echo guard (playback reference + tail) | basic | strengthened (digital-loopback mute, room-reflection tail) | tests #1; morning physical check |
| Barge-in (interrupt mid-reply) | basic | **verified twice today**: real Qwen brain injected barge-in + live mic barge-in event | checks 9, 11 |
| Resilience to VPN stalls on long LLM turns | single non-retrying request (180 s timeout) | **bounded 2-attempt retry** on ReadTimeout/ConnectionError | Section 6, check 10 |
| Luna (GPT-5.6) provider selection | no | yes (code + tests; live test blocked: no OPENAI_API_KEY on this machine) | tests #1 |
| Workstation-first acceptance gate | introduced 10-27 | **re-passed today with fresh packaging** | check 2 |
| Bounded project tools (files/web/commands/timers/status/Git/comm-gated outbound) | yes | yes (23-25 tools depending on surface) | checks 3, 5 |
| Pi deployment target | deployed (historical path) | accepted release staged, `pending_sync` (no scriptable SSH endpoint yet) | deployment records |

## 5. What "100 % without the Pi" covers and does not cover

**Fully verified without the Pi today (post-fix release
`me3-20261031-115057-b31a77fe`):** software test suite (48/48), acceptance
gate (8/8), MCP/audio smoke, Qwen text endpoint (VPN), both UI surfaces with
live Qwen turns, full injected voice loops for both audio variants
(VAD→ASR→Qwen tools→TTS), **real-brain barge-in** (injected), local
full-duplex listener on real hardware (LISTENING, ambient handling, live
barge-in event), presenter surface, and the voice-built demo web app served
from the project workspace.

**Cannot be verified without the Pi (or the user):** a human spoken turn at
the microphone (needs the user; the listener is live and correct), a
physical-speaker barge-in with a real interrupting voice (injected barge-in
is verified; the physical path uses the same detection code), Pi promotion
(no scriptable SSH endpoint; `deploy_to_pi.py` remains gated on the
acceptance record), and Luna live mode (no OpenAI key on this workstation).
No physical-speaker, paper-grade FDB v1.5, or Pi hardware claims are made
from any of the above.

## 6. The bounded-retry fix applied during this test

`me3/brain.py` `ToolBrain._request()` performed a single non-streaming
`requests.post(..., timeout=180 s)` with no retry. On this machine the VPN
path to the Qwen endpoint intermittently stalls on long multi-tool turns:
2 of 4 such turns on 2026-10-31 hit the 180 s read timeout and the run
failed, while every short request (probes, UI turns, ASR/TTS calls)
succeeded within seconds. The fix wraps the request in a 2-attempt loop that
retries **only** `requests.exceptions.ReadTimeout` and
`requests.exceptions.ConnectionError` (2 s pause, one warning log line with
no sensitive data); any other error raises immediately. Return shape,
timeouts, and everything else are unchanged.

Why it is safe: a chat completion whose response never arrived carries no
server-side side effect worth preserving, and the conversation state lives
client-side, so a resend is idempotent. Verified: 48/48 tests, fresh
acceptance gate PASS, and check 10 where the exact recurring stall was
recovered by the retry inside a 487 s passing run.
