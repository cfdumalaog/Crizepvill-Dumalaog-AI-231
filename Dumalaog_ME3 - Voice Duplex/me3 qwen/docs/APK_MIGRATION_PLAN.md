# ME3 Android parity plan

## Goal

Convert the Pi-hosted ME3 Qwen Assistant into a presentable Android APK while
preserving the voice-first, Codex-like interaction model:

- continuous voice capture remains the main interaction;
- the operator can type, approve, answer setup prompts, and inspect progress
  while a voice or software-development task is running;
- the same ToolBrain, MCP tools, clock actions, Spotify/Calendar/Telegram/
  GitHub adapters, model-layer selectors, and restart controls remain visible;
- the app reconnects after Wi-Fi/VPN changes and never creates a second competing
  audio client without telling the operator.

## Current architecture and parity boundary

The current Android project is a native shell around the Pi web client. The Pi
owns the authoritative microphone, VAD, ASR/TTS provider calls, model/tool
loop, credentials, task subprocesses, and browser PCM speaker WebSocket. The
phone supplies the Codex-like conversation UI and may act as the selected PCM
speaker. This already preserves nearly all ME3 functionality without copying
Python or model weights to a phone.

The existing Android client has passed local Gradle/lint/metadata checks, but a
physical phone installation, LAN reconnect, microphone permission flow, and
speaker playback are still open evidence gates. The APK is therefore a client
parity build, not yet a physically accepted full replacement for the Pi.

## Functionality map

| ME3 capability | APK implementation target | Acceptance evidence |
|---|---|---|
| Voice-first conversation | WebView/Native UI connected to Pi `/api/state`, `/api/chat`, and browser PCM WebSocket | Speak from Pi Fifine; typed follow-up appears in the same conversation and is executed |
| Live task progress | Conversation events, tool activity, runtime terminal, background-process panel | Start a bounded software task and observe phase updates while the task continues |
| ASR/TTS/VAD selectors | Reuse `/api/layers`; show current layer, save, and reconnect after Pi restart | Select each available backend; Pi status reports the selected engines |
| Qwen/OpenAI stack choice | Reuse contextual private setup prompts and `/api/provider/status` | Switch only after provider key is stored on Pi; verify a short turn |
| Files, search, web, shell, Git, website tools | Keep ToolBrain and MCP servers on Pi; APK renders results and artifacts | Build a disposable website, run tests, and expose the verified URL/artifact |
| Music, calendar, reminders, alarms, Telegram | Keep provider/device adapters on Pi; use chat prompts for setup and approvals | Execute one real test per authorized integration; record provider result |
| Echo protection/full duplex | Pi remains the acoustic authority; APK is one speaker sink at a time | Run Pi Fifine + phone speaker overlap test and inspect echo/barge-in counters |
| Restart and recovery | Native reconnect screen, health poll, service/loop restart buttons | Restart Pi service and verify the APK returns to the live conversation |

## Credential design

Provider keys are **not** seeded into the APK, source tree, release ZIP, or
Android preferences. An APK is extractable and any embedded key would be
recoverable and difficult to revoke. In the current mode, setup prompts send a
credential directly to the Pi's loopback/private setup route through the
operator's trusted tunnel; the Pi stores it in its mode-0600 private config.
The UI receives only configured/not-configured status.

If a future direct-phone provider mode is required, use Android Keystore with
hardware-backed storage where available, Android app-private storage, provider
scope minimization, key rotation, and no logging. The Pi mode should remain the
default for the demo because it keeps the model/tool permissions and the
institutional VPN boundary in one host. Never put the Qwen, OpenAI, Deepgram,
ElevenLabs, Spotify, Google, Telegram, or GitHub secret values in a manifest,
resource, Java/Kotlin constant, WebView local storage, or Git commit.

## Recommended implementation phases

1. **Transport hardening:** add a pairing code or short-lived session token,
   WebSocket reconnect/backoff, an explicit connection-health banner, and a
   Pi endpoint editor. Prefer HTTPS/reverse proxy when the service leaves the
   trusted LAN; the current HTTP service must stay behind LAN/VPN controls.
2. **Native voice shell:** add an Android foreground audio service only if the
   phone will capture audio. Request `RECORD_AUDIO` and notification permission,
   keep one active capture/speaker client, preserve the Pi echo reference, and
   show a persistent “ME3 listening” notification. Do not silently start a
   second listener when the WebView reloads.
3. **Task workspace:** add project/repository selection, a plan/execute toggle,
   a live task timeline, cancel/stop/skip/steer controls, artifact links,
   diff/test/build summaries, retry with bounded context, and a clear current
   working directory. These are the highest-value improvements for automated
   software development.
4. **Safety and recovery:** add branch/commit checkpoints, a dry-run mode for
   destructive commands, explicit scope display for file and shell actions,
   automatic log redaction, provider timeout/fallback badges, and a task journal
   that survives an app reconnect without exposing hidden reasoning.
5. **Release acceptance:** build signed debug/release variants, install on the
   target phone, test portrait/landscape and background/foreground transitions,
   run the Pi Fifine-to-phone-speaker physical overlap test, check every layer
   selector, and capture APK hash plus a reproducible install/run record.

## What would make ME3 strong for automated software development

The current tools are enough for bounded coding tasks. For a more capable
agent, add these as explicit product features rather than relying on one long
voice prompt:

- a named workspace/repository selector and visible branch/commit context;
- **Plan**, **Execute**, and **Review** modes, with the plan shown in chat;
- a task timeline with current step, elapsed time, last tool, next step, and
  one-minute progress updates;
- automatic test/build/lint gates before a completion claim;
- patch/diff previews and an artifact panel for websites, PDFs, APKs, logs, and
  benchmark results;
- resumable jobs with bounded retries, cancellation, and “continue from last
  checkpoint” after a provider or network outage;
- a compact memory/project brief that records decisions and constraints, not
  raw secrets or unbounded transcripts;
- specialist MCP workers for research, coding, QA, and release packaging, with
  the main ToolBrain retaining ownership of scope and final verification;
- a provider health matrix for ASR, LLM, TTS, VAD, and integrations so a
  fallback is visible rather than sounding like a mysterious voice change.

## Open acceptance gates

- physical Android install and playback are not claimed until a phone is
  attached and tested;
- Qwen/DGX3 reachability still depends on the Pi VPN/DILNET route;
- external Spotify, Google Calendar, Telegram, and GitHub side effects need
  their provider authorization and a recorded live test;
- the class paper-native full-duplex benchmark remains separate from direct
  injection/regression tests.
