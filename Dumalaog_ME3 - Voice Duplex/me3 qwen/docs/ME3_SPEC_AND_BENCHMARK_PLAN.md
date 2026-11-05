# ME3 — Full-Duplex Voice Agent on Raspberry Pi: Spec & Benchmark Plan

> **Status: DRAFT v0.1 for class confirmation (2026-10-24).**
> Consolidated from the **Oct 22, 2026 ME3 class meeting**. Items marked
> **[DRAFT]** need group confirmation. This file is the versioned copy in the
> repo; the group Google Doc remains the collaborative source of truth.
> Meeting passcodes/credentials are intentionally NOT copied here (see Section 9).

## 1. Aim
Build a **full-duplex, voice-enabled AI agent that lives on a Raspberry Pi 4/5**,
behaves like GPT-Realtime (natural, conversational, interruptible), and can
handle real-world tasks.

### Responsibilities (from the assignment)
1. Install an **AI agent harness with MCP support** on the RPi (onit, pipecat, crew, hermes, pi.dev, opencode, ...).
2. **Build & run the MCP servers for SOTA ASR and TTS** on the RPi. Commercial APIs are allowed for the *models*, but the MCP servers themselves are ours, on the RPi.
3. Install **VAD on the RPi** for voice detection.
4. **Run a full-duplex agent** that can do real-world tasks.
5. **Build a latency benchmark (classwork):** a standardized, reproducible latency-measurement method for natural, conversational, interruptible speech.

## 2. System architecture

### 2.1 Roles
- **RPi = the "receptionist."** It owns the audio path (mic + speaker, full-duplex), a **fast LLM**, and orchestration. It **avoids heavy model inference on the Pi**.
- **Fast LLM (owned, on Pi):** handles conversation, easy tasks, and quick tool calls. When a task is complex, it **delegates to a harness**.
- **Harness (complex tasks):** a separate agent framework (Hermes / pi.dev / Onit / ...) running on the Pi **with access to the Pi's folders**, reached **via MCP**. It does heavy work (build a website, produce a report). The person can **keep talking while it works in the background**; the agent reports the result back.
- If a task is beyond the harness, that is not the Pi's fault.

### 2.2 Full-duplex loop (already built)
- Mic capture and speaker playback stay **open simultaneously**; speech during TTS **stops playback and starts a new turn** (barge-in).
- **Backchanneling** as acknowledgement / delegation cues: "Mhm?", "Yes?", "Let me check that" → hand off to the harness.
- **Wake word is optional; VAD is required.** Strategy goal: **only your voice is detected** (useful for the demo).
- Memory: use the harness's memory, or a lightweight owned memory.

### 2.3 Owned local MCP services (on the RPi)
- **ASR**, **TTS**, **VAD** run as local MCP servers (models may be commercial APIs). See `plan/ME3_Voice_Duplex_Plan.pdf`.

### 2.4 Hardware / audio-capture note (benchmark-relevant)
- **Use a wired speaker.** Bluetooth speakers break echo-cancellation (VCM can't reliably stop music, and it inflates latency). During benchmarking, if we inject the stimulus audio anyway, the echo variable is eliminated regardless.
- The benchmark capture path must **hear the played stimulus**: Windows = Stereo Mix; RPi = pass `--input-device` / set `ME3_BENCH_INPUT` to an ALSA loopback or explicit capture device. A physical mic is fine for the live agent but is not an exact internal-loopback benchmark.

### 2.5 Harness connectivity (via MCP)
- Must connect to available agentic frameworks: **Onit, Hermes, OpenCode, pi.dev** (and similar).

### 2.6 Speech-to-speech [DRAFT / OPEN]
- **Leaning OUT** (NVIDIA Moshiko/Personaplex s2s 4b) — it could not delegate real tasks when tried with Hermes. Not a final decision yet.

## 3. What we tried (and verdicts)
| Option | Verdict |
|---|---|
| **Pipecat** | End-to-end; basic prompts work out of the box. Agentic tasks weak; timer / tool calling uncertain. |
| **Crew** | Enterprise $$$ — too expensive. |
| **ASR/TTS** | Options: Deepgram, Cartesia, MLX, Whisper. |
| **Moshiko / Personaplex (NVIDIA voicechat, s2s 4b)** | Considered but leaning out; could not delegate real tasks. |

## 4. Benchmark design (the "classwork")
A standardized, reproducible method for **natural, conversational, interruptible** speech. Follows **Lin et al. 2026 / Full-Duplex-Bench** (arXiv 2503.04721; repo `DanielLin94144/Full-Duplex-Bench`) plus the **FDB v1.5** overlap contract already in `refs/`.

### 4.1 Command bank (unified)
- One **bank of commands**; **sample a subset per trial**.
- **Read out the script on the Pi for live testing** (injected audio).
- Natural, conversational, **interruptible**; includes **injected pauses, backchannels, and interruptions** (Lin et al. method).
- **Focus (per Sir):** start with the **same top-10 most-common ME2 tasks**, then **top up with more complex tasks** (e.g., "build a website", "give a report").

### 4.2 Final set of variables [DRAFT — needs confirmation]

**A. Latency**
- **L1 — End-of-speech → start-acting** *(primary)*: the processing time of the voice command itself. Harness internals are pre-baked / out of scope; benchmark only what we can optimize.
- **L2 — Response latency to a genuine interruption**: how fast it reacts. Sensitive to mic/speaker; hard if the agent hears itself.
- **L3 — Turnaround between harness turns**: harness treated as a **black box**; measure only the observable turnaround.
- **L4 — Time-to-first-acknowledgement**: how fast it emits a backchannel ("let me check that") before/while delegating.

**B. Overlap & interruption handling**
- **O1 — Overlap classification**: label every overlap **backchannel vs genuine steering** (maps to FDB v1.5 `RESPOND`/`RESUME`/`UNCERTAIN`/`UNKNOWN`).
- **O2 — Silence after overlap detection**: how long it stays silent once it detects an overlap, before responding to the new command.
- **O3 — False-interruption handling**: speech that is *not* an interruption → should not stop (ideally), or stop-then-resume. The assistant is always listening.
- **O4 — Backchannel rate / intrusiveness**: how much it tries to respond (is it annoying?). Model may pause-and-continue *or* ignore.

**C. Naturalness (LLM-as-a-judge)**
- **N1 — Conversational naturalness** score.
- **N2 — Tone consistency** before vs. after an overlap event.
- **N3 — Pause detection**: are pauses appropriate?
- **N4 — Interruption-handling quality**: backchannel vs steering judged correctly?

**D. Task capability**
- **T1 — Easy-task tool-call**: success + latency.
- **T2 — Complex-task delegation**: does it hand off to the harness correctly and report back?
- **T3 — Task set**: top-10 ME2 tasks (baseline) + complex tasks (website, report).
- **T4 — Tokens spent per task** *(optional, personal target if using paid models)*.

> Tool-call note: a tool call means the LLM is hit **twice** (decide to call → get tool output → final reply); more complex tasks = more round-trips. We count **L1** (end-of-speech → start-acting), not the harness's internal round-trips.

### 4.3 Measurement protocol
- Log **timestamps** for: end-of-speech, start-of-acting, first TTS, interruption start, overlap-detect, first-backchannel, and harness-turn boundaries.
- Capture **overlap artifacts** for the FDB v1.5 evaluator (`me3/fdb_v15.py` / `scripts/fdb_v15_eval.py`).
- **Evidence boundary** (from `docs/PAPER_ALIGNMENT.md`): a valid **v1.5 claim** requires captured overlap trials + an explicit behavior category for every trial + the paper-native evaluator. The live loop currently uses "VAD speech during playback" as a barge-in signal, which is fine for interruption responsiveness but does **not** by itself classify backchannels / off-axis / background speech as `RESUME`. **No v1.5 claim** until the 499 required utterances, behavior accuracy, prosodic adaptation, and Pi latency artifacts exist.

### 4.4 Roles & ownership
- **Standard benchmark codebase** maintained by **2–3 students**; the instructor keeps **write access**.

## 5. Agent capabilities in scope [DRAFT — decide on Saturday]
Which of these are in scope given cost? (Some cost money.)
1. **Planning & multi-step execution**
2. **Tool use** (retrieval, function/API calling, code execution, browser/computer use)
3. **Computer & browser use**
4. **Long-running work** (hours/days; coordinate with other agents)
5. **Multi-agent teams**
6. **Memory & skills** (long-term memory; MCP + Skills)
7. **Enterprise integration** (trigger workflows, call APIs, move data)

## 6. Open questions for Sir Rowel
- Can we allocate an **A100 24/7** for inference so we don't pay per-call for ASR/TTS?
- A **lightweight LLM** so we don't clog the QWEN 3.8 for simple tests?
- Which **agent-capability scope** (Section 5) is in bounds?
- **Speech-to-speech**: in or out?

## 7. Action items
- Standard benchmark codebase (2–3 student maintainers + instructor write access).
- Free up time after AI 231 class.
- Consult Sir Rowel after class.
- Collate as many questions as possible (group Google Doc).
- Read & understand FullDuplexBench (arXiv 2503.04721; `DanielLin94144/Full-Duplex-Bench`) and the V1.5 (King) overlap spec.

## 8. Repository map (what exists)
| Area | Files |
|---|---|
| Full-duplex loop / barge-in | `me3/agent.py`, `me3/audio_io.py` |
| FDB v1.5 contract (4 scenarios, counts, categories, timing) | `me3/fdb_v15.py` |
| Overlap-trial evaluator | `scripts/fdb_v15_eval.py` |
| Engineering latency driver (5 scenarios; NOT a v1.5 result) | `me3/bench_driver.py` |
| Contract check (papers, MCP servers, full-duplex config, Pi surface) | `scripts/check_paper_alignment.py` |
| Dev → ship workflow | `scripts/dev_env_check.py`, `local_smoke_test.py`, `package_release.py`, `deploy_to_pi.py` |
| RPi deploy plan (owned ASR/TTS/VAD MCP, systemd) | `plan/ME3_Voice_Duplex_Plan.pdf` |
| FDB reference (Lin et al. 2503.04721) | `refs/FDB_paper_2503.04721.pdf`, `refs/full-duplex-bench/v1_v1.5/` |
| Evidence boundary | `docs/PAPER_ALIGNMENT.md` |

**Current release state:** `deployment/latest_release.json` → `me3-20261021-055257-f2b1d376` (**prepared, not synced to a Pi**). A `--brain stub` control run (`runs/fdb_benchmark/20261021-054444/`, 25/25 trials) exists and is **diagnostic-only**.

## 9. External references (credentials intentionally not linked here)
- Group **Google Doc** (research papers, benchmarks, repos; anonymized questions for Sir) — link in the meeting thread.
- **Figma architecture board** (for drawing the architecture) — link + password in the meeting thread.
- **Zoom recording** of the meeting — link + passcode in the meeting thread.
- Papers in the repo: `Building AI Voice Agents — Summary.pdf`, `FULL-DUPLEX-BENCH V1.5 — EVALUATING OVERLAP.pdf`.

## 10. Changelog
- **2026-10-24 — DRAFT v0.1.** Consolidated from the Oct 22 ME3 class meeting. The final-set-of-variables (4.2) and capability-scope (5) sections are drafts pending group confirmation.
