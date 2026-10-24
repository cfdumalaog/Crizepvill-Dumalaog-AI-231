#!/usr/bin/env python3
"""Stdlib-only generator for the ME3 Voice Duplex plan PDF.
No third-party deps (no reportlab/weasyprint). Emits a valid 10-page PDF.
Run:  python make_pdf.py   ->  ME3_Voice_Duplex_Plan.pdf
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ME3_Voice_Duplex_Plan.pdf")

# ---- palette -------------------------------------------------------------
NAVY   = (0.10, 0.16, 0.30)
BLUE   = (0.15, 0.35, 0.65)
ACCENT = (0.85, 0.45, 0.10)
GREY   = (0.35, 0.38, 0.42)
LIGHT  = (0.90, 0.92, 0.95)
WHITE  = (1.0, 1.0, 1.0)
INK    = (0.12, 0.13, 0.15)

PAGE_W, PAGE_H = 612.0, 792.0   # US Letter
ML, MR, MT, MB = 54.0, 54.0, 60.0, 54.0
CW = PAGE_W - ML - MR           # content width
CY0 = PAGE_H - MT               # top of content
CY1 = MB                        # bottom of content

# ---- low-level PDF primitives -------------------------------------------
def esc(s):
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

class Page:
    def __init__(self):
        self.ops = []          # list of (content_stream_bytes)
    def add(self, stream):
        self.ops.append(stream)

def rect(x, y, w, h, rgb, fill=True):
    c = "%0.3f %0.3f %0.3f rg" % rgb
    if fill:
        return ("%s %.2f %.2f %.2f %.2f re f\n" % (c, x, y, w, h)).encode()
    return ("%s %.2f %.2f %.2f %.2f re S\n" % (c, x, y, w, h)).encode()

def line(x1, y1, x2, y2, rgb, w=1.0):
    return ("%0.3f %0.3f %0.3f RG %.2f w %.2f %.2f m %.2f %.2f l S\n" % (rgb[0], rgb[1], rgb[2], w, x1, y1, x2, y2)).encode()

def text(x, y, s, size, rgb=INK, bold=False, font=None):
    f = font or ("F2" if bold else "F1")
    return ("BT /%s %0.2f Tf %0.3f %0.3f %0.3f rg %.2f %.2f Td (%s) Tj ET\n" % (f, size, rgb[0], rgb[1], rgb[2], x, y, esc(s))).encode()

# ---- layout helpers ------------------------------------------------------
class Doc:
    def __init__(self):
        self.pages = []
        self.cur = Page()
        self.y = CY0
    def new_page(self):
        self.pages.append(self.cur)
        self.cur = Page()
        self.y = CY0
    def ensure(self, need):
        if self.y - need < CY1 + 6:
            self.new_page()
    def bar(self, h=6, rgb=BLUE):
        self.cur.add(rect(0, PAGE_H - h, PAGE_W, h, rgb))
    def footer(self, idx, total):
        self.cur.add(line(ML, 40, PAGE_W - MR, 40, LIGHT, 1.0))
        self.cur.add(text(ML, 28, "ME3 \u2014 Voice Duplex Agent Plan", 8, GREY))
        self.cur.add(text(PAGE_W - MR - 90, 28, "Dumalaog  |  AI 231  |  2026-10-19", 8, GREY))
        self.cur.add(text(PAGE_W - MR - 24, 28, "%d / %d" % (idx, total), 8, BLUE, bold=True))
    def h1(self, s, color=NAVY):
        self.ensure(46)
        self.y -= 20
        self.cur.add(text(ML, self.y, s, 19, color, bold=True))
        self.y -= 8
        self.cur.add(rect(ML, self.y - 2, CW, 2.4, ACCENT))
        self.y -= 16
    def h2(self, s, color=BLUE):
        self.ensure(30)
        self.y -= 16
        self.cur.add(text(ML, self.y, s, 12.5, color, bold=True))
        self.y -= 14
    def para(self, s, size=9.6, leading=13.4, color=INK, indent=0.0):
        words = s.split()
        x0 = ML + indent
        xw = CW - indent
        line_h = leading
        cur = ""
        first = True
        for w in words:
            trial = (cur + " " + w).strip()
            if self._width(trial, size) > xw and cur:
                self._emit_line(cur, x0, size, color, line_h, first)
                first = False
                cur = w
            else:
                cur = trial
        if cur:
            self._emit_line(cur, x0, size, color, line_h, first)
    def _emit_line(self, s, x0, size, color, line_h, first):
        if not first:
            self.ensure(line_h)
        self.y -= line_h
        self.cur.add(text(x0, self.y, s, size, color))
    def _width(self, s, size):
        return len(s) * size * 0.52
    def bullet(self, s, size=9.4, leading=13.0, marker="\u2022"):
        self.ensure(leading)
        self.y -= leading
        self.cur.add(text(ML, self.y, marker, size, ACCENT, bold=True))
        self.para(s, size=size, leading=leading, indent=14)
    def chip(self, x, y, w, h, label, sub, rgb=LIGHT):
        self.cur.add(rect(x, y, w, h, rgb))
        self.cur.add(rect(x, y, w, 3.0, BLUE))
        self.cur.add(text(x + 8, y + h - 16, label, 9.5, NAVY, bold=True))
        self.cur.add(text(x + 8, y + 8, sub, 7.6, GREY))

# ============================================================================
# CONTENT  (each function draws one logical page; Doc.footer adds page number)
# ============================================================================
TOTAL = 10
d = Doc()

# ---------------- PAGE 1 : TITLE ------------------------------------------
d.bar(h=10, rgb=NAVY)
d.cur.add(rect(0, PAGE_H - 10, PAGE_W, 10, NAVY))
d.cur.add(rect(ML, 600, CW, 2.0, ACCENT))
d.cur.add(text(ML, 640, "ME3", 46, BLUE, bold=True))
d.cur.add(text(ML, 596, "Voice Duplex \u2014 Full-Duplex Voice AI Agent on Raspberry Pi", 19, NAVY, bold=True))
d.cur.add(text(ML, 566, "An agent-oriented engineering plan: every step, microstep, owner, and gate", 11.5, GREY))
d.cur.add(text(ML, 540, "AI 231  \u00b7  Dumalaog  \u00b7  MEng AI Notebooks", 10.5, BLUE))
d.cur.add(rect(ML, 470, CW, 1.2, LIGHT))
meta = [
    ("Prepared by", "OnIt (agent)  \u00b7  coordinated with Codex (thinker/auditor) + Antigravity (implementer)"),
    ("Date", "2026-10-19  (Asia/Manila)"),
    ("Target hardware", "Raspberry Pi 4 (4/8 GB) or Pi 5 (4/8/16 GB) \u2014 64-bit Raspberry Pi OS Lite"),
    ("Classwork deliverable", "Standardized voice-interaction latency benchmark (Section 9)"),
    ("Status", "Planning \u2014 greenfield (no prior ME3 scaffolding in workspace)"),
]
yy = 448
for k, v in meta:
    d.cur.add(text(ML, yy, k.upper(), 8.2, ACCENT, bold=True))
    d.cur.add(text(ML + 130, yy, v, 9.6, INK))
    yy -= 20
d.cur.add(rect(ML, 300, CW, 1.2, LIGHT))
d.cur.add(text(ML, 280, "EXECUTIVE SUMMARY", 10, NAVY, bold=True))
summ = ("This plan specifies a full-duplex, interruptible, voice-enabled AI agent that runs on a "
        "Raspberry Pi. It installs an MCP-capable agent harness (Pipecat 1.13.0), builds two local "
        "MCP servers wrapping SOTA ASR (faster-whisper 1.2.1) and TTS (Piper 1.8.0), installs a "
        "production VAD (Silero VAD 6.2.3), and runs the agent against real-world tasks. Commercial "
        "APIs may supply the LLM brain, but the ASR/TTS/VAD MCP servers are built and run on the Pi. "
        "A standardized latency benchmark quantifies voice-interaction quality. Work is split across "
        "three agents with explicit ownership and verification gates.")
d.para(summ, size=9.6, leading=14.0, indent=0)
d.footer(1, TOTAL)
d.new_page()

# ---------------- PAGE 2 : MISSION, SCOPE, CONSTRAINTS ---------------------
d.bar()
d.h1("1  Mission, Scope & Constraints")
d.para("Goal: a conversational, natural, interruptible voice agent on Pi 4/5 that listens and speaks "
       "simultaneously (full-duplex), can be barged into mid-sentence, and completes real-world tasks "
       "via MCP tool calls. The system must be reproducible, measurable, and safe to hand off between agents.")
d.h2("1.1  In scope")
for s in [
    "Agent harness with MCP support (Pipecat + MCP Python SDK v2).",
    "Two locally-built MCP servers: ASR (faster-whisper) and TTS (Piper), running on the Pi.",
    "Silero VAD for voice activity detection and endpointing, exposed to the pipeline.",
    "Full-duplex audio: simultaneous capture + playback, barge-in, turn-taking.",
    "Real-world task execution through MCP tools (weather, timers, ME2 command model, etc.).",
    "Standardized latency benchmark (classwork) with a fixed measurement protocol.",
]:
    d.bullet(s)
d.h2("1.2  Out of scope / deferred")
for s in [
    "Retraining ASR/TTS/VAD models (use SOTA pre-trained weights; fine-tune only if required later).",
    "Multi-speaker diarization, translation, and vision (multimodal) in v1.",
    "Cloud-hosted agent (target is on-Pi; cloud is used only for the optional LLM brain).",
]:
    d.bullet(s)
d.h2("1.3  Hard constraints (from workspace AGENTS.md)")
for s in [
    "Single shared environment (.venv) on the dev host; on the Pi, one dedicated venv for ME3.",
    "No secrets in files/logs/chat/commits; API keys in OS keychain / env, referenced by name only.",
    "Preserve existing Pi deployments (VCM at ~/Desktop/dandan, Kiko) \u2014 ME3 lives in its own folder.",
    "Every claim of 'ran / verified' must be backed by an actually-executed command and recorded evidence.",
    "Mandatory handoff update after material changes (Update-HandoffIndex.ps1 + HANDOFF_INDEX.md).",
]:
    d.bullet(s)
d.footer(2, TOTAL)
d.new_page()

# ---------------- PAGE 3 : SYSTEM ARCHITECTURE ------------------------------
d.bar()
d.h1("2  System Architecture")
d.para("The agent is a Pipecat pipeline over a full-duplex audio transport. Three on-Pi MCP servers "
       "(ASR, TTS, VAD) are wrapped so the LLM brain can call them as tools; the same models also feed "
       "the realtime voice loop. The LLM brain is swappable: local (Ollama/llama.cpp) for offline, or a "
       "commercial/remote API (e.g. the existing vLLM Qwen endpoint) for capability.")
# architecture diagram (boxes)
gap = 12
bx, bw, bh = ML, (CW - 3*gap)/4, 46
labels = [("MIC / I2S", "audio in"), ("VAD", "Silero 6.2.3"), ("ASR", "faster-whisper"), ("LLM", "brain + MCP")]
yy = CY0 - 40
for i,(t,s) in enumerate(labels):
    x = bx + i*(bw+gap)
    d.chip(x, yy-bh, bw, bh, t, s)
# arrows row
ar_y = yy - bh - 14
for i in range(3):
    x = bx + (i+1)*bw + i*gap + gap*0.5
    d.cur.add(text(x-6, ar_y, "\u2192", 14, BLUE, bold=True))
labels2 = [("TOOL CALL", "MCP tools"), ("TTS", "Piper 1.8.0"), ("SPEAKER", "audio out"), ("BARGE-IN", "interrupt loop")]
yy2 = ar_y - 34
for i,(t,s) in enumerate(labels2):
    x = bx + i*(bw+gap)
    d.chip(x, yy2-bh, bw, bh, t, s, rgb=(0.95,0.90,0.82))
d.para("Data flow: MIC \u2192 VAD (frames speech vs silence) \u2192 ASR (streaming partials) \u2192 LLM brain "
       "(decides reply + MCP tool calls) \u2192 TTS (synthesizes speech) \u2192 SPEAKER. While the agent "
       "speaks, the VAD keeps listening; a detected user utterance cancels pending TTS (barge-in) and "
       "feeds the new input to the LLM \u2014 this is the full-duplex loop.", size=9.4, leading=13.2)
d.h2("2.1  Component inventory (pinned, verified on PyPI 2026-10-19)")
inv = [
    ("Harness", "pipecat-ai", "1.13.0", "realtime voice pipeline, transports, MCP tool calling"),
    ("MCP SDK", "mcp (Python)", "2.3.0", "official Model Context Protocol, FastMCP server builder"),
    ("VAD", "silero-vad", "6.2.3", "voice activity detection, endpointing, ONNX/TorchScript"),
    ("ASR", "faster-whisper", "1.2.1", "CTranslate2 Whisper; int8, streaming, on-Pi capable"),
    ("TTS", "piper-tts", "1.8.0", "low-latency neural TTS, many voices, real-time on Pi"),
    ("Audio I/O", "sounddevice / PortAudio", "latest", "full-duplex capture + playback"),
    ("LLM (opt.)", "ollama / remote vLLM", "\u2014", "swappable brain; local or commercial API"),
]
ty = CY0 - 232
d.cur.add(rect(ML, ty-7*16-6, CW, 7*16+6, LIGHT))
d.cur.add(text(ML+8, ty-2, "COMPONENT", 7.6, GREY, bold=True))
d.cur.add(text(ML+150, ty-2, "PACKAGE", 7.6, GREY, bold=True))
d.cur.add(text(ML+300, ty-2, "VERSION", 7.6, GREY, bold=True))
d.cur.add(text(ML+372, ty-2, "ROLE", 7.6, GREY, bold=True))
ry = ty - 18
for comp, pkg, ver, role in inv:
    d.cur.add(text(ML+8, ry, comp, 8.4, NAVY, bold=True))
    d.cur.add(text(ML+150, ry, pkg, 8.4, INK))
    d.cur.add(text(ML+300, ry, ver, 8.4, BLUE, bold=True))
    d.cur.add(text(ML+372, ry, role, 7.8, GREY))
    ry -= 16
d.footer(3, TOTAL)
d.new_page()

# ---------------- PAGE 4 : AGENT ROLES & DIVISION OF LABOR ------------------
d.bar()
d.h1("3  Multi-Agent Division of Labor")
d.para("Per the workspace model, three agents collaborate with strict roles. Each task below names an "
       "owner and a verifier so no agent claims success it did not earn. OnIt (this planner) produces the "
       "plan and coordinates; Codex reasons and audits; Antigravity implements and reports evidence.")
roles = [
    ("ONIT  (planner/coordinator)", NAVY,
     "Owns this plan, task sequencing, and the AGENT_TASK_STATE.json contract. Decides scope from "
     "evidence, writes the handoff, and escalates user-only decisions. Does not silently claim Pi results."),
    ("CODEX  (thinker / auditor)", BLUE,
     "Decides scope from evidence, reconciles requirements, reviews every diff, runs independent checks, "
     "and updates the handoff. The gatekeeper: no step is 'verified' until Codex re-runs the check."),
    ("ANTIGRAVITY  (implementer)", ACCENT,
     "Inspects before changing, implements the agreed task, and reports exact files/commands/results. "
     "Flags uncertainty instead of inferring success. Builds the MCP servers, VAD integration, and benchmark."),
]
ry = CY0 - 44
for name, col, desc in roles:
    d.cur.add(rect(ML, ry-70, CW, 70, LIGHT))
    d.cur.add(rect(ML, ry-70, 4, 70, col))
    d.cur.add(text(ML+14, ry-14, name, 10.5, col, bold=True))
    d.para(desc, size=9.2, leading=12.6, indent=14)
    d.cur.add(rect(ML, ry-70, CW, 70, col, fill=False))
    ry -= 82
d.h2("3.1  Collaboration protocol (every task)")
for s in [
    "Codex writes the task spec + acceptance criteria into AGENT_TASK_STATE.json (owner, next_step).",
    "Antigravity implements, records exact commands + output, marks status 'implemented' (not 'verified').",
    "Codex independently re-runs the acceptance check; only then sets status 'verified' with evidence ref.",
    "OnIt updates HANDOFF_INDEX.md (change log, current state, open items) before the task closes.",
    "Limit-stall watchdog (30-min schedule) resumes a stalled owner from next_step using the shared .venv.",
]:
    d.bullet(s)
d.footer(4, TOTAL)
d.new_page()

# ---------------- PAGE 5 : TASK 1 - HARNESS INSTALL -------------------------
d.bar()
d.h1("4  Task 1 \u2014 Install MCP-Capable Harness on Pi")
d.para("Owner: Antigravity  \u00b7  Auditor: Codex  \u00b7  Gate: imports pass + a hello-pipeline runs end-to-end.")
d.h2("4.1  Pre-flight (microsteps)")
for s in [
    "Confirm Pi model/RAM: `cat /proc/cpuinfo`, `free -m`, `uname -m` (must be arm64).",
    "Confirm OS: 64-bit Raspberry Pi OS Lite; `sudo apt update && sudo apt -y upgrade`.",
    "Install system deps: `sudo apt -y install python3-venv python3-dev portaudio19-dev libasound2-dev`.",
    "Create isolated folder + venv: `mkdir -p ~/me3 && cd ~/me3 && python3 -m venv .venv`.",
    "Activate: `source .venv/bin/activate`; pin `python -m pip install -U pip wheel setuptools`.",
]:
    d.bullet(s)
d.h2("4.2  Harness + MCP install (microsteps)")
for s in [
    "`pip install pipecat-ai==1.13.0 mcp==2.3.0` \u2014 harness + official MCP SDK.",
    "`pip install sounddevice numpy` \u2014 full-duplex audio I/O (PortAudio backend).",
    "Verify: `python -c \"import pipecat, mcp, sounddevice; print('ok')\"`.",
    "Smoke test: run Pipecat's minimal pipeline with a mock transport; assert frames flow in and out.",
    "Record: `pip freeze > requirements.lock`; commit `requirements.lock` + install script.",
]:
    d.bullet(s)
d.h2("4.3  Acceptance criteria (Codex verifies)")
for s in [
    "All imports succeed inside ~/me3/.venv on the Pi (evidence: command output saved).",
    "A 10-second synthetic-audio pipeline runs without exception and logs frame counts.",
    "MCP SDK can spawn a trivial in-process server and list its tools (proves MCP path works).",
    "No secrets in any committed file; keys referenced by env var name only.",
]:
    d.bullet(s)
d.footer(5, TOTAL)
d.new_page()

# ---------------- PAGE 6 : TASK 2 - MCP SERVERS (ASR + TTS) ------------------
d.bar()
d.h1("5  Task 2 \u2014 Build ASR & TTS MCP Servers (on Pi)")
d.para("Owner: Antigravity  \u00b7  Auditor: Codex  \u00b7  These are OUR servers (built + run on the Pi), "
       "wrapping SOTA models. Commercial APIs are allowed for the LLM brain only \u2014 not for ASR/TTS.")
d.h2("5.1  ASR MCP server (faster-whisper 1.2.1)")
for s in [
    "`pip install faster-whisper==1.2.1` \u2014 CTranslate2 Whisper, int8, on-Pi capable.",
    "Pick model size by RAM: Pi 4 (4GB) \u2192 `small`/`base.en`; Pi 5 (8GB+) \u2192 `medium`/`small` int8.",
    "Wrap with FastMCP: `from mcp.server.fastmcp import FastMCP; mcp = FastMCP(\"asr\")`.",
    "Expose tools: `transcribe(wav_path_or_bytes) -> text`, `transcribe_stream(chunk) -> partial_text`.",
    "Serve over stdio (local) and/or SSE (network) so Pipecat can call it as an MCP tool.",
    "Test: feed a known WAV, assert exact expected transcript; measure RTF (real-time factor).",
]:
    d.bullet(s)
d.h2("5.2  TTS MCP server (Piper 1.8.0)")
for s in [
    "`pip install piper-tts==1.8.0` \u2014 low-latency neural TTS, real-time on Pi.",
    "Download a voice model (e.g. en_US lessac/amy low) into `~/me3/tts/voices/` (no key needed).",
    "Wrap with FastMCP: `FastMCP(\"tts\")`; expose `synthesize(text) -> wav_bytes`.",
    "Stream chunks so Pipecat can start speaking before synthesis completes (latency win).",
    "Test: synthesize a sentence, assert non-empty WAV at 22050 Hz; measure chars/sec and RTF.",
]:
    d.bullet(s)
d.h2("5.3  Acceptance criteria (Codex verifies)")
for s in [
    "Both servers register and are discoverable by the Pipecat pipeline via MCP (tool list non-empty).",
    "ASR transcribes a 5 s sample correctly; RTF < 1.0 on target Pi (evidence: timing log).",
    "TTS emits valid audio; RTF < 1.0 (faster-than-realtime) on target Pi (evidence: timing log).",
    "Servers run as supervised processes (systemd) and restart on crash; logs captured.",
]:
    d.bullet(s)
d.footer(6, TOTAL)
d.new_page()

# ---------------- PAGE 7 : TASK 3 - VAD + FULL-DUPLEX ENGINE -----------------
d.bar()
d.h1("6  Task 3 \u2014 VAD + Full-Duplex Engine")
d.para("Owner: Antigravity  \u00b7  Auditor: Codex  \u00b7  Full-duplex = listen and speak at once, with "
       "barge-in. This is the hardest part; it is decomposed into verifiable microsteps.")
d.h2("6.1  VAD (Silero 6.2.3)")
for s in [
    "`pip install silero-vad==6.2.3 torch` (CPU) \u2014 production VAD, endpointing built in.",
    "Load model; configure: sampling 16 kHz, `threshold=0.5`, `min_silence_ms=500`, `speech_pad_ms=300`.",
    "Feed 512-sample frames; emit `speech_start`/`speech_end` events with timestamps.",
    "Expose to pipeline as a frame processor (and optionally as an MCP tool for benchmarking).",
    "Tune on real room noise: sweep threshold, log false-trigger + miss rates on a 2-min sample.",
]:
    d.bullet(s)
d.h2("6.2  Full-duplex loop (microsteps)")
for s in [
    "Open a duplex stream: `sounddevice.RawInputStream(16k)` + `RawOutputStream(22050)` concurrently.",
    "Maintain a ring buffer of mic frames; VAD runs on it in a worker thread.",
    "On `speech_start` while TTS is playing \u2192 CANCEL pending TTS queue (barge-in) and flush speaker.",
    "On `speech_end` \u2192 slice the utterance, send to ASR, feed transcript to LLM brain.",
    "LLM reply \u2192 TTS \u2192 speaker, streamed in chunks; keep mic live throughout (no half-duplex gap).",
    "Echo/self-suppression: subtract or gate the agent's own output so it is not re-transcribed.",
]:
    d.bullet(s)
d.h2("6.3  Acceptance criteria (Codex verifies)")
for s in [
    "Agent speaks while mic stays open; a spoken interrupt stops the agent within ~300 ms (timed).",
    "Conversational exchange of \u2265 5 turns with no dead-air lockup and no self-transcription.",
    "VAD precision/recall logged on a fixed 2-min noisy sample; thresholds recorded in config.",
    "A 3-minute soak run produces no crash, leak, or buffer overflow (evidence: memory log).",
]:
    d.bullet(s)
d.footer(7, TOTAL)
d.new_page()

# ---------------- PAGE 8 : TASK 4 - REAL-WORLD TASKS + DEPLOYMENT ------------
d.bar()
d.h1("7  Task 4 \u2014 Real-World Tasks + Deployment")
d.para("Owner: Antigravity  \u00b7  Auditor: Codex  \u00b7  The agent must do useful things, not just talk. "
       "Tasks are exposed as MCP tools the LLM brain can invoke during conversation.")
d.h2("7.1  Real-world task tools (MCP)")
for s in [
    "Weather: wrap a public/commercial weather API as an MCP tool `get_weather(city)` (key in env).",
    "Time/date: `now()`, `set_timer(seconds, label)` with a spoken confirmation + alarm tone.",
    "System control: `status()` (RAM/temp/free space), `play_tone(freq, ms)` for feedback.",
    "ME2 bridge (optional): call the existing ME2 voice-command model as a tool for offline intents.",
    "Each tool: typed schema, docstring, error handling, and a unit test with a mocked dependency.",
]:
    d.bullet(s)
d.h2("7.2  Deployment & supervision")
for s in [
    "systemd units: `me3-asr.service`, `me3-tts.service`, `me3-agent.service` (Restart=always).",
    "Boot persistence: enable all three; verify `systemctl status` after reboot.",
    "Config: single `config.yaml` (models, thresholds, transport, brain endpoint) \u2014 no hardcoded secrets.",
    "Logging: structured JSON logs to `~/me3/logs/`, rotated; include per-event timestamps for the benchmark.",
    "Safety: respect existing Pi folders (dandan VCM, Kiko); ME3 is self-contained in `~/me3`.",
]:
    d.bullet(s)
d.h2("7.3  Acceptance criteria (Codex verifies)")
for s in [
    "Agent completes \u2265 3 distinct real-world tasks by voice (e.g. 'what's the weather in Manila?', "
    "'set a 5-minute timer', 'how much free space?') with correct tool output (evidence: logs).",
    "All services survive a reboot and self-heal on crash (evidence: journalctl excerpt).",
    "Tool-call trace shows the LLM selected the right MCP tool for each request (evidence: trace log).",
]:
    d.bullet(s)
d.footer(8, TOTAL)
d.new_page()

# ---------------- PAGE 9 : LATENCY BENCHMARK (CLASSWORK) --------------------
d.bar()
d.h1("8  Latency Benchmark (Classwork Deliverable)")
d.para("A standardized, reproducible method to measure voice-interaction latency. Same fixtures, same "
       "clocks, same stats every run \u2014 so results are comparable across Pi 4 vs Pi 5 and across configs.")
d.h2("8.1  Metric definitions (fixed vocabulary)")
for s in [
    "VAD latency: speech onset \u2192 `speech_start` event.",
    "Endpoint latency: speech offset \u2192 `speech_end` event (governed by min_silence_ms).",
    "ASR latency: utterance end \u2192 final transcript (and first-partial latency separately).",
    "LLM TTFT: transcript ready \u2192 first token of the reply.",
    "TTS TTFP: first reply token \u2192 first audio chunk out of the speaker (time-to-first-playback).",
    "E2E voice-to-voice: user speech offset \u2192 agent's first audible reply (the headline number).",
]:
    d.bullet(s)
d.h2("8.2  Standardized method (the protocol)")
for s in [
    "Fixtures: a frozen set of N=30 utterances (varied length/noise) stored as WAV; identical every run.",
    "Clocks: monotonic clock (`time.monotonic_ns`) stamped at each pipeline boundary; no wall-clock.",
    "Runs: 5 warm-up + 20 measured trials per fixture; report median, p95, and max (never mean alone).",
    "Isolation: single-task load, CPU governor `performance`, record `vcgencmd measure_temp` + `free -m`.",
    "Output: a JSON report `{fixture, metric, n, median_ms, p95_ms, max_ms, hw, cfg}` + a summary table.",
]:
    d.bullet(s)
d.h2("8.3  Reporting & pass thresholds")
for s in [
    "Deliver `benchmark.py` (runs the protocol) + `report.md` (table + interpretation) + raw JSON.",
    "Target (indicative, Pi 5 / int8): E2E voice-to-voice p95 < 1.5 s; TTS TTFP < 400 ms; ASR RTF < 1.0.",
    "Compare Pi 4 vs Pi 5 side-by-side in the same table; state which component dominates latency.",
    "Reproducibility gate: two consecutive runs differ < 10% on E2E p95, else investigate variance source.",
]:
    d.bullet(s)
d.footer(9, TOTAL)
d.new_page()

# ---------------- PAGE 10 : TIMELINE, RISKS, HANDOFF -------------------------
d.bar()
d.h1("9  Timeline, Risks & Handoff")
d.h2("9.1  Sequenced plan (with owners)")
steps = [
    ("S1", "Pi pre-flight + venv + harness/MCP install", "Antigravity", "imports + smoke pipeline"),
    ("S2", "ASR MCP server (faster-whisper)", "Antigravity", "correct transcript, RTF<1"),
    ("S3", "TTS MCP server (Piper)", "Antigravity", "valid audio, RTF<1"),
    ("S4", "VAD (Silero) + tuning on room noise", "Antigravity", "precision/recall logged"),
    ("S5", "Full-duplex loop + barge-in", "Antigravity", "5-turn convo, ~300ms barge-in"),
    ("S6", "Real-world MCP task tools", "Antigravity", "3 tasks by voice"),
    ("S7", "systemd deploy + reboot persistence", "Antigravity", "survives reboot"),
    ("S8", "Latency benchmark run + report", "Antigravity", "JSON + report.md"),
    ("S9", "Independent audit + handoff update", "Codex/OnIt", "all gates green"),
]
ty = CY0 - 44
d.cur.add(rect(ML, ty-9*15-6, CW, 9*15+6, LIGHT))
d.cur.add(text(ML+8, ty-2, "ID", 7.6, GREY, bold=True))
d.cur.add(text(ML+34, ty-2, "STEP", 7.6, GREY, bold=True))
d.cur.add(text(ML+330, ty-2, "OWNER", 7.6, GREY, bold=True))
d.cur.add(text(ML+430, ty-2, "GATE", 7.6, GREY, bold=True))
ry = ty - 17
for sid, stp, own, gate in steps:
    d.cur.add(text(ML+8, ry, sid, 8.2, ACCENT, bold=True))
    d.cur.add(text(ML+34, ry, stp, 8.2, INK))
    d.cur.add(text(ML+330, ry, own, 8.2, BLUE, bold=True))
    d.cur.add(text(ML+430, ry, gate, 7.8, GREY))
    ry -= 15
d.h2("9.2  Top risks & mitigations")
for s in [
    "Pi 4 (4GB) too slow for medium Whisper \u2192 mitigate: int8 + `small`/`base.en`; benchmark proves fit.",
    "Echo/self-transcription corrupts loop \u2192 mitigate: AEC/gate + self-output suppression, verify in S5.",
    "Barge-in race (TTS not cancelling) \u2192 mitigate: explicit cancel-on-speech-start, timed test < 300 ms.",
    "Commercial API cost/latency for LLM \u2192 mitigate: swappable brain; local Ollama fallback for offline.",
    "Secrets leakage \u2192 mitigate: env/keychain only; Codex greps diffs for key patterns before commit.",
]:
    d.bullet(s)
d.h2("9.3  Handoff & completion definition")
for s in [
    "Definition of Done: S1\u2013S9 all 'verified' by Codex with evidence; benchmark report published.",
    "OnIt updates HANDOFF_INDEX.md (current state, open items, change log) + runs Update-HandoffIndex.ps1.",
    "AGENT_TASK_STATE.json set to status 'done'; next_step cleared; watchdog confirms 'no active task'.",
    "Left for user: physical Pi hardware access for live mic/speaker runs (agents cannot claim audio results "
    "without an executed, evidenced run).",
]:
    d.bullet(s)
d.footer(10, TOTAL)

# ============================================================================
# ASSEMBLE PDF
# ============================================================================
def build():
    # finalize last page
    d.pages.append(d.cur)
    objs = []
    def add_obj(body):
        objs.append(body)
        return len(objs)
    # 1: catalog, 2: pages, 3: fonts F1,F2 (two objects), then per page: page obj + content obj
    font1 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    font2 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    page_obj_nums = []
    content_obj_nums = []
    for pg in d.pages:
        # placeholder page object
        page_obj_nums.append(len(objs) + 1)  # will be appended after content
        stream = b"".join(pg.ops)
        content_obj_nums.append(None)  # filled below
    # We need deterministic numbering: reserve page+content pairs
    # Simpler: build sequentially now that counts are known
    objs = []
    def add_obj2(body):
        objs.append(body)
        return len(objs)
    f1 = add_obj2(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    f2 = add_obj2(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    # reserve catalog + pages numbers
    catalog_num = None
    pages_num = None
    # We'll compute: after fonts, catalog and pages come next
    catalog_num = len(objs) + 1
    pages_num = len(objs) + 2
    kids = []
    for pg in d.pages:
        page_num = len(objs) + 3 + (len(kids) * 2)      # page obj
        content_num = page_num + 1                        # content obj follows
        kids.append(page_num)
        stream = b"".join(pg.ops)
        objs.append(None)  # page placeholder
        objs.append(("stream", stream))  # content placeholder
    # now insert catalog + pages at reserved positions (shift)
    # Rebuild cleanly:
    objs = []
    def A(body):
        objs.append(body); return len(objs)
    f1 = A(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    f2 = A(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    # precompute page/content numbers
    n = len(d.pages)
    # catalog = 3, pages = 4, then pairs start at 5
    cat = 3
    pages = 4
    page_nums = []
    for i in range(n):
        pn = 5 + i * 2
        page_nums.append(pn)
    kids_arr = " ".join("%d 0 R" % p for p in page_nums)
    A(b"<< /Type /Catalog /Pages %d 0 R >>" % pages)                 # obj 3
    A(("pages", kids_arr, n))                                        # obj 4
    for i, pg in enumerate(d.pages):
        pn = page_nums[i]
        cn = pn + 1
        A(("page", cn))                                              # page obj
        A(("stream", b"".join(pg.ops)))                              # content obj
    # serialize
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        if isinstance(o, bytes):
            out += ("%d 0 obj\n" % i).encode() + o + b"\nendobj\n"
        elif isinstance(o, tuple) and o[0] == "pages":
            _, kids_str, cnt = o
            body = ("<< /Type /Pages /Kids [%s] /Count %d >>" % (kids_str, cnt)).encode()
            out += ("%d 0 obj\n" % i).encode() + body + b"\nendobj\n"
        elif isinstance(o, tuple) and o[0] == "page":
            _, cn = o
            body = ("<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                    "/Resources << /Font << /F1 %d 0 R /F2 %d 0 R >> >> "
                    "/Contents %d 0 R >>" % (pages, f1, f2, cn)).encode()
            out += ("%d 0 obj\n" % i).encode() + body + b"\nendobj\n"
        elif isinstance(o, tuple) and o[0] == "stream":
            data = o[1]
            head = ("<< /Length %d >>\nstream\n" % len(data)).encode()
            out += ("%d 0 obj\n" % i).encode() + head + data + b"\nendstream\nendobj\n"
    # xref
    xref_pos = len(out)
    out += ("xref\n0 %d\n" % (len(objs) + 1)).encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += ("%010d 00000 n \n" % off).encode()
    out += ("trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, cat, xref_pos)).encode()
    with open(OUT, "wb") as fh:
        fh.write(bytes(out))
    return OUT, len(d.pages)

if __name__ == "__main__":
    path, npages = build()
    print("WROTE %s (%d pages, %d bytes)" % (path, npages, os.path.getsize(path)))
