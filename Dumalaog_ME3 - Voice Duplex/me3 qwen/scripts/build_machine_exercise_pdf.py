"""Build a filled ME3 Qwen Instruct machine-exercise PDF from the supplied style.

The source template remains untouched.  The output is a new two-page, 960x540
PDF that keeps the template's hierarchy, palette, spacing, table structure,
and footer treatment while replacing placeholders with verified Qwen Instruct
details.  Mean/p95 cells are deliberately blank until the updated direct-feed
class benchmark is run.
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "machine_exercise_2page_ME3_Qwen_Instruct.pdf"
W, H = 960, 540

NAVY = colors.HexColor("#182742")
ORANGE = colors.HexColor("#F47B24")
TEAL = colors.HexColor("#078A91")
TEXT = colors.HexColor("#202C3D")
MUTED = colors.HexColor("#8795A9")
LIGHT = colors.HexColor("#EEF3F8")
GRID = colors.HexColor("#C8D4E2")
WHITE = colors.white


def pstyle(name: str, size: float, leading: float, color=TEXT,
           font: str = "Helvetica", align=TA_LEFT) -> ParagraphStyle:
    return ParagraphStyle(name, fontName=font, fontSize=size,
                          leading=leading, textColor=color, alignment=align,
                          spaceAfter=0, spaceBefore=0)


def para(c: canvas.Canvas, text: str, x: float, top: float, width: float,
         size: float = 9, leading: float | None = None, color=TEXT,
         font: str = "Helvetica", align=TA_LEFT, max_height: float = 1000) -> float:
    """Draw wrapped ASCII-safe text with its top edge at ``top``."""
    leading = leading or size * 1.28
    style = pstyle(f"p-{size}-{font}-{color}", size, leading, color, font, align)
    safe = escape(str(text)).replace("\n", "<br/>")
    block = Paragraph(safe, style)
    _, height = block.wrap(width, max_height)
    block.drawOn(c, x, top - height)
    return height


def section(c: canvas.Canvas, number: str, title: str, note: str,
            x: float, y: float) -> None:
    c.setFillColor(ORANGE)
    c.roundRect(x, y - 3, 17, 17, 4, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(x + 8.5, y + 2, number)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(x + 23, y, title)
    if note:
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(x + 23 + stringWidth(title, "Helvetica-Bold", 15) + 12,
                     y + 1, note)


def bullet_list(c: canvas.Canvas, items: list[str], x: float, top: float,
                width: float, size: float = 9.2, leading: float = 13.5,
                gap: float = 2.5) -> float:
    y = top
    for item in items:
        c.setFillColor(ORANGE)
        c.setFont("Helvetica-Bold", size)
        c.drawString(x, y - size + 1, "-")
        h = para(c, item, x + 10, y, width - 10, size=size,
                 leading=leading, color=TEXT)
        y -= h + gap
    return y


def footer(c: canvas.Canvas, page: int) -> None:
    c.setStrokeColor(GRID)
    c.setLineWidth(0.7)
    c.line(48, 27, 912, 27)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8.5)
    c.drawString(48, 13, "Machine Exercise - Full-Duplex Voice Agent on Raspberry Pi")
    c.setFont("Helvetica-Oblique", 8.5)
    c.drawCentredString(480, 13, "ME3 Qwen Instruct | verified implementation details only")
    c.setFont("Helvetica-Bold", 8.5)
    c.drawRightString(912, 13, f"2-page submission | {page}/2")


def header(c: canvas.Canvas, title: str, right: str, eyebrow: str | None = None) -> None:
    c.setFillColor(NAVY)
    c.rect(0, 454, W, 86, fill=1, stroke=0)
    c.setFillColor(ORANGE)
    c.rect(0, 454, W, 4, fill=1, stroke=0)
    if eyebrow:
        c.setFillColor(ORANGE)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(48, 514, eyebrow)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 21)
    c.drawString(48, 486, title)
    c.setFillColor(colors.HexColor("#C4CDDB"))
    c.setFont("Helvetica-Oblique", 10)
    c.drawRightString(912, 488, right)


def diagram(c: canvas.Canvas, x: float, y: float, w: float, h: float) -> None:
    """Draw the actual Qwen pipeline inside the template diagram box."""
    c.setFillColor(LIGHT)
    c.setStrokeColor(GRID)
    c.setDash(4, 3)
    c.roundRect(x, y, w, h, 8, fill=1, stroke=1)
    c.setDash()

    # Small boundary labels keep the cloud/Pi split legible without overcrowding.
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 7.2)
    c.drawString(x + 10, y + h - 13, "RASPBERRY PI SERVICE")
    c.setFillColor(colors.HexColor("#596B83"))
    c.drawString(x + 242, y + h - 13, "QWEN VLLM OVER VPN")

    boxes = [
        (x + 9, "Fifine mic\nEcho guard + VAD", TEAL),
        (x + 101, "Qwen2-Audio\nASR MCP", colors.HexColor("#4C6FFF")),
        (x + 193, "Qwen ToolBrain\nMCP tasks/tools", TEAL),
        (x + 285, "Qwen3-TTS Aiden\nPCM browser sink", colors.HexColor("#4C6FFF")),
    ]
    bw, bh = 80, 39
    by = y + 73
    for bx, label, fill in boxes:
        c.setFillColor(WHITE)
        c.setStrokeColor(fill)
        c.setLineWidth(1.0)
        c.roundRect(bx, by, bw, bh, 5, fill=1, stroke=1)
        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold", 6.9)
        lines = label.split("\n")
        c.drawCentredString(bx + bw / 2, by + 24, lines[0])
        c.setFont("Helvetica", 6.6)
        c.drawCentredString(bx + bw / 2, by + 12, lines[1])
    c.setStrokeColor(NAVY)
    c.setFillColor(NAVY)
    c.setLineWidth(1.0)
    for start in (x + 89, x + 181, x + 273):
        c.line(start, by + 19, start + 9, by + 19)
        c.line(start + 6, by + 22, start + 9, by + 19)
        c.line(start + 6, by + 16, start + 9, by + 19)

    # The bottom strip is the explicit browser-to-laptop output path.
    c.setFillColor(NAVY)
    c.roundRect(x + 9, y + 22, w - 18, 26, 5, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Courier-Bold", 7.3)
    c.drawCentredString(x + w / 2, y + 32,
                        "final reply -> browser PCM WebSocket -> LAPTOP SPEAKER")
    c.setFillColor(ORANGE)
    c.setFont("Courier-Bold", 6.8)
    c.drawCentredString(x + w / 2, y + 9,
                        "echo match rejected | genuine overlap queued | stop / skip / steer")


def prompt_box(c: canvas.Canvas, x: float, y: float, w: float, h: float) -> None:
    c.setFillColor(LIGHT)
    c.setStrokeColor(GRID)
    c.setLineWidth(1)
    c.roundRect(x, y, w, h, 8, fill=1, stroke=1)
    c.setFillColor(TEAL)
    c.setFont("Courier-Bold", 9)
    c.drawString(x + 10, y + h - 17, "SYSTEM PROMPT (QWEN FAST MODE) - EXCERPT")
    content = (
        "Filipino / English / Taglish voice assistant for a Philippines customer-support pilot.\n"
        "Formal, respectful, concise spoken replies; never flirtatious or suggestive.\n"
        "Use bounded project tools when asked; confirm calls, SMS, Telegram, or email first.\n"
        "Never expose credentials or claim an external action succeeded without tool evidence.\n"
        "For long work: \"I am processing your request now.\""
    )
    para(c, content, x + 10, y + h - 31, w - 20, size=7.7, leading=11.2,
         color=TEXT, font="Courier")


def context_columns(c: canvas.Canvas, x: float, y: float, w: float) -> None:
    gap = 18
    col = (w - gap) / 2
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 9.2)
    c.drawString(x, y, "GOES IN:")
    c.drawString(x + col + gap, y, "MANAGED:")
    bullet_list(c, [
        "system persona + PH language policy",
        "ASR transcript and VAD endpoint",
        "rolling history + tool results",
        "overlap, echo, and control events",
    ], x, y - 13, col, size=8.5, leading=12, gap=1)
    bullet_list(c, [
        "fast mode; hidden thinking off",
        "24-round tool budget + safe summary",
        "1-3 spoken sentences; short history",
        "latest-prompt queue + background status",
        "sanitized terminal; no credentials or CoT",
    ], x + col + gap, y - 13, col, size=8.5, leading=12, gap=1)


def draw_table(c: canvas.Canvas, x: float, y_top: float, widths: list[float],
               rows: list[list[str]], row_h: float, font_size: float = 7.4,
               header_rows: int = 1) -> float:
    total = sum(widths)
    y = y_top
    for ridx, row in enumerate(rows):
        fill = NAVY if ridx < header_rows else (LIGHT if ridx % 2 else WHITE)
        c.setFillColor(fill)
        c.setStrokeColor(GRID)
        c.rect(x, y - row_h, total, row_h, fill=1, stroke=1)
        xx = x
        for j, width in enumerate(widths):
            c.setStrokeColor(GRID)
            c.line(xx, y, xx, y - row_h)
            txt = row[j] if j < len(row) else ""
            color = WHITE if ridx < header_rows else TEXT
            font = "Helvetica-Bold" if ridx < header_rows else "Helvetica"
            para(c, txt, xx + 5, y - 5, width - 10, size=font_size,
                 leading=font_size * 1.18, color=color, font=font,
                 max_height=row_h - 5)
            xx += width
        c.line(xx, y, xx, y - row_h)
        y -= row_h
    return y


def panel(c: canvas.Canvas, x: float, y: float, w: float, h: float,
          title: str, lines: list[str], color=TEAL) -> None:
    c.setFillColor(LIGHT)
    c.setStrokeColor(GRID)
    c.roundRect(x, y, w, h, 7, fill=1, stroke=1)
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 9.3)
    c.drawString(x + 9, y + h - 17, title)
    yy = y + h - 31
    for line in lines:
        used = para(c, line, x + 9, yy, w - 18, size=7.7, leading=10.2,
                    color=TEXT if not line.startswith("Not ") else MUTED)
        yy -= used + 2


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(W, H), pageCompression=1)
    c.setTitle("ME3 Qwen Instruct - Machine Exercise Full-Duplex Voice Agent")
    c.setAuthor("ME3 Qwen Instruct project")
    c.setSubject("Filled two-page machine exercise based on the supplied template")

    # Page 1
    header(c, "Full-Duplex Voice Agent on a Raspberry Pi",
           "QWEN INSTRUCT BUILD | 06 NOV 2026",
           "MACHINE EXERCISE - ME3 QWEN INSTRUCT")
    section(c, "1", "SYSTEM DIAGRAM", "input -> Pi -> agent -> laptop speaker", 48, 418)
    diagram(c, 48, 255, 430, 145)
    bullet_list(c, [
        "Fifine mic is captured by the Pi service; laptop browser is the speaker sink in the intended demo route.",
        "Pi owns echo guard, Silero VAD, MCP servers, task safety, and duplex state; Qwen services are reached over VPN.",
        "Qwen2-Audio ASR -> Qwen ToolBrain / MCP -> Qwen3-TTS Aiden; output is 16-kHz PCM to the browser.",
        "The agent supports web/file/build tools and confirmation-gated channels; no model training is required.",
        "Protected overlap keeps listening open: matched echo is rejected; genuine speech is queued; stop, skip, and steer are explicit controls.",
    ], 500, 395, 410, size=8.8, leading=12.5, gap=2)
    c.setFillColor(NAVY)
    c.roundRect(48, 228, 430, 20, 5, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Courier-Bold", 7.4)
    c.drawCentredString(263, 235, "voice in -> VAD -> ASR MCP -> Qwen agent + tools -> TTS MCP -> browser -> speaker")
    c.setFillColor(ORANGE)
    c.setFont("Courier-Bold", 7.1)
    c.drawCentredString(263, 215, "benchmark mode: injected audio only | no physical mic/speaker | tools and fillers disabled")

    section(c, "2", "PROMPT", "versioned in config/config.yaml", 48, 190)
    prompt_box(c, 48, 57, 430, 119)
    section(c, "3", "CONTEXT ENGINEERING", "short, safe, observable turns", 500, 190)
    context_columns(c, 510, 164, 395)
    footer(c, 1)
    c.showPage()

    # Page 2
    header(c, "MCP Services, Evaluation & Model Serving",
           "build + run owned MCP services on the RPi")
    section(c, "4", "MCP SERVICES", "+ model serving: one table, one truth", 48, 428)
    rows = [
        ["Component", "MCP server / transport", "Tools", "Model", "Served where / notes"],
        ["ASR", "me3-asr / FastMCP", "transcribe(audio) -> text", "Qwen2-Audio-7B-Instruct", "Pi service; Qwen vLLM over VPN; final at VAD endpoint"],
        ["TTS", "me3-tts / FastMCP", "speak(text) -> audio", "Qwen3-TTS-12Hz-1.7B-CustomVoice; Aiden", "Pi service; Qwen vLLM over VPN; one formal voice"],
        ["LLM", "ToolBrain / MCP client", "function calling", "Qwen/Qwen3.8-27B", "Remote vLLM; fast_mode=true; hidden thinking=false"],
        ["VAD", "me3-vad / FastMCP", "detect_speech(frames)", "Silero ONNX", "On-device Pi CPU; 16 kHz mono; endpointing"],
        ["Task + channels", "me3-tasks / FastMCP + REST", "files, web, build, timers, ME2 parity, channels", "n/a / provider APIs", "Pi-owned tools; live IoT/provider effects require confirmation"],
    ]
    draw_table(c, 48, 405, [63, 150, 165, 160, 326], rows, 18, font_size=6.8)

    section(c, "5", "EVALUATION & BENCHMARK", "updated class direct-feed protocol", 48, 274)
    bullet_list(c, [
        "One clock: benchmark direct feed uses the assistant machine clock and records stage marks per turn.",
        "Injection only: no physical microphone or speaker; no acoustic feedback in the latency run.",
        "Benchmark mode: no tool schemas, external agent/harness calls, fillers, backchannels, or real channel side effects.",
        "Report mean + p95 over the updated stage-latency phrases and preserve JSON, CSV, HTML, and latency chart artifacts.",
    ], 48, 250, 425, size=8.3, leading=11.2, gap=1.5)

    metrics = [
        ["Metric", "Definition", "Target", "Mean", "p95"],
        ["E2E latency", "end of speech -> audio out", "4.0 s", "", ""],
        ["ASR final", "VAD end -> transcript", "0.8 s", "", ""],
        ["LLM TTFT", "transcript -> first token", "1.5 s", "", ""],
        ["TTS TTFB", "text -> first audio chunk", "0.6 s", "", ""],
        ["Barge-in", "speech -> playback stopped", "0.8 s", "", ""],
        ["WER", "ASR accuracy on test set", "15%", "", ""],
    ]
    draw_table(c, 507, 262, [83, 160, 52, 47, 47], metrics, 17, font_size=6.9)

    panel(c, 48, 37, 277, 96, "DEMO (deck 2)", [
        "Available now: injected Qwen voice/tool turn, web-app creation, and ME2 parity tools for music, weather, media, IoT state, timers, reminders, and channels.",
        "The intended physical demo is Fifine mic on Pi -> Pi ME3 -> browser -> laptop speaker.",
        "Not claimed here: a clean human Pi spoken turn and physical speaker observation.",
    ])
    panel(c, 339, 37, 277, 96, "RESULTS (deck 2)", [
        "Workstation acceptance: 70 tests passed; all four MCP health checks passed; Qwen injected voice/tool path passed with zero errors.",
        "Qwen fixture evidence: ASR 431.9 ms and TTS first-byte 851.6 ms on the recorded comparison run.",
        "Class mean/p95 cells above remain blank until the updated upstream direct-feed run is executed.",
    ])
    panel(c, 630, 37, 282, 96, "CHALLENGES & REFERENCES", [
        "Pi audio: restore Fifine ALSA capture; run a human overlap/echo check.",
        "Benchmark: run commit 57eaddb; attach stage marks and latency chart.",
        "Telegram text is implemented; ordinary voice calls need a separately reviewed MTProto/TDLib bridge.",
        "Refs: MCP, Pipecat, Qwen vLLM, upstream fd-benchmark, ME3 repo/logs.",
    ])
    footer(c, 2)
    c.save()
    return OUTPUT


if __name__ == "__main__":
    print(build())
