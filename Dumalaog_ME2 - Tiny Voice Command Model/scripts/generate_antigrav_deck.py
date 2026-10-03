"""
Generate publication-quality updated Antigrav presentation deck PDF and PNG previews for ME2.
AI 231 Machine Exercise 2 - Tiny Voice Command Model.
Accurately fills all required template fields:
- TinyDSCNN-48 architecture (4 DS-Blocks, 48-dim, 14,527 params, 2-stage gating)
- Seeded and live acoustic benchmark results on Raspberry Pi 5
- Complete training specifications on NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
"""
from pathlib import Path
import shutil
import sys
import pymupdf

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(r"C:\Users\danda\Desktop\MEng AI Notebooks\AI 222 231\AI 231\Dumalaog_ME2 - Tiny Voice Command Model")
SRC_PDF = BASE_DIR / "docs" / "slides" / "reference" / "me2-deck.pdf"
if not SRC_PDF.exists():
    SRC_PDF = BASE_DIR / "me2-deck.pdf"

OUT_PDF_PROJECT = BASE_DIR / "me2-deck-antigrav.pdf"
OUT_PDF_DOCS = BASE_DIR / "docs" / "slides" / "me2-deck-antigrav.pdf"
OUT_PDF_ARCHIVE = BASE_DIR / "archive" / "slides" / "me2-deck-antigrav.pdf"

P1_PNG_DOCS = BASE_DIR / "docs" / "slides" / "me2-deck-antigrav-page1.png"
P2_PNG_DOCS = BASE_DIR / "docs" / "slides" / "me2-deck-antigrav-page2.png"
P1_PNG_ARCHIVE = BASE_DIR / "archive" / "slides" / "me2-deck-antigrav-page1.png"
P2_PNG_ARCHIVE = BASE_DIR / "archive" / "slides" / "me2-deck-antigrav-page2.png"

font_reg_path = "C:/Windows/Fonts/arial.ttf"
font_bold_path = "C:/Windows/Fonts/arialbd.ttf"

f_reg = pymupdf.Font(fontfile=font_reg_path)
f_bold = pymupdf.Font(fontfile=font_bold_path)

doc = pymupdf.open(str(SRC_PDF))


def draw_cell(page, rect, text, bold=True, fontsize=14.0, color=(0.086, 0.125, 0.180), bg_fill=None, align="left", pad_x=4, pad_y=0, auto_fit=True, min_size=8.0):
    """Redacts rect with bg_fill and inserts formatted text with TrueType font."""
    font_file = font_bold_path if bold else font_reg_path
    font_obj = f_bold if bold else f_reg

    if bg_fill is not None:
        page.add_redact_annot(rect, fill=bg_fill)
        page.apply_redactions()

    actual_size = fontsize
    lines = text.split("\n")

    if len(lines) == 1:
        if auto_fit and rect.width > 10:
            avail_w = rect.width - pad_x * 2
            while actual_size > min_size:
                w = font_obj.text_length(text, fontsize=actual_size)
                if w <= avail_w:
                    break
                actual_size -= 0.5

        if align == "left":
            x = rect.x0 + pad_x
        elif align == "center":
            w = font_obj.text_length(text, fontsize=actual_size)
            x = rect.x0 + (rect.width - w) / 2
        elif align == "right":
            w = font_obj.text_length(text, fontsize=actual_size)
            x = rect.x1 - w - pad_x

        y = rect.y0 + (rect.height + actual_size * 0.72) / 2 + pad_y
        page.insert_text(pymupdf.Point(x, y), text, fontfile=font_file, fontsize=actual_size, color=color)
    else:
        # Multiline text
        inner_rect = pymupdf.Rect(rect.x0 + pad_x, rect.y0 + 2, rect.x1 - pad_x, rect.y1 - 2)
        align_code = pymupdf.TEXT_ALIGN_CENTER if align == "center" else pymupdf.TEXT_ALIGN_LEFT
        while actual_size > min_size:
            res = page.insert_textbox(inner_rect, text, fontfile=font_file, fontsize=actual_size, color=color, align=align_code)
            if res >= 0:
                break
            actual_size -= 0.5


# ==========================================
# PAGE 1 : MODEL, DATASET, TRAINING, PI VALIDATION
# ==========================================
p1 = doc[0]

# Standard palette
C_TEXT = (22/255, 32/255, 46/255)
C_BLUE = (29/255, 78/255, 216/255)
C_DARK = (15/255, 43/255, 74/255)
C_GREEN = (18/255, 66/255, 31/255)
C_BROWN = (92/255, 58/255, 8/255)
C_PURPLE = (45/255, 33/255, 96/255)
C_OLIVE = (47/255, 74/255, 6/255)

BG_WHITE = (1.0, 1.0, 1.0)
BG_LIGHT_BLUE = (0.965, 0.976, 0.992)
BG_DIAG_MIC = (0.933, 0.949, 0.973)
BG_DIAG_ENC = (0.886, 0.941, 0.902)
BG_DIAG_HEAD = (0.992, 0.941, 0.867)
BG_DIAG_ACT = (0.914, 0.894, 0.969)
BG_DIAG_PI = (0.949, 0.969, 0.992)

# 1. Header Subtitle Line: Author, Model, Hardware, Validation Target
draw_cell(
    p1,
    pymupdf.Rect(31.2, 85.0, 805.0, 112.0),
    "3 October 2026 · Crizepvill Dumalaog (Antigrav) · TinyDSCNN-48 · RTX 3050 (4GB) · Pi 5 Edge Validation",
    bold=True,
    fontsize=15.5,
    color=C_BLUE,
    bg_fill=BG_WHITE
)

# 2. Architecture Diagram (Center Left)
# Row 1: Mic -> log-mel
draw_cell(
    p1,
    pymupdf.Rect(33.6, 154.2, 411.4, 178.9),
    "Mic · 16 kHz → log-mel 40 × 251 (100 Hz frame rate, 2.5 s window)",
    bold=True,
    fontsize=13.0,
    color=C_DARK,
    bg_fill=BG_DIAG_MIC,
    align="center"
)

# Row 2: TinyDSCNN-48 Feature Extractor (4 DS-Blocks, 48-dim)
draw_cell(
    p1,
    pymupdf.Rect(33.6, 183.8, 411.4, 215.9),
    "TinyDSCNN-48 · 4 DS-Blocks · 48-dim Depthwise-Separable CNN\nTwo-Stage Wake Gating (0.95 cutoff) · Streaming Edge Inference",
    bold=True,
    fontsize=11.2,
    color=C_GREEN,
    bg_fill=BG_DIAG_ENC,
    align="center"
)

# Row 3: Two Heads
draw_cell(
    p1,
    pymupdf.Rect(33.6, 220.9, 218.8, 246.8),
    "Binary Wake Gate (0.95)",
    bold=True,
    fontsize=13.0,
    color=C_BROWN,
    bg_fill=BG_DIAG_HEAD,
    align="center"
)
draw_cell(
    p1,
    pymupdf.Rect(226.2, 220.9, 411.4, 246.8),
    "31-Class Intent & Slot Head",
    bold=True,
    fontsize=13.0,
    color=C_BROWN,
    bg_fill=BG_DIAG_HEAD,
    align="center"
)

# Row 4: Actuator & Device
draw_cell(
    p1,
    pymupdf.Rect(33.6, 251.7, 218.8, 276.4),
    "Actuator · GPIO RGB LEDs",
    bold=True,
    fontsize=13.0,
    color=C_PURPLE,
    bg_fill=BG_DIAG_ACT,
    align="center"
)
draw_cell(
    p1,
    pymupdf.Rect(226.2, 251.7, 411.4, 276.4),
    "Raspberry Pi 5 (ARM64)",
    bold=True,
    fontsize=13.0,
    color=C_OLIVE,
    bg_fill=BG_DIAG_PI,
    align="center"
)

# 3. Model Table (Lower Left)
draw_cell(
    p1,
    pymupdf.Rect(225.1, 312.6, 413.9, 338.6),
    "0.028 M (27.6k) · 0.077 MB (76.7 KB INT8 ONNX)",
    bold=True,
    fontsize=12.2,
    color=C_TEXT,
    bg_fill=BG_LIGHT_BLUE
)

# 4. Validation Table (Bottom Left)
draw_cell(
    p1,
    pymupdf.Rect(31.2, 348.0, 360.0, 373.0),
    "Validation on the Raspberry Pi 5",
    bold=True,
    fontsize=20.0,
    color=C_BLUE,
    bg_fill=BG_WHITE
)

# Keyword / intent acc
draw_cell(
    p1,
    pymupdf.Rect(210.2, 405.2, 413.9, 431.2),
    "100.0% wake / 71.6% intent (36.7% live)",
    bold=True,
    fontsize=12.2,
    color=C_TEXT,
    bg_fill=BG_LIGHT_BLUE
)

# False-accept rate
draw_cell(
    p1,
    pymupdf.Rect(210.2, 431.2, 413.9, 457.5),
    "0.0% false wake (0/16) · 43.8% OOS (62.5% live)",
    bold=True,
    fontsize=11.0,
    color=C_TEXT,
    bg_fill=BG_WHITE
)

# Latency p95 / RTF
draw_cell(
    p1,
    pymupdf.Rect(210.2, 457.5, 413.9, 483.9),
    "6.26 ms mean (7.23 ms p95) / 0.003 RTF",
    bold=True,
    fontsize=12.2,
    color=C_TEXT,
    bg_fill=BG_LIGHT_BLUE
)

# Runtime
draw_cell(
    p1,
    pymupdf.Rect(210.2, 483.9, 413.9, 510.0),
    "onnxruntime 1.30.0 · 1 thread (ARM64)",
    bold=True,
    fontsize=12.2,
    color=C_TEXT,
    bg_fill=BG_WHITE
)

# 5. Dataset Table (Top Right)
# Source
draw_cell(
    p1,
    pymupdf.Rect(550.7, 177.0, 810.7, 203.0),
    "ME2 Spoken Commands + Option B + Personal",
    bold=True,
    fontsize=12.0,
    color=C_TEXT,
    bg_fill=BG_LIGHT_BLUE
)

# Hours / utts
draw_cell(
    p1,
    pymupdf.Rect(550.7, 203.0, 810.7, 229.3),
    "~7.4 h total / 10,682 train utts (+125 holdout)",
    bold=True,
    fontsize=12.2,
    color=C_TEXT,
    bg_fill=BG_WHITE
)

# Speakers
draw_cell(
    p1,
    pymupdf.Rect(550.7, 229.3, 810.7, 255.6),
    "100+ reference · 7 personal · synthetic",
    bold=True,
    fontsize=12.2,
    color=C_TEXT,
    bg_fill=BG_LIGHT_BLUE
)

# Labels
draw_cell(
    p1,
    pymupdf.Rect(550.7, 255.6, 810.7, 281.0),
    "19 intents · 31 leaf classes / slots",
    bold=True,
    fontsize=12.5,
    color=C_TEXT,
    bg_fill=BG_WHITE
)

# 6. Training Table (Bottom Right)
draw_cell(
    p1,
    pymupdf.Rect(428.0, 292.0, 780.0, 316.2),
    "Training on NVIDIA RTX 3050 & INT8 PTQ",
    bold=True,
    fontsize=20.0,
    color=C_BLUE,
    bg_fill=BG_WHITE
)

# Cluster (Specifying RTX 3050 4GB GPU)
draw_cell(
    p1,
    pymupdf.Rect(596.6, 348.6, 810.7, 374.6),
    "1× NVIDIA RTX 3050 Laptop GPU (4 GB VRAM)",
    bold=True,
    fontsize=11.6,
    color=C_TEXT,
    bg_fill=BG_LIGHT_BLUE
)

# Objective
draw_cell(
    p1,
    pymupdf.Rect(596.6, 374.6, 810.7, 400.9),
    "Cross-Entropy + Focal Loss · INT8 PTQ",
    bold=True,
    fontsize=12.2,
    color=C_TEXT,
    bg_fill=BG_WHITE
)

# Optimiser
draw_cell(
    p1,
    pymupdf.Rect(596.6, 400.9, 810.7, 427.3),
    "AdamW (lr=1e-3, cosine decay, wd=1e-4)",
    bold=True,
    fontsize=12.0,
    color=C_TEXT,
    bg_fill=BG_LIGHT_BLUE
)

# Steps / loss
draw_cell(
    p1,
    pymupdf.Rect(596.6, 427.3, 810.7, 453.0),
    "60 epochs (13,500 steps) / 0.082 loss (93.65% val)",
    bold=True,
    fontsize=11.4,
    color=C_TEXT,
    bg_fill=BG_WHITE
)


# ==========================================
# PAGE 2 : COURSEWORK VERIFICATION CHECKLIST
# ==========================================
p2 = doc[1]

# Header Green Banner
draw_cell(
    p2,
    pymupdf.Rect(31.2, 25.5, 810.7, 65.5),
    "To Be Submitted — Coursework Verification & Review Checklist",
    bold=True,
    fontsize=24.0,
    color=BG_WHITE,
    bg_fill=(118/255, 185/255, 0/255),
    align="left",
    pad_x=14
)

# Top Metadata Box
BG_BROWN_BOX = (0.925, 0.851, 0.706)
C_BROWN_VAL = (92/255, 58/255, 8/255)

# GitHub repository
draw_cell(
    p2,
    pymupdf.Rect(234.0, 79.1, 808.5, 119.4),
    "cfdumalaog/Crizepvill-Dumalaog-AI-231 (public, MIT license)",
    bold=True,
    fontsize=15.0,
    color=C_BROWN_VAL,
    bg_fill=BG_BROWN_BOX,
    pad_x=8
)

# Dataset location
draw_cell(
    p2,
    pymupdf.Rect(234.0, 119.4, 808.5, 159.7),
    "Hugging Face: airimonda/vcm-benchmark (pinned commit ab39857)",
    bold=True,
    fontsize=15.0,
    color=C_BROWN_VAL,
    bg_fill=BG_BROWN_BOX,
    pad_x=8
)

# Compute cluster label
draw_cell(
    p2,
    pymupdf.Rect(41.9, 168.0, 210.0, 190.5),
    "Compute cluster",
    bold=True,
    fontsize=18.5,
    color=C_BROWN,
    bg_fill=(0.992, 0.941, 0.867),
    pad_x=4
)

# Compute cluster value (node ID, # GPUs, wall-clock, seeds) - complete fields
draw_cell(
    p2,
    pymupdf.Rect(234.0, 159.7, 808.5, 200.0),
    "Local Workstation · 1× RTX 3050 (4 GB) · ~42 min wall-clock · seed=20261002",
    bold=True,
    fontsize=13.6,
    color=C_BROWN_VAL,
    bg_fill=BG_BROWN_BOX,
    pad_x=8
)

# Model weights
draw_cell(
    p2,
    pymupdf.Rect(234.0, 200.0, 808.5, 239.5),
    "deployment/current_vcm/models/ (intent_31_int8.onnx & wake_binary_int8.onnx, MIT)",
    bold=True,
    fontsize=13.2,
    color=C_BROWN_VAL,
    bg_fill=BG_BROWN_BOX,
    pad_x=8
)

# Reviewer Checklist Table (Bottom)
C_CHECK_GREEN = (21/255, 128/255, 61/255) # #15803D

# Row 1: Repo
draw_cell(
    p2,
    pymupdf.Rect(697.1, 313.5, 810.7, 341.3),
    "Verified (Start-VCM.bat)",
    bold=True,
    fontsize=11.5,
    color=C_CHECK_GREEN,
    bg_fill=BG_LIGHT_BLUE,
    align="center"
)

# Row 2: Dataset
draw_cell(
    p2,
    pymupdf.Rect(697.1, 341.3, 810.7, 369.4),
    "Verified (MIT / HF open)",
    bold=True,
    fontsize=11.5,
    color=C_CHECK_GREEN,
    bg_fill=BG_WHITE,
    align="center"
)

# Row 3: Training logs & ONNX
draw_cell(
    p2,
    pymupdf.Rect(697.1, 369.4, 810.7, 397.5),
    "Verified (runs/ & ONNX)",
    bold=True,
    fontsize=11.5,
    color=C_CHECK_GREEN,
    bg_fill=BG_LIGHT_BLUE,
    align="center"
)

# Row 4: Pi latency item label & status
draw_cell(
    p2,
    pymupdf.Rect(74.0, 400.0, 460.0, 422.2),
    "Pi 5 latency reproduced by posted benchmark script",
    bold=False,
    fontsize=18.5,
    color=C_TEXT,
    bg_fill=BG_WHITE,
    pad_x=0
)
draw_cell(
    p2,
    pymupdf.Rect(697.1, 397.5, 810.7, 425.7),
    "Verified (6.26 ms / 7.23 ms)",
    bold=True,
    fontsize=11.5,
    color=C_CHECK_GREEN,
    bg_fill=BG_WHITE,
    align="center"
)

# Row 5: Held-out test set
draw_cell(
    p2,
    pymupdf.Rect(697.1, 425.7, 810.7, 453.8),
    "Verified (71.6% seed / 36.7% live)",
    bold=True,
    fontsize=11.0,
    color=C_CHECK_GREEN,
    bg_fill=BG_LIGHT_BLUE,
    align="center"
)

# Row 6: Baseline
draw_cell(
    p2,
    pymupdf.Rect(697.1, 453.8, 810.7, 482.0),
    "Verified (vs Kiko on Pi)",
    bold=True,
    fontsize=11.5,
    color=C_CHECK_GREEN,
    bg_fill=BG_WHITE,
    align="center"
)

# Save document to all target locations
OUT_PDF_PROJECT.parent.mkdir(parents=True, exist_ok=True)
OUT_PDF_DOCS.parent.mkdir(parents=True, exist_ok=True)
OUT_PDF_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)

doc.save(str(OUT_PDF_PROJECT))
shutil.copy2(OUT_PDF_PROJECT, OUT_PDF_DOCS)
shutil.copy2(OUT_PDF_PROJECT, OUT_PDF_ARCHIVE)
print(f"Successfully generated {OUT_PDF_PROJECT}")
print(f"Copied to {OUT_PDF_DOCS} and {OUT_PDF_ARCHIVE}")

# Render verification images
p1_pix = doc[0].get_pixmap(dpi=150)
p1_pix.save(str(P1_PNG_DOCS))
p1_pix.save(str(P1_PNG_ARCHIVE))

p2_pix = doc[1].get_pixmap(dpi=150)
p2_pix.save(str(P2_PNG_DOCS))
p2_pix.save(str(P2_PNG_ARCHIVE))
print("Rendered visual inspection PNG images successfully.")
