from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "ME2-VCM-Dataset-Comparison-and-Next-Steps.pdf"
REVISION = "a90b8d106349b02c5570a1a258503386043f63b2"
DATASET_URL = (
    "https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/"
    f"blob/{REVISION}/README.md"
)
BENCHMARK_URL = (
    "https://github.com/airimonda/vcm-benchmark/blob/"
    "3bd722173a040356cc71a4a902188af183540b43/README.md"
)

NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#DCEAF7")
PALE = colors.HexColor("#F3F6F9")
BORDER = colors.HexColor("#AAB7C4")
TEXT = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#506070")
GREEN = colors.HexColor("#E2F0E8")


def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCustom", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=19, leading=22, textColor=NAVY, alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCustom", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=12, textColor=MUTED, spaceAfter=7,
        ),
        "h1": ParagraphStyle(
            "H1Custom", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=13, leading=15, textColor=NAVY, spaceBefore=4,
            spaceAfter=5,
        ),
        "h2": ParagraphStyle(
            "H2Custom", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=10, leading=12, textColor=NAVY, spaceBefore=4,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyCustom", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.2, leading=10.5, textColor=TEXT, spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "SmallCustom", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.2, leading=9, textColor=TEXT,
        ),
        "cell": ParagraphStyle(
            "CellCustom", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.2, leading=9, textColor=TEXT,
        ),
        "cell_head": ParagraphStyle(
            "CellHeadCustom", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=7.3, leading=9, textColor=colors.white,
        ),
        "callout": ParagraphStyle(
            "CalloutCustom", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=8.5, leading=11, textColor=NAVY, backColor=BLUE,
            borderPadding=6, spaceBefore=3, spaceAfter=6,
        ),
        "center": ParagraphStyle(
            "CenterCustom", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.2, leading=9, textColor=TEXT, alignment=TA_CENTER,
        ),
    }


def p(text, style):
    return Paragraph(text, style)


def cell(text, style):
    return p(text, style)


def make_table(rows, widths, header=True, repeat_rows=1):
    table = Table(rows, colWidths=widths, repeatRows=repeat_rows if header else 0,
                  hAlign="LEFT", splitByRow=1)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ]
    table.setStyle(TableStyle(commands))
    return table


def footer(canvas, doc):
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(38, 29, width - 38, 29)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(38, 18, "ME2 - VCM on Raspberry Pi 5 | Dataset comparison")
    canvas.drawRightString(width - 38, 18, str(doc.page))
    canvas.restoreState()


def build():
    styles = make_styles()
    output_dir = OUTPUT.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, rightMargin=38, leftMargin=38,
        topMargin=38, bottomMargin=38, title="ME2 - VCM on Raspberry Pi 5: Dataset comparison",
        author="ME2 VCM project",
        subject="Plain-language comparison of the deployed baseline and latest dataset revision",
    )
    story = []

    # Page 1: the practical difference between the deployed model and the data.
    story += [
        p("ME2 - VCM on Raspberry Pi 5", styles["title"]),
        p("A plain-language comparison of the running VCM, the earlier dataset snapshot, and the latest linked dataset revision.", styles["subtitle"]),
        p("<b>Important:</b> This is a dataset update, not a model update. The latest snapshot has not been used to train or score a replacement.", styles["callout"]),
        p("Decision in plain language", styles["h1"]),
        p("Keep the deployed Pi demo and its separate wake model as the working version. Treat the latest dataset as the source for a future, isolated candidate. Do not overwrite or sync a model from this pull. Data permissions and safe rejection behavior still need decisions.", styles["body"]),
        p("Your brief lists ten everyday command families. The shared dataset splits those into 19 intent names and 31 supported command/value labels; for example, media becomes pause/next/stop/volume, while lights become on/off/brightness/color. These are finer actions inside the listed families, not 19 unrelated use cases.", styles["body"]),
        p("Version comparison", styles["h1"]),
    ]
    headers = ["Topic", "Current VCM", "Earlier linked snapshot", "Latest repull"]
    rows = [[cell(x, styles["cell_head"]) for x in headers]]
    comparison = [
        ("What it is",
         "Running two-model system: binary wake detector, then a 31-label TinyDSCNN-48 command classifier.",
         "Dataset snapshot only; not itself a model release.",
         "Dataset snapshot only; not a trained replacement."),
        ("Data / training",
         "Intent model trained from scratch on the earlier project command corpus plus personal recordings. Run record: 14,370 base training draws per epoch and 374 unique personal training clips.",
         "10,682 train; 4,418 test; 196 holdout; 66,390 separate numeral clips.",
         "Same training, test, and numeral bytes as the earlier snapshot. Same row totals."),
        ("Labels",
         "31 combined command/value outputs. Each supported value is a separate label.",
         "19 broad intents; 31 allowed intent/value combinations; 93 phrase variants.",
         "Same 93-row phrase sheet and same 31 allowed combinations. Current labels map exactly."),
        ("What changed",
         "This is the deployed baseline; no new weights were made in this audit.",
         "Previous data revision: 25111444.",
         "README clarified the 158 other-slot-value rows. Holdout changed: 59 rows replaced, including 56 in-scope real-voice clips and 3 OOS clips."),
        ("Performance / gates",
         "Previously saved: 96.89% accuracy / 96.90% macro F1 on a reused reference test; 92.17% / 92.44% on 115 personal clips. Wake cutoff 0.95; intent gate 0.68 confidence plus 0.15 margin.",
         "No new score was produced by this data snapshot.",
         "No model trained or scored on this pull. No new accuracy, F1, or threshold."),
        ("Deployment",
         "Paired INT8 models total about 76.7 KB. Existing Pi and Desktop copies preserved.",
         "No deployment from this dataset snapshot.",
         "No ONNX, Pi, model threshold, or active release changed."),
    ]
    for row in comparison:
        rows.append([cell(v, styles["cell"]) for v in row])
    story.append(make_table(rows, [68, 158, 145, 161]))
    story += [
        Spacer(1, 6),
        p("The old-model scores are saved-file results, not live-microphone guarantees. Personal partitions overlap speakers. The new 4,418-row test has not been scored in this audit.", styles["small"]),
        Spacer(1, 4),
        p(f"Revision {REVISION}. All 13 files matched the Hub; all 81,686 WAVs decoded. There were zero row errors, cross-label collisions, and cross-split audio groups. One duplicate-PCM group is confined to a single label and split.", styles["small"]),
        PageBreak(),
    ]

    # Page 2: explain broad intents, value labels, and the unusual rows.
    story += [
        p("How the labels fit together", styles["h1"]),
        p("The dataset has 19 intent names. Some intents carry a value, or slot, such as a color or timer duration. The current model combines each allowed intent and value into one of 31 labels. Three spoken phrasings can express the same command/value pair, so the 93 phrases are not 93 different device actions.", styles["body"]),
    ]
    tax_rows = [[cell(x, styles["cell_head"]) for x in ["Dataset intent", "Current combined model labels", "Meaning"]]]
    tax = [
        ("COLOR", "COLOR_RED, COLOR_GREEN, COLOR_BLUE", "One intent; three allowed colors."),
        ("TIMER", "TIMER_10s, TIMER_30s, TIMER_1m", "One intent; three allowed durations."),
        ("ALARM", "ALARM_6_00AM, ALARM_8_00AM, ALARM_9_00PM", "One intent; three allowed times."),
        ("BRIGHTNESS", "BRIGHTNESS_20, BRIGHTNESS_60, BRIGHTNESS_100", "One intent; three allowed levels."),
        ("TEMPERATURE", "TEMPERATURE_18, TEMPERATURE_22, TEMPERATURE_26", "One intent; three allowed settings."),
        ("CREATE_REMINDER", "CREATE_REMINDER_DRINK_WATER, CREATE_REMINDER_EXERCISE, CREATE_REMINDER_STUDY", "One intent; three allowed items."),
        ("Fixed-value intents", "PLAY_MUSIC, LIGHT_ON, LIGHT_OFF, PAUSE, NEXT, STOP, VOLUME_UP, VOLUME_DOWN, WEATHER, TIME, CALL, MESSAGE, LIST_REMINDERS", "13 intents with one fixed command each."),
    ]
    for row in tax:
        tax_rows.append([cell(x, styles["cell"]) for x in row])
    story += [
        make_table(tax_rows, [105, 225, 207]),
        Spacer(1, 5),
        p("<b>Wake is separate.</b> The wake detector is a two-class wake/not-wake model, not one of the 19 command intents.", styles["body"]),
        p("What are the 158 unusual rows?", styles["h1"]),
        p("The dataset README documents these as recordings for an in-scope intent with a slot value outside the allowed list. Example: a timer request with a duration outside the three supported choices. They remain marked out_of_scope=0 and sit in an other-slot-value bucket. They are not extra intents and are not data errors.", styles["body"]),
    ]
    unusual = [[cell(x, styles["cell_head"]) for x in ["Category", "Train", "Test", "Holdout", "Meaning"]]]
    for row in [
        ("Supported commands", "10,481", "4,371", "186", "Allowed intents and values."),
        ("True out of scope", "201", "47", "10", "Requests outside the 19 intents."),
        ("Other slot value", "158", "0", "0", "In-scope intent; unsupported value."),
    ]:
        unusual.append([cell(v, styles["cell"]) for v in row])
    story += [
        make_table(unusual, [112, 55, 55, 55, 260]),
        Spacer(1, 6),
        p("Do not relabel the 158 as true out of scope or force them to the nearest allowed value. Decide how to safely reject unsupported values. The benchmark accepts either an explicit out-of-scope result or no response; it does not require a 32nd output class.", styles["callout"]),
        p("The 196-row holdout was refreshed: two clips for each of the 93 supported variations plus 10 true OOS clips. Keep it sealed for the agreed live demo; do not train on it or tune thresholds with it.", styles["small"]),
        PageBreak(),
    ]

    # Page 3: directly answer whether to keep extra commands and what next.
    story += [
        p("Can the project keep extra intents?", styles["h1"]),
        p("Yes, as a clearly separated extension. For the course benchmark, keep the scored model aligned to the agreed 19 intents and their 31 supported command/value outputs. Extra spoken actions should not be folded into a core label or added to the core accuracy number.", styles["body"]),
    ]
    option_rows = [[cell(x, styles["cell_head"]) for x in ["Choice", "Use it for", "How to report it"]]]
    for row in [
        ("Agreed core (recommended for course score)", "19 intents and 31 allowed intent/value outputs. Safe reject/no-action behavior for true OOS and unsupported values.", "Report intent and command-level scores separately, with per-class support and false accepts."),
        ("Optional extra commands", "Additional user-requested actions outside the agreed list, after collecting labeled audio and testing them separately.", "Call them an extension. Publish extension metrics separately; do not mix them into the core benchmark score."),
        ("Wake word", "Separate binary wake model that decides when command recognition begins.", "Report wake detection and false-wake behavior separately. It is not an extra command intent."),
    ]:
        option_rows.append([cell(v, styles["cell"]) for v in row])
    story += [
        make_table(option_rows, [120, 221, 195]),
        Spacer(1, 6),
        p("What the audit found", styles["h1"]),
        p("The current 31 command labels match the dataset's 31 supported command/value combinations. The current label list has no extra command intent. The wake model is separate. If an old recorder screen or archived folder exposes a command outside the agreed list, keep it out of the core model unless you deliberately create a separately measured extension.", styles["body"]),
        p("Recommended next steps", styles["h1"]),
    ]
    next_rows = [[cell(x, styles["cell_head"]) for x in ["Step", "Action"]]]
    for row in [
        ("1", "Keep the latest snapshot immutable. It is downloaded and hash-verified; the previous snapshot is preserved."),
        ("2", "Resolve source permissions and speaker consent before using the full corpus. The audit has unresolved license and voice-consent items."),
        ("3", "Agree how to handle true OOS speech and in-scope intents with unsupported slot values. The 158 rows are a separate category."),
        ("4", "If approved, train a separate scratch candidate using train only; carve validation from train speakers/voices. Leave the deployed pair untouched."),
        ("5", "Freeze the candidate and thresholds, then run the frozen test once. Use holdout only for the approved live Raspberry Pi demo."),
        ("6", "Package for Pi only after candidate audit and local checks pass; preserve the current release and rollback path."),
    ]:
        next_rows.append([cell(row[0], styles["center"]), cell(row[1], styles["cell"])])
    story += [
        make_table(next_rows, [35, 501]),
        Spacer(1, 6),
        p("<b>You do not need to change the core taxonomy right now.</b> Your current 31 command labels already match the agreed command/value list. The remaining choices are permission to use the data and how to reject unknown requests safely.", styles["callout"]),
        p("Sources and evidence", styles["h2"]),
        p(f'Dataset README at the <link href="{DATASET_URL}" color="#145E9B">exact repulled revision</link>; benchmark README at the <link href="{BENCHMARK_URL}" color="#145E9B">inspected commit</link>. Local full-scan report: docs/classagreementvcm-repull-audit-20261002.md and runs/classagreementvcm-audit-20261002-repull/gold_dataset_audit.json.', styles["small"]),
        Spacer(1, 3),
        p("No new training, frozen test scoring, live microphone trial, or Pi test was performed for the repull.", styles["small"]),
    ]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build()