"""Build a PDF reference of ME2 intents, dataset phrase variations, and recorder prompts."""

from __future__ import annotations

import csv
from collections import defaultdict
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "data" / "ai231-me2-voice-commands-hf-a90b8d10"
VARIATIONS = DATASET / "variations.csv"
PHRASE_DIR = PROJECT / "docs" / "ground_truth_phrases"
OUTPUT = PROJECT / "outputs" / "pdf" / "ME2_VCM_Intents_and_Phrase_Variations.pdf"
DATASET_REVISION = "a90b8d106349b02c5570a1a258503386043f63b2"


SLOT_TO_COMMAND_LABEL = {
    ("ALARM", "6:00 AM"): "ALARM_6_00AM",
    ("ALARM", "8:00 AM"): "ALARM_8_00AM",
    ("ALARM", "9:00 PM"): "ALARM_9_00PM",
    ("BRIGHTNESS", "20 percent"): "BRIGHTNESS_20",
    ("BRIGHTNESS", "60 percent"): "BRIGHTNESS_60",
    ("BRIGHTNESS", "100 percent"): "BRIGHTNESS_100",
    ("COLOR", "Red"): "COLOR_RED",
    ("COLOR", "Green"): "COLOR_GREEN",
    ("COLOR", "Blue"): "COLOR_BLUE",
    ("CREATE_REMINDER", "Drink water"): "CREATE_REMINDER_DRINK_WATER",
    ("CREATE_REMINDER", "Exercise"): "CREATE_REMINDER_EXERCISE",
    ("CREATE_REMINDER", "Study"): "CREATE_REMINDER_STUDY",
    ("TEMPERATURE", "18 degrees"): "TEMPERATURE_18",
    ("TEMPERATURE", "22 degrees"): "TEMPERATURE_22",
    ("TEMPERATURE", "26 degrees"): "TEMPERATURE_26",
    ("TIMER", "10 seconds"): "TIMER_10s",
    ("TIMER", "30 seconds"): "TIMER_30s",
    ("TIMER", "1 minute"): "TIMER_1m",
}


def command_label(intent: str, value: str) -> str:
    if value:
        try:
            return SLOT_TO_COMMAND_LABEL[(intent, value)]
        except KeyError as exc:
            raise ValueError(f"Unmapped intent/slot combination: {intent} / {value}") from exc
    return intent


def load_reference() -> tuple[list[str], dict[str, list[dict[str, str]]], dict[str, str]]:
    prompts = {
        path.stem: path.read_text(encoding="utf-8").strip()
        for path in sorted(PHRASE_DIR.glob("*.txt"))
    }
    with VARIATIONS.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 93:
        raise ValueError(f"Expected 93 dataset variation rows, got {len(rows)}")

    intents: list[str] = []
    by_command: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        intent = row["label"].strip()
        if intent not in intents:
            intents.append(intent)
        label = command_label(intent, row["value"].strip())
        by_command[label].append(
            {
                "intent": intent,
                "value": row["value"].strip(),
                "variation": row["variation"].strip(),
                "phrase": row["phrase"].strip(),
            }
        )

    if len(intents) != 19:
        raise ValueError(f"Expected 19 top-level intents, got {len(intents)}")
    if len(by_command) != 31:
        raise ValueError(f"Expected 31 supported command labels, got {len(by_command)}")
    if set(prompts) != set(by_command):
        missing_prompts = sorted(set(by_command) - set(prompts))
        extra_prompts = sorted(set(prompts) - set(by_command))
        raise ValueError(f"Recorder phrase files disagree with labels: missing={missing_prompts}, extra={extra_prompts}")

    for label, variants in by_command.items():
        variants.sort(key=lambda item: int(item["variation"]))
        ids = [item["variation"] for item in variants]
        if ids != ["1", "2", "3"]:
            raise ValueError(f"Expected variation IDs 1, 2, 3 for {label}; got {ids}")
    return intents, dict(by_command), prompts


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def draw_page(canvas, doc) -> None:
    canvas.saveState()
    width, height = landscape(letter)
    canvas.setFillColor(colors.HexColor("#17324D"))
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(40, height - 24, "ME2 - VCM on Raspberry Pi 5")
    canvas.setFillColor(colors.HexColor("#62748A"))
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(width - 40, height - 24, "Intent and phrase reference")
    canvas.setStrokeColor(colors.HexColor("#CFD8E3"))
    canvas.setLineWidth(0.5)
    canvas.line(40, height - 31, width - 40, height - 31)
    canvas.line(40, 31, width - 40, 31)
    canvas.setFillColor(colors.HexColor("#62748A"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(40, 19, f"Dataset revision {DATASET_REVISION[:12]} - source: variations.csv and ground_truth_phrases/")
    canvas.drawRightString(width - 40, 19, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf() -> Path:
    intents, by_command, prompts = load_reference()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    base = getSampleStyleSheet()
    styles = {
        "kicker": ParagraphStyle(
            "Kicker", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8,
            leading=10, textColor=colors.HexColor("#276C92"), spaceAfter=5,
        ),
        "title": ParagraphStyle(
            "TitleME2", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22,
            leading=26, textColor=colors.HexColor("#17324D"), alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleME2", parent=base["Normal"], fontName="Helvetica", fontSize=10,
            leading=14, textColor=colors.HexColor("#4D6074"), spaceAfter=10,
        ),
        "body": ParagraphStyle(
            "BodyME2", parent=base["BodyText"], fontName="Helvetica", fontSize=8.7,
            leading=12, textColor=colors.HexColor("#26384A"), spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallME2", parent=base["BodyText"], fontName="Helvetica", fontSize=7.5,
            leading=10, textColor=colors.HexColor("#4D6074"),
        ),
        "section": ParagraphStyle(
            "SectionME2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13,
            leading=16, textColor=colors.HexColor("#17324D"), spaceBefore=6,
            spaceAfter=7,
        ),
        "table_head": ParagraphStyle(
            "TableHeadME2", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.3,
            leading=8.5, textColor=colors.white, alignment=TA_LEFT,
        ),
        "table": ParagraphStyle(
            "TableME2", parent=base["Normal"], fontName="Helvetica", fontSize=7.2,
            leading=9, textColor=colors.HexColor("#26384A"),
        ),
        "table_bold": ParagraphStyle(
            "TableBoldME2", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.2,
            leading=9, textColor=colors.HexColor("#17324D"),
        ),
        "match": ParagraphStyle(
            "MatchME2", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.2,
            leading=9, textColor=colors.HexColor("#22704A"), alignment=TA_CENTER,
        ),
        "nomatch": ParagraphStyle(
            "NoMatchME2", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.2,
            leading=9, textColor=colors.HexColor("#9A5A1B"), alignment=TA_CENTER,
        ),
        "intent_small": ParagraphStyle(
            "IntentSmallME2", parent=base["Normal"], fontName="Helvetica", fontSize=7,
            leading=8.7, textColor=colors.HexColor("#26384A"),
        ),
    }

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(letter),
        leftMargin=40,
        rightMargin=40,
        topMargin=46,
        bottomMargin=40,
        title="ME2 VCM Intents and Phrase Variations",
        author="ME2 VCM Project",
        subject="All dataset intents, supported command labels, phrase variations, and recorder prompts",
    )

    story = [
        paragraph("REFERENCE GUIDE", styles["kicker"]),
        paragraph("Intents and Phrase Variations", styles["title"]),
        paragraph(
            "The complete command phrase inventory for the ME2 Spoken Command Dataset, aligned to the project's 31-label recorder.",
            styles["subtitle"],
        ),
    ]

    stats = [
        [paragraph("19", styles["title"]), paragraph("31", styles["title"]), paragraph("93", styles["title"]), paragraph("31", styles["title"])],
        [paragraph("top-level intents", styles["small"]), paragraph("supported command labels", styles["small"]), paragraph("dataset phrase variations", styles["small"]), paragraph("recorder ground-truth prompts", styles["small"])],
    ]
    stats_table = Table(stats, colWidths=[172, 172, 172, 172], rowHeights=[31, 18])
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CFD8E3")),
        ("INNERGRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#D8E1EA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([stats_table, Spacer(1, 10)])
    story.append(paragraph(
        f"Dataset source: `airimonda/ai231-me2-voice-commands`, immutable revision {DATASET_REVISION}. "
        "The dataset's `variations.csv` supplies three numbered phrase examples for each of the 31 supported intent/slot labels. "
        "The recorder prompt is listed separately from those variations and is read from the canonical ground-truth phrase file for that label.",
        styles["body"],
    ))

    exact_count = sum(
        prompts[label] in [entry["phrase"] for entry in variants]
        for label, variants in by_command.items()
    )
    story.append(paragraph(
        f"Prompt alignment: {exact_count}/31 recorder prompts are text-identical to one of that label's three dataset variations. "
        "For the others, the recorder prompt remains the project's canonical phrase from `docs/ground_truth_phrases/`; it is shown exactly as recorded. "
        "A suggested prompt is not a recovered transcript for older human recordings.",
        styles["body"],
    ))

    story.append(paragraph("Intent inventory", styles["section"]))
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for label, variants in by_command.items():
        intent = variants[0]["intent"]
        value = variants[0]["value"]
        grouped[intent].append((label, value))
    inventory_data = []
    for intent in intents:
        leaves = grouped[intent]
        descriptions = []
        for label, value in leaves:
            descriptions.append(f"<b>{escape(label)}</b>" + (f" ({escape(value)})" if value else ""))
        inventory_data.append((intent, len(leaves), "; ".join(descriptions)))

    def inventory_table(rows):
        data = [[
            paragraph("Intent", styles["table_head"]),
            paragraph("Labels", styles["table_head"]),
            paragraph("Supported labels and slot values", styles["table_head"]),
        ]]
        data.extend([
            [paragraph(escape(intent), styles["table_bold"]), paragraph(str(count), styles["table"],), paragraph(text, styles["intent_small"])]
            for intent, count, text in rows
        ])
        table = Table(data, colWidths=[88, 42, 224], repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEE8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    split_at = (len(inventory_data) + 1) // 2
    left_table = inventory_table(inventory_data[:split_at])
    right_table = inventory_table(inventory_data[split_at:])
    inventory = Table([[left_table, right_table]], colWidths=[354, 354], hAlign="LEFT")
    inventory.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(inventory)
    story.append(Spacer(1, 8))
    story.append(paragraph(
        "Scope note: the binary wake/not-wake model is separate from these command intents. "
        "The phrase-variation file contains no wake-positive phrase rows, so wake words and background labels are not presented as command intents here.",
        styles["small"],
    ))

    story.extend([PageBreak(), paragraph("Complete phrase reference", styles["section"])])
    story.append(paragraph(
        "Each row is one supported command label. The variations column preserves the dataset's three numbered examples verbatim; "
        "the final columns show the recorder prompt and whether its exact text matches one of those three examples.",
        styles["body"],
    ))

    header = [
        paragraph("Intent", styles["table_head"]),
        paragraph("Command label", styles["table_head"]),
        paragraph("Slot value", styles["table_head"]),
        paragraph("Dataset phrase variations", styles["table_head"]),
        paragraph("Recorder canonical prompt", styles["table_head"]),
        paragraph("Exact variation match", styles["table_head"]),
    ]
    table_data = [header]
    label_order = []
    for intent in intents:
        label_order.extend(label for label, variants in by_command.items() if variants[0]["intent"] == intent)
    for label in label_order:
        variants = by_command[label]
        variations_text = "<br/>".join(
            f"<b>{escape(item['variation'])}.</b> {escape(item['phrase'])}"
            for item in variants
        )
        prompt = prompts[label]
        matches = prompt in [item["phrase"] for item in variants]
        table_data.append([
            paragraph(escape(variants[0]["intent"]), styles["table"]),
            paragraph(escape(label), styles["table_bold"]),
            paragraph(escape(variants[0]["value"] or "Fixed command"), styles["table"]),
            paragraph(variations_text, styles["table"]),
            paragraph(escape(prompt), styles["table_bold"]),
            paragraph("Yes" if matches else "No", styles["match"] if matches else styles["nomatch"]),
        ])

    detail_table = Table(
        table_data,
        colWidths=[87, 146, 72, 212, 130, 65],
        repeatRows=1,
        splitByRow=1,
        hAlign="LEFT",
    )
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEE8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(detail_table)
    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())
