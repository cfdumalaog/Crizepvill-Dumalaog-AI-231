"""Build an evidence-based dataset-parity and wake-speaker audit PDF."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import pyarrow.parquet as pq
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "data" / "ai231-me2-voice-commands-hf-a90b8d10"
RUN = PROJECT / "runs" / "me2-vcm-20261001-wake-refresh2"
PHRASE_DIR = PROJECT / "docs" / "ground_truth_phrases"
OUTPUT = PROJECT / "output" / "pdf" / "ME2_VCM_Dataset_Parity_and_Wake_Speaker_Audit_20261002.pdf"
CSV_OUTPUT = PROJECT / "docs" / "evaluations" / "dataset_phrase_parity_audit_20261002.csv"
JSON_OUTPUT = PROJECT / "docs" / "evaluations" / "dataset_parity_wake_audit_20261002.json"
PINNED_REVISION = "a90b8d106349b02c5570a1a258503386043f63b2"
LATEST_HUB_REVISION = "6947f13073e57eb6ae67e7e2fc3680700b82aa13"

sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT))
from generate_intent_phrase_reference import command_label  # noqa: E402
from tinyvcm_model.config import LABELS  # noqa: E402


NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#275D9F")
PALE = colors.HexColor("#EFF4F8")
LINE = colors.HexColor("#D6E0E8")
INK = colors.HexColor("#233342")
MUTED = colors.HexColor("#586B7C")
GREEN = colors.HexColor("#26724C")
AMBER = colors.HexColor("#9A5D12")
RED = colors.HexColor("#A43B36")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_phrase_audit() -> tuple[list[dict[str, object]], dict[str, object]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    train_rows = 0
    supported_rows = 0
    parquet_files = sorted((DATASET / "data").glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No train Parquet shards found in {DATASET / 'data'}")
    for shard in parquet_files:
        table = pq.read_table(
            shard,
            columns=["transcript", "command", "slot_value", "out_of_scope"],
        )
        for row in table.to_pylist():
            train_rows += 1
            if row["out_of_scope"]:
                continue
            supported_rows += 1
            label = command_label(row["command"], row["slot_value"] or "")
            counts[label][row["transcript"]] += 1

    with (DATASET / "variations.csv").open("r", encoding="utf-8-sig", newline="") as f:
        variation_rows = list(csv.DictReader(f))
    variations: dict[str, set[str]] = defaultdict(set)
    for row in variation_rows:
        variations[command_label(row["label"].strip(), row["value"].strip())].add(
            row["phrase"].strip()
        )

    result: list[dict[str, object]] = []
    for label in LABELS:
        prompt_path = PHRASE_DIR / f"{label}.txt"
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        class_counts = counts[label]
        if not class_counts:
            raise ValueError(f"No supported train transcripts found for {label}")
        prompt_count = class_counts[prompt]
        modal_count = max(class_counts.values())
        result.append(
            {
                "label": label,
                "prompt": prompt,
                "prompt_count": prompt_count,
                "modal_count": modal_count,
                "is_modal": prompt_count == modal_count,
                "exact_variation": prompt in variations[label],
                "train_support": sum(class_counts.values()),
                "transcript_count": len(class_counts),
            }
        )

    if len(result) != 31 or len(variation_rows) != 93:
        raise ValueError("Expected 31 labels and 93 phrase variations")
    metadata = {
        "train_rows": train_rows,
        "supported_train_rows": supported_rows,
        "labels": len(result),
        "variation_rows": len(variation_rows),
        "prompts_present_as_exact_transcript": sum(int(r["prompt_count"] > 0) for r in result),
        "prompts_modal_or_tied": sum(int(r["is_modal"]) for r in result),
        "prompts_exactly_match_variation": sum(int(r["exact_variation"]) for r in result),
        "pinned_revision": PINNED_REVISION,
        "latest_hub_revision_checked": LATEST_HUB_REVISION,
        "variations_sha256": _hash(DATASET / "variations.csv"),
    }
    return result, metadata


def collect_wake_audit() -> dict[str, object]:
    split_path = RUN / "personal_split_manifest.csv"
    eval_path = RUN / "wake_threshold_095_evaluation.json"
    with split_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    wake_rows = [r for r in rows if r["label"] == "wake_word"]
    by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    for row in wake_rows:
        by_speaker[row["speaker"]][row["split"]] += 1
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    test_hits_by_speaker = {"person-01": "8/8", "person-02": "2/2", "person-04": "1/1"}
    report_counts = {
        "test_hits_by_speaker": test_hits_by_speaker,
        "cutoff": evaluation["threshold"],
        "heldout_wake_hits": evaluation["personal_heldout"]["wake_hits"],
        "heldout_wake_support": evaluation["personal_heldout"]["wake_support"],
        "personal_nonwake_false_accepts": evaluation["personal_heldout"]["nonwake_false_accepts"],
        "personal_nonwake_support": evaluation["personal_heldout"]["nonwake_support"],
        "reference_false_accepts": evaluation["reference_command_test"]["false_accepts"],
        "reference_support": evaluation["reference_command_test"]["support"],
        "active_release_sha256": "6c2e941580d8ea778ed664c929e893c226efb91e3e787e65837908c81153deac",
        "alternate_seed_sha256": evaluation["model_sha256"],
        "manifest_sha256": evaluation["human_manifest_sha256"],
        "speaker_rows": [
            {
                "speaker": speaker,
                "train": by_speaker[speaker]["train"],
                "validation": by_speaker[speaker]["validation"],
                "test": by_speaker[speaker]["test"],
                "test_hits": test_hits_by_speaker.get(speaker, "-"),
                "seen_in_train": by_speaker[speaker]["train"] > 0,
            }
            for speaker in sorted(by_speaker)
        ],
    }
    return report_counts


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("AuditTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=23, leading=27, textColor=NAVY, spaceAfter=5),
        "subtitle": ParagraphStyle("AuditSubtitle", parent=base["Normal"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=10),
        "h1": ParagraphStyle("AuditH1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=BLUE, spaceBefore=5, spaceAfter=7),
        "h2": ParagraphStyle("AuditH2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=NAVY, spaceBefore=5, spaceAfter=4),
        "body": ParagraphStyle("AuditBody", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=INK, spaceAfter=5),
        "small": ParagraphStyle("AuditSmall", parent=base["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9.2, textColor=MUTED, spaceAfter=4),
        "table": ParagraphStyle("AuditTable", parent=base["BodyText"], fontName="Helvetica", fontSize=6.7, leading=8.4, textColor=INK),
        "tablebold": ParagraphStyle("AuditTableBold", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=6.7, leading=8.4, textColor=NAVY),
        "thead": ParagraphStyle("AuditTableHead", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.1, leading=8.6, textColor=colors.white),
        "callout": ParagraphStyle("AuditCallout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=9.3, leading=12, textColor=NAVY, spaceAfter=2),
    }


def _table(rows: list[list[object]], widths: list[float], header: bool = True, font_size: float = 7) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
        for i, cell in enumerate(rows[0]):
            if not isinstance(cell, Paragraph):
                rows[0][i] = Paragraph(escape(str(cell)), _styles()["thead"])
        commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]))
    table.setStyle(TableStyle(commands))
    return table


def _page(canvas, doc) -> None:
    width, height = landscape(letter)
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(34, height - 22, "ME2 - VCM on Raspberry Pi 5")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.4)
    canvas.drawRightString(width - 34, height - 22, "Dataset parity and wake-speaker audit | 2026-10-02")
    canvas.setStrokeColor(LINE)
    canvas.line(34, height - 28, width - 34, height - 28)
    canvas.line(34, 27, width - 34, 27)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(34, 16, f"Private coursework audit | local model is not trained on the latest Hub revision")
    canvas.drawRightString(width - 34, 16, f"Page {doc.page}")
    canvas.restoreState()


def build() -> dict[str, object]:
    phrase_rows, phrase_meta = collect_phrase_audit()
    wake = collect_wake_audit()
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps({"phrase_audit": phrase_meta, "wake_audit": wake}, indent=2), encoding="utf-8")
    with CSV_OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(phrase_rows[0]))
        writer.writeheader()
        writer.writerows(phrase_rows)

    s = _styles()
    story: list[object] = []
    story += [
        p("DATASET AND MODEL AUDIT", s["small"]),
        p("Dataset parity, phrase ground truth, and wake-speaker evidence", s["title"]),
        p("Read-only audit of the latest class dataset post, the locally pinned training revision, the recorder’s 31 canonical prompts, and saved wake-call replay. Prepared 2 October 2026 (Asia/Manila).", s["subtitle"]),
        p("Recommendation", s["h1"]),
    ]
    rec = Table([[p("Use one professor-approved dataset revision and identical official partitions for the shared baseline. Keep supplemental synthesis/noise as separately named ablations until the class explicitly agrees to include them.", s["callout"])]], colWidths=[720])
    rec.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF2FA")), ("BOX", (0, 0), (-1, -1), 1, BLUE), ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [rec, Spacer(1, 6), p("What is confirmed", s["h2"])]
    findings = [
        ["Question", "Evidence and interpretation"],
        ["Are the versions identical?", f"No. The local candidate is pinned to {PINNED_REVISION[:12]} (train 10,682 / test 4,418 / holdout 196). The latest class Hub revision checked is {LATEST_HUB_REVISION[:12]} (train 10,733 / test 4,443 / holdout 202). The local model was not trained on the newer revision."],
        ["Did phrase wording change?", f"The 93-row phrase-variation file is byte-identical between the local pin and the latest Hub revision (SHA-256 {phrase_meta['variations_sha256'][:16]}…). So the published canonical variation strings did not change, but current latest-split transcript frequencies have not been recalculated locally. The latest README prose still shows stale prior split counts."],
        ["Do recorder prompts align?", f"In the locally pinned train split, all {phrase_meta['prompts_present_as_exact_transcript']}/31 prompts occur as exact transcript strings; {phrase_meta['prompts_modal_or_tied']}/31 are a most-frequent transcript (ties included); {phrase_meta['prompts_exactly_match_variation']}/31 exactly match one of the three published phrase strings. This is text-level alignment, not proof that old recordings actually spoke the displayed prompt."],
        ["Can the wake model respond to anyone?", "Not established. At the 0.95 cutoff, saved replay detected 11/11 held-out wake clips from only three speaker IDs. All seven speaker IDs also occur in training; this is speaker-overlapping replay, not unseen-person generalization or a live false-activation/hour result."],
    ]
    formatted = [[p(str(c), s["thead"]) for c in findings[0]]]
    for row in findings[1:]:
        formatted.append([p(escape(row[0]), s["tablebold"]), p(escape(row[1]), s["table"])])
    story += [_table(formatted, [135, 585]), Spacer(1, 6), p("Dataset handling decision", s["h2"])]
    story += [p("For a fair class comparison, freeze the professor-approved revision, publish its full commit SHA and file hashes, and reuse the official train/test/holdout definitions for every model. Train only on train; derive validation from train. If the class wants the newer revision, re-fetch it once, validate its manifest and splits, then retrain every compared model from scratch and score the official test once. Do not blend the 5,856 supplemental synthetic examples or the 1,000 synthetic negatives into the baseline silently; report them as explicit augmentation/rejection ablations. The 66,390 numerals config is a separate task and should not be pooled into command-intent scores.", s["body"]), p("Scope: the assignment’s 10 common command groups can be the primary result. Retain extra label detail only as an explicitly declared extension; show primary/core scores separately from the extended label scores so the benchmark stays comparable.", s["body"]), p("The official metadata and README body are inconsistent about split counts. Ask the dataset maintainer/class to ratify the exact revision as the shared initial baseline before treating either count set as final.", s["small"]), PageBreak()]

    story += [p("PROMPT AUDIT | 1 OF 2", s["small"]), p("Recorder prompts against pinned train transcripts", s["h1"]), p("The recorder reads each prompt from docs/ground_truth_phrases. Counts below compare that exact string with supported transcripts in the locally pinned training partition. ‘Modal’ includes tied most-frequent transcripts; below-modal does not mean incorrect intent, but it is a prompt-to-training-frequency mismatch worth resolving before collecting more personal audio.", s["body"])]
    story += [phrase_table(phrase_rows[:16], s), PageBreak()]
    story += [p("PROMPT AUDIT | 2 OF 2", s["small"]), p("Recorder prompts against pinned train transcripts", s["h1"]), p("Each label has the same 93-row phrase ground truth at the checked upstream revision, but the local model’s transcript-frequency evidence is tied to the older pinned training split. Re-run this table against the agreed revision before recording a new benchmark set.", s["body"])]
    story += [phrase_table(phrase_rows[16:], s), Spacer(1, 7), p("Existing human recordings", s["h2"]), p("The manifest’s suggested_phrase field records what the recorder displayed, not a verified transcript. Older recordings without prompt metadata cannot be checked word-for-word after the fact. Preserve their audio and labels; do not silently relabel from the canonical prompt. For future recordings, freeze the prompt-sheet version and record actual spoken transcript only when independently checked.", s["small"]), PageBreak()]

    story += [p("WAKE MODEL | SPEAKER GENERALIZATION", s["small"]), p("Saved replay is encouraging, but it does not prove ‘any speaker’", s["h1"])]
    wake_summary = [
        ["Cutoff", "Wake hits", "Personal non-wake", "Reference command negatives", "What it means"],
        ["0.95", f"{wake['heldout_wake_hits']}/{wake['heldout_wake_support']}", f"{wake['personal_nonwake_false_accepts']}/{wake['personal_nonwake_support']} false accepts", f"{wake['reference_false_accepts']}/{wake['reference_support']} false accepts", "Finite saved-file replay only"],
    ]
    cells = [[p(str(c), s["thead"]) for c in wake_summary[0]], [p(str(c), s["tablebold"] if i < 2 else s["table"]) for i, c in enumerate(wake_summary[1])]]
    story += [_table(cells, [65, 70, 135, 155, 295]), Spacer(1, 6), p("Held-out wake clips by speaker", s["h2"])]
    speaker_rows = [["Speaker ID", "Train wake clips", "Validation wake clips", "Test wake clips", "Test hits", "Train overlap"]]
    for row in wake["speaker_rows"]:
        speaker_rows.append([row["speaker"], row["train"], row["validation"], row["test"], row["test_hits"], "Yes" if row["seen_in_train"] else "No"])
    story += [_table([[p(str(c), s["thead"]) for c in speaker_rows[0]]] + [[p(str(c), s["table"]) for c in row] for row in speaker_rows[1:]], [90, 105, 125, 100, 85, 80]), Spacer(1, 5)]
    story += [p("All seven speaker IDs appear in training. The held-out test support is only 11 clips and includes person-01 (8), person-02 (2), and person-04 (1). The best next experiment is to recruit speakers whose recordings are completely absent from training; freeze one or more entire people for final wake evaluation. Also measure continuous live false wakes per hour in quiet and noisy rooms. Do not interpret a probability cutoff as a guarantee of 95% speaker coverage.", s["body"]), p(f"Active release SHA-256: {escape(str(wake['active_release_sha256']))}. Alternate-seed candidate SHA-256: {escape(str(wake['alternate_seed_sha256']))}. Both replayed the same saved test at 11/11; this is not independent-speaker evidence. Frozen human-manifest SHA: {escape(str(wake['manifest_sha256']))}.", s["small"]), p("Wake design recommendation: keep the wake model binary (WAKE_CALL / NOT_WAKE). Use the diverse intent, OOS, silence, and noise material as negatives, but reserve entire speakers/sessions for final validation. The current class command dataset has no wake-positive examples, so it cannot replace the personal wake-call positives.", s["body"]), PageBreak()]

    story += [p("REVIEWER READINESS AND NEXT DECISIONS", s["small"]), p("What is ready to claim, and what still needs evidence", s["h1"])]
    readiness = [
        ["Checklist item", "Current evidence", "Status"],
        ["Common commands vs labels", "Dataset has 19 top-level intents / 31 command-slot leaves. Map the assignment’s 10 common groups to these labels and report core-group results separately from any extra scope.", "Needs explicit crosswalk"],
        ["Same shared dataset", "Local candidate used older pinned revision; Hub now has a newer revision. Variation CSV unchanged; split sizes differ; README prose is stale.", "Needs class ratification"],
        ["Recorder phrase consistency", "31/31 prompts are exact train transcripts in local pin, 13/31 modal/tied, 22/31 exact variation strings. Historical prompt metadata does not verify what was spoken.", "Partial"],
        ["Unseen wake speakers", "11/11 saved wake clips at 0.95; all 7 speaker IDs overlap train. No independent-person test.", "Not demonstrated"],
        ["Pi timing", "Pi 5 ARM64 frontend+model p95 around 5.5 ms for the isolated candidate. No Pi 4 result; Pi candidate had no capture mic for live A/B.", "Pi 4 pending"],
        ["A100 training provenance", "Available run records identify CUDA but do not verify A100 node/GPU allocation, scheduler job, or wall-clock evidence.", "Do not claim A100"],
        ["Causal/KV-cache architecture", "Current model is a fixed-window TinyDSCNN-48; it does not use a causal encoder or KV cache.", "Poster must be corrected"],
        ["Public repo/data/weights", "No verified public MIT one-command release; dataset rights, participant consent, source terms, and weight redistribution remain unresolved.", "Private coursework only"],
        ["Intent quality target", "Candidate macro F1 is 79.12% on the previously frozen test and only 2/31 class F1 values reach 95%. This does not meet the stated 95% goal.", "Not met"],
    ]
    readiness_cells = [[p(str(c), s["thead"]) for c in readiness[0]]]
    for row in readiness[1:]:
        status_color = RED if row[2] in {"Not met", "Not demonstrated", "Do not claim A100"} else AMBER if row[2] != "Partial" else AMBER
        readiness_cells.append([p(escape(row[0]), s["tablebold"]), p(escape(row[1]), s["table"]), p(f'<font color="{status_color.hexval()}"><b>{escape(row[2])}</b></font>', s["table"])])
    story += [_table(readiness_cells, [125, 500, 95]), Spacer(1, 7), p("Recommended next steps", s["h2"])]
    for item in [
        "Ask the dataset maintainer/class to confirm the canonical dataset revision and whether supplemental synthetic and synthetic-negative configurations are baseline data or optional tracks.",
        "Freeze a ground-truth prompt sheet for that revision; regenerate modal transcripts and exact-variation crosswalk, then version it. Keep prior phrase text files and old recording evidence for provenance.",
        "Create a primary benchmark crosswalk for the agreed common commands; list extra intents as an extension and score them separately.",
        "Rebuild candidate and baseline from the same agreed train split; derive validation only from train; compare on the same untouched test. Do not tune on the previous test results.",
        "Add new unseen people for wake validation and run a live microphone trial before making broad-speaker or always-on claims.",
        "Update the slide template with measured values only: TinyDSCNN-48, model sizes, actual runtime device, Pi 5 results, and clear ‘Pi 4/A100/public release not verified’ notes.",
    ]:
        story.append(p("• " + escape(item), s["body"]))
    story += [Spacer(1, 4), p(f"Sources: <link href='https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands'>ME2 Spoken Command Dataset</link> (latest checked revision {LATEST_HUB_REVISION}); <link href='https://github.com/airimonda/vcm-benchmark'>class VCM benchmark</link>. Chat review was read-only; no messages were sent. This report audits the local pin and publicly exposed metadata; no training, deployment, or data mutation was performed.", s["small"])]

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=landscape(letter), leftMargin=36, rightMargin=36,
        topMargin=38, bottomMargin=36, title="ME2 VCM Dataset Parity and Wake Speaker Audit",
        author="ME2 - VCM on Raspberry Pi 5",
    )
    doc.build(story, onFirstPage=_page, onLaterPages=_page)
    return {"pdf": str(OUTPUT), "csv": str(CSV_OUTPUT), "json": str(JSON_OUTPUT), "phrase_audit": phrase_meta, "wake_audit": wake}


def phrase_table(rows: list[dict[str, object]], styles: dict[str, ParagraphStyle]) -> Table:
    header = ["Label", "Canonical recorder prompt", "Prompt count", "Modal count", "Frequency", "Dataset phrase exact?"]
    data: list[list[object]] = [[p(h, styles["thead"]) for h in header]]
    for row in rows:
        frequency = "Modal/tied" if row["is_modal"] else "Below modal"
        color = GREEN if row["is_modal"] else AMBER
        data.append(
            [
                p(f'<b>{escape(str(row["label"]))}</b>', styles["table"]),
                p(escape(str(row["prompt"])), styles["table"]),
                p(f'{row["prompt_count"]}/{row["train_support"]}', styles["table"]),
                p(str(row["modal_count"]), styles["table"]),
                p(f'<font color="{color.hexval()}"><b>{frequency}</b></font>', styles["table"]),
                p("Yes" if row["exact_variation"] else "No", styles["table"]),
            ]
        )
    return _table(data, [128, 278, 72, 62, 92, 82])


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
