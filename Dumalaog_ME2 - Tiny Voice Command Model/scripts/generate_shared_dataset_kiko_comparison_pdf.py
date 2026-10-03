"""Build a concise, evidence-linked dataset and model comparison PDF."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PROJECT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT / "docs" / "evaluations" / "kiko-pi-comparison-20261002" / "metrics.json"
OUTPUT = PROJECT / "output" / "pdf" / "ME2_VCM_Shared_Dataset_and_Kiko_Comparison_20261002.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#275D9F")
PALE = colors.HexColor("#EFF4F8")
LINE = colors.HexColor("#D6E0E8")
INK = colors.HexColor("#233342")
MUTED = colors.HexColor("#586B7C")
GREEN = colors.HexColor("#26724C")
AMBER = colors.HexColor("#9A5D12")
RED = colors.HexColor("#A43B36")


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("VCMTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=NAVY, alignment=TA_LEFT, spaceAfter=6),
        "subtitle": ParagraphStyle("VCMSubtitle", parent=base["Normal"], fontSize=9, leading=12, textColor=MUTED, spaceAfter=8),
        "h1": ParagraphStyle("VCMH1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=BLUE, spaceBefore=3, spaceAfter=6),
        "h2": ParagraphStyle("VCMH2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=NAVY, spaceBefore=4, spaceAfter=3),
        "body": ParagraphStyle("VCMBody", parent=base["BodyText"], fontName="Helvetica", fontSize=9.1, leading=12, textColor=INK, spaceAfter=5),
        "small": ParagraphStyle("VCMSmall", parent=base["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10, textColor=MUTED, spaceAfter=4),
        "cell": ParagraphStyle("VCMCell", parent=base["BodyText"], fontName="Helvetica", fontSize=7.6, leading=9.2, textColor=INK),
        "cellbold": ParagraphStyle("VCMCellBold", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.6, leading=9.2, textColor=NAVY),
        "thead": ParagraphStyle("VCMHead", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.6, leading=9.2, textColor=colors.white),
        "callout": ParagraphStyle("VCMCallout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=9.1, leading=11.5, textColor=NAVY, spaceAfter=2),
        "tiny": ParagraphStyle("VCMTiny", parent=base["BodyText"], fontName="Helvetica", fontSize=7.1, leading=8.6, textColor=INK),
    }


def para(value: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(value), style)


def table(rows: list[list[object]], widths: list[float], *, header: bool = True, size: float = 7) -> Table:
    s = styles()
    converted: list[list[object]] = []
    for ridx, row in enumerate(rows):
        out = []
        for cell in row:
            if isinstance(cell, Paragraph):
                out.append(cell)
            else:
                st = s["thead"] if header and ridx == 0 else s["cell"]
                out.append(para(escape(str(cell)), st))
        converted.append(out)
    tbl = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.3, LINE),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE])]
    tbl.setStyle(TableStyle(cmds))
    return tbl


def page(canvas, doc) -> None:
    width, height = landscape(letter)
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(34, height - 22, "ME2 - VCM on Raspberry Pi 5")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.2)
    canvas.drawRightString(width - 34, height - 22, "Shared dataset and model comparison | 2026-10-02")
    canvas.setStrokeColor(LINE)
    canvas.line(34, height - 28, width - 34, height - 28)
    canvas.line(34, 27, width - 34, 27)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(34, 16, "Private coursework analysis | class baseline requires revision ratification")
    canvas.drawRightString(width - 34, 16, f"Page {doc.page}")
    canvas.restoreState()


def add_label_table(rows: list[list[object]], s: dict[str, ParagraphStyle]) -> Table:
    data: list[list[object]] = [[para(h, s["thead"]) for h in rows[0]]]
    for row in rows[1:]:
        data.append([para(escape(str(v)), s["tiny"]) for v in row])
    return table(data, [136, 66, 66, 66])


def build() -> Path:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    s = styles()
    story: list[object] = []

    # Page 1: what the dataset contains and how to keep the class comparison fair.
    story += [
        para("ME2 - VCM on Raspberry Pi 5", s["small"]),
        para("Shared dataset, phrase ground truth, and fair comparison", s["title"]),
        para("Prepared 2 October 2026 (Asia/Manila) from the opened Hugging Face dataset card/viewer, the class benchmark repository, the local pinned training snapshot, and the direct Kiko ONNX Pi diagnostic.", s["subtitle"]),
        para("Recommendation", s["h1"]),
        para("Start with the exact professor-approved shared dataset revision as the baseline for every student model. Discuss and document any filtering or added data, then report it as a separate experiment. Keep the class holdout untouched and run the same Pi benchmark procedure for every model.", s["callout"]),
        Spacer(1, 5),
        para("What is in the dataset currently shown in the browser", s["h1"]),
    ]
    snapshot = [
        ["Configuration / snapshot", "Contents shown", "How to use it"],
        ["Default command dataset, latest viewer", "Train 10,733 · test 4,443 · holdout 202; separate numerals config 66,390", "Current card/viewer reports a newer split count than the checked local pin. Ask the class to ratify exact revision and split before baseline training."],
        ["Local pinned copy used by our models", "Train 10,682 · test 4,418 · holdout 196; SHA a90b8d106349b02c5570a1a258503386043f63b2", "This is the revision behind current local models and the Kiko diagnostic. It is not the newest viewer snapshot."],
        ["Main default audio", "Mix of human/group and open-source recordings plus generated speech; the card describes synthetic fill and human recordings.", "This mixed source is the fairest initial baseline if the class confirms it. Report real vs synthetic source slices."],
        ["supplemental_synth", "5,856 generated clips; voice_split: train 3,983 · test 1,612 · holdout 261", "Synthetic, not real voices. Optional augmentation: add only its train partition to training; retain test/holdout partitions for their matching evaluation splits."],
        ["synthetic_negatives", "1,000 train + 250 test generated out-of-scope clips", "Keep it as an explicit rejection-data track; do not silently mix evaluation clips into training."],
    ]
    story += [table(snapshot, [145, 250, 270]), Spacer(1, 5)]
    story += [
        para("The displayed README prose still gives the older 10,682 / 4,418 / 196 counts while the current viewer metadata gives 10,733 / 4,443 / 202. The variation phrase file was unchanged between the checked local revision and the latest revision, but the audio split changed. This mismatch needs maintainer/class confirmation before claiming everyone used the same release.", s["small"]),
        para("Phrase/label match", s["h2"]),
        para("Our 31 recorder prompts map to the model’s 31 command/value outputs. In the pinned train split, all 31 prompt strings occur as exact transcripts; 13 are a most-frequent or tied transcript, and 22 exactly match a phrase in the 93-row variations file. So label alignment is present, but not every prompt is the dominant wording. Keep the exact revision’s variations file as shared ground truth and report its hash.", s["body"]),
        para("The base dataset covers 19 top-level intents and 93 command variations. Preserve those for primary class scoring. Extra intents can remain as a declared extension, but should not replace or be mixed into the shared 19/93 results.", s["body"]),
        PageBreak(),
    ]

    # Page 2: common protocol and aggregate Pi comparison.
    story += [
        para("FAIR VALIDATION AND MODEL COMPARISON", s["small"]),
        para("How to compare without changing the class benchmark", s["h1"]),
    ]
    steps = [
        "Ask the dataset maintainer/class to name the canonical revision, exact train/test/holdout files, and benchmark-script commit. Do not combine the older and newer split counts.",
        "Train every baseline only on the agreed training partition. Derive validation from training data; use it for early stopping, threshold selection, and any model choice. Never tune on the class test or holdout.",
        "Score every model on the same locked class holdout using the same benchmark repository and Pi procedure: same mic, placement, distance, gain, wake enrollment, randomized commands, out-of-scope clips, and response logging.",
        "For an added-data experiment, retain the agreed base training data and add only supplemental_synth rows marked train. Keep its test and holdout partitions out of training. Label the experiment and report source-specific results.",
        "For a real-only experiment, filter synthetic training rows only after the class agrees this is a declared data change. Score it on the same untouched shared holdout and include real-vs-synthetic slices.",
        "The published test in our pinned snapshot has already been used for model comparison and the Kiko diagnostic. Final claims need a new locked holdout or an unreleased common set; do not tune or select models from reused test results.",
    ]
    for i, item in enumerate(steps, 1):
        story.append(para(f"<b>{i}.</b> {escape(item)}", s["body"]))
    story += [para("Installed Kiko and ME2 models: diagnostic operating points", s["h1"])]
    ours = metrics["ours_saved_comparison"]
    base = ours["baseline"]
    cand = ours["candidate"]
    kiko = metrics["kiko"]
    base_top = base["top_level_19"]["accuracy"]
    cand_top = cand["top_level_19"]["accuracy"]
    base_raw = base["raw_supported"]
    cand_raw = cand["raw_supported"]
    base_gate = base["validation_calibrated_gate"]
    cand_gate = cand["validation_calibrated_gate"]
    kiko_gate = kiko["confidence_gate"]
    comparison = [
        ["Model / gate", "Top-level intent accuracy, argmax", "Raw leaf accuracy / macro F1", "Correct-action coverage under gate", "Accepted supported coverage / accuracy", "OOS false actions", "ONNX bytes"],
        [f"ME2 active baseline / val gate {base_gate['confidence']:.2f}", f"{base_top:.2%}", f"{base_raw['accuracy']:.2%} / {base_raw['macro_f1']:.2%}", f"{base_gate['results']['correct_action_coverage']:.2%}", f"{(base_gate['results']['supported_support'] - base_gate['results']['supported_rejects']) / base_gate['results']['supported_support']:.2%} / {base_gate['results']['accepted_supported_accuracy']:.2%}", f"{base_gate['results']['noncommand_false_actions']}/47", "39,315"],
        [f"ME2 dataset candidate / val gate {cand_gate['confidence']:.2f}", f"{cand_top:.2%}", f"{cand_raw['accuracy']:.2%} / {cand_raw['macro_f1']:.2%}", f"{cand_gate['results']['correct_action_coverage']:.2%}", f"{(cand_gate['results']['supported_support'] - cand_gate['results']['supported_rejects']) / cand_gate['results']['supported_support']:.2%} / {cand_gate['results']['accepted_supported_accuracy']:.2%}", f"{cand_gate['results']['noncommand_false_actions']}/47", "39,315"],
        [f"Kiko ONNX on Pi / fixed gate {kiko_gate:.2f}", f"{kiko['top_level']['accuracy_raw']:.2%}", "Not retained", f"{kiko['correct_action_coverage']:.2%}", f"{kiko['accepted_supported_coverage']:.2%} / {kiko['accuracy_among_accepted_actions']:.2%}", f"{kiko['oos_false_actions']}/47", "83,163"],
    ]
    story += [table(comparison, [128, 84, 93, 93, 117, 66, 55]), Spacer(1, 5)]
    story += [
        para("Reading the result", s["h2"]),
        para("Kiko responded to more supported test clips, but its accepted actions were much less reliable and it fired on 43/47 out-of-scope clips. Our saved validation gates traded coverage for precision: the active baseline (0.98 gate) correctly acted on 65.41% of supported clips, was correct on 98.59% of accepted clips, and fired on 2/47 out-of-scope clips; the candidate (0.76 gate) correctly acted on 61.36%, was correct on 98.82% of accepted clips, and fired on 0/47. Kiko’s correct-action coverage was 73.14%, accepted accuracy 76.94%, and false-action count 43/47. These are current engineering diagnostics, not final ranking evidence.", s["body"]),
        para("Kiko’s saved raw logits are unavailable, so an ungated exact-leaf accuracy cannot be reconstructed. Its per-class F1 below includes the fixed 0.30 gate; ME2 per-class F1 is raw argmax. Do not treat those columns as an apples-to-apples F1 ranking. Both the dataset test and Kiko’s model-training overlap are uncertain/reused, further limiting a winner claim.", s["small"]),
        para("Pi evidence: Kiko ONNX ran on the Raspberry Pi 5 CPU (model-only p95 1.858 ms, max 3.040 ms). Its 64-band feature generation ran on Windows because the Pi environment lacks librosa; this is not end-to-end Pi latency. Kiko uses a separate pretrained Hey Jarvis wake detector, so wake behavior was not compared to our from-scratch custom wake model.", s["small"]),
        PageBreak(),
    ]

    # Page 3: every leaf label with careful score interpretation.
    story += [
        para("PER-LABEL DIAGNOSTIC", s["small"]),
        para("F1 by command/value label", s["h1"]),
        para("The common leaf taxonomy contains 31 labels. Kiko uses intent and slot heads composed into leaves; its F1 is measured after the fixed 0.30 intent gate. The ME2 columns are raw argmax. Different gating means these figures are useful for finding weak classes, but not for declaring which system wins.", s["body"]),
    ]
    labels = sorted(kiko["per_leaf"])
    per_class = [["Label", "Kiko F1\n@ 0.30 gate", "ME2 baseline\nraw F1", "ME2 candidate\nraw F1"]]
    for label in labels:
        k = kiko["per_leaf"][label]["f1-score"]
        v = ours["per_class_delta"][label]
        per_class.append([label, f"{k:.1%}", f"{v['baseline_f1']:.1%}", f"{v['candidate_f1']:.1%}"])
    left_rows = per_class[:17]
    right_rows = [per_class[0]] + per_class[17:]
    two_col = Table(
        [[add_label_table(left_rows, s), add_label_table(right_rows, s)]],
        colWidths=[360, 360],
        hAlign="LEFT",
    )
    two_col.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 1), ("RIGHTPADDING", (0, 0), (-1, -1), 1)]))
    story += [two_col, Spacer(1, 5)]
    story += [
        para("Most urgent Kiko-side diagnostic classes", s["h2"]),
        para("Among the Kiko gated scores, the weakest leaves include PLAY_MUSIC, LIGHT_OFF, STOP, WEATHER, PAUSE, VOLUME_UP, LIGHT_ON, CALL, VOLUME_DOWN, TIME, and COLOR_RED. Our own weakest classes also include several of these, so the most productive next step is to inspect the benchmark’s expected→predicted confusions and the exact phrases/audio for those labels—not merely collect more recordings without reviewing errors.", s["body"]),
        para("Sources and artifacts", s["h2"]),
        para("Dataset card: <link href='https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands'>airimonda/ai231-me2-voice-commands</link> · class SOP: <link href='https://github.com/airimonda/vcm-benchmark'>airimonda/vcm-benchmark</link> · detailed report and per-class CSV: docs/evaluations/kiko-pi-comparison-20261002/. The Kiko folder itself was read-only; no deployment or GPIO action was performed.", s["small"]),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=landscape(letter), leftMargin=35, rightMargin=35,
        topMargin=38, bottomMargin=35,
        title="ME2 VCM Shared Dataset and Kiko Comparison",
        author="ME2 - VCM on Raspberry Pi 5",
    )
    doc.build(story, onFirstPage=page, onLaterPages=page)
    return OUTPUT


if __name__ == "__main__":
    print(build())
