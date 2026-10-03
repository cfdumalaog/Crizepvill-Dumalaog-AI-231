from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "classagreementvcm-20261002-a90b8d10-r3"
RECEIPT = ROOT / "deployment" / "classagreementvcm_candidate_deployment.json"
DEFAULT_OUTPUT = ROOT / "docs" / "evaluations" / "ME2_VCM_Dataset_Comparison_20261002.pdf"
NAVY = colors.HexColor("#0d2738")
TEAL = colors.HexColor("#168c9d")
PALE = colors.HexColor("#e8f4f5")
INK = colors.HexColor("#263746")
MUTED = colors.HexColor("#586c7d")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build(output: Path) -> None:
    comparison = load(RUN / "frozen_test_comparison.json")
    metadata = load(RUN / "candidate_metadata.json")
    training = load(RUN / "training_summary.json")
    receipt = load(RECEIPT)
    quant = load(RUN / "quantization_validation.json")

    font_regular = Path(r"C:\Windows\Fonts\arial.ttf")
    font_bold = Path(r"C:\Windows\Fonts\arialbd.ttf")
    if font_regular.is_file() and font_bold.is_file():
        pdfmetrics.registerFont(TTFont("ME2Arial", str(font_regular)))
        pdfmetrics.registerFont(TTFont("ME2Arial-Bold", str(font_bold)))
        regular, bold = "ME2Arial", "ME2Arial-Bold"
    else:
        regular, bold = "Helvetica", "Helvetica-Bold"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleME2", parent=styles["Title"], fontName=bold,
        fontSize=21, leading=25, alignment=TA_LEFT, textColor=NAVY,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="SubME2", parent=styles["Normal"], fontName=regular,
        fontSize=9, leading=12, textColor=MUTED, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="HeadME2", parent=styles["Heading2"], fontName=bold,
        fontSize=13, leading=16, textColor=NAVY, spaceBefore=9, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="BodyME2", parent=styles["BodyText"], fontName=regular,
        fontSize=8.5, leading=12, textColor=INK, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="SmallME2", parent=styles["BodyText"], fontName=regular,
        fontSize=7.3, leading=9.2, textColor=MUTED, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="CellME2", parent=styles["BodyText"], fontName=regular,
        fontSize=7.4, leading=9, textColor=INK,
    ))
    styles.add(ParagraphStyle(
        name="CellBoldME2", parent=styles["BodyText"], fontName=bold,
        fontSize=7.4, leading=9, textColor=colors.white,
    ))

    def para(text: str, style: str = "BodyME2") -> Paragraph:
        return Paragraph(text, styles[style])

    def table(rows, widths, repeat=1, font=7.3, padding=4):
        obj = Table(rows, colWidths=widths, repeatRows=repeat, hAlign="LEFT")
        obj.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), bold),
            ("FONTNAME", (0, 1), (-1, -1), regular),
            ("FONTSIZE", (0, 0), (-1, -1), font),
            ("LEADING", (0, 0), (-1, -1), font + 1.5),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d7df")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), padding),
            ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ]))
        return obj

    def footer(canvas, doc):
        canvas.saveState()
        width, _ = letter
        canvas.setStrokeColor(colors.HexColor("#d3e0e6"))
        canvas.line(doc.leftMargin, 0.48 * inch, width - doc.rightMargin, 0.48 * inch)
        canvas.setFont(regular, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 0.31 * inch, "ME2 - VCM on Raspberry Pi 5 | Dataset comparison | 2026-10-02")
        canvas.drawRightString(width - doc.rightMargin, 0.31 * inch, f"Page {doc.page}")
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output), pagesize=letter, rightMargin=0.62 * inch,
        leftMargin=0.62 * inch, topMargin=0.55 * inch, bottomMargin=0.65 * inch,
        title="ME2 - VCM on Raspberry Pi 5: Dataset Comparison",
        author="ME2 VCM project",
    )
    story = [
        para("ME2 - VCM on Raspberry Pi 5", "TitleME2"),
        para("Dataset-trained candidate: results, gates and Pi trial", "SubME2"),
        para("<b>Decision:</b> The new intent model is installed as a separate Raspberry Pi trial, but the current model remains the default. On the single frozen test pass, the candidate scored slightly lower overall. It did better on smaller real/varied-source slices and had fewer out-of-scope actions at its own validation-selected gate. The Pi app is running, but live speech cannot be tested until a microphone is attached."),
        para("Why this dataset was not used earlier", "HeadME2"),
        para("The first pass audited the repulled files but did not fit a model because several source licenses and participant/voice permissions were not evidenced. The user then directed a private coursework training and comparison. Training/scoring ran locally; only model/runtime files were sent to the user's Pi, not raw dataset audio. Rights and consent still need confirmation before broader redistribution."),
        para("Training and evaluation", "HeadME2"),
        para("Source: <b>ME2 Spoken Command Dataset</b>, pinned revision a90b8d106349b02c5570a1a258503386043f63b2. Intent model: TinyDSCNN-48, 14,527 parameters, random initialization, no pretrained weights, selected epoch 82 of 100 on RTX 3050. Fit used 8,201 supported training clips and 2,122 source/speaker-group validation clips. The 66,390 numeral clips and 196-row holdout remained untouched."),
        para("The model, gates and scorer were frozen before the single paired evaluation of 4,418 published test clips: 4,371 supported commands and 47 true out-of-scope clips. The 158 unsupported-slot train rows were kept separate from true OOS. No wake-positive examples are present in this dataset, so the candidate retains the existing binary wake model byte-identically."),
        para(f"INT8 intent size: 39,315 bytes; paired wake+intent size: 76,740 bytes (74.9 KiB). Validation FP32/INT8 prediction agreement: {quant['candidate_fp32_int8_prediction_parity']:.2%}."),
        para("Paired test results", "HeadME2"),
    ]

    base = comparison["baseline"]
    cand = comparison["candidate"]
    rows = [[para(x, "CellBoldME2") for x in ["Model", "Accuracy", "Macro F1", "Top-level accuracy", "Top-level macro F1"]]]
    for name, result in (("Current intent model", base), ("New dataset candidate", cand)):
        rows.append([para(name, "CellME2"),
                     para(f"{result['raw_supported']['accuracy']:.2%}", "CellME2"),
                     para(f"{result['raw_supported']['macro_f1']:.2%}", "CellME2"),
                     para(f"{result['top_level_19']['accuracy']:.2%}", "CellME2"),
                     para(f"{result['top_level_19']['macro_f1']:.2%}", "CellME2")])
    story.append(table(rows, [1.55*inch, .86*inch, .86*inch, 1.12*inch, 1.17*inch]))
    story.extend([
        Spacer(1, 5),
        para("Current leads aggregate accuracy and macro F1 by 0.69 and 0.35 percentage points. Candidate-minus-current top-level accuracy is +0.87 points; top-level macro F1 is -0.18 points. The source/speaker bootstrap intervals overlap, so this test does not establish an overall winner."),
        para("Accuracy by test source", "HeadME2"),
    ])
    source_rows = [[para(x, "CellBoldME2") for x in ["Source slice", "n", "Current", "Candidate"]]]
    for name, result in comparison["source_slices"].items():
        source_rows.append([para(name.replace("_", " "), "CellME2"),
                            para(str(result["support"]), "CellME2"),
                            para(f"{result['baseline_accuracy']:.2%}", "CellME2"),
                            para(f"{result['candidate_accuracy']:.2%}", "CellME2")])
    story.append(table(source_rows, [2.8*inch, .55*inch, 1.0*inch, 1.0*inch]))
    story.append(para("Candidate performance is higher in several small real/varied-source slices and lower on group-synthetic speech (the largest slice). Source mix is imbalanced; use these values to plan live evaluation, not to claim a general real-world win.", "SmallME2"))

    story.extend([PageBreak(), para("Action gates and per-class results", "HeadME2")])
    story.append(para("Confidence and margin are accept/reject operating gates. They are not accuracy thresholds."))
    gates = [[para(x, "CellBoldME2") for x in ["Model / gate", "Confidence", "Margin", "Correct-action coverage", "OOS false actions"]]]
    gate_specs = [
        ("Current deployed", .68, .15, .7730, "22 / 47"),
        ("Current, validation-calibrated", .98, 0.0, .6541, "2 / 47"),
        ("Candidate, validation-calibrated", .76, 0.0, .6136, "0 / 47"),
    ]
    for name, conf, margin, coverage, false_actions in gate_specs:
        gates.append([para(name, "CellME2"), para(f"{conf:.2f}", "CellME2"),
                      para(f"{margin:.2f}", "CellME2"), para(f"{coverage:.2%}", "CellME2"),
                      para(false_actions, "CellME2")])
    story.append(table(gates, [2.05*inch, .78*inch, .7*inch, 1.2*inch, 1.15*inch]))
    story.append(para("Candidate accepted-command accuracy: 98.82% at 61.36% correct-action coverage. It rejects more supported speech than the deployed current gate. The validation policy allowed at most 1% false actions on reject examples; only 27 such examples were available for gate selection."))
    wake = comparison["wake"]
    story.append(para(f"Wake remains at threshold {wake['threshold']:.2f}. On {wake['negative_support']:,} wake-negative test clips, there was {wake['false_accepts']} false accept ({wake['false_accept_rate']:.2%}). There are no wake positives, so wake recall cannot be measured here."))
    story.append(para("Per-class F1 (141 examples per label)", "HeadME2"))
    class_rows = [[para(x, "CellBoldME2") for x in ["Label", "Current F1", "Candidate F1", "Change (pp)"]]]
    for label, result in comparison["per_class"].items():
        label_para = para(label, "CellME2")
        class_rows.append([label_para,
                           para(f"{result['baseline_f1']:.2%}", "CellME2"),
                           para(f"{result['candidate_f1']:.2%}", "CellME2"),
                           para(f"{result['baseline_candidate_f1_delta_pp']:+.2f}", "CellME2")])
    story.append(table(class_rows, [3.55*inch, 1.0*inch, 1.05*inch, 1.0*inch], font=6.9, padding=1.8))
    story.append(para("Only 2 of 31 candidate labels reach 95% F1. The per-class 95% target is unmet; the weakest candidate classes are PLAY_MUSIC (48.43%), LIGHT_OFF (49.80%), VOLUME_UP (54.62%), and COLOR_BLUE (66.41%).", "SmallME2"))

    story.extend([PageBreak(), para("Raspberry Pi trial", "HeadME2")])
    story.append(para(f"Installed and manually running at {receipt['install_path']} on {receipt['pi_hostname']}, bound to loopback port 7865. Assistant: {receipt['local_loopback_url']}. Autostart is off and GPIO is disabled, so the trial does not change the existing LED or default VCM state."))
    checks = receipt["pi_verification"]
    deployment_rows = [[para(x, "CellBoldME2") for x in ["Check", "Result"]]]
    for name, value in [
        ("Bundle integrity", f"{checks['verified_payload_files'] if 'verified_payload_files' in checks else receipt['package']['verified_payload_files']} payload hashes passed"),
        ("ARM64 ONNX CPU inference", "passed"),
        ("Wake frontend + model p95", f"{checks['wake_frontend_plus_model_p95_ms']:.3f} ms"),
        ("Intent frontend + model p95", f"{checks['intent_frontend_plus_model_p95_ms']:.3f} ms"),
        ("Loaded model hashes", "match frozen candidate and retained wake"),
        ("API and Assistant page", "HTTP 200"),
        ("Microphone", checks["microphone"]),
    ]:
        deployment_rows.append([para(name, "CellME2"), para(value, "CellME2")])
    story.append(table(deployment_rows, [2.45*inch, 4.1*inch]))
    story.append(Spacer(1, 9))
    story.append(para("The Pi app and CPU models are running, but the Pi reports no microphone and zero active capture. Do not count this as a live speech test. The Windows SSH tunnel currently makes the candidate UI available at 127.0.0.1:7865."))
    story.append(para("Next step", "HeadME2"))
    story.append(para("Connect a microphone, then compare both model versions with the same speaker, microphone placement, exact prompts, room noise, and randomized order. Record wake hits/false wakes per hour, intent actions/rejects, and lights-off/color mistakes. Keep the current release as default unless the personal trial shows a useful gain without worse OOS rejection. Do not tune against the already-scored frozen test."))
    story.append(para("Evidence: frozen_test_comparison.json, candidate training/quantization outputs in runs/classagreementvcm-20261002-a90b8d10-r3, and deployment/classagreementvcm_candidate_deployment.json. The full Markdown report includes interpretation and the complete source-slice table.", "SmallME2"))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
