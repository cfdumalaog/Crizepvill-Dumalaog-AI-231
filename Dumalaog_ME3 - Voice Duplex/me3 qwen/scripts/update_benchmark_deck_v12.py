"""Create the ME3 Qwen evidence deck with the complete runtime pipeline.

The v11 deck is preserved.  This update reuses its slide geometry, typography,
colors, and footer, then replaces only evidence text so the class submission
keeps the established visual language while making the WER and physical-audio
boundaries explicit.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pptx import Presentation
from pptx.util import Pt


PROJECT = Path(__file__).resolve().parents[1]
PRESENTATION = PROJECT / "docs" / "presentation"
SOURCE = PRESENTATION / "ME3_Voice_Duplex_Presentation_v11_Benchmark_20261119.pptx"
TARGET = PRESENTATION / "ME3_Voice_Duplex_Presentation_v12_Pipeline_WER_20261120.pptx"


def replace_text(shape, value: str, *, size: float | None = None) -> None:
    frame = shape.text_frame
    paragraphs = frame.paragraphs
    if not paragraphs:
        frame.text = value
    else:
        runs = paragraphs[0].runs
        if runs:
            runs[0].text = value
            for run in runs[1:]:
                run.text = ""
        else:
            paragraphs[0].add_run().text = value
        for paragraph in list(paragraphs[1:]):
            paragraph._element.getparent().remove(paragraph._element)
    if size is not None and frame.paragraphs[0].runs:
        frame.paragraphs[0].runs[0].font.size = Pt(size)


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    shutil.copy2(SOURCE, TARGET)
    deck = Presentation(TARGET)

    # Slide 2: the experience remains a single Codex-style conversation while
    # voice is active and tools run in the background.
    slide = deck.slides[1]
    replace_text(slide.shapes[23], "Voice stays active during playback; typed chat shares the same turn history.")
    replace_text(slide.shapes[24], "Setup, approvals, tool progress, and verified artifacts appear in that conversation.")

    # Slide 3: compress the complete Pi-to-speaker data path into the existing
    # four architecture cards without changing their geometry.
    slide = deck.slides[2]
    replace_text(slide.shapes[1], "FULL PIPELINE", size=27)
    replace_text(slide.shapes[2], "Capture and turn-taking stay on the Pi; model inference may use DGX3/API providers; actions remain owned by ME3.")
    replace_text(slide.shapes[8], "PI CAPTURE")
    replace_text(slide.shapes[9], "Fifine mic\nPCM 16 kHz → VAD\n→ echo guard", size=10.5)
    replace_text(slide.shapes[10], "input + endpointing")
    replace_text(slide.shapes[13], "INFERENCE")
    replace_text(slide.shapes[14], "Qwen2-Audio ASR\n→ Qwen3.6 ToolBrain", size=10.5)
    replace_text(slide.shapes[15], "DGX3/API or fallback")
    replace_text(slide.shapes[18], "MCP ACTIONS")
    replace_text(slide.shapes[19], "files • web • jobs\ncalendar • Telegram", size=10.5)
    replace_text(slide.shapes[20], "bounded policy")
    replace_text(slide.shapes[23], "PI PLAYBACK")
    replace_text(slide.shapes[24], "Qwen3-TTS Aiden\n→ USB speaker\n→ echo reference", size=10.5)
    replace_text(slide.shapes[25], "browser optional")
    replace_text(slide.shapes[27], "One always-listening loop: audio in → transcript → tool reasoning → verified result → audio out → echo-aware listening.")

    # Slide 6: make WER a first-class corpus metric and distinguish the
    # injected class benchmark from the physical acoustic gate.
    slide = deck.slides[5]
    replace_text(slide.shapes[2], "The class protocol is direct-feed/injection-only; the Pi acoustic test is reported separately so WER is not confused with echo evidence.")
    replace_text(slide.shapes[5], "•  40/40 planned trials retained: 10 phrases × 3 stage runs + 1 full round")
    replace_text(slide.shapes[6], "•  Direct feed uses one benchmark clock; no physical mic/speaker, tools, or fillers")
    replace_text(slide.shapes[7], "•  E2E 4.30 s mean / 7.19 s p95; target 4.0 s")
    replace_text(slide.shapes[8], "•  ASR final 2.68 s / 5.32 s p95; target 0.8 s")
    replace_text(slide.shapes[11], "WER + ACOUSTIC GATE")
    replace_text(slide.shapes[12], "•  Corpus WER 17.6% (n=29); target 15%. WER is aggregate word edits, not p95.")
    replace_text(slide.shapes[13], "•  TTS TTFB 0.63 s / 0.62 s p95; LLM TTFT n=0; barge-in n=0")
    replace_text(slide.shapes[14], "•  Pi defaults: Fifine + USB Audio; playback test: 4,429 canceled / 375 suspect frames; corr 0.807")
    replace_text(slide.shapes[15], "DGX3 timed out during the run; score is fallback/stub diagnostic. Physical human overlap is still open.", size=10.5)

    # Slide 7: current demo voice and language behavior.
    slide = deck.slides[6]
    replace_text(slide.shapes[5], "English by default\nFilipino + Taglish understood")
    replace_text(slide.shapes[6], "Qwen3-TTS Aiden is the single active demo voice; Piper is the local fallback.")

    # Slide 8: make the autonomous next gates actionable.
    slide = deck.slides[7]
    replace_text(slide.shapes[3], "ME3 combines full-duplex audio, a Codex-style conversation, provider-selectable layers, MCP actions, and evidence-bound evaluation on Raspberry Pi.")
    replace_text(slide.shapes[6], "Next: restore DGX3 → rerun clean Qwen ASR/TTS → repeat human overlap and barge-in → reduce WER")
    replace_text(slide.shapes[7], "Documentation: ELI5/deep manual • ACM two-column paper • source + benchmark receipts")

    deck.save(TARGET)
    print(TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
