"""Create the next ME3 benchmark deck from the preserved v10 template.

Only evidence copy changes. The source deck remains untouched so the previous
presentation is recoverable. This fallback uses python-pptx because the local
presentation session is unavailable in the current desktop runtime.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pptx import Presentation
from pptx.util import Pt


PROJECT = Path(__file__).resolve().parents[1]
PRESENTATION = PROJECT / "docs" / "presentation"
SOURCE = PRESENTATION / "ME3_Voice_Duplex_Presentation_v10_Benchmark.pptx"
TARGET = PRESENTATION / "ME3_Voice_Duplex_Presentation_v11_Benchmark_20261119.pptx"


def replace_text(shape, value: str) -> None:
    """Replace text while retaining the first run's existing visual style."""
    frame = shape.text_frame
    paragraphs = frame.paragraphs
    if not paragraphs:
        frame.text = value
        return
    runs = paragraphs[0].runs
    if runs:
        runs[0].text = value
        for run in runs[1:]:
            run.text = ""
    else:
        paragraphs[0].add_run().text = value
    # Targeted source shapes are single paragraphs with explicit line breaks.
    # Remove stale extra paragraphs without changing shape geometry.
    for paragraph in list(paragraphs[1:]):
        paragraph._element.getparent().remove(paragraph._element)


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    shutil.copy2(SOURCE, TARGET)
    deck = Presentation(TARGET)

    # Slide 2: the conversation is voice-first but remains usable while a job
    # runs, matching the new Codex-style center chat surface.
    slide = deck.slides[1]
    replace_text(slide.shapes[23], "Voice stays active during playback; chat remains available while tools run.")
    replace_text(slide.shapes[24], "Typed approvals and follow-ups share the same live conversation history.")

    # Slide 5: reflect the now-selectable model layers and artifact-oriented
    # software-development workflow.
    slide = deck.slides[4]
    replace_text(slide.shapes[5], "Qwen or OpenAI chat + multi-step tool loop")
    replace_text(slide.shapes[21], "Every action returns bounded status, evidence, or a verified artifact.")

    # Slide 6: exact values from fdbench run 2026-11-19_02-24-57. The run is
    # intentionally labeled as fallback/stub diagnostic because DGX3 refused
    # connections; no Qwen-provider claim is made.
    slide = deck.slides[5]
    replace_text(slide.shapes[2], "The 17-minute Pi direct-feed run retained all 40 planned trials and aggregated repeated runs.")
    replace_text(slide.shapes[5], "•  Pi direct-feed connected; laptop↔Pi RTT 5 ms; fdbench used no local mic/speaker")
    replace_text(slide.shapes[6], "•  10 phrases × 3 stage runs + 1 full-duplex round = 40 trials; all retained")
    replace_text(slide.shapes[7], "•  E2E 4.30 s mean / 7.19 s p95; target 4.0 s")
    replace_text(slide.shapes[8], "•  ASR final 2.68 s / 5.32 s p95; WER 17.6% (n=29)")
    replace_text(slide.shapes[9], "")
    replace_text(slide.shapes[11], "RESULTS + LIMITS")
    replace_text(slide.shapes[12], "•  TTS TTFB 0.63 s / 0.62 s p95; target 0.6 s")
    replace_text(slide.shapes[13], "•  LLM TTFT: no stage marks (n=0); barge-in: no data (n=0)")
    replace_text(slide.shapes[14], "•  No judge; overlap cases include silent stops. Scoring used faster-whisper/small.")
    replace_text(slide.shapes[15], "DGX3 unavailable. Fallback/stub diagnostic only. Pi fixture: no VAD turn (peak 173, RMS 25.2).")
    slide.shapes[15].text_frame.paragraphs[0].runs[0].font.size = Pt(11)

    # Slide 7: the active demo defaults to English; Filipino/Taglish support is
    # available, while ElevenLabs remains optional rather than silently active.
    slide = deck.slides[6]
    replace_text(slide.shapes[5], "fil-PH\nEnglish by default\nFilipino + Taglish understood")
    replace_text(slide.shapes[6], "Qwen3-TTS Aiden is the active demo voice. ElevenLabs Filipino remains an optional selectable layer.")
    replace_text(slide.shapes[13], "Pilot first. Measure quality, safety, cost, and handoff.")
    slide.shapes[13].text_frame.paragraphs[0].runs[0].font.size = Pt(11)

    # Slide 8: make the next evidence gate explicit.
    slide = deck.slides[7]
    replace_text(slide.shapes[3], "ME3 combines full-duplex audio, a Codex-style voice and chat workspace, provider-selectable layers, typed tools, and verification into a Raspberry Pi workflow.")
    replace_text(slide.shapes[6], "Next: restore DGX3, rerun fdbench, then capture physical Pi echo + barge-in")

    deck.save(TARGET)
    print(TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
