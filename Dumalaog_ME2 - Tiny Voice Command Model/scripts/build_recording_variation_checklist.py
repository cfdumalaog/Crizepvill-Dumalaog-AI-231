"""Build a cautious prompt-coverage checklist from the frozen ASR audit."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
VARIATIONS = PROJECT / "docs" / "ground_truth_phrases" / "ME2_variations.csv"
AUDIT = PROJECT / "runs" / "personal-transcript-audit-20261002" / "closest_canonical_prompt_audit.csv"
OUT = PROJECT / "docs" / "evaluations" / "recording-variation-checklist-20261002.md"


def main() -> None:
    candidate_counts: Counter[tuple[str, str]] = Counter()
    with AUDIT.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if float(row.get("original_variation_similarity") or 0) >= 0.70:
                candidate_counts[(row.get("original_variation_candidate_label", ""), row.get("original_variation_candidate", ""))] += 1
    by_intent: dict[str, list[tuple[str, str, int]]] = {}
    with VARIATIONS.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            label, phrase = row["label"], row["phrase"].strip()
            by_intent.setdefault(label, []).append((row["variation"], phrase, candidate_counts[(label, phrase)]))
    lines = [
        "# Human recording checklist for the 93 ME2 command variations",
        "",
        "Generated 2026-10-02 from the saved personal transcript audit and the published `variations.csv`.",
        "",
        "## How to read this",
        "",
        "The personal manifest has 1,025 rows, including 796 command-labelled rows representing 611 unique command waveforms. Historic takes generally have no saved `suggested_phrase`; therefore we cannot know which wording the recorder showed when they were captured. The earlier local Whisper audit is only a transcript suggestion. Counts below are unique waveforms whose ASR transcript selected that exact official variation with similarity >= 0.70. A positive count is not human-verified proof; zero means the audit did not confidently identify it, not proof that it was never spoken.",
        "",
        "**Collection rule:** Use the recorder’s phrase dropdown and record every phrase below at least 5 times, preferably across 3+ consenting speakers and quiet-near, fan-near, and quiet-far conditions. Record one take per selected phrase, listen back, then save. Keep a complete speaker out of training for personal validation. Do not relabel or delete old audio from ASR alone.",
        "",
        "`[ ]` is deliberately left unchecked: the historic clip-to-prompt link is unknown, and all 93 prompts should be covered prospectively. The count is only a hint about possible prior wording coverage.",
        "",
        "## Checklist",
        "",
    ]
    for label in sorted(by_intent):
        lines.extend([f"### {label}", ""])
        for variant, phrase, count in by_intent[label]:
            lines.append(f"- [ ] Variation {variant}: **{phrase}** — provisional ASR matches: {count}")
        lines.append("")
    lines += [
        "## Separate audio-label issues to review",
        "",
        "- 102 legacy-labeled command rows are unmapped to the current 31-label taxonomy; inspect audio and transcript before deciding whether they fit a current class.",
        "- 105 mapped rows have an ASR/fuzzy phrase candidate that differs from or weakly matches the folder label; listen before relabeling.",
        "- One identical PCM waveform appears under both `CREATE_REMINDER_STUDY` and `MESSAGE`. The active training audit excluded this collision; preserve the original files and resolve the correct spoken meaning by listening.",
        "- Prior ASR text showed likely boundary confusions for light on/off, temperature values, and reminder subtypes. These are review targets, not proven labeling errors.",
        "",
        "## Recorder changes",
        "",
        "The recorder now offers the established canonical prompt plus the official dataset wording variations for each of the 31 model labels. Every new manifest row records the exact selected wording in `suggested_phrase` and its source ID (`canonical` or published `v1`/`v2`/`v3`) in `phrase_variant`. Existing rows remain unchanged; empty historical prompt fields remain unknown.",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} with {sum(map(len, by_intent.values()))} official variation prompts")


if __name__ == "__main__":
    main()
