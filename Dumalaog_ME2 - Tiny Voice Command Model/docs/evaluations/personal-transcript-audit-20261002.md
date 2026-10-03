# Personal recording transcript audit — 2026-10-02

## Recommendation

Do not restart/retrain the personal-recording model yet. First use the manual review queue to verify the recordings whose phrase-based candidate differs from the folder's mapped training label, plus old labels that cannot be mapped. A local ASR transcript is useful for finding likely issues, but is not reliable ground truth for short commands and paraphrases. The existing labels and training manifest were not changed.

## Audit method

Transcribed command-labeled rows from `data/human/manifest.csv` with the already-cached `small.en` faster-whisper model, CPU int8, local-files-only mode, English decode, and no label/prompt context. Identical waveform hashes reused one transcript. Compared transcripts to the dataset phrase variations and their parent-intent/slot-value mapping to the 31 supported leaf labels. This produces a phrase-based candidate only; it does not confirm what was spoken or the intended action.

## Results

- 1,025 manifest rows total; 796 command-labeled rows audited. Wake/noise/silence rows (229) were skipped.
- 611 unique waveforms transcribed; duplicate WAV rows reused the same transcript (185 rows); zero file/transcription errors.
- 694 command rows map to the current 31-class label set; 102 legacy rows remain unmapped.
- Among mapped rows, the phrase matcher suggests the folder-mapped label for 589 and suggests a different/low-match label for 105. These are manifest-row counts, not independent recordings or verified label accuracy.
- No labels changed. The manifest, training split, model weights, and Pi deployment were untouched; no retraining was run.

### Closest canonical recorder prompt

Compared each unique transcript to the 31 canonical prompt text files in `docs/ground_truth_phrases/` as a second, separate nearest-text check. For 561 unique waveform groups with a single mapped label, the nearest prompt label matched the mapped folder label for 451 and differed for 110. One additional waveform is duplicated under two different labels (`CREATE_REMINDER_STUDY` and `MESSAGE`); its audio must be reviewed and the group excluded from any training snapshot until resolved. The 110 nearest-prompt differences are not confirmed label errors: this text similarity method confuses short, acoustically/lexically similar commands and slot values.

The current active training audit already detected that cross-label duplicate and excluded both conflicting paths from the model splits. Therefore, this audit has not identified evidence that requires replacing the active model immediately. If the reviewer confirms additional wrong labels among clips actually used in the current model, create a corrected, frozen dataset snapshot and retrain as a candidate, then compare against untouched evaluation data before promotion.

## Examples showing why listening is required

- Several `LIGHT_OFF` recordings transcribe as “Turn off the lights,” but the phrase matcher selects `LIGHT_ON` because that exact wording is absent from its available `LIGHT_OFF` variants. The legacy recording prompt makes the `LIGHT_OFF` folder label plausible; this is a phrase-ground-truth gap, not proof of a bad recording.
- Several `PAUSE` files yield inconsistent short-audio ASR output (“Fast/Boss music”) or wording such as “Pause the music” not listed in the current variants. Listen to the clips before deciding whether the label is correct.
- `TIME` clips can be transcribed as “What’s the time?” while the phrase list has only nearby forms. Again, phrase mismatch alone does not justify relabeling.

## Review artifacts and next step

- `runs/personal-transcript-audit-20261002/personal_transcript_audit.csv` — all manifest command rows, transcript, mapped folder label, phrase candidate, similarity, and review status.
- `runs/personal-transcript-audit-20261002/manual_review_queue.csv` — unique waveform groups needing review, with all associated paths and labels preserved together.
- `runs/personal-transcript-audit-20261002/closest_canonical_prompt_audit.csv` — one row per waveform, nearest canonical recorder prompt, and all associated paths/labels (including the conflicting duplicate).
- `runs/personal-transcript-audit-20261002/summary.json` — machine-readable counts and limitations.

Listen to the queued unique waveforms, fill `reviewer_confirmed_label` and `reviewer_notes`, and only then decide which confirmed examples belong in a new training snapshot. Preserve an immutable snapshot and speaker/session-separated validation/test data before retraining. Keep this ASR audit tool outside the deployed VCM: it is not part of wake or intent inference.
