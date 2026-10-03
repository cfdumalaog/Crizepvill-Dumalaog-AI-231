"""Build decoded-audio, source-deduplicated, speaker-disjoint SLURP manifests.

Only the official human ``slurp_real`` archive is used. See the project handoff
for the source citation and CC BY-NC 4.0 restrictions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import re
import sys

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLURP_ANNOTATIONS = PROJECT_ROOT / "data" / "external" / "slurp" / "annotations"
SLURP_AUDIO_DIR = PROJECT_ROOT / "data" / "external" / "slurp" / "audio" / "slurp_real"
OUTPUT_DIR = PROJECT_ROOT / "data" / "slurp"
sys.path.insert(0, str(PROJECT_ROOT))
from tinyvcm_model.config import LABELS, WAKE_LABELS, SAMPLES, SR  # noqa: E402

DATE_QUERY_RE = re.compile(
    r"\b(?:date|day|weekday|weekdays|month|months|year|years|calendar|birthday|easter|holiday)\b",
    re.IGNORECASE,
)
TIME_DIFFERENCE_RE = re.compile(r"\btime difference\b", re.IGNORECASE)
CLOCK_WORD_RE = re.compile(r"\b(?:time|clock)\b", re.IGNORECASE)


def entity_surface(prompt: dict, entity: dict) -> str:
    tokens = prompt.get("tokens", [])
    pieces = []
    for token_index in entity.get("span", []):
        try:
            pieces.append(str(tokens[int(token_index)].get("surface", "")))
        except (IndexError, TypeError, ValueError, AttributeError):
            return ""
    return " ".join(pieces).strip().lower()


def map_prompt(prompt: dict, exclusions: Counter) -> str | None:
    """Map only an exact VCM command class; never infer a missing slot."""
    intent = str(prompt.get("intent", ""))
    entities = prompt.get("entities", [])
    if intent.startswith("lists_"):
        exclusions["lists_not_reminders"] += 1
        exclusions[f"unsupported_intent:{intent}"] += 1
        return None
    if intent == "play_music":
        return "PLAY_MUSIC"
    if intent == "weather_query":
        return "WEATHER"
    if intent == "datetime_query":
        sentence = str(prompt.get("sentence", ""))
        if not CLOCK_WORD_RE.search(sentence):
            exclusions["datetime_not_clock_time"] += 1
            return None
        if DATE_QUERY_RE.search(sentence):
            exclusions["datetime_calendar_query"] += 1
            return None
        if TIME_DIFFERENCE_RE.search(sentence):
            exclusions["datetime_time_difference"] += 1
            return None
        return "TIME"
    if intent == "iot_hue_lighton":
        return "LIGHT_ON"
    if intent in ("iot_hue_lightoff", "hue_lightoff"):
        return "LIGHT_OFF"
    if intent == "audio_volume_up":
        return "VOLUME_UP"
    if intent == "audio_volume_down":
        return "VOLUME_DOWN"
    if intent == "iot_hue_lightchange":
        color_entities = [entity for entity in entities if entity.get("type") == "color_type"]
        conflicting_entities = [
            entity for entity in entities if entity.get("type") in ("color_type", "change_amount")
        ]
        if len(color_entities) != 1 or len(conflicting_entities) != 1:
            exclusions["ambiguous_or_mixed_color_entities"] += 1
            return None
        value = entity_surface(prompt, color_entities[0])
        mapped = {"red": "COLOR_RED", "green": "COLOR_GREEN", "blue": "COLOR_BLUE"}.get(value)
        if mapped is None:
            exclusions[f"unsupported_color_value:{value or '<empty>'}"] += 1
        return mapped
    if intent == "alarm_set":
        time_entities = [entity for entity in entities if entity.get("type") == "time"]
        if len(time_entities) != 1:
            exclusions["ambiguous_alarm_time_entities"] += 1
            return None
        value = entity_surface(prompt, time_entities[0])
        alarm_labels = {
            "6 am": "ALARM_6_00AM",
            "6:00 am": "ALARM_6_00AM",
            "six am": "ALARM_6_00AM",
            "6:00am": "ALARM_6_00AM",
            "6am": "ALARM_6_00AM",
            "8 am": "ALARM_8_00AM",
            "8:00 am": "ALARM_8_00AM",
            "eight am": "ALARM_8_00AM",
            "8:00am": "ALARM_8_00AM",
            "8am": "ALARM_8_00AM",
            "9 pm": "ALARM_9_00PM",
            "9:00 pm": "ALARM_9_00PM",
            "nine pm": "ALARM_9_00PM",
            "9:00pm": "ALARM_9_00PM",
            "9pm": "ALARM_9_00PM",
        }
        mapped = alarm_labels.get(value)
        if mapped is None:
            exclusions[f"unsupported_alarm_time:{value or '<empty>'}"] += 1
        return mapped

    exclusions[f"unsupported_intent:{intent or '<empty>'}"] += 1
    return None


def read_annotations() -> tuple[list[dict], dict]:
    prompts = []
    for name in ("train.jsonl", "devel.jsonl", "test.jsonl"):
        source = SLURP_ANNOTATIONS / name
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("r", encoding="utf-8") as stream:
            prompts.extend(json.loads(line) for line in stream if line.strip())
    metadata_path = SLURP_ANNOTATIONS / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return prompts, metadata


def speaker_split(speakers: list[str], seed: int) -> dict[str, set[str]]:
    ordered = sorted(speakers)
    random.Random(seed).shuffle(ordered)
    n_train = int(round(len(ordered) * 0.80))
    n_val = int(round(len(ordered) * 0.10))
    split = {
        "train": set(ordered[:n_train]),
        "val": set(ordered[n_train : n_train + n_val]),
        "test": set(ordered[n_train + n_val :]),
    }
    assert not (split["train"] & split["val"] or split["train"] & split["test"] or split["val"] & split["test"])
    return split


def decode_candidate(filename: str, check_decode: bool) -> tuple[Path, int | None, int | None, int | None] | None:
    root = SLURP_AUDIO_DIR.resolve()
    path = (SLURP_AUDIO_DIR / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    if not check_decode:
        # Validation/test content stays unopened until the final evaluation.
        return path, None, None, None
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return None
    if sample_rate != SR or audio.size == 0:
        return None
    frames, channels = audio.shape
    return path, int(sample_rate), int(channels), int(frames)


def build_manifests(seed: int = 231) -> dict[str, list[dict]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts, metadata = read_annotations()
    speakers = sorted(
        {
            str(info["usrid"])
            for item in metadata.values()
            for info in item.get("recordings", {}).values()
            if info.get("usrid") is not None
        }
    )
    splits = speaker_split(speakers, seed)
    prompt_by_id = {str(prompt["slurp_id"]): prompt for prompt in prompts}

    exclusion_counts: Counter = Counter()
    mapped = {}
    for prompt in prompts:
        label = map_prompt(prompt, exclusion_counts)
        if label is not None:
            mapped[str(prompt["slurp_id"])] = label

    expected_mapped = 2574
    if len(mapped) != expected_mapped:
        raise AssertionError(f"Strict prompt mapping changed: expected {expected_mapped}, got {len(mapped)}")

    # Collapse duplicate prompt references to the true recording identity. Keep
    # all labels until conflicts are detected; never let the label split a key.
    sources: dict[tuple[str, str], dict] = defaultdict(lambda: {"labels": set(), "files": {}, "prompt_ids": set()})
    metadata_recordings = 0
    for prompt_id, label in mapped.items():
        item = metadata.get(prompt_id)
        if item is None:
            exclusion_counts["mapped_prompt_missing_metadata"] += 1
            continue
        for filename, info in item.get("recordings", {}).items():
            metadata_recordings += 1
            usrid = str(info.get("usrid", ""))
            recid = str(info.get("recid", ""))
            if not usrid or not recid:
                exclusion_counts["recording_missing_speaker_or_recid"] += 1
                continue
            key = (usrid, recid)
            sources[key]["labels"].add(label)
            sources[key]["prompt_ids"].add(prompt_id)
            sources[key]["files"][str(filename)] = info

    label_collisions = []
    selected = []
    decode_failures = Counter()
    cropped_count = 0
    for index, ((usrid, recid), group) in enumerate(sorted(sources.items()), start=1):
        if len(group["labels"]) != 1:
            label_collisions.append(
                {"usrid": usrid, "recid": recid, "labels": sorted(group["labels"]), "prompt_ids": sorted(group["prompt_ids"])}
            )
            continue
        label = next(iter(group["labels"]))
        split = next(name for name, ids in splits.items() if usrid in ids)
        options = sorted(group["files"])
        options.sort(key=lambda filename: ("-headset" in filename.lower(), filename))
        decoded = None
        chosen_filename = None
        for filename in options:
            result = decode_candidate(filename, check_decode=(split == "train"))
            if result is not None:
                decoded = result
                chosen_filename = filename
                break
            decode_failures["missing_or_invalid_or_wrong_rate"] += 1
        if decoded is None:
            decode_failures["source_without_decodable_16khz_audio"] += 1
            continue
        audio_path, sample_rate, channels, frames = decoded
        if frames is not None and frames > SAMPLES:
            cropped_count += 1
        try:
            relative = audio_path.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError as error:
            raise RuntimeError(f"Selected audio escaped project root: {audio_path}") from error
        selected.append(
            {
                "path": relative,
                "slurp_ids": ";".join(sorted(group["prompt_ids"])),
                "source_prompt_count": len(group["prompt_ids"]),
                "recid": recid,
                "usrid": usrid,
                "label": label,
                "class_idx": WAKE_LABELS.index(label),
                "is_headset": int("-headset" in chosen_filename.lower()),
                "sample_rate": sample_rate,
                "channels": channels,
                "duration_sec": frames / sample_rate if frames is not None and sample_rate else None,
                "will_crop_to_2_5s": int(frames > SAMPLES) if frames is not None else None,
                "audio_checked": int(frames is not None),
                "split": split,
            }
        )
        if index % 1000 == 0:
            print(f"Decoded {index:,}/{len(sources):,} source groups; kept {len(selected):,}.", flush=True)

    if label_collisions:
        exclusion_counts["cross_label_source_groups"] = len(label_collisions)
    result: dict[str, list[dict]] = {name: [] for name in ("train", "val", "test")}
    for row in selected:
        result[row["split"]].append(row)

    fields = [
        "path", "slurp_ids", "source_prompt_count", "recid", "usrid", "label", "class_idx",
        "is_headset", "sample_rate", "channels", "duration_sec", "will_crop_to_2_5s", "audio_checked", "split",
    ]
    class_support = {}
    split_summary = {}
    for split_name, rows in result.items():
        rows.sort(key=lambda row: (row["usrid"], row["recid"]))
        output = OUTPUT_DIR / f"manifest_slurp_{split_name}.csv"
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        class_support[split_name] = dict(sorted(Counter(row["label"] for row in rows).items()))
        split_summary[split_name] = {
            "speakers": len({row["usrid"] for row in rows}),
            "recordings": len(rows),
            "source_groups": len({(row["usrid"], row["recid"]) for row in rows}),
        }
        state = "decoded" if split_name == "train" else "selected (audio deferred)"
        print(f"[SAVED] {output.name}: {len(rows):,} {state} source recordings from {split_summary[split_name]['speakers']} speakers")

    audit = {
        "seed": seed,
        "source": "SLURP official slurp_real archive only",
        "source_repository": "https://github.com/pswietojanski/slurp",
        "audio_license": "CC BY-NC 4.0",
        "raw_audio_used": "real human recordings only; no slurp_synth files",
        "total_prompts": len(prompts),
        "metadata_speakers": len(speakers),
        "mapped_prompts": len(mapped),
        "mapped_classes": sorted(set(mapped.values())),
        "mapped_class_counts": dict(sorted(Counter(mapped.values()).items())),
        "datetime_clock_time_only": {
            "rule": "requires time or clock; excludes calendar/day/month/year and time-difference questions",
            "mapped_count": sum(label == "TIME" for label in mapped.values()),
            "excluded_non_clock_or_calendar": exclusion_counts.get("datetime_not_clock_time", 0)
            + exclusion_counts.get("datetime_calendar_query", 0)
            + exclusion_counts.get("datetime_time_difference", 0),
        },
        "metadata_recording_entries_before_recid_dedup": metadata_recordings,
        "unique_speaker_recid_groups_before_audio_decode": len(sources),
        "duplicate_prompt_source_groups": sum(len(group["prompt_ids"]) > 1 for group in sources.values()),
        "cross_label_source_exclusions": label_collisions,
        "speaker_split_counts_from_metadata": {name: len(ids) for name, ids in splits.items()},
        "decoded_deduplicated_split_summary": split_summary,
        "class_support_by_split": class_support,
        "duration_over_2_5s_cropped_by_frontend": cropped_count,
        "audio_access_policy": "only train audio decoded during manifest build; validation/test audio remains unopened until final evaluation",
        "decoded_train_records": sum(row["audio_checked"] == 1 for row in selected),
        "deferred_validation_test_records": sum(row["audio_checked"] == 0 for row in selected),
        "decode_failures": dict(decode_failures),
        "exclusions": dict(exclusion_counts),
    }
    (OUTPUT_DIR / "slurp_exclusion_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(
        f"Mapped {len(mapped):,} prompts to {len(audit['mapped_classes'])} classes; "
        f"selected {len(selected):,} unique speaker/recid sources; decoded train rows only: "
        f"{audit['decoded_train_records']:,}; train clips over 2.5 s: {cropped_count:,}."
    )
    return result


if __name__ == "__main__":
    build_manifests(seed=231)
