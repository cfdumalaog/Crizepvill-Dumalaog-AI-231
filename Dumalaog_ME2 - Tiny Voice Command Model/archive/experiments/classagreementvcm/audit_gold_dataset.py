"""Read-only audit of a pinned ME2 Spoken Command Dataset Parquet snapshot.

This script never writes into the source dataset and never fits a model. Its
report is suitable for the candidate-only classagreementvcm audit namespace.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parents[1]
SNAPSHOT = PROJECT / "data" / "ai231-me2-voice-commands-hf-25111444"
REVISION = "25111444af3adff7588d86ab27c304bb080895cf"
EXPECTED_ROWS = {"train": 10682, "test": 4418, "holdout": 196, "numerals": 66390}
DATASET = "ME2 Spoken Command Dataset"

sys.path.insert(0, str(PROJECT))
from tinyvcm_model.config import LABELS as MODEL_LABELS  # noqa: E402


# Dataset README source names are not necessarily equal to the manifest source
# values. Keep this inventory explicit and fail closed for unmapped sources.
LICENSE_INVENTORY = {
    "SLURP": "Collated README says CC BY 4.0; upstream audio terms require reconciliation",
    "Google Speech Commands v2": "CC BY 4.0; retain attribution",
    "Common Voice 19 (en)": "CC0",
    "Fluent Speech Commands": "CC BY-NC-ND 4.0; academic research only; no commercial use",
    "SNIPS SLU": "Sonos license: non-commercial academic/research only; privacy obligations apply",
    "Timers and Such": "unverified; README points to source license file",
    "MLEnd spoken numerals": "unverified; README points to dataset terms; numerals excluded",
    "Multi-Sensor Voice Command (Xela VCM)": "CC BY 4.0 plus GDPR download-tracking obligation; verify compliance",
    "Group synthetic set": "conditional; confirm rights/consent for both reference voice sources",
    "Group recordings": "conditional; README says do not share outside class without each speaker's consent",
}
UPSTREAM_LICENSE_AUDIT = {
    "SLURP": "CONFLICT: collated README states CC BY 4.0; upstream SLURP repository states audio is CC BY-NC 4.0. Treat as non-commercial only pending dataset-owner correction/confirmation.",
    "Google Speech Commands v2": "CC BY 4.0 per Google; numerals are excluded from command training, with 11 command examples in train.",
    "Common Voice 19 (en)": "CC0 per Mozilla; do not mirror or redistribute the dataset outside the designated distribution platform.",
    "Fluent Speech Commands": "CC BY-NC-ND 4.0 and academic research only per Fluent.ai; no commercial use.",
    "SNIPS SLU": "Sonos terms permit internal training/research only for non-commercial academic use and impose personal-data/GDPR responsibilities; acceptance and local compliance evidence are unavailable here.",
    "Timers and Such": "Zenodo record marks the dataset other-open; the exact license file is not present in this snapshot, so rights remain unverified.",
    "Multi-Sensor Voice Command (Xela VCM)": "CC-BY-4.0; KU Leuven RDR terms require tracking full and partial downloads/derivatives for GDPR compliance; this project's tracking record is not found.",
    "Group synthetic set": "Blocked pending exact SilencioPH reference dataset/version/license and voice-cloning/synthetic-output permission evidence.",
    "Group recordings": "Blocked pending speaker consent evidence covering model training and this use; do not share outside class without each speaker's consent.",
    "MLEnd spoken numerals": "Not used by the command candidate; source terms are not audited for this excluded partition.",
}
SOURCE_INVENTORY = {
    "SLURP": "SLURP",
    "SpeechCommands_v2": "Google Speech Commands v2",
    "CommonVoice_en": "Common Voice 19 (en)",
    "FluentSpeechCommands": "Fluent Speech Commands",
    "SNIPS": "SNIPS SLU",
    "TimersAndSuch": "Timers and Such",
    "MLEnd_numerals": "MLEnd spoken numerals",
    "xela_Multi-Sensor": "Multi-Sensor Voice Command (Xela VCM)",
    "xela_SET_TEMPERATURE_REAL": "Multi-Sensor Voice Command (Xela VCM)",
    "group_synthetic": "Group synthetic set",
    "real_voice": "Group recordings",
}

COMMAND_VALUES = {
    "ALARM": {"6:00 AM": "ALARM_6_00AM", "8:00 AM": "ALARM_8_00AM", "9:00 PM": "ALARM_9_00PM"},
    "BRIGHTNESS": {"20 percent": "BRIGHTNESS_20", "60 percent": "BRIGHTNESS_60", "100 percent": "BRIGHTNESS_100"},
    "COLOR": {"Blue": "COLOR_BLUE", "Green": "COLOR_GREEN", "Red": "COLOR_RED"},
    "CREATE_REMINDER": {"Drink water": "CREATE_REMINDER_DRINK_WATER", "Exercise": "CREATE_REMINDER_EXERCISE", "Study": "CREATE_REMINDER_STUDY"},
    "TEMPERATURE": {"18 degrees": "TEMPERATURE_18", "22 degrees": "TEMPERATURE_22", "26 degrees": "TEMPERATURE_26"},
    "TIMER": {"10 seconds": "TIMER_10s", "1 minute": "TIMER_1m", "30 seconds": "TIMER_30s"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_json(url: str) -> dict | list:
    request = Request(url, headers={"User-Agent": "ME2-VCM-dataset-audit/1.0"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def verify_remote_snapshot() -> dict:
    """Compare the pinned Hub revision, file inventory and source hashes."""
    base = "https://huggingface.co/api/datasets/airimonda/ai231-me2-voice-commands"
    resolve = "https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/resolve"
    try:
        repo = fetch_json(f"{base}/revision/{REVISION}")
        if repo.get("sha") != REVISION:
            return {"status": "FAIL", "reason": "Hub returned a commit different from the pinned revision", "observed_revision": repo.get("sha")}
        siblings = {item.get("rfilename") for item in repo.get("siblings", [])}
        expected = {".gitattributes", "README.md", "variations.csv"}
        local_parquets = sorted((SNAPSHOT / "data").glob("*.parquet"))
        expected.update(f"data/{path.name}" for path in local_parquets)
        missing_remote = sorted(expected - siblings)
        unexpected_remote = sorted(siblings - expected)
        if missing_remote or unexpected_remote:
            return {"status": "FAIL", "reason": "Local snapshot file inventory differs from the pinned Hub revision", "missing_remote_files": missing_remote, "unexpected_remote_files": unexpected_remote}

        root_tree = fetch_json(f"{base}/tree/{REVISION}")
        data_tree = fetch_json(f"{base}/tree/{REVISION}/data")
        entries = {item.get("path"): item for item in [*root_tree, *data_tree] if item.get("type") == "file"}
        mismatches = []
        unverifiable = []
        checked = []
        for relative in sorted(expected):
            local = SNAPSHOT / Path(relative)
            if not local.is_file():
                mismatches.append({"path": relative, "reason": "missing locally"})
                continue
            item = entries.get(relative)
            if item is None:
                # The root API may return the file in the commit sibling list
                # without an expanded tree item; query its direct parent.
                parent = relative.rsplit("/", 1)[0] if "/" in relative else ""
                parent_tree = fetch_json(f"{base}/tree/{REVISION}/{quote(parent, safe='/')}") if parent else root_tree
                item = next((entry for entry in parent_tree if entry.get("path") == relative), None)
            if item is None:
                mismatches.append({"path": relative, "reason": "no file metadata in pinned Hub tree"})
                continue
            remote_size = item.get("size") or (item.get("lfs") or {}).get("size")
            if remote_size is not None and int(remote_size) != local.stat().st_size:
                mismatches.append({"path": relative, "reason": "size mismatch", "local_size": local.stat().st_size, "remote_size": int(remote_size)})
                continue
            remote_sha = (item.get("lfs") or {}).get("oid")
            if remote_sha:
                local_sha = sha256_file(local)
                if local_sha != remote_sha:
                    mismatches.append({"path": relative, "reason": "LFS SHA-256 mismatch", "local_sha256": local_sha, "remote_sha256": remote_sha})
                    continue
                checked.append(relative)
            elif relative in {"README.md", "variations.csv", ".gitattributes"}:
                request = Request(f"{resolve}/{REVISION}/{quote(relative, safe='/')}", headers={"User-Agent": "ME2-VCM-dataset-audit/1.0"})
                with urlopen(request, timeout=30) as response:
                    remote_sha = hashlib.sha256(response.read()).hexdigest()
                local_sha = sha256_file(local)
                if local_sha != remote_sha:
                    mismatches.append({"path": relative, "reason": "SHA-256 mismatch", "local_sha256": local_sha, "remote_sha256": remote_sha})
                    continue
                checked.append(relative)
            else:
                unverifiable.append(relative)
        return {
            "status": "PASS" if not mismatches and not unverifiable and not unexpected_remote and len(checked) == len(expected) else "BLOCKED",
            "repository": "airimonda/ai231-me2-voice-commands",
            "revision": REVISION,
            "revision_matches": repo.get("sha") == REVISION,
            "public": repo.get("private") is False and repo.get("gated") is False,
            "dataset_card_license_field": (repo.get("cardData") or {}).get("license"),
            "expected_files": len(expected),
            "checked_files": checked,
            "mismatches": mismatches,
            "unverifiable_files": unverifiable,
            "unexpected_remote_files": unexpected_remote,
        }
    except Exception as exc:
        return {"status": "BLOCKED", "reason": f"Hub verification failed: {type(exc).__name__}: {exc}"}


def derive_leaf(command: str, slot_value: str) -> str:
    if command == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE"
    values = COMMAND_VALUES.get(command)
    if values is None:
        return command
    return values.get(slot_value, "UNMAPPED")


def canonical_path(value: str) -> bool:
    path = Path(value.replace("\\", "/"))
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def read_phrase_schema() -> dict:
    path = SNAPSHOT / "variations.csv"
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    phrases_by_command: dict[str, set[str]] = defaultdict(set)
    values_by_command: dict[str, set[str]] = defaultdict(set)
    phrase_value: dict[tuple[str, str], str] = {}
    leaf_phrases: dict[str, set[str]] = defaultdict(set)
    errors = []
    for row in rows:
        command = row["label"].strip()
        phrase = row["phrase"].strip()
        value = row["value"].strip()
        phrases_by_command[command].add(phrase)
        phrase_value[(command, phrase)] = value
        if value:
            values_by_command[command].add(value)
        leaf = derive_leaf(command, value)
        if leaf == "UNMAPPED":
            errors.append({"command": command, "value": value, "phrase": phrase})
        else:
            leaf_phrases[leaf].add(phrase)
    schema_leaves = sorted(leaf_phrases)
    return {
        "row_count": len(rows),
        "commands": sorted(phrases_by_command),
        "phrases_by_command": phrases_by_command,
        "values_by_command": values_by_command,
        "phrase_value": phrase_value,
        "leaf_phrases": leaf_phrases,
        "leaves": schema_leaves,
        "mapping_errors": errors,
        "file_sha256": sha256_file(path),
    }


def iter_shards(split: str):
    paths = sorted((SNAPSHOT / "data").glob(f"{split}-*.parquet"))
    for path in paths:
        parquet = pq.ParquetFile(path)
        yield path, parquet


def audit(max_rows: int | None = None, owner_confirmed_holdout_oos_rows: bool = False) -> dict:
    schema = read_phrase_schema()
    hub = verify_remote_snapshot()
    split_rows = Counter()
    labels = defaultdict(Counter)
    sources = defaultdict(Counter)
    accents = defaultdict(Counter)
    transcript_sources = defaultdict(Counter)
    speaker_groups = defaultdict(set)
    raw_hash_groups = defaultdict(list)
    pcm_hash_groups = defaultdict(list)
    file_names = defaultdict(Counter)
    split_digest = {split: hashlib.sha256() for split in EXPECTED_ROWS}
    audio_checks = Counter()
    row_errors = []
    speaker_cross_split = defaultdict(set)
    command_speaker_cross_split = defaultdict(set)
    bytes_by_split = Counter()
    total_duration = Counter()
    mapping_counts = Counter()
    command_counts = defaultdict(Counter)
    other_slot_counts = Counter()
    oos_flag_mismatches = []
    processed = 0

    for split in EXPECTED_ROWS:
        shard_paths = sorted((SNAPSHOT / "data").glob(f"{split}-*.parquet"))
        if not shard_paths:
            row_errors.append({"kind": "missing_split_shard", "split": split})
            continue
        for shard_path, parquet in iter_shards(split):
            print(f"Auditing {split}: {shard_path.name} ({parquet.metadata.num_rows} rows)", flush=True)
            for batch in parquet.iter_batches(
                batch_size=24,
                columns=[
                    "audio", "file", "command", "variation", "slot_value", "out_of_scope",
                    "bucket", "speaker_id", "source", "is_synthetic", "accent_group", "duration_s",
                    "transcript", "transcript_source", "variation_match", "whisper_check",
                ],
            ):
                for row in batch.to_pylist():
                    if max_rows is not None and processed >= max_rows:
                        break
                    processed += 1
                    split_rows[split] += 1
                    command = (row.get("command") or "").strip()
                    slot_value = (row.get("slot_value") or "").strip()
                    out_of_scope = int(row.get("out_of_scope") or 0)
                    bucket = (row.get("bucket") or "").strip()
                    is_non_schema_slot = (
                        not out_of_scope
                        and command != "OUT_OF_SCOPE"
                        and bucket.endswith("(other slot value)")
                    )
                    leaf = "NON_SCHEMA_SLOT_VALUE" if is_non_schema_slot else derive_leaf(command, slot_value)
                    if split == "numerals":
                        leaf = "NUMERAL_EXCLUDED"
                    elif out_of_scope:
                        leaf = "OUT_OF_SCOPE"
                    mapping_counts[leaf] += 1
                    command_counts[split][command] += 1
                    if is_non_schema_slot:
                        other_slot_counts[split] += 1
                    if out_of_scope != int(command == "OUT_OF_SCOPE") and split != "numerals":
                        oos_flag_mismatches.append({"split": split, "file": row.get("file"), "command": command, "out_of_scope": out_of_scope, "bucket": bucket})
                        row_errors.append({"kind": "out_of_scope_command_flag_mismatch", **oos_flag_mismatches[-1]})
                    if leaf == "UNMAPPED" or (not out_of_scope and not is_non_schema_slot and leaf not in schema["leaves"]):
                        row_errors.append({"kind": "unmapped_in_scope_label", "split": split, "file": row.get("file"), "command": command, "slot_value": slot_value})
                    if not out_of_scope and not is_non_schema_slot and split != "numerals":
                        phrase = (row.get("variation") or "").strip()
                        if phrase not in schema["phrases_by_command"].get(command, set()):
                            row_errors.append({"kind": "variation_phrase_not_in_schema", "split": split, "file": row.get("file"), "command": command, "variation": phrase})
                        elif schema["phrase_value"].get((command, phrase), "") != slot_value:
                            row_errors.append({"kind": "variation_slot_value_mismatch", "split": split, "file": row.get("file"), "command": command, "variation": phrase, "slot_value": slot_value, "schema_value": schema["phrase_value"].get((command, phrase), "")})
                    if split != "numerals" and not out_of_scope and not is_non_schema_slot and not (leaf in MODEL_LABELS and leaf in schema["leaves"]):
                        row_errors.append({"kind": "intent_label_not_in_model_schema", "split": split, "file": row.get("file"), "leaf": leaf})
                    labels[split][leaf] += 1
                    source = (row.get("source") or "").strip()
                    sources[split][source] += 1
                    accents[split][(row.get("accent_group") or "").strip()] += 1
                    transcript_sources[split][(row.get("transcript_source") or "").strip()] += 1
                    filename = (row.get("file") or "").strip()
                    audio_obj = row.get("audio") or {}
                    audio_bytes = audio_obj.get("bytes")
                    embedded_path = (audio_obj.get("path") or "").strip()
                    file_names[split][filename] += 1
                    if not canonical_path(filename) or not filename.lower().endswith(".wav"):
                        row_errors.append({"kind": "unsafe_or_nonwav_manifest_path", "split": split, "file": filename})
                    if Path(filename).name != Path(embedded_path).name:
                        row_errors.append({"kind": "manifest_audio_path_mismatch", "split": split, "file": filename, "audio_path": embedded_path})
                    if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
                        row_errors.append({"kind": "missing_audio_payload", "split": split, "file": filename})
                        continue
                    raw_sha = hashlib.sha256(audio_bytes).hexdigest()
                    raw_hash_groups[raw_sha].append((split, leaf, filename))
                    bytes_by_split[split] += len(audio_bytes)
                    try:
                        info = sf.info(io.BytesIO(audio_bytes))
                        samples, rate = sf.read(io.BytesIO(audio_bytes), dtype="int16", always_2d=True)
                        if rate != 16000 or info.samplerate != 16000:
                            row_errors.append({"kind": "wrong_sample_rate", "split": split, "file": filename, "rate": rate})
                        if samples.shape[1] != 1 or info.channels != 1:
                            row_errors.append({"kind": "wrong_channel_count", "split": split, "file": filename, "channels": int(samples.shape[1])})
                        if info.subtype != "PCM_16":
                            row_errors.append({"kind": "wrong_pcm_subtype", "split": split, "file": filename, "subtype": info.subtype})
                        if samples.shape[0] == 0:
                            row_errors.append({"kind": "empty_waveform", "split": split, "file": filename})
                        if not np.isfinite(samples).all():
                            row_errors.append({"kind": "nonfinite_waveform", "split": split, "file": filename})
                        if len(samples) and not np.any(samples):
                            audio_checks["zero_waveforms"] += 1
                        if len(samples) and np.any(np.abs(samples[:, 0].astype(np.int32)) >= 32760):
                            audio_checks["near_clipped_waveforms"] += 1
                        actual_duration = len(samples) / rate
                        total_duration[split] += actual_duration
                        expected_duration = float(row.get("duration_s") or 0.0)
                        if abs(actual_duration - expected_duration) > 0.02:
                            row_errors.append({"kind": "duration_mismatch", "split": split, "file": filename, "manifest_s": expected_duration, "wav_s": round(actual_duration, 5)})
                        pcm_digest = hashlib.sha256()
                        pcm_digest.update(int(rate).to_bytes(4, "little", signed=False))
                        pcm_digest.update(int(samples.shape[1]).to_bytes(2, "little", signed=False))
                        pcm_digest.update(samples.tobytes(order="C"))
                        pcm_sha = pcm_digest.hexdigest()
                        pcm_hash_groups[pcm_sha].append((split, leaf, filename))
                        audio_checks["decoded_waveforms"] += 1
                    except Exception as exc:  # Preserve full evidence, don't silently drop.
                        row_errors.append({"kind": "wav_decode_error", "split": split, "file": filename, "error": f"{type(exc).__name__}: {exc}"})
                    speaker = (row.get("speaker_id") or "").strip()
                    synthetic = int(row.get("is_synthetic") or 0)
                    group = f"{'synthetic' if synthetic else 'human'}:{speaker}"
                    speaker_groups[split].add(group)
                    speaker_cross_split[group].add(split)
                    if split != "numerals":
                        command_speaker_cross_split[group].add(split)
                    total_duration[split] += 0  # Keep Counter key for empty duration.
                    split_digest[split].update(
                        f"{filename}\0{leaf}\0{speaker}\0{raw_sha}\n".encode("utf-8", errors="replace")
                    )
                if max_rows is not None and processed >= max_rows:
                    break
            if max_rows is not None and processed >= max_rows:
                break
        if max_rows is not None and processed >= max_rows:
            break

    raw_duplicates = [items for items in raw_hash_groups.values() if len(items) > 1]
    pcm_duplicates = [items for items in pcm_hash_groups.values() if len(items) > 1]
    duplicate_label_collisions = [items for items in pcm_duplicates if len({item[1] for item in items}) > 1]
    cross_split_audio = [items for items in pcm_duplicates if len({item[0] for item in items}) > 1]
    speaker_leakage_all = {group: sorted(splits) for group, splits in speaker_cross_split.items() if len(splits) > 1}
    speaker_leakage = {group: sorted(splits) for group, splits in command_speaker_cross_split.items() if len(splits) > 1}

    local_file_hashes = {}
    for path in [SNAPSHOT / "README.md", SNAPSHOT / "variations.csv", *sorted((SNAPSHOT / "data").glob("*.parquet"))]:
        if path.is_file():
            local_file_hashes[str(path.relative_to(SNAPSHOT)).replace("\\", "/")] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    source_names = sorted({name for split in sources.values() for name in split})
    source_disposition = {
        source: {
            "readme_source": SOURCE_INVENTORY.get(source),
            "license_or_terms": LICENSE_INVENTORY.get(SOURCE_INVENTORY.get(source, ""), "unmapped source; training blocked"),
            "rows_by_split": {split: counts.get(source, 0) for split, counts in sources.items()},
        }
        for source in source_names
    }
    report = {
        "dataset_name": DATASET,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only local snapshot audit; no model training, inference, test scoring, promotion or deployment",
        "source_snapshot": {
            "local_path": str(SNAPSHOT),
            "declared_huggingface_revision": REVISION,
            "remote_revision_verification": hub,
            "file_hashes": local_file_hashes,
            "expected_shard_count": 10,
            "observed_shard_count": len(list((SNAPSHOT / "data").glob("*.parquet"))),
        },
        "readme_license_inventory": LICENSE_INVENTORY,
        "upstream_license_audit": UPSTREAM_LICENSE_AUDIT,
        "source_rights_by_manifest_source": source_disposition,
        "training_rights_gate": "BLOCKED: SLURP license conflict, Timers and Such exact terms, Xela download/derivative tracking record, classmate consent and synthetic voice/source rights remain unresolved. Academic-only sources must remain non-commercial; MLEnd numerals are excluded from the command candidate.",
        "holdout_oos_confirmation": {
            "owner_confirmation_recorded": owner_confirmed_holdout_oos_rows,
            "evidence": [
                "Course-group dataset owner stated that OOS items were added to test and holdout, then asked members to re-pull.",
                "Linked benchmark README at inspected commit documents the full 196-row holdout as 186 in-scope commands plus 10 OOS clips.",
            ] if owner_confirmed_holdout_oos_rows else [],
        },
        "schema": {
            "phrase_rows": schema["row_count"],
            "commands": schema["commands"],
            "mapped_leaf_labels": schema["leaves"],
            "schema_mapping_errors": schema["mapping_errors"],
            "phrase_schema_issues": [error for error in row_errors if error["kind"] in {"variation_phrase_not_in_schema", "variation_slot_value_mismatch"}],
            "expected_current_model_labels": MODEL_LABELS,
            "model_schema_label_set_matches_variations": set(MODEL_LABELS) == set(schema["leaves"]),
            "benchmark_top_level_intent_count": len(schema["commands"]),
            "benchmark_variation_command_count": schema["row_count"],
            "model_leaf_outputs_decode_to_intent_and_slot": set(MODEL_LABELS) == set(schema["leaves"]),
            "explicit_out_of_scope_model_output_present": "OUT_OF_SCOPE" in MODEL_LABELS,
            "reject_policy_decided": False,
            "variations_csv_sha256": schema["file_sha256"],
        },
        "scan": {
            "partial": max_rows is not None,
            "max_rows": max_rows,
            "rows_processed": processed,
            "rows_by_split": dict(split_rows),
            "expected_rows_by_split": EXPECTED_ROWS,
            "row_counts_match_expected": max_rows is None and dict(split_rows) == EXPECTED_ROWS,
            "intent_leaf_counts_by_split": {split: dict(sorted(counts.items())) for split, counts in labels.items()},
            "command_counts_by_split": {split: dict(sorted(counts.items())) for split, counts in command_counts.items()},
            "non_schema_slot_rows_by_split": dict(other_slot_counts),
            "out_of_scope_flag_mismatch_rows": oos_flag_mismatches,
            "source_counts_by_split": {split: dict(sorted(counts.items())) for split, counts in sources.items()},
            "accent_counts_by_split": {split: dict(sorted(counts.items())) for split, counts in accents.items()},
            "transcript_source_counts_by_split": {split: dict(sorted(counts.items())) for split, counts in transcript_sources.items()},
            "row_issue_counts_by_split": {split: dict(sorted(Counter(error["kind"] for error in row_errors if error.get("split") == split).items())) for split in EXPECTED_ROWS},
            "invalid_in_scope_rows_by_split": {split: len({error.get("file") for error in row_errors if error.get("split") == split and error["kind"] == "unmapped_in_scope_label"}) for split in EXPECTED_ROWS},
            "observed_source_values": source_names,
            "all_sources_mapped_to_readme_sources": all(source in SOURCE_INVENTORY for source in source_names),
            "speaker_groups_by_split": {split: len(groups) for split, groups in speaker_groups.items()},
            "speaker_identity_leakage": speaker_leakage,
            "speaker_identity_overlap_including_numerals": speaker_leakage_all,
            "raw_duplicate_groups": len(raw_duplicates),
            "normalized_pcm_duplicate_groups": len(pcm_duplicates),
            "cross_label_audio_collision_groups": len(duplicate_label_collisions),
            "cross_split_audio_groups_command_partitions": len([items for items in pcm_duplicates if len({item[0] for item in items if item[0] != "numerals"}) > 1]),
            "cross_split_audio_groups_including_numerals": len(cross_split_audio),
            "waveform_checks": dict(audio_checks),
            "payload_bytes_by_split": dict(bytes_by_split),
            "decoded_duration_hours_by_split": {split: round(seconds / 3600, 3) for split, seconds in total_duration.items()},
            "file_name_duplicates_by_split": {split: sum(count - 1 for count in counts.values() if count > 1) for split, counts in file_names.items()},
            "split_inventory_sha256": {split: digest.hexdigest() for split, digest in split_digest.items()},
            "errors": row_errors,
        },
        "gates": {
            "local_snapshot_readable": all(split_rows.values()) if max_rows is None else processed > 0,
            "snapshot_revision_confirmed_against_hub": hub.get("status") == "PASS",
            "schema_crosswalk_complete": not schema["mapping_errors"] and set(MODEL_LABELS) == set(schema["leaves"]) and not any(error["kind"] == "unmapped_in_scope_label" for error in row_errors),
            "out_of_scope_flags_consistent": not oos_flag_mismatches,
            "hub_revision_and_content_hashes_verified": hub.get("status") == "PASS",
            "audio_integrity_pass": max_rows is None and not [error for error in row_errors if error["kind"] in {"missing_audio_payload", "unsafe_or_nonwav_manifest_path", "manifest_audio_path_mismatch", "wrong_sample_rate", "wrong_channel_count", "wrong_pcm_subtype", "empty_waveform", "nonfinite_waveform", "duration_mismatch", "wav_decode_error"}],
            "split_leakage_pass": max_rows is None and not speaker_leakage and not [items for items in pcm_duplicates if len({item[0] for item in items if item[0] != "numerals"}) > 1],
            "duplicate_label_collision_free": max_rows is None and not duplicate_label_collisions,
            "dataset_owner_confirmed_holdout_oos_rows": owner_confirmed_holdout_oos_rows,
            "out_of_scope_32nd_class_approved": False,
            "reject_policy_decided": False,
            "source_training_rights_confirmed": False,
        },
    }
    return report


def main() -> int:
    global SNAPSHOT, REVISION
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "runs" / "classagreementvcm-audit-20261002" / "gold_dataset_audit.json")
    parser.add_argument("--max-rows", type=int, default=None, help="smoke-test only; a partial scan is never a pass")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT, help="local immutable snapshot directory")
    parser.add_argument("--revision", default=REVISION, help="immutable Hugging Face commit SHA")
    parser.add_argument("--owner-confirmed-holdout-oos-rows", action="store_true", help="record owner confirmation observed in the class handoff")
    args = parser.parse_args()
    SNAPSHOT = args.snapshot.resolve()
    REVISION = args.revision
    report = audit(args.max_rows, args.owner_confirmed_holdout_oos_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    scan = report["scan"]
    print(json.dumps({
        "report": str(args.output),
        "rows_by_split": scan["rows_by_split"],
        "expected_rows_by_split": scan["expected_rows_by_split"],
        "audio_checks": scan["waveform_checks"],
        "errors": len(scan["errors"]),
        "speaker_leakage_groups": len(scan["speaker_identity_leakage"]),
        "cross_split_audio_groups": scan["cross_split_audio_groups_command_partitions"],
        "invalid_in_scope_rows": scan["invalid_in_scope_rows_by_split"],
        "non_schema_slot_rows": scan["non_schema_slot_rows_by_split"],
        "out_of_scope_flag_mismatches": len(scan["out_of_scope_flag_mismatch_rows"]),
        "hub_revision_verification": report["source_snapshot"]["remote_revision_verification"].get("status"),
        "training_rights_gate": report["training_rights_gate"],
        "gates": report["gates"],
    }, indent=2, ensure_ascii=False))
    critical_errors = {"missing_audio_payload", "unsafe_or_nonwav_manifest_path", "manifest_audio_path_mismatch", "wrong_sample_rate", "wrong_channel_count", "wrong_pcm_subtype", "empty_waveform", "nonfinite_waveform", "duration_mismatch", "wav_decode_error", "unmapped_in_scope_label", "out_of_scope_command_flag_mismatch"}
    return 0 if (
        report["scan"]["row_counts_match_expected"]
        and report["source_snapshot"]["remote_revision_verification"].get("status") == "PASS"
        and report["gates"]["schema_crosswalk_complete"]
        and report["gates"]["audio_integrity_pass"]
        and report["gates"]["split_leakage_pass"]
        and report["gates"]["duplicate_label_collision_free"]
        and report["gates"]["out_of_scope_flags_consistent"]
        and not any(error["kind"] in critical_errors for error in report["scan"]["errors"])
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
