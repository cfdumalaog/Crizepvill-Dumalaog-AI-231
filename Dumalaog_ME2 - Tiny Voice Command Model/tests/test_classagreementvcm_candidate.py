import numpy as np

from scripts.classagreementvcm.train_candidate import (
    LABELS,
    choose_gate,
    classify_row,
    derive_leaf,
    score_gate,
    top_level_label,
)


def test_dataset_slot_crosswalk_keeps_unsupported_distinct_from_true_oos():
    assert derive_leaf("LIGHT_ON", "", "Turn on the lights") == "LIGHT_ON"
    assert derive_leaf("TIMER", "1 minute", "Set a one-minute timer") == "TIMER_1m"
    assert derive_leaf("TIMER", "5 minutes", "Set a five-minute timer") is None
    assert classify_row({"command": "OUT_OF_SCOPE", "out_of_scope": 1}) == ("true_oos", None)
    assert classify_row({"command": "TIMER", "slot_value": "5 minutes", "variation": "", "out_of_scope": 0}) == ("unsupported_slot", None)


def test_validation_gate_limits_false_actions_and_keeps_clear_commands():
    targets = np.asarray([LABELS.index("LIGHT_ON")] * 10 + [-1] * 20, dtype=np.int64)
    probabilities = np.zeros((30, len(LABELS)), dtype=np.float32)
    probabilities[:, 1] = 0.005
    probabilities[:10, LABELS.index("LIGHT_ON")] = 0.95
    probabilities[10:, LABELS.index("LIGHT_OFF")] = 0.80
    probabilities[:, -1] += 1.0 - probabilities.sum(axis=1)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    gate = choose_gate(probabilities, targets)
    observed = score_gate(probabilities, targets, gate["confidence_threshold"], gate["margin_threshold"])
    assert observed["unknown_false_actions"] == 0
    assert observed["known_correct_actions"] == 10
    assert gate["selection_rule"].startswith("maximize correctly accepted")


def test_top_level_aggregation_keeps_reminders_as_one_intent():
    assert top_level_label("CREATE_REMINDER_DRINK_WATER") == "CREATE_REMINDER"
    assert top_level_label("TEMPERATURE_22") == "TEMPERATURE"
    assert top_level_label("PLAY_MUSIC") == "PLAY_MUSIC"
