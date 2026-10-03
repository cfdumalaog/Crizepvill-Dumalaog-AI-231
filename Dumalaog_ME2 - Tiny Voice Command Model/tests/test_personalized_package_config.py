from pathlib import Path

import pytest

from deployment.package_personalized_rpi import patch_intent_gate_config


def test_release_thresholds_are_written_to_packaged_config_only():
    source = Path("tinyvcm_model/config.py").read_bytes()
    packaged = patch_intent_gate_config(source, 0.76, 0.0)
    assert b"INTENT_CONFIDENCE_THRESHOLD = 0.76" in packaged
    assert b"INTENT_MARGIN_THRESHOLD = 0" in packaged
    assert b"INTENT_CONFIDENCE_THRESHOLD = 0.68" in source
    assert b"INTENT_MARGIN_THRESHOLD = 0.15" in source


def test_release_threshold_patch_refuses_ambiguous_config():
    with pytest.raises(ValueError, match="exactly one INTENT_CONFIDENCE_THRESHOLD"):
        patch_intent_gate_config(
            b"INTENT_CONFIDENCE_THRESHOLD = 0.68\nINTENT_CONFIDENCE_THRESHOLD = 0.68\nINTENT_MARGIN_THRESHOLD = 0.15\n",
            0.76,
            0.0,
        )
