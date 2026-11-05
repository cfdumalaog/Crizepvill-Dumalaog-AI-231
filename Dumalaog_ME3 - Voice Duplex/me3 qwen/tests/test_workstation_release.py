from __future__ import annotations

from scripts.deploy_to_pi import validate_workstation_acceptance


def _release() -> dict:
    return {
        "release_id": "me3-test",
        "source_fingerprint": "source-test",
        "zip_sha256": "zip-test",
    }


def _acceptance(**overrides) -> dict:
    value = {
        "acceptance_id": "workstation-test",
        "status": "pass",
        "pi_deploy_allowed": True,
        "release_id": "me3-test",
        "source_fingerprint": "source-test",
        "zip_sha256": "zip-test",
        "pi_contacted": False,
        "external_side_effects": False,
    }
    value.update(overrides)
    return value


def test_matching_workstation_acceptance_allows_promotion():
    assert validate_workstation_acceptance(_release(), _acceptance()) is None


def test_stale_acceptance_is_rejected():
    reason = validate_workstation_acceptance(
        _release(), _acceptance(release_id="me3-old"))
    assert reason and "release_id" in reason


def test_acceptance_with_side_effects_is_rejected():
    reason = validate_workstation_acceptance(
        _release(), _acceptance(external_side_effects=True))
    assert reason and "side effects" in reason
