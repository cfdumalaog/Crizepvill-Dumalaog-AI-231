import pytest

from vcm_app import parse_volume_percent


@pytest.mark.parametrize(("value", "expected"), [(0, 0), (35, 35), (99.6, 100), (100, 100)])
def test_volume_request_accepts_and_rounds_valid_percentages(value, expected):
    assert parse_volume_percent({"percent": value}) == expected


@pytest.mark.parametrize("payload", [None, [], {"percent": True}, {"percent": "35"}, {"percent": float("nan")}, {"percent": float("inf")}, {"percent": -1}, {"percent": 101}, {}])
def test_volume_request_rejects_invalid_values(payload):
    with pytest.raises(ValueError):
        parse_volume_percent(payload)
