import record_dataset


def test_clear_take_resets_audio_for_next_recording():
    assert record_dataset.clear_take() is None
