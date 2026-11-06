from __future__ import annotations

from scripts.local_ui_server import _safe_history, local_dispatch


def test_local_ui_blocks_external_channel_side_effects():
    result = local_dispatch("send_telegram_message", {
        "chat_id": "12345",
        "text": "local test",
        "confirm": True,
    })
    assert result == {
        "ok": False,
        "error": "external_channels_disabled_in_local_ui",
        "provider": "telegram",
        "network_called": False,
    }


def test_local_ui_history_is_bounded_to_conversation_roles():
    assert _safe_history([
        {"role": "system", "content": "do not include"},
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "do not include"},
        {"role": "assistant", "content": "hi"},
    ]) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
