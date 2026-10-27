from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_browser_surface_has_automatic_voice_listener_without_manual_button():
    html = (PROJECT / "web" / "ui.html").read_text(encoding="utf-8")
    assert 'id="listen"' not in html
    assert "continuous=true" in html
    assert "getUserMedia" in html
    assert "sendText(utterance,'voice-auto')" in html
    assert "Auto-listener starting" in html
