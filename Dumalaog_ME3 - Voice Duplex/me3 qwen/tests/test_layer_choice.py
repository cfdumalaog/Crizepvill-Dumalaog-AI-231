from __future__ import annotations

import json
import urllib.request

from me3 import layer_choice, secure_config
from me3.agent_status import LoopStatus, StatusServer


def test_layer_choices_are_validated_and_private(monkeypatch, tmp_path):
    monkeypatch.setenv("ME3_CREDENTIALS_FILE", str(tmp_path / "credentials.json"))
    assert layer_choice.choose({"asr": "deepgram", "tts": "piper", "vad": "energy"})["ok"]
    assert layer_choice.selected() == {"asr": "deepgram", "tts": "piper", "vad": "energy"}
    assert layer_choice.choose({"asr": "not-a-provider"})["error"] == "unsupported_asr_provider"
    path = secure_config._path().with_name("layers.json")
    assert json.loads(path.read_text(encoding="utf-8"))["asr"] == "deepgram"
    assert "api" not in path.read_text(encoding="utf-8").lower()


def test_layer_routes_return_options_and_save(monkeypatch, tmp_path):
    monkeypatch.setenv("ME3_CREDENTIALS_FILE", str(tmp_path / "credentials.json"))
    server = StatusServer(
        LoopStatus(), html_dir=tmp_path, get_state=lambda: "idle",
        get_stats=lambda: object(), host="127.0.0.1", port=0,
        layer_status=layer_choice.status, layer_select=layer_choice.choose,
    )
    assert server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with urllib.request.urlopen(base + "/api/layers", timeout=3) as response:
            payload = json.loads(response.read())
        assert payload["ok"] and "deepgram" in payload["options"]["asr"]
        assert "custom_vllm" not in payload["options"]["asr"]
        assert payload["layers"]["brain"] == "qwen"
        assert "qwen" in payload["options"]["brain"]
        request = urllib.request.Request(
            base + "/api/layers",
            data=json.dumps({"brain": "qwen", "asr": "faster_whisper", "tts": "piper", "vad": "silero"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=3) as response:
            saved = json.loads(response.read())
        assert saved["ok"] and saved["layers"]["asr"] == "faster_whisper"
    finally:
        server.stop()
