import base64
from http.client import HTTPConnection
import io
import threading

import numpy as np
import soundfile as sf

import remote_vcm_eval as remote


class FakePredictor:
    model_name = "fake two-stage model"
    labels = ["LIGHT_ON", "TIME"]
    wake_threshold = 0.9

    def predict(self, audio, mode="command"):
        assert audio.shape == (remote.SAMPLES,)
        label = "WAKE_WORD" if mode == "wake" else "TIME"
        return {
            "label": label, "confidence": 0.95, "wake_probability": 0.95,
            "latency_ms": 0.3, "top3": {label: 0.95},
        }


def _auth(username, password):
    value = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {value}"}


def _wav():
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(remote.SAMPLES, dtype=np.float32), remote.SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _server(monkeypatch):
    username, password = "reviewer", "test-password-long-enough"
    salt = b"0123456789abcdef"
    digest = remote.hashlib.pbkdf2_hmac("sha256", password.encode(), salt, remote.AUTH_ITERATIONS, dklen=32)
    monkeypatch.setenv("VCM_REMOTE_USER", username)
    monkeypatch.setenv("VCM_REMOTE_SALT", salt.hex())
    monkeypatch.setenv("VCM_REMOTE_PASSWORD_HASH", digest.hex())
    server = remote.create_server(FakePredictor(), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, username, password


def test_remote_evaluator_requires_auth_and_exposes_only_evaluation_routes(monkeypatch):
    server, thread, username, password = _server(monkeypatch)
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port)
        conn.request("GET", "/")
        response = conn.getresponse()
        assert response.status == 401
        assert response.getheader("WWW-Authenticate")
        response.read()

        conn.request("GET", "/", headers=_auth(username, password))
        response = conn.getresponse()
        page = response.read().decode()
        assert response.status == 200
        assert "ME2 - VCM on Raspberry Pi 5" in page
        assert "/api/evaluate" in page
        assert "/api/wake" not in page
        assert "five temporal placements" in page
        assert "script-src 'unsafe-inline'" in response.getheader("Content-Security-Policy")

        conn.request("GET", "/health", headers=_auth(username, password))
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("Cache-Control") == "no-store"
        assert b'"ok": true' in response.read()
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_remote_evaluator_runs_only_explicit_wake_and_command_prediction(monkeypatch):
    server, thread, username, password = _server(monkeypatch)
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port)
        for mode, expected in (("wake", "WAKE_WORD"), ("command", "TIME")):
            conn.request(
                "POST", f"/api/evaluate?mode={mode}", body=_wav(),
                headers={**_auth(username, password), "Content-Type": "audio/wav"},
            )
            response = conn.getresponse()
            body = response.read()
            assert response.status == 200, body
            result = remote.json.loads(body)
            assert result["label"] == expected
            assert result["mode"] == mode
            if mode == "wake":
                assert result["wake_threshold"] == 0.9
                assert result["window_count"] == 5
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_remote_evaluator_rejects_bad_auth_content_and_non_loopback(monkeypatch):
    server, thread, username, password = _server(monkeypatch)
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port)
        conn.request("POST", "/api/evaluate?mode=command", body=_wav(), headers={"Content-Type": "audio/wav"})
        response = conn.getresponse()
        assert response.status == 401
        response.read()
        conn.request(
            "POST", "/api/evaluate?mode=command", body=b"nope",
            headers={**_auth(username, password), "Content-Type": "application/octet-stream"},
        )
        response = conn.getresponse()
        assert response.status == 415
        response.read()
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    with __import__("pytest").raises(ValueError, match="loopback"):
        remote.create_server(FakePredictor(), host="0.0.0.0", port=0)
