from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from me3.agent import TurnStats
from me3.agent_status import LoopStatus, StatusServer, load_release_info

PROJECT = Path(__file__).resolve().parents[1]


def _fake_stats() -> TurnStats:
    s = TurnStats()
    s.n_turns = 2
    s.n_barge_in = 1
    s.n_errors = 0
    s.asr_ms = [120.0, 140.0]
    s.llm_ms = [900.0]
    s.tts_ms = [400.0]
    s.e2e_ms = [1500.0]
    return s


def test_snapshot_shape_after_simulated_events():
    st = LoopStatus(release_info={"release_id": "me3-test"},
                    brain_name="stub", variant="current")
    st.report_started()
    st.report_transcript("hello me3")
    st.report_reply("hi there", "reply")
    st.report_tool("tool_started", "system_info")
    st.report_tool("tool_finished", "system_info", ok=True)
    st.report_barge_in()
    st.report_models(vad_ms=10.0, asr_ms=100.0, asr_model="small",
                     tts_ms=50.0, tts_voice="en_GB-alba-medium", tts_hz=22050)

    snap = st.snapshot("listening", _fake_stats())
    assert snap["ok"] is True
    assert snap["loop"] is True
    assert snap["state"] == "listening"
    assert snap["brain"] == "stub"
    assert snap["variant"] == "current"
    assert snap["last_transcript"]["text"] == "hello me3"
    assert snap["last_reply"]["text"] == "hi there"
    assert snap["last_reply"]["kind"] == "reply"
    assert snap["last_barge_in_at"] is not None
    assert snap["stats"]["n_turns"] == 2
    assert snap["stats"]["n_barge_in"] == 1
    assert snap["stats"]["latency"]["asr_ms"]["last"] == 140.0
    assert snap["stats"]["latency"]["asr_ms"]["median"] == 130.0
    assert snap["tools"]["active"] is None
    assert [e["name"] for e in snap["tools"]["recent"]] == [
        "system_info", "system_info"]
    assert snap["tools"]["recent"][-1]["ok"] is True
    assert snap["models"]["asr_model"] == "small"
    assert snap["release"]["release_id"] == "me3-test"
    assert snap["uptime_s"] is not None

    st.report_stopped()
    snap2 = st.snapshot("idle", _fake_stats())
    assert snap2["loop"] is False
    assert snap2["stopped_at"] is not None


def test_tool_active_tracking_and_bounded_history():
    st = LoopStatus()
    snap = st.snapshot("thinking", TurnStats())
    assert snap["tools"]["active"] is None
    assert snap["tools"]["recent"] == []

    st.report_tool("tool_started", "run_command")
    assert st.snapshot("thinking", TurnStats())["tools"]["active"] == \
        "run_command"

    for i in range(12):  # history must stay bounded to 8
        st.report_tool("tool_finished", f"tool_{i}", ok=True)
    snap = st.snapshot("thinking", TurnStats())
    assert len(snap["tools"]["recent"]) == 8

    # a finish for a different tool must not clear the active name
    st.report_tool("tool_started", "fetch_url")
    st.report_tool("tool_finished", "something_else", ok=False)
    snap = st.snapshot("thinking", TurnStats())
    assert snap["tools"]["active"] == "fetch_url"
    assert snap["tools"]["recent"][-1]["ok"] is False


def test_status_server_serves_state_and_pages():
    st = LoopStatus(release_info={"release_id": "me3-test"})
    st.report_started()
    st.report_transcript("t")
    server = StatusServer(
        st, html_dir=PROJECT / "web",
        get_state=lambda: "speaking", get_stats=lambda: _fake_stats(),
        port=0,
    )
    assert server.start() is True
    try:
        base = f"http://127.0.0.1:{server.port}"
        with urllib.request.urlopen(base + "/api/state", timeout=5) as r:
            assert r.status == 200
            assert r.headers.get("Access-Control-Allow-Origin") == "*"
            data = json.loads(r.read().decode("utf-8"))
            assert data["state"] == "speaking"
            assert data["loop"] is True
            assert data["release"]["release_id"] == "me3-test"
        with urllib.request.urlopen(base + "/api/health", timeout=5) as r:
            assert json.loads(r.read())["ok"] is True
        with urllib.request.urlopen(base + "/", timeout=5) as r:
            assert r.status == 200
            html = r.read().decode("utf-8")
            assert "orbCanvas" in html and "ME3" in html
        with urllib.request.urlopen(base + "/roadmap", timeout=5) as r:
            assert r.status == 200
            assert "left to do" in r.read().decode("utf-8")
    finally:
        server.stop()
    assert server._httpd is None


def test_load_release_info_reads_deployment_records():
    info = load_release_info(PROJECT)
    assert str(info["release"].get("release_id", "")).startswith("me3-")
    # The status is the result of the most recent gate, so a failed prior
    # attempt must not make the next gate unrecoverable.  The server only
    # needs the receipt to expose a bounded, well-formed status block.
    assert info["acceptance"].get("status") in {"pass", "fail", None}
    assert 0 <= info["acceptance"]["checks_passed"] <= \
        info["acceptance"]["checks_total"]
    assert "status" in info["pi"]
def test_status_server_release_block_follows_deployment(tmp_path):
    dep = tmp_path / "deployment"
    dep.mkdir()
    rel = dep / "latest_release.json"
    rel.write_text(json.dumps({"release_id": "me3-aaa",
                               "status": "prepared"}),
                   encoding="utf-8")

    st = LoopStatus()
    server = StatusServer(
        st, html_dir=PROJECT / "web",
        get_state=lambda: "idle", get_stats=lambda: _fake_stats(),
        port=0, project=tmp_path,
    )
    assert server.start() is True
    try:
        base = f"http://127.0.0.1:{server.port}"
        with urllib.request.urlopen(base + "/api/state", timeout=5) as r:
            first = json.loads(r.read().decode("utf-8"))
        assert first["release"]["release"]["release_id"] == "me3-aaa"

        # The acceptance gate rewrites the record while the server runs.
        rel.write_text(json.dumps({"release_id": "me3-bbb",
                                   "status": "prepared"}),
                       encoding="utf-8")
        rel.touch()
        with urllib.request.urlopen(base + "/api/state", timeout=5) as r:
            second = json.loads(r.read().decode("utf-8"))
        assert second["release"]["release"]["release_id"] == "me3-bbb"
    finally:
        server.stop()
