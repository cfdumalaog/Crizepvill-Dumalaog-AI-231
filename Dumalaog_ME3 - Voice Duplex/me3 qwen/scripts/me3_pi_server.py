"""Run ME3 on a Pi with an ME2-style LAN UI and browser speaker output.

The Pi owns the microphone and the inference/tool loop.  The browser is only
the operator console and, when selected, a PCM speaker sink.  This keeps the
voice input path on the Pi while allowing a laptop on the same LAN to hear the
agent without installing PortAudio there.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from me3_bootstrap import configure_me3_path  # noqa: E402

configure_me3_path(PROJECT)

from me3.agent import StubBrain, LocalVoiceLoop  # noqa: E402
from me3.agent_status import LoopStatus, StatusServer, load_release_info  # noqa: E402
from me3.audio_io import FullDuplexAudio  # noqa: E402
from me3.brain import build_system_prompt, build_tool_brain  # noqa: E402
from me3.config import load_config  # noqa: E402
from me3.web_audio import BrowserAudioHub, BrowserOutputAudio  # noqa: E402

log = logging.getLogger("me3.pi_server")


class AgentController:
    def __init__(self, cfg: dict, brain, status: LoopStatus, hub: BrowserAudioHub,
                 input_device=None, output_mode: str = "browser",
                 output_device=None):
        self.cfg = cfg
        self.brain = brain
        self.status = status
        self.hub = hub
        self.input_device = input_device
        self.output_mode = output_mode
        self.output_device = output_device
        self.loop: LocalVoiceLoop | None = None
        self.thread: threading.Thread | None = None
        self.lock = threading.RLock()

    def _config(self) -> dict:
        cfg = dict(self.cfg)
        audio = dict(cfg.get("audio", {}))
        audio["input_device"] = self.input_device
        if self.output_mode == "pi":
            audio["output_device"] = self.output_device
        cfg["audio"] = audio
        return cfg

    def _run(self, loop: LocalVoiceLoop) -> None:
        try:
            loop.run()
        except Exception:  # noqa: BLE001 - status stays reachable on a bad device
            log.exception("ME3 agent loop stopped with an error")

    def start(self) -> None:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return
            cfg = self._config()
            if self.output_mode == "browser":
                audio = BrowserOutputAudio(cfg, self.hub)
                loop = LocalVoiceLoop(cfg, brain=self.brain, audio=audio,
                                      status=self.status)
            else:
                loop = LocalVoiceLoop(cfg, brain=self.brain,
                                      device=self.input_device,
                                      status=self.status)
            self.loop = loop
            self.thread = threading.Thread(target=self._run, args=(loop,),
                                           daemon=True, name="me3-agent-loop")
            self.thread.start()

    def stop(self) -> None:
        with self.lock:
            loop, thread = self.loop, self.thread
            self.loop = None
            self.thread = None
        if loop is not None:
            loop.stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)

    def select(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("selection must be a JSON object")
        output = str(payload.get("output", "browser")).strip()
        if output == "browser":
            output_mode, output_device = "browser", None
        else:
            output_mode, output_device = "pi", output
            if output_device in {"", "auto", "null", "none"}:
                output_device = None
        input_device = payload.get("input")
        if isinstance(input_device, str) and input_device.strip().lower() in {
                "", "auto", "default", "none"}:
            input_device = None
        self.stop()
        with self.lock:
            self.input_device = input_device
            self.output_mode = output_mode
            self.output_device = output_device
        self.start()
        return self.audio_info()

    def audio_info(self) -> dict:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            inputs, outputs = [], []
            for idx, dev in enumerate(devices):
                row = {
                    "id": idx,
                    "name": str(dev.get("name", ""))[:160],
                    "sample_rate": int(dev.get("default_samplerate") or 0),
                }
                if dev.get("max_input_channels", 0) > 0:
                    inputs.append({**row, "channels": int(dev.get("max_input_channels", 0))})
                if dev.get("max_output_channels", 0) > 0:
                    outputs.append({**row, "channels": int(dev.get("max_output_channels", 0))})
            with self.lock:
                selected_input = self.input_device
                selected_output = ("browser" if self.output_mode == "browser"
                                   else self.output_device)
                loop = self.loop
            echo_guard = {}
            capture = getattr(getattr(loop, "audio", None), "capture", None)
            if capture is None:
                capture = getattr(loop, "audio", None)
            if callable(getattr(capture, "echo_snapshot", None)):
                echo_guard = capture.echo_snapshot()
            return {
                "ok": True,
                "available": True,
                "inputs": inputs,
                "outputs": outputs,
                "selected_input": selected_input,
                "selected_output": selected_output,
                "browser_output": {
                    "id": "browser",
                    "name": "This browser / connected PC speaker",
                    "sample_rate": 16000,
                },
                "browser_audio": self.hub.snapshot(),
                "echo_guard": echo_guard,
            }
        except Exception as exc:  # noqa: BLE001 - UI can still show the error
            return {"ok": False, "available": False,
                    "error": type(exc).__name__, "message": str(exc)[:300],
                    "browser_audio": self.hub.snapshot(), "echo_guard": {}}


def build_brain(cfg: dict, selection: str):
    if selection == "stub":
        return StubBrain()
    selected = "qwen" if selection == "remote" else selection
    return build_tool_brain(
        cfg, selected, build_system_prompt(cfg["agent"]["system_prompt"]),
        on_tool=lambda kind, name, payload: log.info(
            "tool %s %s ok=%s", kind, name,
            payload.get("ok") if isinstance(payload, dict) else True),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0",
                        help="LAN bind address for the browser UI")
    parser.add_argument("--port", type=int, default=8126,
                        help="HTTP status/UI port")
    parser.add_argument("--audio-port", type=int, default=8127,
                        help="WebSocket PCM speaker port")
    parser.add_argument("--brain", choices=("auto", "remote", "qwen", "luna", "stub"),
                        default="auto")
    parser.add_argument("--variant", choices=("current", "qwen_instruct"),
                        default="qwen_instruct")
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output", choices=("browser", "pi"), default="browser")
    parser.add_argument("--output-device", default=None)
    args = parser.parse_args()
    os.environ["ME3_VARIANT"] = args.variant
    cfg = load_config()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    try:
        brain = build_brain(cfg, args.brain)
    except RuntimeError:
        if args.brain != "auto":
            raise
        brain = StubBrain()
    status = LoopStatus(
        release_info=load_release_info(PROJECT),
        brain_name=str(getattr(brain, "name", "brain")),
        variant=str(cfg.get("runtime_variant", "current")),
    )
    hub = BrowserAudioHub(args.host, args.audio_port)
    hub.start()
    controller = AgentController(
        cfg, brain, status, hub,
        input_device=args.input_device,
        output_mode=args.output,
        output_device=args.output_device,
    )
    http = StatusServer(
        status, html_dir=PROJECT / "web", get_state=lambda: controller.loop.state
        if controller.loop else "idle",
        get_stats=lambda: controller.loop.stats if controller.loop else type("S", (), {})(),
        port=args.port, host=args.host, project=PROJECT,
        audio_info=controller.audio_info, audio_select=controller.select,
    )
    if not http.start():
        hub.stop()
        return 2
    controller.start()
    print(f"ME3 LAN UI      : http://<PI-IP>:{http.port}/", flush=True)
    print(f"Browser audio   : ws://<PI-IP>:{hub.port}/audio", flush=True)
    print("Pi microphone   : selected through the UI or --input-device", flush=True)
    print("Output route    : browser (connected PC) or pi", flush=True)
    try:
        while True:
            threading.Event().wait(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        http.stop()
        hub.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
