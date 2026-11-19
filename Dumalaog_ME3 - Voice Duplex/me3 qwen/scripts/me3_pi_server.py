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
import re
import shutil
import subprocess
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
from me3.audio_io import AudioDeviceError, FullDuplexAudio  # noqa: E402
from me3.brain import build_system_prompt, build_tool_brain  # noqa: E402
from me3.config import load_config  # noqa: E402
from me3 import (google_calendar, me2_commands, scheduled_actions,
                 spotify_player, deepgram_asr, github_pages, provider_choice,
                 setup_profiles, permission_policy)  # noqa: E402
from me3 import layer_choice  # noqa: E402
from me3.web_audio import BrowserAudioHub, BrowserOutputAudio  # noqa: E402

log = logging.getLogger("me3.pi_server")


class AgentController:
    MIC_RETRY_S = 10.0

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
        # Browser output is normally the laptop speaker while capture remains
        # on the Pi.  That is an acoustic, cross-device path: a VAD-positive
        # speaker reflection must not immediately become a barge-in before the
        # protected-playback echo/keyword gate can inspect it.  Keep automatic
        # barge-in available through the UI toggle, but make the safe mode the
        # default for the Pi/browser transport. Direct Pi speaker output keeps
        # the configured automatic mode.
        configured_mode = (
            "automatic" if bool(cfg.get("agent", {}).get("barge_in", True))
            else "keyword_only"
        )
        self.duplex_mode = (
            "keyword_only" if output_mode == "browser" else configured_mode
        )
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
        while True:
            try:
                loop.run()
                return
            except AudioDeviceError as exc:
                # USB microphones can enumerate a few seconds after boot or
                # reconnect. Keep the UI and model routes alive and retry the
                # same capture loop, so plugging the Fifine back in does not
                # require a service restart.
                log.exception("ME3 microphone unavailable; retrying capture")
                self.status.report_error("AUDIO", exc)
                self.status.report_state_reason(
                    f"microphone unavailable; retrying in {self.MIC_RETRY_S:g} seconds")
                stop_event = getattr(loop, "_stop_event", None)
                if stop_event is not None and stop_event.wait(self.MIC_RETRY_S):
                    return
                with self.lock:
                    if self.loop is not loop:
                        return
            except Exception as exc:  # noqa: BLE001 - UI remains reachable
                self.status.report_error("LOOP", exc)
                log.exception("ME3 agent loop stopped with an error")
                return

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
            # The UI mode is runtime state and survives loop/audio restarts,
            # but does not rewrite the protected YAML configuration.
            loop.barge_in = self.duplex_mode == "automatic"
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

    def restart(self) -> dict:
        """Restart only the agent loop while keeping UI/audio online."""
        self.stop()
        self.start()
        with self.lock:
            running = bool(self.thread and self.thread.is_alive())
        return {"loop_running": running,
                "variant": self.cfg.get("runtime_variant", "qwen_instruct")}

    def announce_scheduled(self, item: dict) -> None:
        """Queue a durable reminder/alarm/timer on the live Qwen voice loop."""
        from me3.scheduled_actions import due_announcement
        with self.lock:
            loop = self.loop
        if loop is not None:
            loop.queue_announcement(due_announcement(item))

    def submit_chat(self, text: str) -> dict:
        with self.lock:
            loop = self.loop
            running = bool(self.thread and self.thread.is_alive())
        if not running or loop is None:
            return {"ok": False, "error": "voice_loop_unavailable"}
        return loop.queue_text_turn(text)

    def duplex_mode_info(self) -> dict:
        with self.lock:
            mode = self.duplex_mode
            thread = self.thread
        return {
            "ok": True,
            "mode": mode,
            "barge_in": mode == "automatic",
            "loop_running": bool(thread and thread.is_alive()),
            "description": (
                "Genuine speech interrupts TTS; echo guard rejects ME3 output."
                if mode == "automatic" else
                "Only stop processing, skip, or steer interrupts TTS."
            ),
        }

    def set_duplex_mode(self, payload: dict) -> dict:
        mode = str(payload.get("mode", "")).strip().lower()
        aliases = {
            "auto": "automatic",
            "automatic_barge_in": "automatic",
            "full_duplex": "automatic",
            "keyword": "keyword_only",
            "protected": "keyword_only",
            "protected_playback": "keyword_only",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"automatic", "keyword_only"}:
            raise ValueError("mode must be automatic or keyword_only")
        with self.lock:
            self.duplex_mode = mode
            loop = self.loop
            if loop is not None:
                loop.barge_in = mode == "automatic"
                self.status.report_state_reason(
                    "automatic barge-in enabled" if mode == "automatic"
                    else "keyword-only protected playback enabled"
                )
        return self.duplex_mode_info()

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
            if not inputs:
                # Some Pi PortAudio builds expose a USB microphone through
                # arecord but report zero input channels.  Surface those ALSA
                # devices so the UI still offers an explicit selectable mic.
                try:
                    listed = subprocess.run(
                        ["arecord", "-l"], capture_output=True, text=True,
                        timeout=5, check=False,
                    )
                    for card, _card_label, name, device in re.findall(
                            r"(?m)^card\s+(\d+):\s+([^\[]+)\[([^\]]+)\].*?device\s+(\d+):",
                            listed.stdout):
                        inputs.append({
                            "id": f"hw:{card},{device}",
                            "name": f"{name.strip()} (ALSA hw:{card},{device})",
                            "sample_rate": 16000,
                            "channels": 1,
                        })
                except (OSError, subprocess.SubprocessError):
                    pass
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


def schedule_systemd_restart(service_name: str = "me3-agent.service") -> dict:
    """Schedule a restart of the canonical Pi user service.

    The HTTP response must leave the process before systemd stops it, so the
    command is launched from a short-lived daemon timer. This endpoint is
    wired only by the Pi server; workstation status servers do not expose it.
    """
    if not shutil.which("systemctl"):
        raise RuntimeError("systemctl is unavailable on this host")

    def _restart() -> None:
        try:
            subprocess.Popen(
                ["systemctl", "--user", "restart", service_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:  # noqa: BLE001 - the service will expose its own logs
            log.exception("could not launch systemd restart for %s", service_name)

    timer = threading.Timer(0.35, _restart)
    timer.daemon = True
    timer.start()
    return {"service": service_name, "restart_scheduled": True}


def choose_layers(payload: dict) -> dict:
    """Save layer choices and restart the Pi service when systemd is present."""
    result = layer_choice.choose(payload)
    if not result.get("ok"):
        return result
    if shutil.which("systemctl"):
        result.update(schedule_systemd_restart())
    else:
        result["restart_scheduled"] = False
        result["restart_note"] = "restart the ME3 service to load the new layers"
    return result


def activate_setup_profile(profile_id: str) -> dict:
    """Activate an adaptive provider profile and reload the Pi service."""
    result = setup_profiles.activate_profile(profile_id)
    if not result.get("ok"):
        return result
    profile = result.get("profile") or {}
    role = str(profile.get("role") or "").lower()
    if role == "llm":
        selected = provider_choice.choose("custom_vllm")
    else:
        selected = layer_choice.choose({role: "custom_vllm"})
    if not selected.get("ok"):
        return selected
    result["activation"] = selected
    result.update(schedule_systemd_restart())
    return result


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
    parser.add_argument("--brain", choices=("auto", "remote", "qwen", "openai", "luna", "stub"),
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
        brain = build_brain(cfg, cfg["runtime_provider"] if args.brain == "auto" else args.brain)
    except RuntimeError:
        if args.brain != "auto":
            raise
        brain = StubBrain()
    status = LoopStatus(
        release_info=load_release_info(PROJECT),
        brain_name=str(getattr(brain, "name", "brain")),
        variant=str(cfg.get("runtime_variant", "qwen_instruct")),
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
        restart_agent=controller.restart,
        restart_service=schedule_systemd_restart,
         duplex_mode_get=controller.duplex_mode_info,
         duplex_mode_set=controller.set_duplex_mode,
        demo_light_get=me2_commands.simulated_light_state,
        demo_light_control=lambda payload: me2_commands.control_lights(**payload),
        notifications_pending=me2_commands.pending_notifications,
        notification_ack=me2_commands.acknowledge_notification,
        spotify_status=spotify_player.health,
        spotify_connect=lambda: _spotify_connect_result(),
        spotify_callback=spotify_player.complete_oauth,
        spotify_client_id_set=spotify_player.configure_client_id,
        google_status=google_calendar.health,
        google_connect=lambda: _google_connect_result(),
        google_callback=google_calendar.complete_oauth,
        google_client_set=google_calendar.configure_client,
        telegram_status=me2_commands.telegram_status,
        telegram_token_set=me2_commands.save_telegram_bot_token,
        telegram_connect_target=me2_commands.connect_telegram_target,
        deepgram_status=deepgram_asr.health,
        deepgram_key_set=deepgram_asr.configure_api_key,
        provider_status=provider_choice.status,
        openai_key_set=provider_choice.save_openai_key,
        provider_select=lambda payload: provider_choice.choose(payload.get("provider")),
        layer_status=layer_choice.status,
        layer_select=choose_layers,
        permission_status=permission_policy.status,
        permission_set=permission_policy.set_mode,
        chat_submit=controller.submit_chat,
        github_status=github_pages.health,
        github_token_set=github_pages.configure_token,
        schedules_get=me2_commands.clock_schedule_snapshot,
        schedule_create=me2_commands.create_clock_action,
        schedule_cancel=me2_commands.cancel_clock_action,
        setup_plan=setup_profiles.setup_plan,
        setup_profile_save=setup_profiles.save_profile,
        setup_profile_activate=activate_setup_profile,
    )
    if not http.start():
        hub.stop()
        return 2
    controller.start()
    scheduled_runner = me2_commands.configure_schedule_delivery(
        controller.announce_scheduled)
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
        scheduled_runner.stop()
        me2_commands.set_device_action_dispatcher(None)
        controller.stop()
        http.stop()
        hub.stop()
    return 0


def _spotify_connect_result() -> dict:
    url, error = spotify_player.begin_oauth()
    if error:
        return error
    return {"ok": True, "url": url}


def _google_connect_result() -> dict:
    url, error = google_calendar.begin_oauth()
    if error:
        return error
    return {"ok": True, "url": url}


if __name__ == "__main__":
    raise SystemExit(main())
