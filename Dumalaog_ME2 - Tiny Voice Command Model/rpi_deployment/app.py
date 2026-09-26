"""
Interactive Web Application for Tiny Voice Command Model (VCM).
Features Wake-Word ("Hi" / "Hello") Activation, Live Microphone Audio Streaming,
Realistic Virtual OLED (SSD1306) Display, RGB PWM LED Simulation,
and Smart Device State Dashboard.
"""

import sys
import os
import time
import io
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# Setup import path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import threading
import sounddevice as sd

from src.config import (
    SAMPLE_RATE,
    NUM_SAMPLES,
    STREAM_HOP_SEC,
    MODELS_DIR,
    DATA_DIR,
    ASSETS_DIR,
    WAKE_WORD_CLASS,
    WAKE_TIMEOUT_SEC,
    COMMAND_CLASSES
)
from src.model import build_model
from src.hal import VirtualHardware
from src.actions import SmartDeviceController
from src.engine import StreamingVCMEngine
from src.mic_stream import RobustMicrophoneStreamer

# ---------------------------------------------------------------------------
# Global Assistant System State
# ---------------------------------------------------------------------------
class SmartAssistantApp:
    def __init__(self):
        self.hw = VirtualHardware(verbose=False)
        self.controller = SmartDeviceController(hardware=self.hw)
        
        # Load BC-ResNet-1 model
        checkpoint_path = MODELS_DIR / "bc_resnet_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}.")

        self.model = build_model("bc_resnet")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.eval()

        self.engine = StreamingVCMEngine(
            model=self.model,
            controller=self.controller,
            confidence_threshold=0.50,
            wake_confidence=0.48,
            wake_timeout_sec=WAKE_TIMEOUT_SEC,
            holdoff_seconds=0.35,
            device="cpu",
            require_wake_word=True
        )

        self.event_history = []
        self.last_mel_spec = None
        self.last_top3 = {}
        self.mel_updated = False
        self.top3_updated = False
        self.last_audio_feedback = None
        self.last_status_message = "Assistant Ready. Standing by for 'Hi Dandan' or 'Hello Dandan'."
        self.last_audio_to_play = None

        # Continuous background hardware microphone stream
        self.mic_streaming = True
        self.mic_device_name = "Detecting microphone..."
        self._start_hardware_mic()

    def _audio_callback(self, chunk_16k: np.ndarray):
        if not self.mic_streaming:
            return
        event = self.engine.feed_audio_chunk(chunk_16k)
        if event:
            evt_type = event.get("event")
            cmd = event.get("command", "")
            conf = event.get("confidence", 0.0)
            lat = event.get("latency_ms", 0.0)
            msg = event.get("action_message", "")
            self.last_top3 = event.get("top3", {})
            self.top3_updated = True
            if self.engine.last_mel_spec is not None:
                self.last_mel_spec = self.engine.last_mel_spec
                self.mel_updated = True

            if evt_type == "COMPOUND_COMMAND_EXECUTED":
                self.last_status_message = f"⚡ Compound One-Shot: {msg} (conf: {conf*100:.1f}%, latency: {lat:.1f}ms)"
                self.log_event("COMPOUND_CMD", f"[{cmd}] -> {msg}", conf, lat)
                if cmd == "play_music":
                    self.last_audio_to_play = str(ASSETS_DIR / "sample_music.wav")
                else:
                    resp_wav = ASSETS_DIR / "voice_responses" / f"{cmd}.wav"
                    self.last_audio_to_play = str(resp_wav) if resp_wav.exists() else None
            elif evt_type == "WAKE_WORD_DETECTED":
                self.last_status_message = f"👋 Wake Word Recognized! ('{cmd}', conf: {conf*100:.1f}%, latency: {lat:.1f}ms). Assistant is now LISTENING!"
                self.log_event("WAKE_DETECTED", f"Woke up from standby ('{cmd}')", conf, lat)
                self.last_audio_to_play = str(ASSETS_DIR / "voice_responses" / "wake_word.wav")
            elif evt_type == "COMMAND_EXECUTED":
                trigger = event.get("trigger", "KWS")
                self.last_status_message = f"✅ Command Executed ({trigger}): {msg} (conf: {conf*100:.1f}%, latency: {lat:.1f}ms)"
                self.log_event("COMMAND", f"[{cmd}] -> {msg}", conf, lat)
                if cmd == "play_music":
                    self.last_audio_to_play = str(ASSETS_DIR / "sample_music.wav")
                else:
                    resp_wav = ASSETS_DIR / "voice_responses" / f"{cmd}.wav"
                    self.last_audio_to_play = str(resp_wav) if resp_wav.exists() else None
            elif evt_type == "TIMEOUT":
                self.last_status_message = "⏱️ Listening window expired -> Assistant returned to STANDBY."
                self.log_event("TIMEOUT", "Listening window expired", 1.0, 0.0)
                self.last_audio_to_play = str(ASSETS_DIR / "voice_responses" / "timeout.wav")

    def _start_hardware_mic(self):
        """Continuously streams audio from host microphone with ZERO button clicks."""
        try:
            self.streamer = RobustMicrophoneStreamer(
                target_sample_rate=SAMPLE_RATE,
                hop_seconds=STREAM_HOP_SEC,
                callback=self._audio_callback,
                apply_agc=True
            )
            self.mic_device_name = f"{self.streamer.device_name} ({self.streamer.native_sample_rate}Hz via {self.streamer.host_api_name})"
            self.streamer.start()

            def _timeout_poll():
                while self.mic_streaming:
                    to_evt = self.engine.check_timeout()
                    if to_evt:
                        self.last_status_message = "⏱️ Listening window expired -> Assistant returned to STANDBY."
                        self.log_event("TIMEOUT", "Listening window expired", 1.0, 0.0)
                        self.last_audio_to_play = str(ASSETS_DIR / "voice_responses" / "timeout.wav")
                    time.sleep(0.15)

            t = threading.Thread(target=_timeout_poll, daemon=True)
            t.start()
        except Exception as e:
            self.mic_device_name = f"Hardware Mic Stream Error: {e}"

    def reset_state(self):
        """Resets assistant to initial standby state."""
        self.engine.state = StreamingVCMEngine.STATE_STANDBY
        self.engine.ring_buffer.reset()
        self.controller.reset_to_standby()
        self.event_history.clear()
        self.last_status_message = "Assistant reset to STANDBY (Awaiting 'Hi Dandan' / 'Hello Dandan')"
        self.log_event("SYSTEM", "Assistant reset to STANDBY (Awaiting 'Hi' / 'Hello')", 1.0, 0.0)

    def log_event(self, event_type: str, details: str, conf: float, latency: float):
        ts = datetime.now().strftime("%H:%M:%S")
        self.event_history.insert(0, {
            "Time": ts,
            "Event": event_type,
            "Details": details,
            "Confidence": f"{conf * 100:.1f}%" if conf > 0 else "-",
            "Latency": f"{latency:.2f} ms" if latency > 0 else "-"
        })
        if len(self.event_history) > 30:
            self.event_history.pop()


# Instantiate global app state
APP = SmartAssistantApp()
APP.reset_state()


# ---------------------------------------------------------------------------
# UI Helpers & HTML Renderers
# ---------------------------------------------------------------------------
def render_oled_html(display_lines: List[str]) -> str:
    """Renders retro blue-black SSD1306 128x64 OLED display mockup."""
    l1 = display_lines[0] if len(display_lines) > 0 else ""
    l2 = display_lines[1] if len(display_lines) > 1 else ""
    l3 = display_lines[2] if len(display_lines) > 2 else ""
    l4 = display_lines[3] if len(display_lines) > 3 else ""

    html = f"""
    <div style="
        background-color: #050b14;
        border: 4px solid #1a2333;
        border-radius: 12px;
        padding: 16px 20px;
        font-family: 'Courier New', Courier, monospace;
        color: #64d2ff;
        text-shadow: 0 0 8px rgba(100, 210, 255, 0.7);
        box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.9), 0 4px 15px rgba(0, 0, 0, 0.5);
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        user-select: none;
    ">
        <div style="font-size: 13px; font-weight: bold; border-bottom: 1px dashed #204060; padding-bottom: 4px; display: flex; justify-content: space-between;">
            <span>[I2C SSD1306 OLED 128x64]</span>
            <span style="color: #4cd964;">VCM ACTIVE</span>
        </div>
        <div style="font-size: 15px; font-weight: bold; letter-spacing: 0.5px; color: #ffffff; text-shadow: 0 0 8px rgba(255,255,255,0.8);">{l1}</div>
        <div style="font-size: 14px; color: #64d2ff;">{l2}</div>
        <div style="font-size: 12px; color: #8ab4f8; display: flex; justify-content: space-between;">
            <span>{l3}</span>
            <span style="color: #ffcc00;">{l4}</span>
        </div>
    </div>
    """
    return html


def render_led_html(led_state: Dict[str, Any]) -> str:
    """Renders realistic glowing RGB PWM LED bulb."""
    is_on = led_state.get("on", False)
    r = int(led_state.get("r", 0.0) * 255)
    g = int(led_state.get("g", 0.0) * 255)
    b = int(led_state.get("b", 0.0) * 255)
    brightness = led_state.get("brightness", 0.0)

    if not is_on or brightness < 0.01:
        color_rgba = "rgba(40, 45, 55, 0.4)"
        glow_css = "box-shadow: inset 0 0 10px rgba(0,0,0,0.8);"
        label = "LED OFF"
    else:
        color_rgba = f"rgba({r}, {g}, {b}, {max(0.6, brightness)})"
        glow_css = f"box-shadow: 0 0 {int(20 + 35 * brightness)}px rgba({r}, {g}, {b}, 0.9), inset 0 0 12px rgba(255,255,255,0.6);"
        label = f"RGB: ({r}, {g}, {b}) - {int(brightness * 100)}%"

    html = f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 16px;
        background: #111827;
        padding: 14px 20px;
        border-radius: 12px;
        border: 1px solid #1f2937;
    ">
        <div style="
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: {color_rgba};
            border: 2px solid rgba(255, 255, 255, 0.3);
            {glow_css}
            transition: all 0.3s ease;
        "></div>
        <div>
            <div style="font-size: 13px; color: #9ca3af; text-transform: uppercase; font-weight: 600;">RGB PWM LED (GPIO 17, 27, 22)</div>
            <div style="font-size: 15px; font-weight: bold; color: #f3f4f6; margin-top: 2px;">{label}</div>
        </div>
    </div>
    """
    return html


def render_state_badge(state: str) -> str:
    """Renders visual status pill badge for assistant state."""
    if state == StreamingVCMEngine.STATE_STANDBY:
        badge_color = "#374151"
        text_color = "#9ca3af"
        icon = "💤"
        title = "STANDBY / SLEEPING"
        desc = "Always listening. Say <b>'Hi Dandan'</b> or <b>'Hello Dandan'</b> to wake up."
    elif state == StreamingVCMEngine.STATE_LISTENING:
        badge_color = "#0369a1"
        text_color = "#38bdf8"
        icon = "👂"
        title = "LISTENING FOR COMMAND"
        desc = "Assistant is awake! Say a command (e.g. <b>'play music'</b>, <b>'turn on lights'</b>)."
    else:
        badge_color = "#065f46"
        text_color = "#34d399"
        icon = "⚡"
        title = "COMMAND EXECUTING"
        desc = "Recognized voice command and actuating hardware."

    html = f"""
    <div style="
        background: {badge_color};
        color: {text_color};
        padding: 12px 18px;
        border-radius: 10px;
        border-left: 5px solid {text_color};
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    ">
        <span style="font-size: 26px;">{icon}</span>
        <div>
            <div style="font-size: 14px; font-weight: 800; letter-spacing: 0.5px;">{title}</div>
            <div style="font-size: 13px; opacity: 0.9; margin-top: 2px;">{desc}</div>
        </div>
    </div>
    """
    return html


def render_device_metrics(controller: SmartDeviceController) -> str:
    """Renders smart home status cards (lights, AC, media, timer)."""
    lights_str = f"ON ({int(controller.light_brightness * 100)}%)" if controller.lights_on else "OFF"
    media_str = f"Playing ({controller.media_volume}%)" if controller.media_playing else "Paused"
    
    html = f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 8px;">
        <div style="background: #1e293b; padding: 10px 14px; border-radius: 8px; border: 1px solid #334155;">
            <div style="font-size: 11px; color: #94a3b8;">💡 LIGHTS</div>
            <div style="font-size: 14px; font-weight: bold; color: {'#facc15' if controller.lights_on else '#94a3b8'};">{lights_str}</div>
        </div>
        <div style="background: #1e293b; padding: 10px 14px; border-radius: 8px; border: 1px solid #334155;">
            <div style="font-size: 11px; color: #94a3b8;">🌡️ THERMOSTAT</div>
            <div style="font-size: 14px; font-weight: bold; color: #38bdf8;">{controller.temperature}°F</div>
        </div>
        <div style="background: #1e293b; padding: 10px 14px; border-radius: 8px; border: 1px solid #334155;">
            <div style="font-size: 11px; color: #94a3b8;">🎵 MEDIA</div>
            <div style="font-size: 14px; font-weight: bold; color: {'#a7f3d0' if controller.media_playing else '#94a3b8'};">{media_str}</div>
        </div>
        <div style="background: #1e293b; padding: 10px 14px; border-radius: 8px; border: 1px solid #334155;">
            <div style="font-size: 11px; color: #94a3b8;">⏰ ALARM</div>
            <div style="font-size: 14px; font-weight: bold; color: #f472b6;">{controller.alarm_set}</div>
        </div>
    </div>
    """
    return html


def render_voice_speech_card(last_speech: str) -> str:
    """Renders prominent monitor subtitle banner showing spoken assistant response."""
    msg = last_speech if last_speech else "Assistant is in standby. Speak 'Hi Dandan' to activate."
    return f"""
    <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border: 2px solid #0284c7; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25);">
        <div style="font-size: 11px; font-weight: bold; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">
            🔊 Spoken Voice Response & Monitor Subtitle
        </div>
        <div style="font-size: 18px; font-weight: 700; color: #f8fafc; line-height: 1.4;">
            "{msg}"
        </div>
    </div>
    """


def plot_mel_spectrogram(mel_matrix: Optional[np.ndarray]) -> matplotlib.figure.Figure:
    """Draws acoustic Log-Mel spectrogram."""
    fig, ax = plt.subplots(figsize=(6, 2.2), dpi=100)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    if mel_matrix is not None:
        im = ax.imshow(mel_matrix, aspect='auto', origin='lower', cmap='inferno')
        ax.set_title("Log-Mel Spectrogram (40 filterbanks × 151 frames)", color='#e2e8f0', fontsize=10, pad=6)
        ax.set_xlabel("Time Frames (10ms hop)", color='#94a3b8', fontsize=8)
        ax.set_ylabel("Mel Bins (20-8000 Hz)", color='#94a3b8', fontsize=8)
        ax.tick_params(colors='#64748b', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#334155')
    else:
        ax.text(0.5, 0.5, "Speak or click a command to view spectrogram",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='#64748b', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Audio Processing Handler
# ---------------------------------------------------------------------------
def process_audio_input(audio_tuple, require_wake_word_toggle: bool):
    """
    Handles speech input from live microphone or uploaded WAV audio.
    """
    APP.engine.check_timeout()
    APP.engine.set_require_wake_word(require_wake_word_toggle)

    if audio_tuple is None:
        hw_state = APP.hw.get_state()
        return (
            render_state_badge(APP.engine.state),
            render_voice_speech_card(hw_state.get("last_speech", "")),
            render_oled_html(hw_state["display"]),
            render_led_html(hw_state["led"]),
            render_device_metrics(APP.controller),
            plot_mel_spectrogram(APP.last_mel_spec),
            {},
            "No audio provided.",
            APP.event_history,
            None
        )

    sr, audio_arr = audio_tuple

    # Convert audio to float32 normalized in [-1, 1]
    if audio_arr.dtype == np.int16:
        audio_arr = audio_arr.astype(np.float32) / 32768.0
    elif audio_arr.dtype == np.int32:
        audio_arr = audio_arr.astype(np.float32) / 2147483648.0
    elif audio_arr.dtype != np.float32:
        audio_arr = audio_arr.astype(np.float32)

    if audio_arr.ndim > 1:
        audio_arr = np.mean(audio_arr, axis=1)

    # Process through streaming engine
    events = APP.engine.process_audio_clip(audio_arr, sample_rate=sr)

    # Compute spectrogram of central 1.5s window for display
    if len(audio_arr) >= NUM_SAMPLES:
        disp_win = audio_arr[-NUM_SAMPLES:]
    else:
        disp_win = np.pad(audio_arr, (0, NUM_SAMPLES - len(audio_arr)))

    cmd_name, conf, lat, top3, mel_matrix = APP.engine.classify_window(disp_win)
    APP.last_mel_spec = mel_matrix

    status_message = "Listening..."
    audio_chime_to_play = None

    if len(events) == 0:
        # Check if silence/noise or below threshold
        if APP.engine.state == StreamingVCMEngine.STATE_STANDBY and require_wake_word_toggle:
            status_message = f"Audio analyzed: '{cmd_name}' (conf: {conf*100:.1f}%). Assistant is in STANDBY — speak 'Hi Dandan' or 'Hello Dandan' to wake up."
            APP.log_event("IGNORED (STANDBY)", f"Detected '{cmd_name}' (conf: {conf*100:.1f}%) while sleeping", conf, lat)
        else:
            status_message = f"Detected: '{cmd_name}' (conf: {conf*100:.1f}% < threshold). No action triggered."
            APP.log_event("LOW_CONFIDENCE", f"Detected '{cmd_name}'", conf, lat)
    else:
        for evt in events:
            evt_type = evt.get("event")
            c_name = evt.get("command", "")
            c_conf = evt.get("confidence", 0.0)
            c_lat = evt.get("latency_ms", 0.0)
            msg = evt.get("action_message", "")

            if evt_type == "COMPOUND_COMMAND_EXECUTED":
                status_message = f"⚡ Compound One-Shot Executed: {msg} (conf: {c_conf*100:.1f}%, latency: {c_lat:.1f}ms)"
                APP.log_event("COMPOUND_CMD", f"[{c_name}] -> {msg}", c_conf, c_lat)
                if c_name == "play_music":
                    audio_chime_to_play = str(ASSETS_DIR / "sample_music.wav")
                else:
                    voice_wav = ASSETS_DIR / "voice_responses" / f"{c_name}.wav"
                    audio_chime_to_play = str(voice_wav) if voice_wav.exists() else None
            elif evt_type == "WAKE_WORD_DETECTED":
                status_message = f"👋 Wake Word Recognized! (conf: {c_conf*100:.1f}%, latency: {c_lat:.1f}ms). Assistant is now LISTENING!"
                APP.log_event("WAKE_DETECTED", "Woke up from standby ('Hi Dandan' / 'Hello Dandan')", c_conf, c_lat)
                audio_chime_to_play = str(ASSETS_DIR / "voice_responses" / "wake_word.wav")
            elif evt_type == "COMMAND_EXECUTED":
                trigger = evt.get("trigger", "KWS")
                status_message = f"✅ Command Executed ({trigger}): {msg} (conf: {c_conf*100:.1f}%, latency: {c_lat:.1f}ms)"
                APP.log_event("COMMAND", f"[{c_name}] -> {msg}", c_conf, c_lat)
                if c_name == "play_music":
                    audio_chime_to_play = str(ASSETS_DIR / "sample_music.wav")
                else:
                    voice_wav = ASSETS_DIR / "voice_responses" / f"{c_name}.wav"
                    audio_chime_to_play = str(voice_wav) if voice_wav.exists() else None
            elif evt_type == "TIMEOUT":
                status_message = "⏱️ Active listening timed out. Assistant returned to STANDBY."
                APP.log_event("TIMEOUT", "Listening window expired", 1.0, 0.0)
                audio_chime_to_play = str(ASSETS_DIR / "voice_responses" / "timeout.wav")

    hw_state = APP.hw.get_state()
    vu_str, energy = APP.streamer.get_vu_meter(num_bars=10) if hasattr(APP, "streamer") else ("[----------]", 0.0)
    mic_banner = render_mic_banner_html(APP.mic_device_name, vu_str, energy, APP.engine.state)
    return (
        mic_banner,
        render_state_badge(APP.engine.state),
        render_voice_speech_card(hw_state.get("last_speech", "")),
        render_oled_html(hw_state["display"]),
        render_led_html(hw_state["led"]),
        render_device_metrics(APP.controller),
        plot_mel_spectrogram(APP.last_mel_spec),
        top3,
        status_message,
        APP.event_history,
        audio_chime_to_play
    )


def simulate_preset_command(cmd_name: str, require_wake_word_toggle: bool):
    """Feeds a real recorded audio utterance from dataset directly into the engine."""
    APP.engine.set_require_wake_word(require_wake_word_toggle)

    cls_dir = DATA_DIR / "dataset" / cmd_name
    if not cls_dir.exists():
        return process_audio_input(None, require_wake_word_toggle)

    clean_files = list(cls_dir.glob("*clean.wav"))
    all_files = clean_files if clean_files else list(cls_dir.glob("*.wav"))
    if not all_files:
        return process_audio_input(None, require_wake_word_toggle)

    wav_file = all_files[0]
    data, sr = sf.read(str(wav_file))
    return process_audio_input((sr, data), require_wake_word_toggle)


def render_mic_banner_html(mic_name: str, vu_str: str, energy: float, state: str) -> str:
    color = "#10b981" if state == StreamingVCMEngine.STATE_LISTENING else "#0ea5e9"
    bg = "linear-gradient(135deg, #064e3b, #065f46)" if state == StreamingVCMEngine.STATE_LISTENING else "linear-gradient(135deg, #0f172a, #1e293b)"
    mode_text = "🟢 ACTIVE LISTENING — Speak your command now!" if state == StreamingVCMEngine.STATE_LISTENING else "🔵 STANDBY ALWAYS-LISTENING — Say 'Hi Dandan' or 'Hello Dandan'"
    return f"""
    <div style="background: {bg}; border: 2px solid {color}; border-radius: 12px; padding: 14px 20px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);">
        <div style="display: flex; align-items: center; gap: 14px;">
            <span style="font-size: 28px;">🎙️</span>
            <div>
                <div style="font-weight: 800; color: #f8fafc; font-size: 15px; letter-spacing: 0.5px;">
                    {mode_text}
                </div>
                <div style="color: #cbd5e1; font-size: 13px; margin-top: 2px;">
                    Active Mic: <b>{mic_name}</b> &nbsp;|&nbsp; Signal VU: <span style="font-family: monospace; background: #020617; padding: 2px 8px; border-radius: 4px; color: #38bdf8;">{vu_str}</span> (RMS: {energy:.4f})
                </div>
            </div>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(255,255,255,0.1); color: #f8fafc; font-size: 12px; font-weight: 700; padding: 6px 14px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.2);">
                ZERO BUTTONS TO CLICK — JUST SPEAK INTO ROOM
            </span>
        </div>
    </div>
    """


def handle_reset():
    """Resets the state machine."""
    APP.reset_state()
    hw_state = APP.hw.get_state()
    vu_str, energy = APP.streamer.get_vu_meter(num_bars=10) if hasattr(APP, "streamer") else ("[----------]", 0.0)
    mic_banner = render_mic_banner_html(APP.mic_device_name, vu_str, energy, APP.engine.state)
    return (
        mic_banner,
        render_state_badge(APP.engine.state),
        render_voice_speech_card("System reset to STANDBY mode. Say 'Hi Dandan' to wake up."),
        render_oled_html(hw_state["display"]),
        render_led_html(hw_state["led"]),
        render_device_metrics(APP.controller),
        plot_mel_spectrogram(None),
        {},
        "System reset to STANDBY mode.",
        APP.event_history,
        None
    )


def auto_refresh_dashboard(require_wake_word_toggle: bool):
    """Auto-polls hardware state machine and streams updates with zero button clicks."""
    APP.engine.set_require_wake_word(require_wake_word_toggle)
    to_evt = APP.engine.check_timeout()
    if to_evt:
        APP.last_status_message = "⏱️ Listening window expired -> Assistant returned to STANDBY."
        APP.log_event("TIMEOUT", "Listening window expired", 1.0, 0.0)
        APP.last_audio_to_play = str(ASSETS_DIR / "voice_responses" / "timeout.wav")

    hw_state = APP.hw.get_state()
    audio_chime = APP.last_audio_to_play
    APP.last_audio_to_play = None

    spec_plot = plot_mel_spectrogram(APP.last_mel_spec) if APP.mel_updated else gr.update()
    APP.mel_updated = False

    top3_val = APP.last_top3 if APP.top3_updated else gr.update()
    APP.top3_updated = False

    vu_str, energy = APP.streamer.get_vu_meter(num_bars=10) if hasattr(APP, "streamer") else ("[----------]", 0.0)
    mic_banner = render_mic_banner_html(APP.mic_device_name, vu_str, energy, APP.engine.state)

    return (
        mic_banner,
        render_state_badge(APP.engine.state),
        render_voice_speech_card(hw_state.get("last_speech", "")),
        render_oled_html(hw_state["display"]),
        render_led_html(hw_state["led"]),
        render_device_metrics(APP.controller),
        spec_plot,
        top3_val,
        APP.last_status_message,
        list(APP.event_history),
        audio_chime
    )


# ---------------------------------------------------------------------------
# Gradio Modern UI Definition
# ---------------------------------------------------------------------------
custom_css = """
body { background-color: #0b0f19; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
.gradio-container { max-width: 1200px !important; margin: auto; }
.card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 18px; }
button.primary-btn { background: #2563eb !important; color: white !important; font-weight: bold; border-radius: 8px; }
button.wake-btn { background: #0891b2 !important; color: white !important; font-weight: bold; border-radius: 8px; }
@keyframes pulse { 0% { opacity: 0.6; } 50% { opacity: 1; } 100% { opacity: 0.6; } }
"""

with gr.Blocks(title="Tiny Voice Command Model (VCM)") as demo:
    gr.Markdown(
        """
        # 🎙️ Tiny Voice Command Model (VCM) Smart Assistant
        ### Standalone, 100% On-Device Neural Voice Classifier with Wake-Word ("Hi Dandan" / "Hello Dandan") Activation
        *Powered by BC-ResNet-1 (27,642 parameters, 181 KB INT8 quantized, 8.13 ms latency on Edge CPU)*
        """
    )

    # 100% Hands-Free Live Mic Status Banner
    mic_banner_html = gr.HTML(
        value=f"""
        <div style="background: linear-gradient(135deg, #064e3b, #065f46); border: 2px solid #10b981; border-radius: 12px; padding: 14px 20px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="font-size: 28px;">🎙️</span>
                <div>
                    <div style="font-weight: 800; color: #a7f3d0; font-size: 15px; letter-spacing: 0.5px;">
                        🟢 100% HANDS-FREE ZERO-BUTTON MICROPHONE ACTIVE & LISTENING
                    </div>
                    <div style="color: #d1fae5; font-size: 13px; margin-top: 2px;">
                        Continuous Audio Capture on: <b>{APP.mic_device_name}</b> (16 kHz continuous background stream)
                    </div>
                </div>
            </div>
            <div style="text-align: right;">
                <span style="background: #047857; color: #ecfdf5; font-size: 12px; font-weight: 700; padding: 6px 14px; border-radius: 20px; border: 1px solid #34d399;">
                    ZERO BUTTONS TO CLICK — JUST SPEAK!
                </span>
            </div>
        </div>
        """
    )

    with gr.Row():
        # Left Column: Virtual Hardware & State Dashboard
        with gr.Column(scale=5):
            state_badge_html = gr.HTML(value=render_state_badge(APP.engine.state))
            voice_card_html = gr.HTML(value=render_voice_speech_card(APP.hw.get_state().get("last_speech", "")))
            oled_html = gr.HTML(value=render_oled_html(APP.hw.display_lines))
            led_html = gr.HTML(value=render_led_html(APP.hw.get_state()["led"]))
            metrics_html = gr.HTML(value=render_device_metrics(APP.controller))

            with gr.Row():
                require_wake_toggle = gr.Checkbox(
                    value=True,
                    label="🔒 Require Wake-Word ('Hi Dandan' / 'Hello Dandan') to Activate",
                    info="When checked, commands are ignored until you say 'Hi Dandan' or 'Hello Dandan'."
                )
                reset_btn = gr.Button("🔄 Reset Assistant", variant="secondary", size="sm")

            audio_output_player = gr.Audio(label="🔊 Speaker Feedback", type="filepath", interactive=False)

        # Right Column: Acoustic Analytics & Visualizer
        with gr.Column(scale=5):
            status_text = gr.Textbox(label="Assistant Status Message", value="System Ready. Standby mode active.", interactive=False)
            spectrogram_plot = gr.Plot(value=plot_mel_spectrogram(None), label="Real-Time Log-Mel Spectrogram")
            prediction_labels = gr.Label(label="Top Prediction Confidences", num_top_classes=3)

    gr.Markdown("---")

    # Audio Inputs Section
    with gr.Tabs():
        with gr.TabItem("🎙️ Hands-Free Zero-Button Mode (Active & Listening)"):
            gr.Markdown(
                """
                ### 🗣️ Zero-Button Voice Control Workflow
                **The system is streaming audio from your microphone right now. You DO NOT need to click anything:**

                1. **Wake Up (Hands-Free):** Speak into your room: **"Hi Dandan"** or **"Hello Dandan"**.
                   - *Feedback:* The assistant halo turns cyan, SSD1306 OLED mirrors `[WAKE: HI DANDAN]`, and the speaker responds: *"Hi Dandan! I'm listening. What can I do for you?"*
                2. **Give Any Command:** Within 7 seconds, speak your command naturally:
                   - **Music:** *"Play music"* / *"Pause music"* / *"Volume up"*
                   - **Lighting:** *"Turn on the lights"* / *"Turn off the lights"* / *"Dim lights fifty percent"*
                   - **Questions:** *"What's the weather"* / *"What time is it"*
                   - **Timers & Alarms:** *"Set timer for five minutes"* / *"Set alarm for seven AM"*
                   - **Thermostat:** *"Make it cooler"* / *"Make it warmer"*
                   - **Productivity & Communication:** *"Check my reminders"* / *"Call Mom"*
                3. **Automatic Action & Return to Sleep:**
                   - The system executes the device action, announces the spoken voice confirmation, and goes back to sleep automatically!
                """
            )

        with gr.TabItem("⚡ 1-Click Quick-Test Voice Bank"):
            gr.Markdown("Click any button below to instantly feed authentic multi-speaker audio into the system:")
            
            with gr.Row():
                btn_wake_hello = gr.Button("👋 'Hello Dandan' (Wake)", variant="primary")
                btn_wake_hi = gr.Button("👋 'Hi Dandan' (Wake)", variant="primary")
                btn_noise = gr.Button("🔇 Room Noise (Rejection)", variant="secondary")

            gr.Markdown("**Commands (Ranked 1 to 10):**")
            with gr.Row():
                btn_music = gr.Button("🎵 #1: Play Music")
                btn_weather = gr.Button("🌤️ #2: What's the Weather")
                btn_time = gr.Button("⏰ #2: What Time is It")
                btn_lights_on = gr.Button("💡 #3: Turn On Lights")
                btn_lights_off = gr.Button("💡 #3: Turn Off Lights")

            with gr.Row():
                btn_dim_50 = gr.Button("🌓 #4: Dim Lights 50%")
                btn_timer = gr.Button("⏱️ #5: Timer (5 min)")
                btn_alarm = gr.Button("🔔 #6: Set Alarm (7 AM)")
                btn_cooler = gr.Button("❄️ #7: Make it Cooler")
                btn_warmer = gr.Button("🔥 #7: Make it Warmer")

            with gr.Row():
                btn_pause = gr.Button("⏸️ #8: Pause Music")
                btn_vol_up = gr.Button("🔊 #8: Volume Up")
                btn_reminders = gr.Button("📝 #9: Check Reminders")
                btn_call = gr.Button("📞 #10: Call Mom")

        with gr.TabItem("📁 Upload WAV Audio File"):
            file_input = gr.Audio(sources=["upload"], type="numpy", label="Upload 16kHz WAV Audio")
            file_submit_btn = gr.Button("🚀 Process Audio File", variant="primary")

        with gr.TabItem("🖱️ Browser HTML5 Mic (Manual Sandbox Fallback)"):
            gr.Markdown(
                """
                > [!NOTE]
                > **Why does this tab have a record button?**
                > Web browsers enforce strict security sandboxes that prevent web pages from silently recording your microphone without an explicit user click.
                > **For 100% zero-button operation:** Use the **Hands-Free** tab above (or run `monitor_display.py` / `live_listen.py` directly in Python), which stream directly from the OS audio driver with **zero button clicks**!
                """
            )
            mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Browser Sandbox Recording")
            mic_submit_btn = gr.Button("🚀 Process Browser Recording", variant="primary")

    gr.Markdown("---")

    # Chronological Event Log
    with gr.Accordion("📜 Real-Time Hardware & Interaction Event Log", open=True):
        history_df = gr.Dataframe(
            headers=["Time", "Event", "Details", "Confidence", "Latency"],
            datatype=["str", "str", "str", "str", "str"],
            value=APP.event_history,
            interactive=False
        )

    # -----------------------------------------------------------------------
    # Wire UI Events & Zero-Button Live Refresh Timer
    # -----------------------------------------------------------------------
    outputs = [
        mic_banner_html,
        state_badge_html,
        voice_card_html,
        oled_html,
        led_html,
        metrics_html,
        spectrogram_plot,
        prediction_labels,
        status_text,
        history_df,
        audio_output_player
    ]

    # Automatic background timer: streams hardware mic detections straight to the UI
    auto_timer = gr.Timer(value=0.4, active=True)
    auto_timer.tick(
        fn=auto_refresh_dashboard,
        inputs=[require_wake_toggle],
        outputs=outputs
    )

    mic_submit_btn.click(
        fn=process_audio_input,
        inputs=[mic_input, require_wake_toggle],
        outputs=outputs
    )

    file_submit_btn.click(
        fn=process_audio_input,
        inputs=[file_input, require_wake_toggle],
        outputs=outputs
    )

    reset_btn.click(
        fn=handle_reset,
        inputs=[],
        outputs=outputs
    )

    # Preset button events
    btn_wake_hello.click(fn=lambda w: simulate_preset_command("wake_word", w), inputs=[require_wake_toggle], outputs=outputs)
    btn_wake_hi.click(fn=lambda w: simulate_preset_command("wake_word", w), inputs=[require_wake_toggle], outputs=outputs)
    btn_noise.click(fn=lambda w: simulate_preset_command("_background_noise_", w), inputs=[require_wake_toggle], outputs=outputs)

    btn_music.click(fn=lambda w: simulate_preset_command("play_music", w), inputs=[require_wake_toggle], outputs=outputs)
    btn_weather.click(fn=lambda w: simulate_preset_command("question_weather", w), inputs=[require_wake_toggle], outputs=outputs)
    btn_time.click(fn=lambda w: simulate_preset_command("question_time", w), inputs=[require_wake_toggle], outputs=outputs)
    btn_lights_on.click(fn=lambda w: simulate_preset_command("lights_on", w), inputs=[require_wake_toggle], outputs=outputs)
    btn_lights_off.click(fn=lambda w: simulate_preset_command("lights_off", w), inputs=[require_wake_toggle], outputs=outputs)

    btn_dim_50.click(fn=lambda w: simulate_preset_command("dim_lights_50", w), inputs=[require_wake_toggle], outputs=outputs)
    btn_timer.click(fn=lambda w: simulate_preset_command("timer_5min", w), inputs=[require_wake_toggle], outputs=outputs)
    btn_alarm.click(fn=lambda w: simulate_preset_command("alarm_set", w), inputs=[require_wake_toggle], outputs=outputs)
    btn_cooler.click(fn=lambda w: simulate_preset_command("temp_cooler", w), inputs=[require_wake_toggle], outputs=outputs)
    btn_warmer.click(fn=lambda w: simulate_preset_command("temp_warmer", w), inputs=[require_wake_toggle], outputs=outputs)

    btn_pause.click(fn=lambda w: simulate_preset_command("media_pause", w), inputs=[require_wake_toggle], outputs=outputs)
    btn_vol_up.click(fn=lambda w: simulate_preset_command("volume_up", w), inputs=[require_wake_toggle], outputs=outputs)
    btn_reminders.click(fn=lambda w: simulate_preset_command("reminders_check", w), inputs=[require_wake_toggle], outputs=outputs)
    btn_call.click(fn=lambda w: simulate_preset_command("call_mom", w), inputs=[require_wake_toggle], outputs=outputs)


if __name__ == "__main__":
    print("=" * 78)
    print("  LAUNCHING TINY VOICE COMMAND MODEL (VCM) INTERACTIVE APPLICATION")
    print("=" * 78)
    print("  Local URL: http://127.0.0.1:7860")
    print("  Featuring Wake-Word ('Hi Dandan' / 'Hello Dandan') Activation + Live Mic Input")
    print("=" * 78)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=custom_css
    )
