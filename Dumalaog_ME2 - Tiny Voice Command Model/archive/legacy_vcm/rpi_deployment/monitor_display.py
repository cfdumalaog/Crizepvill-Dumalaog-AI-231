#!/usr/bin/env python3
"""
Native Desktop Monitor Display & Kiosk Interface for Raspberry Pi 5.
Runs standalone on external HDMI monitor connected to Raspberry Pi or PC.
Features:
- Fullscreen / Kiosk mode toggle (F11)
- Animated Glowing Assistant Halo (shifts color with device state)
- Big across-the-room Spoken Voice Subtitle Card
- Simulated 128x64 OLED Screen Mirror
- Real-time Hardware Telemetry (Lights, Thermostat, Media, Timer)
- Hands-Free Always-Listening Engine (Triggered by 'Hi Dandan' / 'Hello Dandan')
- 1-Click On-Screen Command Buttons for touchscreens & mice
"""

import os
import sys
import time
import math
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Setup project root import
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tkinter as tk
from tkinter import ttk, font
import numpy as np
import torch
import sounddevice as sd

from src.config import (
    SAMPLE_RATE,
    STREAM_HOP_SEC,
    MODELS_DIR,
    WAKE_TIMEOUT_SEC
)
from src.model import build_model
from src.hal import get_hardware
from src.actions import SmartDeviceController
from src.engine import StreamingVCMEngine
from src.tts import VOICE_ENGINE
from src.mic_stream import RobustMicrophoneStreamer


class MonitorKioskApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VCM Smart Assistant — Monitor Display")
        self.root.geometry("1240x820")
        self.root.minsize(1024, 700)
        self.root.configure(bg="#0b0f19")

        self.is_fullscreen = False
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.root.bind("<Escape>", lambda e: self.exit_fullscreen())

        # 1. Initialize HAL & Controller
        self.hw = get_hardware(verbose=False)
        self.controller = SmartDeviceController(hardware=self.hw)

        # 2. Load Model
        int8_path = MODELS_DIR / "bc_resnet_int8.pt"
        best_path = MODELS_DIR / "bc_resnet_best.pt"
        self.model = build_model("bc_resnet")
        if int8_path.exists():
            try:
                q = torch.ao.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                )
                q.load_state_dict(torch.load(int8_path, map_location="cpu"))
                self.model = q
            except Exception:
                self.model.load_state_dict(torch.load(best_path, map_location="cpu"))
        elif best_path.exists():
            self.model.load_state_dict(torch.load(best_path, map_location="cpu"))
        self.model.eval()

        # 3. Streaming Engine
        self.engine = StreamingVCMEngine(
            model=self.model,
            controller=self.controller,
            confidence_threshold=0.55,
            wake_confidence=0.50,
            wake_timeout_sec=WAKE_TIMEOUT_SEC,
            holdoff_seconds=0.35,
            device="cpu",
            require_wake_word=True
        )

        # State tracking
        self.last_speech_text = "Assistant Ready. Say 'Hi Dandan' to wake up."
        self.halo_pulse = 0.0
        self.halo_color = "#0284c7"
        self.mic_streaming = True

        # Register speech listener for live subtitles
        VOICE_ENGINE.register_listener(self.on_speech_dispatched)

        self._build_ui()
        self._start_audio_stream()
        self._start_gui_update_loop()

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)

    def exit_fullscreen(self):
        self.is_fullscreen = False
        self.root.attributes("-fullscreen", False)

    def on_speech_dispatched(self, text: str, wav_path: Optional[Path]):
        """Called when assistant speaks."""
        self.last_speech_text = text
        self.root.after(0, self._update_speech_display)

    def _update_speech_display(self):
        self.subtitle_label.config(text=f"🔊 \"{self.last_speech_text}\"")

    def _build_ui(self):
        # Top Header Bar
        header = tk.Frame(self.root, bg="#111827", height=60, bd=0)
        header.pack(fill="x", side="top")

        title_lbl = tk.Label(
            header,
            text="🎙️ TINY VOICE COMMAND MODEL — MONITOR DISPLAY",
            font=("Segoe UI", 16, "bold"),
            fg="#f8fafc",
            bg="#111827"
        )
        title_lbl.pack(side="left", padx=20, pady=12)

        self.clock_lbl = tk.Label(
            header,
            text="--:--:--",
            font=("Consolas", 14, "bold"),
            fg="#38bdf8",
            bg="#111827"
        )
        self.clock_lbl.pack(side="right", padx=20, pady=12)

        fs_btn = tk.Button(
            header,
            text="⛶ Fullscreen (F11)",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#1f2937",
            activebackground="#374151",
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=4,
            command=self.toggle_fullscreen
        )
        fs_btn.pack(side="right", padx=10, pady=12)

        # Main Layout: Two Columns
        main_content = tk.Frame(self.root, bg="#0b0f19")
        main_content.pack(fill="both", expand=True, padx=20, pady=15)

        # Left Stage: Halo, Status, Subtitle Banner
        left_stage = tk.Frame(main_content, bg="#111827", bd=1, relief="solid")
        left_stage.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Animated Halo Canvas
        self.halo_canvas = tk.Canvas(left_stage, width=220, height=220, bg="#111827", bd=0, highlightthickness=0)
        self.halo_canvas.pack(pady=(20, 10))

        # State Pill Badge
        self.status_pill = tk.Label(
            left_stage,
            text="💤 STANDBY (Always Listening)",
            font=("Segoe UI", 12, "bold"),
            fg="#38bdf8",
            bg="#0c4a6e",
            padx=16,
            pady=6
        )
        self.status_pill.pack(pady=5)

        # Big Across-The-Room Voice Subtitle Card
        subtitle_frame = tk.Frame(left_stage, bg="#1e293b", bd=1, relief="solid")
        subtitle_frame.pack(fill="x", padx=20, pady=15)

        sub_title_hdr = tk.Label(
            subtitle_frame,
            text="SPOKEN ASSISTANT RESPONSE",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#1e293b"
        )
        sub_title_hdr.pack(anchor="w", padx=15, pady=(8, 2))

        self.subtitle_label = tk.Label(
            subtitle_frame,
            text=f"🔊 \"{self.last_speech_text}\"",
            font=("Segoe UI", 16, "bold"),
            fg="#f1f5f9",
            bg="#1e293b",
            wraplength=520,
            justify="left",
            pady=10
        )
        self.subtitle_label.pack(anchor="w", padx=15, pady=(0, 8))

        # Prompt hint
        hint_lbl = tk.Label(
            left_stage,
            text="Hands-Free: Speak aloud in the room: \"Hi Dandan\" or \"Hello Dandan\"",
            font=("Segoe UI", 11, "italic"),
            fg="#64748b",
            bg="#111827"
        )
        hint_lbl.pack(pady=(0, 4))

        # Siri/Alexa VAD Energy Level Meter
        self.vad_label = tk.Label(
            left_stage,
            text="🎙️ VAD: [░░░░░░░░░░░░] IDLE (VAD Gated)",
            font=("Consolas", 10),
            fg="#64748b",
            bg="#111827"
        )
        self.vad_label.pack(pady=(0, 12))

        # Right Stage: Telemetry & OLED Mirror
        right_stage = tk.Frame(main_content, bg="#111827", width=460, bd=1, relief="solid")
        right_stage.pack(side="right", fill="both", padx=(10, 0))

        # OLED Display Simulation Card
        oled_box = tk.Frame(right_stage, bg="#000000", bd=3, relief="sunken")
        oled_box.pack(fill="x", padx=15, pady=15)

        oled_hdr = tk.Label(oled_box, text="I2C SSD1306 OLED (128x64) MIRROR", font=("Consolas", 8, "bold"), fg="#38bdf8", bg="#000000")
        oled_hdr.pack(anchor="w", padx=8, pady=(4, 2))

        self.oled_lines = []
        for i in range(4):
            lbl = tk.Label(
                oled_box,
                text=" " if i > 0 else "VCM SMART ASSISTANT",
                font=("Consolas", 12, "bold"),
                fg="#38bdf8",
                bg="#000000",
                anchor="w"
            )
            lbl.pack(fill="x", padx=10, pady=2)
            self.oled_lines.append(lbl)

        # Smart Device Telemetry Grid
        telem_frame = tk.Frame(right_stage, bg="#111827")
        telem_frame.pack(fill="both", expand=True, padx=15, pady=5)

        # Row 1: Lights & Media
        self.card_lights = self._create_telem_card(telem_frame, "💡 IOT LIGHTS", "OFF", "#94a3b8", 0, 0)
        self.card_media = self._create_telem_card(telem_frame, "🎵 MEDIA PLAYBACK", "IDLE (70%)", "#94a3b8", 0, 1)

        # Row 2: Thermostat & Timer
        self.card_thermo = self._create_telem_card(telem_frame, "❄️ THERMOSTAT", "72°F", "#38bdf8", 1, 0)
        self.card_timer = self._create_telem_card(telem_frame, "⏱️ TIMER", "No Timer Active", "#94a3b8", 1, 1)

        # Row 3: Reminders & Alarm
        self.card_reminders = self._create_telem_card(telem_frame, "📝 REMINDERS", "3 Active Items", "#c084fc", 2, 0)
        self.card_alarm = self._create_telem_card(telem_frame, "⏰ ALARM", "07:00 AM", "#f472b6", 2, 1)

        # Bottom Bar: 1-Click Touch Test Utterances
        btn_bar = tk.Frame(self.root, bg="#111827", bd=0, height=80)
        btn_bar.pack(fill="x", side="bottom", padx=20, pady=(0, 15))

        btn_title = tk.Label(
            btn_bar,
            text="⚡ QUICK TOUCH/CLICK UTTERANCE TRIGGERS:",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#111827"
        )
        btn_title.pack(anchor="w", padx=10, pady=(8, 4))

        btn_row = tk.Frame(btn_bar, bg="#111827")
        btn_row.pack(fill="x", padx=10, pady=(0, 8))

        quick_btns = [
            ("👋 'Hi Dandan'", lambda: self._trigger_preset("wake_word"), "#0284c7"),
            ("💡 Lights ON", lambda: self._trigger_preset("lights_on"), "#334155"),
            ("💡 Lights OFF", lambda: self._trigger_preset("lights_off"), "#334155"),
            ("🎵 Play Music", lambda: self._trigger_preset("play_music"), "#334155"),
            ("🌤️ Weather", lambda: self._trigger_preset("question_weather"), "#334155"),
            ("⏱️ Timer 5m", lambda: self._trigger_preset("timer_5min"), "#334155"),
            ("❄️ Cooler", lambda: self._trigger_preset("temp_cooler"), "#334155"),
            ("📞 Call Mom", lambda: self._trigger_preset("call_mom"), "#334155"),
            ("🔄 Reset", lambda: self._reset_state(), "#475569")
        ]

        for text, cmd, col in quick_btns:
            b = tk.Button(
                btn_row,
                text=text,
                font=("Segoe UI", 9, "bold"),
                fg="#ffffff",
                bg=col,
                activebackground="#0ea5e9",
                activeforeground="#ffffff",
                bd=0,
                padx=10,
                pady=6,
                command=cmd
            )
            b.pack(side="left", padx=4)

    def _create_telem_card(self, parent, title: str, initial_val: str, fg_col: str, r: int, c: int):
        f = tk.Frame(parent, bg="#1f2937", bd=1, relief="solid")
        f.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
        parent.grid_columnconfigure(c, weight=1)
        parent.grid_rowconfigure(r, weight=1)

        t = tk.Label(f, text=title, font=("Segoe UI", 8, "bold"), fg="#94a3b8", bg="#1f2937")
        t.pack(anchor="w", padx=10, pady=(8, 2))

        val = tk.Label(f, text=initial_val, font=("Segoe UI", 12, "bold"), fg=fg_col, bg="#1f2937")
        val.pack(anchor="w", padx=10, pady=(0, 8))
        return val

    def _trigger_preset(self, cmd_name: str):
        """Simulate a voice command."""
        self.engine.check_timeout()
        if cmd_name == "wake_word":
            res = self.controller.trigger_wake()
            self.engine.state = StreamingVCMEngine.STATE_LISTENING
            self.engine.wake_time = time.time()
        else:
            if self.engine.state == StreamingVCMEngine.STATE_STANDBY:
                self.last_speech_text = "Assistant is asleep. Say 'Hi Dandan' first."
                self.subtitle_label.config(text=f"⚠️ {self.last_speech_text}")
                return
            res = self.controller.execute_command(cmd_name, 0.98)
            self.engine.state = StreamingVCMEngine.STATE_STANDBY
        self._update_speech_display()

    def _reset_state(self):
        self.controller.reset_to_standby()
        self.engine.state = StreamingVCMEngine.STATE_STANDBY
        self.last_speech_text = "System reset. Standing by for 'Hi Dandan'."
        self._update_speech_display()

    def _start_audio_stream(self):
        """Background continuous microphone stream for hands-free listening."""
        def _audio_cb(chunk_16k: np.ndarray):
            if not self.mic_streaming:
                return
            evt = self.engine.feed_audio_chunk(chunk_16k)
            if evt:
                self.root.after(0, self._handle_engine_event, evt)

        self.streamer = RobustMicrophoneStreamer(
            target_sample_rate=SAMPLE_RATE,
            hop_seconds=STREAM_HOP_SEC,
            callback=_audio_cb,
            apply_agc=True
        )
        print(f"[MONITOR MIC] Active Device: '{self.streamer.device_name}' (Native: {self.streamer.native_sample_rate} Hz, {self.streamer.channels} ch)")
        self.streamer.start()

        def _timeout_poll():
            while self.mic_streaming:
                to_evt = self.engine.check_timeout()
                if to_evt:
                    self.root.after(0, self._handle_engine_event, to_evt)
                time.sleep(0.15)

        t = threading.Thread(target=_timeout_poll, daemon=True)
        t.start()

    def _handle_engine_event(self, evt: Dict[str, Any]):
        evt_type = evt.get("event")
        msg = evt.get("action_message", "")
        self.last_speech_text = self.hw.get_state().get("last_speech", msg)
        self._update_speech_display()

    def _start_gui_update_loop(self):
        """High-refresh 30 FPS GUI animation and telemetry update."""
        def _loop():
            # Update Clock
            now = datetime.now()
            self.clock_lbl.config(text=now.strftime("%I:%M:%S %p | %A, %b %d"))

            # Update OLED lines
            for i, line in enumerate(self.hw.display_lines[:4]):
                self.oled_lines[i].config(text=line)

            # Update Smart Device Telemetry
            lights_txt = f"ON ({int(self.controller.light_brightness*100)}%)" if self.controller.lights_on else "OFF"
            lights_col = "#fef08a" if self.controller.lights_on else "#94a3b8"
            self.card_lights.config(text=lights_txt, fg=lights_col)

            media_txt = f"PLAYING ({self.controller.media_volume}%)" if self.controller.media_playing else f"PAUSED ({self.controller.media_volume}%)"
            media_col = "#38bdf8" if self.controller.media_playing else "#94a3b8"
            self.card_media.config(text=media_txt, fg=media_col)

            self.card_thermo.config(text=f"{self.controller.temperature}°F")
            self.card_alarm.config(text=self.controller.alarm_set)

            # Update Siri/Alexa VAD Energy Level Meter
            energy = getattr(self.engine, "last_audio_energy", 0.0)
            is_speech = getattr(self.engine, "is_speech_active", False)
            bars = int(min(12, max(0, energy * 400)))
            vu = "█" * bars + "░" * (12 - bars)
            status_str = "SPEECH ACTIVE" if is_speech else "IDLE (VAD Gated)"
            status_col = "#34d399" if is_speech else "#64748b"
            self.vad_label.config(
                text=f"🎙️ VAD: [{vu}] {status_str} (Energy: {energy:.4f})",
                fg=status_col
            )

            # Update Halo Color & State Pill
            state = self.engine.state
            self.halo_pulse += 0.12
            pulse_rad = 10 * math.sin(self.halo_pulse)

            if state == StreamingVCMEngine.STATE_LISTENING:
                self.status_pill.config(text="👂 ACTIVE LISTENING (Say any command)", fg="#22d3ee", bg="#164e63")
                self.halo_color = "#06b6d4"
            elif self.controller.lights_on:
                self.status_pill.config(text="💡 LIGHTS ACTIVE — Standby", fg="#fef08a", bg="#713f12")
                self.halo_color = "#eab308"
            else:
                self.status_pill.config(text="💤 STANDBY (Say 'Hi Dandan')", fg="#38bdf8", bg="#0c4a6e")
                self.halo_color = "#0284c7"

            # Draw Animated Halo
            self.halo_canvas.delete("all")
            cx, cy = 110, 110
            r_outer = 75 + pulse_rad
            r_inner = 55 + (pulse_rad * 0.5)

            # Outer glow
            self.halo_canvas.create_oval(
                cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer,
                outline=self.halo_color, width=4
            )
            # Inner core
            self.halo_canvas.create_oval(
                cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner,
                fill=self.halo_color, outline="#f8fafc", width=2
            )
            # Center icon
            icon_char = "👂" if state == StreamingVCMEngine.STATE_LISTENING else "🎙️"
            self.halo_canvas.create_text(cx, cy, text=icon_char, font=("Segoe UI", 24))

            self.root.after(40, _loop)

        self.root.after(100, _loop)

    def close(self):
        self.mic_streaming = False
        self.hw.cleanup()
        self.root.destroy()


def main():
    # Check if test mode flag passed
    test_mode = "--test" in sys.argv

    root = tk.Tk()
    app = MonitorKioskApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)

    if test_mode:
        print("[TEST] Monitor Display initialized successfully in test mode. Auto-closing in 2 seconds...")
        root.after(2000, app.close)

    root.mainloop()


if __name__ == "__main__":
    main()
