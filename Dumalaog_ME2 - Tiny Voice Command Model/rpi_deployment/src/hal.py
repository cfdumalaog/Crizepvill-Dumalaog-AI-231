"""
Hardware Abstraction Layer (HAL) for Voice Command Model (VCM).
Provides a unified interface for both physical Raspberry Pi 4/5 hardware
and Virtual Hardware Simulation for headless/PC development.
"""

import os
import sys
import time
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from .config import (
    GPIO_LED_RED,
    GPIO_LED_GREEN,
    GPIO_LED_BLUE,
    GPIO_BUZZER,
    I2C_BUS,
    OLED_I2C_ADDRESS,
    OLED_WIDTH,
    OLED_HEIGHT,
    ASSETS_DIR
)


class SmartAssistantHardware(ABC):
    """Abstract Base Class for Smart Assistant Physical & Virtual Actuators."""

    @abstractmethod
    def set_rgb_led(self, r: float, g: float, b: float):
        """Set RGB LED color and brightness (values 0.0 to 1.0)."""
        pass

    @abstractmethod
    def turn_off_led(self):
        """Turn off the LED completely."""
        pass

    @abstractmethod
    def beep(self, duration_sec: float = 0.2, freq_hz: int = 1000):
        """Trigger active buzzer chime."""
        pass

    @abstractmethod
    def update_display(self, line1: str, line2: str = "", line3: str = "", line4: str = ""):
        """Render text on the 128x64 OLED display."""
        pass

    @abstractmethod
    def play_sound(self, sound_file: str):
        """Play an audio file through speaker output."""
        pass

    @abstractmethod
    def speak(self, text: str, response_key: Optional[str] = None):
        """Speak text response aloud through speakers or HDMI monitor."""
        pass

    @abstractmethod
    def cleanup(self):
        """Safely release GPIO and peripheral resources."""
        pass


# ---------------------------------------------------------------------------
# 1. Virtual Hardware Simulator (PC / Mac / Linux Virtualization)
# ---------------------------------------------------------------------------

class VirtualHardware(SmartAssistantHardware):
    """
    Virtual Hardware Simulator.
    Simulates all physical peripherals with state tracking, terminal rendering,
    and optional software sound chimes for instant verification without physical hardware.
    """
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.led_r = 0.0
        self.led_g = 0.0
        self.led_b = 0.0
        self.led_on = False
        self.buzzer_active = False
        self.display_lines = ["VCM Smart Assistant", "Ready & Listening...", "", "Mode: Virtual Sim"]
        self.current_playing = None
        self.last_speech = ""
        self.lock = threading.Lock()

        if self.verbose:
            self._render_dashboard("Initialized Virtual Hardware Simulation.")

    def set_rgb_led(self, r: float, g: float, b: float):
        with self.lock:
            self.led_r = max(0.0, min(1.0, float(r)))
            self.led_g = max(0.0, min(1.0, float(g)))
            self.led_b = max(0.0, min(1.0, float(b)))
            self.led_on = (self.led_r > 0 or self.led_g > 0 or self.led_b > 0)
        if self.verbose:
            color_name = self._get_color_name(self.led_r, self.led_g, self.led_b)
            self._render_dashboard(f"LED Color set to: {color_name} (R={self.led_r:.2f}, G={self.led_g:.2f}, B={self.led_b:.2f})")

    def turn_off_led(self):
        with self.lock:
            self.led_r = 0.0
            self.led_g = 0.0
            self.led_b = 0.0
            self.led_on = False
        if self.verbose:
            self._render_dashboard("LED Turned OFF.")

    def beep(self, duration_sec: float = 0.2, freq_hz: int = 1000):
        def _beep_thread():
            with self.lock:
                self.buzzer_active = True
            if self.verbose:
                self._render_dashboard(f"BUZZER BEEP ({freq_hz}Hz for {duration_sec}s)!")
            time.sleep(duration_sec)
            with self.lock:
                self.buzzer_active = False
            if self.verbose:
                self._render_dashboard("Buzzer silence.")

        t = threading.Thread(target=_beep_thread, daemon=True)
        t.start()

    def update_display(self, line1: str, line2: str = "", line3: str = "", line4: str = ""):
        with self.lock:
            self.display_lines = [str(line1), str(line2), str(line3), str(line4)]
        if self.verbose:
            self._render_dashboard("OLED Display Updated.")

    def play_sound(self, sound_file: str):
        sound_path = Path(sound_file)
        self.current_playing = sound_path.name
        if self.verbose:
            self._render_dashboard(f"Audio Playback: '{self.current_playing}'")

    def speak(self, text: str, response_key: Optional[str] = None):
        """Dispatches voiced audio speech output asynchronously."""
        with self.lock:
            self.last_speech = text
        try:
            from .tts import VOICE_ENGINE
            VOICE_ENGINE.speak(text, response_key=response_key, blocking=False)
        except Exception:
            pass
        if self.verbose:
            self._render_dashboard(f"Voiced Output: \"{text}\"")

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "led": {
                    "on": self.led_on,
                    "r": self.led_r,
                    "g": self.led_g,
                    "b": self.led_b,
                    "brightness": max(self.led_r, self.led_g, self.led_b)
                },
                "display": list(self.display_lines),
                "buzzer": self.buzzer_active,
                "current_playing": self.current_playing,
                "last_speech": self.last_speech
            }

    def _get_color_name(self, r: float, g: float, b: float) -> str:
        if r == 0 and g == 0 and b == 0:
            return "OFF"
        if r > 0.6 and g < 0.3 and b < 0.3:
            return "RED (Heating/Alert)"
        if b > 0.6 and r < 0.3 and g < 0.3:
            return "BLUE (Cooling/Active)"
        if g > 0.6 and r < 0.3 and b < 0.3:
            return "GREEN (Success/Active)"
        if r > 0.5 and g > 0.5 and b < 0.3:
            return "YELLOW (Timer/Alarm)"
        if r > 0.4 and g > 0.4 and b > 0.4:
            return "WHITE (Lights ON)"
        return f"CUSTOM_RGB({r:.1f},{g:.1f},{b:.1f})"

    def _render_dashboard(self, event_msg: str = ""):
        """Visual ASCII OLED screen and peripheral status rendering."""
        led_status = f"ON (RGB: {self.led_r:.2f}, {self.led_g:.2f}, {self.led_b:.2f})" if self.led_on else "OFF"
        buzzer_status = "RINGING" if self.buzzer_active else "IDLE"
        playing_status = self.current_playing or "None"

        dashboard = (
            f"\n+================== [ VIRTUAL RASPBERRY PI SMART NODE ] ==================+\n"
            f"| [I2C OLED 128x64 DISPLAY]                                                |\n"
            f"|   Line 1: {self.display_lines[0]:<60} |\n"
            f"|   Line 2: {self.display_lines[1]:<60} |\n"
            f"|   Line 3: {self.display_lines[2]:<60} |\n"
            f"|   Line 4: {self.display_lines[3]:<60} |\n"
            f"|--------------------------------------------------------------------------|\n"
            f"| [PERIPHERAL ACTUATORS]                                                   |\n"
            f"|   RGB LED (GPIO 17,27,22): {led_status:<45} |\n"
            f"|   Active Buzzer (GPIO 23): {buzzer_status:<45} |\n"
            f"|   Audio Output (Speaker) : {playing_status:<45} |\n"
            f"| [LAST EVENT]: {event_msg:<58} |\n"
            f"+==========================================================================+\n"
        )
        print(dashboard)

    def cleanup(self):
        self.turn_off_led()
        if self.verbose:
            print("[HAL] Virtual Hardware cleaned up successfully.")


# ---------------------------------------------------------------------------
# 2. Physical Raspberry Pi 4 / 5 Hardware Driver
# ---------------------------------------------------------------------------

class RaspberryPiHardware(SmartAssistantHardware):
    """
    Physical Hardware Driver for Raspberry Pi 4 / 5.
    Controls hardware PWM/GPIO for RGB LED, Buzzer, and I2C SSD1306 OLED display.
    """
    def __init__(self):
        print("[HAL] Initializing physical Raspberry Pi 4/5 hardware interfaces...")

        # Initialize GPIO using gpiozero or RPi.GPIO
        try:
            from gpiozero import PWMLED, Buzzer
            self.led_r = PWMLED(GPIO_LED_RED)
            self.led_g = PWMLED(GPIO_LED_GREEN)
            self.led_b = PWMLED(GPIO_LED_BLUE)
            self.buzzer = Buzzer(GPIO_BUZZER)
            self.gpio_available = True
            print(f"[HAL] GPIO initialized: RGB=({GPIO_LED_RED},{GPIO_LED_GREEN},{GPIO_LED_BLUE}), Buzzer={GPIO_BUZZER}")
        except Exception as e:
            print(f"[HAL] Warning: Could not initialize gpiozero: {e}")
            self.gpio_available = False

        # Initialize I2C OLED Display (SSD1306)
        try:
            from PIL import Image, ImageDraw, ImageFont
            import board
            import busio
            import adafruit_ssd1306

            i2c = busio.I2C(board.SCL, board.SDA)
            self.oled = adafruit_ssd1306.SSD1306_I2C(OLED_WIDTH, OLED_HEIGHT, i2c, addr=OLED_I2C_ADDRESS)
            self.oled.fill(0)
            self.oled.show()
            self.oled_image = Image.new("1", (OLED_WIDTH, OLED_HEIGHT))
            self.oled_draw = ImageDraw.Draw(self.oled_image)
            self.font = ImageFont.load_default()
            self.oled_available = True
            print(f"[HAL] I2C SSD1306 OLED initialized at address 0x{OLED_I2C_ADDRESS:X}")
        except Exception as e:
            print(f"[HAL] Notice: Physical I2C OLED not available ({e}). Display will output to console.")
            self.oled_available = False

    def set_rgb_led(self, r: float, g: float, b: float):
        if self.gpio_available:
            self.led_r.value = max(0.0, min(1.0, float(r)))
            self.led_g.value = max(0.0, min(1.0, float(g)))
            self.led_b.value = max(0.0, min(1.0, float(b)))

    def turn_off_led(self):
        if self.gpio_available:
            self.led_r.value = 0.0
            self.led_g.value = 0.0
            self.led_b.value = 0.0

    def beep(self, duration_sec: float = 0.2, freq_hz: int = 1000):
        if self.gpio_available:
            def _beep():
                self.buzzer.on()
                time.sleep(duration_sec)
                self.buzzer.off()
            threading.Thread(target=_beep, daemon=True).start()

    def update_display(self, line1: str, line2: str = "", line3: str = "", line4: str = ""):
        if self.oled_available:
            self.oled_draw.rectangle((0, 0, OLED_WIDTH, OLED_HEIGHT), outline=0, fill=0)
            self.oled_draw.text((0, 0), line1[:20], font=self.font, fill=255)
            self.oled_draw.text((0, 16), line2[:20], font=self.font, fill=255)
            self.oled_draw.text((0, 32), line3[:20], font=self.font, fill=255)
            self.oled_draw.text((0, 48), line4[:20], font=self.font, fill=255)
            self.oled.image(self.oled_image)
            self.oled.show()
        else:
            print(f"[OLED] {line1} | {line2} | {line3} | {line4}")

    def play_sound(self, sound_file: str):
        try:
            # On Linux/RPi, use aplay for WAV files without extra dependencies
            if sys.platform.startswith("linux") and os.path.exists(sound_file):
                os.system(f"aplay -q '{sound_file}' &")
            else:
                print(f"[AUDIO] Playing: {sound_file}")
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")

    def speak(self, text: str, response_key: Optional[str] = None):
        """Dispatches voiced audio speech output asynchronously on Raspberry Pi."""
        self.last_speech = text
        try:
            from .tts import VOICE_ENGINE
            VOICE_ENGINE.speak(text, response_key=response_key, blocking=False)
        except Exception as e:
            print(f"[HAL RPI SPEAK ERROR] {e}")

    def cleanup(self):
        self.turn_off_led()
        if self.oled_available:
            self.oled.fill(0)
            self.oled.show()
        print("[HAL] Physical Raspberry Pi hardware cleanup completed.")


# ---------------------------------------------------------------------------
# 3. Factory Auto-Detection
# ---------------------------------------------------------------------------

def is_raspberry_pi() -> bool:
    """Checks whether the current host is a genuine Raspberry Pi."""
    try:
        model_file = Path("/proc/device-tree/model")
        if model_file.exists():
            content = model_file.read_text(encoding="utf-8", errors="ignore").lower()
            return "raspberry pi" in content
    except Exception:
        pass
    return False


def get_hardware(force_virtual: bool = False, verbose: bool = True) -> SmartAssistantHardware:
    """
    Factory function: Returns Physical RaspberryPiHardware if running on RPi,
    otherwise returns VirtualHardware simulation.
    """
    if not force_virtual and is_raspberry_pi():
        try:
            return RaspberryPiHardware()
        except Exception as e:
            print(f"[HAL] Warning: Failed to init physical hardware ({e}). Falling back to virtual simulator.")
    
    return VirtualHardware(verbose=verbose)
