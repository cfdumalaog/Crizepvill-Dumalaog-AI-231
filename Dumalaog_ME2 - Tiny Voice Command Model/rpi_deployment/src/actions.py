"""
Smart Assistant Action Dispatcher for Commands 1 to 10.
Maintains state for lights, media playback, thermostat, timers, alarms, and reminders.
Dispatches state changes to the Hardware Abstraction Layer (HAL).
"""

import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional

from .config import ASSETS_DIR
from .hal import SmartAssistantHardware


class SmartDeviceController:
    """
    Finite State Machine handling real-world smart device commands.
    Directly binds model predictions to physical or virtual hardware actions.
    """
    def __init__(self, hardware: SmartAssistantHardware):
        self.hw = hardware

        # Device States
        self.lights_on = False
        self.light_brightness = 1.0  # 0.0 to 1.0
        self.media_playing = False
        self.media_volume = 70       # 0 to 100%
        self.temperature = 72        # in Fahrenheit
        self.active_timer = None     # {"duration": int, "end_time": float, "thread": Thread}
        self.alarm_set = "07:00 AM"
        self.is_ducked = False
        self.pre_duck_volume = 70
        self.reminders = [
            "Submit AI 231 ME2 on time",
            "Group VCM dataset meeting at 4 PM",
            "Buy groceries"
        ]

    def execute_command(self, command_name: str, confidence: float) -> str:
        """
        Routes the recognized voice command to its concrete smart device action.
        Returns a human-readable action result string.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # -------------------------------------------------------------------
        # Wake Word Activation ("Hi" / "Hello")
        # -------------------------------------------------------------------
        if command_name == "wake_word":
            return self.trigger_wake()

        # -------------------------------------------------------------------
        # Command 1 & 8: Play Music & Media Control
        # -------------------------------------------------------------------
        elif command_name == "play_music":
            self.media_playing = True
            self.is_ducked = False
            self.hw.set_rgb_led(0.0, 0.8, 0.8)  # Cyan for music
            self.hw.update_display("NOW PLAYING:", "Chill Beats - Track 1", f"Vol: {self.media_volume}%", timestamp)
            self.hw.speak("Playing chill beats playlist now.", response_key="play_music")
            self.hw.play_sound(str(ASSETS_DIR / "sample_music.wav"))
            return f"Playing music at volume {self.media_volume}%."

        elif command_name == "media_pause":
            if self.is_ducked:
                self.media_volume = self.pre_duck_volume
                self.is_ducked = False
            self.media_playing = False
            self.hw.update_display("MEDIA PAUSED", "Track 1", f"Vol: {self.media_volume}%", timestamp)
            self.hw.speak("Music playback paused.", response_key="media_pause")
            return "Media playback paused."

        elif command_name == "media_resume":
            self.media_playing = True
            self.hw.update_display("MEDIA RESUMED", "Track 1", f"Vol: {self.media_volume}%", timestamp)
            self.hw.speak("Music playback resumed.", response_key="media_resume")
            return "Media playback resumed."

        elif command_name == "media_next":
            self.hw.beep(0.1, 1200)
            self.hw.update_display("NOW PLAYING:", "Next Track - Track 2", f"Vol: {self.media_volume}%", timestamp)
            self.hw.speak("Skipping to next track.", response_key="media_next")
            return "Skipped to next track."

        elif command_name == "volume_up":
            if self.is_ducked:
                self.pre_duck_volume = min(100, self.pre_duck_volume + 10)
                self.media_volume = self.pre_duck_volume
                self.is_ducked = False
            else:
                self.media_volume = min(100, self.media_volume + 10)
            self.hw.beep(0.08, 1000)
            self.hw.update_display("VOLUME UP", f"Current: {self.media_volume}%", "", timestamp)
            self.hw.speak(f"Volume increased to {self.media_volume} percent.", response_key="volume_up")
            return f"Volume increased to {self.media_volume}%."

        elif command_name == "volume_down":
            if self.is_ducked:
                self.pre_duck_volume = max(0, self.pre_duck_volume - 10)
                self.media_volume = self.pre_duck_volume
                self.is_ducked = False
            else:
                self.media_volume = max(0, self.media_volume - 10)
            self.hw.beep(0.08, 800)
            self.hw.update_display("VOLUME DOWN", f"Current: {self.media_volume}%", "", timestamp)
            self.hw.speak(f"Volume decreased to {self.media_volume} percent.", response_key="volume_down")
            return f"Volume decreased to {self.media_volume}%."

        # -------------------------------------------------------------------
        # Command 2: Ask a Question / Search
        # -------------------------------------------------------------------
        elif command_name == "question_weather":
            self.hw.update_display("WEATHER REPORT", "Diliman, QC: 29 C", "Partly Cloudy, 75% Hum", timestamp)
            self.hw.beep(0.15, 880)
            self.hw.speak("The weather in Diliman is twenty-nine degrees Celsius and partly cloudy with seventy-five percent humidity.", response_key="question_weather")
            return "Weather reported: 29°C, partly cloudy."

        elif command_name == "question_time":
            curr_time = datetime.now().strftime("%I:%M %p")
            curr_date = datetime.now().strftime("%A, %b %d")
            self.hw.update_display("CURRENT TIME", curr_time, curr_date, timestamp)
            self.hw.beep(0.1, 1000)
            self.hw.speak(f"The current time is {curr_time}.", response_key="question_time")
            return f"Current time is {curr_time} on {curr_date}."

        # -------------------------------------------------------------------
        # Command 3: Control Lights (IoT On/Off)
        # -------------------------------------------------------------------
        elif command_name == "lights_on":
            self.lights_on = True
            self.light_brightness = 1.0
            self.hw.set_rgb_led(1.0, 1.0, 1.0)  # Bright white
            self.hw.update_display("LIGHTS: ON", "Brightness: 100%", "Color: White", timestamp)
            self.hw.speak("Turning on the lights.", response_key="lights_on")
            return "Lights turned ON at 100%."

        elif command_name == "lights_off":
            self.lights_on = False
            self.hw.turn_off_led()
            self.hw.update_display("LIGHTS: OFF", "", "", timestamp)
            self.hw.speak("Turning off the lights.", response_key="lights_off")
            return "Lights turned OFF."

        # -------------------------------------------------------------------
        # Command 4: Dim / Color Lights
        # -------------------------------------------------------------------
        elif command_name.startswith("dim_lights_"):
            pct_str = command_name.replace("dim_lights_", "")
            pct = int(pct_str)
            self.lights_on = True
            self.light_brightness = pct / 100.0
            # Set warm white with adjusted duty cycle
            self.hw.set_rgb_led(self.light_brightness, self.light_brightness * 0.9, self.light_brightness * 0.6)
            self.hw.update_display("LIGHTS DIMMED", f"Brightness: {pct}%", "Mode: Warm White", timestamp)
            self.hw.speak(f"Dimming lights to {pct} percent.", response_key=command_name)
            return f"Lights dimmed to {pct}%."

        # -------------------------------------------------------------------
        # Command 5: Set a Timer
        # -------------------------------------------------------------------
        elif command_name.startswith("timer_"):
            mins_str = command_name.replace("timer_", "").replace("min", "")
            duration_mins = int(mins_str)
            self._start_timer(duration_mins)
            self.hw.update_display("TIMER STARTED", f"{duration_mins} minute(s)", "Counting down...", timestamp)
            self.hw.speak(f"{duration_mins} minute timer started.", response_key=command_name)
            return f"Timer set for {duration_mins} minute(s)."

        # -------------------------------------------------------------------
        # Command 6: Set an Alarm
        # -------------------------------------------------------------------
        elif command_name == "alarm_set":
            self.hw.set_rgb_led(0.8, 0.8, 0.0)  # Yellow indicator
            self.hw.update_display("ALARM SET", f"Time: {self.alarm_set}", "Active: Enabled", timestamp)
            self.hw.beep(0.2, 1200)
            self.hw.speak("Alarm set for seven AM.", response_key="alarm_set")
            return f"Alarm set for {self.alarm_set}."

        # -------------------------------------------------------------------
        # Command 7: Adjust Thermostat Temperature
        # -------------------------------------------------------------------
        elif command_name == "temp_cooler":
            self.temperature -= 1
            self.hw.set_rgb_led(0.0, 0.2, 1.0)  # Blue for cooling
            self.hw.update_display("THERMOSTAT", f"Target: {self.temperature} F", "Mode: COOLING", timestamp)
            self.hw.speak(f"Decreasing thermostat temperature to {self.temperature} degrees.", response_key="temp_cooler")
            return f"Decreased temperature to {self.temperature}°F."

        elif command_name == "temp_warmer":
            self.temperature += 1
            self.hw.set_rgb_led(1.0, 0.2, 0.0)  # Red for heating
            self.hw.update_display("THERMOSTAT", f"Target: {self.temperature} F", "Mode: HEATING", timestamp)
            self.hw.speak(f"Increasing thermostat temperature to {self.temperature} degrees.", response_key="temp_warmer")
            return f"Increased temperature to {self.temperature}°F."

        elif command_name == "temp_set_72":
            self.temperature = 72
            self.hw.set_rgb_led(0.0, 1.0, 0.2)  # Green for eco balance
            self.hw.update_display("THERMOSTAT", "Target: 72 F", "Mode: ECO AUTO", timestamp)
            self.hw.speak("Thermostat set to seventy-two degrees Fahrenheit.", response_key="temp_set_72")
            return "Thermostat set to 72°F."

        # -------------------------------------------------------------------
        # Command 9: Reminders and Lists
        # -------------------------------------------------------------------
        elif command_name == "reminders_check":
            r1 = self.reminders[0] if len(self.reminders) > 0 else "No reminders"
            r2 = self.reminders[1] if len(self.reminders) > 1 else ""
            self.hw.update_display("YOUR REMINDERS:", f"1. {r1[:18]}", f"2. {r2[:18]}", timestamp)
            self.hw.beep(0.1, 900)
            self.hw.speak("You have three active reminders. First: Submit A.I. two thirty-one M.E. two on time.", response_key="reminders_check")
            return f"Retrieved {len(self.reminders)} active reminders."

        # -------------------------------------------------------------------
        # Command 10: Calls and Messaging
        # -------------------------------------------------------------------
        elif command_name == "call_mom":
            self.hw.set_rgb_led(0.0, 1.0, 0.0)  # Green for calling
            self.hw.update_display("PHONE CALL", "Calling Mom...", "Status: Ringing...", timestamp)
            self.hw.beep(0.4, 440)  # Dial tone chime
            self.hw.speak("Initiating voice call to Mom.", response_key="call_mom")
            return "Initiating voice call to Mom."

        return f"Unhandled command: {command_name}"

    def duck_media(self) -> bool:
        """
        Ducks media playback volume down to 10% during active listening.
        Prevents acoustic self-interference and speaker feedback into microphone.
        """
        if self.media_playing and not self.is_ducked:
            self.pre_duck_volume = self.media_volume
            self.media_volume = max(5, int(self.pre_duck_volume * 0.15))
            self.is_ducked = True
            return True
        return False

    def unduck_media(self) -> bool:
        """
        Restores media playback volume back to its prior level upon command completion or timeout.
        """
        if self.is_ducked:
            self.media_volume = self.pre_duck_volume
            self.is_ducked = False
            return True
        return False

    def _start_timer(self, minutes: int):
        """Starts asynchronous timer thread with buzzer notification at expiry."""
        def _timer_worker():
            # For demonstration, 1 minute is simulated as 5 seconds countdown in fast-demo mode
            seconds_to_wait = max(3, minutes * 3)
            time.sleep(seconds_to_wait)
            self.hw.set_rgb_led(1.0, 0.0, 0.0)
            self.hw.update_display("TIMER EXPIRED!", f"{minutes} min completed", "Alarm Ringing!", "")
            self.hw.speak(f"{minutes} minute timer completed!", response_key=None)
            # Ring buzzer 3 times
            for _ in range(3):
                self.hw.beep(0.3, 1500)
                time.sleep(0.1)

        t = threading.Thread(target=_timer_worker, daemon=True)
        t.start()

    def trigger_wake(self) -> str:
        """Trigger visual and acoustic chime when wake word ('Hi Dandan' / 'Hello Dandan') is recognized."""
        # Duck media playback during active listening to prevent acoustic self-interference
        self.duck_media()
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.hw.set_rgb_led(0.0, 0.9, 1.0)  # Bright Cyan pulse for listening
        self.hw.beep(0.08, 1200)
        self.hw.beep(0.12, 1600)
        self.hw.update_display("ASSISTANT AWAKE", "Listening for command...", "Say: play music, etc.", timestamp)
        self.hw.speak("Hi Dandan! I'm listening. What can I do for you?", response_key="wake_word")
        return "Wake word ('Hi Dandan' / 'Hello Dandan') detected! Listening for command..."

    def trigger_timeout(self) -> str:
        """Called when listening window times out without receiving a command."""
        self.unduck_media()
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.hw.update_display("STANDBY / SLEEP", "Say 'Hi Dandan'...", "", timestamp)
        if self.lights_on:
            self.hw.set_rgb_led(self.light_brightness, self.light_brightness * 0.9, self.light_brightness * 0.6)
        else:
            self.hw.turn_off_led()
        self.hw.speak("Listening timed out. Going back to sleep.", response_key="timeout")
        return "Listening timed out. Returning to standby."

    def reset_to_standby(self):
        """Puts hardware into initial standby mode."""
        self.unduck_media()
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.hw.update_display("VCM ASSISTANT", "Standby / Sleeping", "Say 'Hi Dandan'...", timestamp)
        if self.lights_on:
            self.hw.set_rgb_led(self.light_brightness, self.light_brightness * 0.9, self.light_brightness * 0.6)
        else:
            self.hw.turn_off_led()
