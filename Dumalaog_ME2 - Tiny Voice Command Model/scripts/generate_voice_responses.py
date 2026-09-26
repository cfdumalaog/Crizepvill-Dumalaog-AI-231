"""
Synthesizes pristine 16 kHz WAV voiced response audio files for all smart assistant
commands, wake phrase acknowledgments, and system alerts.
Output target: rpi_deployment/assets/voice_responses/
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

# Target directory
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "rpi_deployment" / "assets" / "voice_responses"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000

VOICE_RESPONSES = {
    # Wake phrase acknowledgment
    "wake_word": "Hi Dandan! I'm listening. What can I do for you?",
    "timeout": "Listening timed out. Going back to sleep.",

    # Command 1 & 8: Music & Media
    "play_music": "Playing chill beats playlist now.",
    "media_pause": "Music playback paused.",
    "media_resume": "Music playback resumed.",
    "media_next": "Skipping to next track.",
    "volume_up": "Increasing volume to eighty percent.",
    "volume_down": "Decreasing volume to sixty percent.",

    # Command 2: Questions & Queries
    "question_weather": "The weather in Diliman is twenty-nine degrees Celsius and partly cloudy with seventy-five percent humidity.",
    "question_time": "Checking current time for you.",

    # Command 3: Lights On / Off
    "lights_on": "Turning on the lights.",
    "lights_off": "Turning off the lights.",

    # Command 4: Dim Lights
    "dim_lights_25": "Dimming lights to twenty-five percent.",
    "dim_lights_50": "Dimming lights to fifty percent.",
    "dim_lights_75": "Dimming lights to seventy-five percent.",
    "dim_lights_100": "Brightening lights to one hundred percent.",

    # Command 5: Timers
    "timer_1min": "One minute timer started.",
    "timer_5min": "Five minute timer started.",
    "timer_10min": "Ten minute timer started.",

    # Command 6: Alarms
    "alarm_set": "Alarm set for seven AM.",

    # Command 7: Thermostat
    "temp_cooler": "Decreasing thermostat temperature to seventy-one degrees.",
    "temp_warmer": "Increasing thermostat temperature to seventy-three degrees.",
    "temp_set_72": "Thermostat set to seventy-two degrees Fahrenheit.",

    # Command 9: Reminders
    "reminders_check": "You have three active reminders. First: Submit A.I. two thirty-one M.E. two on time.",

    # Command 10: Calls
    "call_mom": "Initiating voice call to Mom."
}


def generate_all_responses():
    print("=" * 70)
    print("   GENERATING PRISTINE 16 kHz VOICED SPEECH RESPONSE ASSETS")
    print("=" * 70)
    print(f"Target Directory: {OUTPUT_DIR}")
    print(f"Total Response Clips: {len(VOICE_RESPONSES)}")

    temp_wav = OUTPUT_DIR / "_temp_speech.wav"

    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        file_stream = win32com.client.Dispatch("SAPI.SpFileStream")
        voices = speaker.GetVoices()
        # Find a natural voice like Zira or Hazel or David
        selected_voice = voices.Item(0)
        for i in range(len(voices)):
            desc = voices.Item(i).GetDescription().lower()
            if "zira" in desc or "hazel" in desc or "female" in desc:
                selected_voice = voices.Item(i)
                break
        speaker.Voice = selected_voice
        speaker.Rate = 0  # Normal conversational speed
        print(f"Using Voice: {selected_voice.GetDescription()}")
    except Exception as e:
        print(f"[ERROR] Could not initialize SAPI: {e}")
        return

    count = 0
    for key, text in VOICE_RESPONSES.items():
        out_file = OUTPUT_DIR / f"{key}.wav"
        try:
            # 3 = SSFMCreateForWrite
            file_stream.Open(str(temp_wav), 3)
            speaker.AudioOutputStream = file_stream
            speaker.Speak(text)
            file_stream.Close()

            # Read and standardize to 16 kHz mono
            data, orig_sr = sf.read(str(temp_wav))
            tensor = torch.from_numpy(data).float()
            if tensor.dim() == 2:
                tensor = tensor.mean(dim=1)
            if orig_sr != SAMPLE_RATE:
                resampler = T.Resample(orig_sr, SAMPLE_RATE)
                tensor = resampler(tensor)

            # Peak normalize with slight headroom
            max_val = torch.max(torch.abs(tensor))
            if max_val > 0:
                tensor = tensor / max_val * 0.92

            sf.write(str(out_file), tensor.numpy().astype(np.float32), SAMPLE_RATE)
            count += 1
            print(f"  [{count:02d}/{len(VOICE_RESPONSES):02d}] Synthesized '{key}.wav' -> \"{text[:45]}...\"")
        except Exception as e:
            print(f"  [FAIL] Failed for {key}: {e}")

    if temp_wav.exists():
        try:
            temp_wav.unlink()
        except Exception:
            pass

    print(f"\nSuccessfully generated {count} voiced audio response files in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_responses()
