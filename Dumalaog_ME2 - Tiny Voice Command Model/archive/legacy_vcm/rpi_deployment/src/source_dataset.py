"""
Dataset Sourcing and Generator for Tiny Voice Command Model (VCM).
Sources and constructs a multi-speaker, phonetically varied 16kHz WAV audio dataset
on disk covering the top 10 smart device commands (24 classes) plus environmental noise.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torchaudio.transforms as T
import torch

from .config import (
    BASE_DIR,
    DATA_DIR,
    SAMPLE_RATE,
    AUDIO_DURATION,
    NUM_SAMPLES,
    COMMAND_CLASSES
)

# Text phrasing variations for each command category (Ranked 1 to 10)
COMMAND_PHRASES = {
    # Command 1 & 8: Music & Media Control
    "play_music": [
        "play music",
        "play some music",
        "start the music",
        "play playlist"
    ],
    "media_pause": [
        "pause",
        "pause the music",
        "stop playback"
    ],
    "media_resume": [
        "resume",
        "continue music",
        "unpause"
    ],
    "media_next": [
        "next",
        "next song",
        "skip track"
    ],
    "volume_up": [
        "volume up",
        "louder",
        "increase volume"
    ],
    "volume_down": [
        "volume down",
        "quieter",
        "decrease volume"
    ],

    # Command 2: Ask Question / Search
    "question_weather": [
        "what's the weather",
        "what is the weather",
        "weather forecast",
        "how is the weather"
    ],
    "question_time": [
        "what time is it",
        "tell me the time",
        "what is the time",
        "current time"
    ],

    # Command 3: Control Lights (IoT)
    "lights_on": [
        "turn on lights",
        "turn on the lights",
        "lights on",
        "switch on lights"
    ],
    "lights_off": [
        "turn off lights",
        "turn off the lights",
        "lights off",
        "switch off lights"
    ],

    # Command 4: Dim / Color Lights
    "dim_lights_25": [
        "dim lights to 25 percent",
        "set brightness to 25 percent"
    ],
    "dim_lights_50": [
        "dim lights to 50 percent",
        "set brightness to 50 percent",
        "dim lights to half"
    ],
    "dim_lights_75": [
        "dim lights to 75 percent",
        "set brightness to 75 percent"
    ],
    "dim_lights_100": [
        "dim lights to 100 percent",
        "lights to maximum",
        "brighten lights to 100 percent"
    ],

    # Command 5: Set a Timer
    "timer_1min": [
        "set a timer for 1 minute",
        "timer for 1 minute",
        "start a 1 minute timer"
    ],
    "timer_5min": [
        "set a timer for 5 minutes",
        "timer for 5 minutes",
        "start a 5 minute timer"
    ],
    "timer_10min": [
        "set a timer for 10 minutes",
        "timer for 10 minutes",
        "start a 10 minute timer"
    ],

    # Command 6: Set an Alarm
    "alarm_set": [
        "set an alarm for 7 am",
        "alarm for 7 am",
        "wake me up at 7 am"
    ],

    # Command 7: Adjust Thermostat Temperature
    "temp_cooler": [
        "make it cooler",
        "decrease temperature",
        "turn down thermostat"
    ],
    "temp_warmer": [
        "make it warmer",
        "increase temperature",
        "turn up thermostat"
    ],
    "temp_set_72": [
        "set temperature to 72 degrees",
        "thermostat 72 degrees"
    ],

    # Command 9: Reminders & Lists
    "reminders_check": [
        "what are my reminders",
        "check my reminders",
        "show my reminders",
        "read my reminders"
    ],

    # Command 10: Calls & Messaging
    "call_mom": [
        "call mom",
        "phone mom",
        "dial mom"
    ],

    # Wake Word (Awakens assistant from standby)
    "wake_word": [
        "hi dandan",
        "hello dandan",
        "hey dandan",
        "hi dan dan",
        "hello dan dan",
        "hey dan dan"
    ]
}


def create_ambient_noise_samples(output_dir: Path, count: int = 50):
    """Synthesizes physical WAV files of ambient noises (fan hum, pink room noise, babble)."""
    noise_dir = output_dir / "_background_noise_"
    silence_dir = output_dir / "_silence_"
    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(silence_dir, exist_ok=True)

    t = np.linspace(0, AUDIO_DURATION, NUM_SAMPLES, endpoint=False)

    # 1. Fan / AC low-frequency motor hum
    for i in range(count // 3):
        f0 = 60.0 + np.random.uniform(-2, 2)
        hum = 0.05 * np.sin(2 * np.pi * f0 * t) + 0.02 * np.sin(2 * np.pi * 2 * f0 * t)
        noise = hum + np.random.normal(0, 0.015, NUM_SAMPLES)
        sf.write(str(noise_dir / f"fan_hum_{i:03d}.wav"), noise.astype(np.float32), SAMPLE_RATE)

    # 2. Pink room noise / ambient air rumble
    for i in range(count // 3):
        white = np.random.normal(0, 0.04, NUM_SAMPLES)
        pink = np.convolve(white, [0.1, 0.2, 0.3, 0.2, 0.1], mode='same')
        sf.write(str(noise_dir / f"pink_noise_{i:03d}.wav"), pink.astype(np.float32), SAMPLE_RATE)

    # 3. Multi-talker babble approximation
    for i in range(count // 3):
        babble = np.zeros(NUM_SAMPLES, dtype=np.float32)
        for _ in range(5):
            freq = np.random.uniform(150, 600)
            babble += 0.01 * np.sin(2 * np.pi * freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
        babble += np.random.normal(0, 0.01, NUM_SAMPLES)
        sf.write(str(noise_dir / f"babble_{i:03d}.wav"), babble.astype(np.float32), SAMPLE_RATE)

    # 4. Digital silence files
    for i in range(count):
        silence = np.random.normal(0, 0.0002, NUM_SAMPLES).astype(np.float32)
        sf.write(str(silence_dir / f"silence_{i:03d}.wav"), silence, SAMPLE_RATE)

    print(f"[DATASET] Sourced background noise and silence files in {noise_dir.name} and {silence_dir.name}")


def source_multi_speaker_dataset(output_dir: Path = DATA_DIR / "dataset"):
    """
    Renders speech audio WAV files for all command classes using distinct SAPI voices
    with varied speech rates and phrase structures.
    """
    os.makedirs(output_dir, exist_ok=True)
    temp_wav_path = output_dir / "_temp_render.wav"

    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        file_stream = win32com.client.Dispatch("SAPI.SpFileStream")
        installed_voices = speaker.GetVoices()
        num_voices = len(installed_voices)
        print(f"[DATASET] Found {num_voices} native acoustic voices installed.")
    except Exception as e:
        print(f"[DATASET] Warning: Could not initialize SAPI ({e}). Falling back to algorithmic synthesis.")
        return

    # Speeds to vary: -2 (slow), 0 (normal), +2 (fast)
    speech_rates = [-2, 0, 2]

    total_files_created = 0
    start_time = time.time()

    print("[DATASET] Sourcing multi-speaker acoustic recordings for all 24 command classes...")
    for cls_name, phrases in COMMAND_PHRASES.items():
        cls_dir = output_dir / cls_name
        os.makedirs(cls_dir, exist_ok=True)
        file_idx = 0

        for voice_idx in range(num_voices):
            speaker.Voice = installed_voices.Item(voice_idx)
            voice_name = installed_voices.Item(voice_idx).GetDescription().split()[1]

            for rate in speech_rates:
                speaker.Rate = rate

                for phrase in phrases:
                    # Render phrase to temporary WAV
                    try:
                        file_stream.Open(str(temp_wav_path), 3)  # 3 = SSFMCreateForWrite
                        speaker.AudioOutputStream = file_stream
                        speaker.Speak(phrase)
                        file_stream.Close()

                        # Read rendered audio and resample / standardize to 16kHz, 1.5s
                        raw_data, orig_sr = sf.read(str(temp_wav_path))
                        raw_tensor = torch.from_numpy(raw_data).float()
                        if raw_tensor.dim() == 2:
                            raw_tensor = raw_tensor.mean(dim=1)

                        # Resample to 16,000 Hz if needed
                        if orig_sr != SAMPLE_RATE:
                            resampler = T.Resample(orig_sr, SAMPLE_RATE)
                            audio_16k = resampler(raw_tensor).numpy()
                        else:
                            audio_16k = raw_tensor.numpy()

                        # Center or fit into exactly 1.5s (24,000 samples)
                        curr_len = len(audio_16k)
                        standard_audio = np.zeros(NUM_SAMPLES, dtype=np.float32)

                        if curr_len >= NUM_SAMPLES:
                            standard_audio = audio_16k[:NUM_SAMPLES]
                        else:
                            # Center the speech utterance in the window
                            start_pad = (NUM_SAMPLES - curr_len) // 2
                            standard_audio[start_pad:start_pad + curr_len] = audio_16k

                        # Peak normalization
                        peak = np.max(np.abs(standard_audio))
                        if peak > 0:
                            standard_audio = (standard_audio / peak) * 0.9

                        base_tensor = torch.from_numpy(standard_audio).float().unsqueeze(0)

                        # Variant 1: Clean centered
                        f1 = cls_dir / f"{cls_name}_{voice_name}_r{rate}_{file_idx:02d}_clean.wav"
                        sf.write(str(f1), standard_audio.astype(np.float32), SAMPLE_RATE)
                        file_idx += 1
                        total_files_created += 1

                        # Variant 2: Time shift left (100ms)
                        shift_samples = int(SAMPLE_RATE * 0.1)
                        shifted_l = np.pad(standard_audio[shift_samples:], (0, shift_samples), mode='constant')
                        f2 = cls_dir / f"{cls_name}_{voice_name}_r{rate}_{file_idx:02d}_shiftL.wav"
                        sf.write(str(f2), shifted_l.astype(np.float32), SAMPLE_RATE)
                        file_idx += 1
                        total_files_created += 1

                        # Variant 3: Time shift right (100ms)
                        shifted_r = np.pad(standard_audio[:-shift_samples], (shift_samples, 0), mode='constant')
                        f3 = cls_dir / f"{cls_name}_{voice_name}_r{rate}_{file_idx:02d}_shiftR.wav"
                        sf.write(str(f3), shifted_r.astype(np.float32), SAMPLE_RATE)
                        file_idx += 1
                        total_files_created += 1

                        # Variant 4: Pitch shift up (+2 semitones)
                        import torchaudio.functional as F_a
                        p_up = F_a.pitch_shift(base_tensor, SAMPLE_RATE, n_steps=2)[0].numpy()
                        p_up = p_up * 0.9 / (np.max(np.abs(p_up)) + 1e-6)
                        f4 = cls_dir / f"{cls_name}_{voice_name}_r{rate}_{file_idx:02d}_pitchUp.wav"
                        sf.write(str(f4), p_up.astype(np.float32), SAMPLE_RATE)
                        file_idx += 1
                        total_files_created += 1

                        # Variant 5: Pitch shift down (-2 semitones)
                        p_dn = F_a.pitch_shift(base_tensor, SAMPLE_RATE, n_steps=-2)[0].numpy()
                        p_dn = p_dn * 0.9 / (np.max(np.abs(p_dn)) + 1e-6)
                        f5 = cls_dir / f"{cls_name}_{voice_name}_r{rate}_{file_idx:02d}_pitchDn.wav"
                        sf.write(str(f5), p_dn.astype(np.float32), SAMPLE_RATE)
                        file_idx += 1
                        total_files_created += 1

                    except Exception as err:
                        print(f"Error rendering '{phrase}': {err}")

    # Clean up temp file
    if temp_wav_path.exists():
        os.remove(str(temp_wav_path))

    # Source background noise and silence
    create_ambient_noise_samples(output_dir, count=90)
    total_files_created += 180

    duration = time.time() - start_time
    print(f"[DATASET] Sourcing complete! Created {total_files_created} physical WAV files in {duration:.1f}s.")
    print(f"[DATASET] Dataset stored at: {output_dir}")


if __name__ == "__main__":
    source_multi_speaker_dataset()
