"""
Configuration file for Tiny Voice Command Model (VCM).
Defines audio properties, command vocabulary, model hyperparameters,
and hardware GPIO pin assignments for Raspberry Pi 4/5.
"""

from pathlib import Path

# --- Base Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "rpi_deployment" / "models"
ASSETS_DIR = BASE_DIR / "rpi_deployment" / "assets"
DATA_DIR = BASE_DIR / "data"

# --- Audio Sampling & Frontend Parameters ---
SAMPLE_RATE = 16000           # 16 kHz standard for speech/command recognition
AUDIO_DURATION = 1.5          # Window length in seconds (24,000 samples)
NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)

# Feature Extraction (Log-Mel Spectrogram)
N_FFT = 512                   # 32 ms window
HOP_LENGTH = 160              # 10 ms frame shift
WIN_LENGTH = 400              # 25 ms analysis window
N_MELS = 40                   # 40 mel frequency bins (standard for tiny edge KWS)
F_MIN = 20.0
F_MAX = 8000.0

# Expected Spectrogram shape: (1, N_MELS, N_FRAMES) -> (1, 40, 151)
EXPECTED_FRAMES = int((NUM_SAMPLES - WIN_LENGTH) / HOP_LENGTH) + 1  # ~151

# --- Smart Device Command Vocabulary (Commands 1 to 10) ---
COMMAND_CLASSES = [
    # Command 1 & 8: Music & Media Control
    "play_music",             # #1: "play music"
    "media_pause",            # #8: "pause", "stop"
    "media_resume",           # #8: "resume", "continue"
    "media_next",             # #8: "next", "skip"
    "volume_up",              # #8: "volume up", "louder"
    "volume_down",            # #8: "volume down", "quieter"
    
    # Command 2: Ask a Question / Search
    "question_weather",       # #2: "what's the weather"
    "question_time",          # #2: "what time is it"
    
    # Command 3: Control Lights (IoT)
    "lights_on",              # #3: "turn on lights"
    "lights_off",             # #3: "turn off lights"
    
    # Command 4: Dim / Color Lights
    "dim_lights_25",          # #4: "dim lights to 25 percent"
    "dim_lights_50",          # #4: "dim lights to 50 percent"
    "dim_lights_75",          # #4: "dim lights to 75 percent"
    "dim_lights_100",         # #4: "dim lights to 100 percent"
    
    # Command 5: Set a Timer
    "timer_1min",             # #5: "set a timer for 1 minute"
    "timer_5min",             # #5: "set a timer for 5 minutes"
    "timer_10min",            # #5: "set a timer for 10 minutes"
    
    # Command 6: Set an Alarm
    "alarm_set",              # #6: "set an alarm for 7 am"
    
    # Command 7: Adjust Thermostat Temperature
    "temp_cooler",            # #7: "make it cooler", "decrease temperature"
    "temp_warmer",            # #7: "make it warmer", "increase temperature"
    "temp_set_72",            # #7: "set temperature to 72 degrees"
    
    # Command 9: Reminders & Lists
    "reminders_check",        # #9: "what are my reminders"
    
    # Command 10: Calls & Messaging
    "call_mom",               # #10: "call mom"
    
    # Wake Word (Awakens assistant from standby)
    "wake_word",              # "hi dandan", "hello dandan", "hey dandan"

    # Non-command classes (essential for continuous edge streaming)
    "_background_noise_",     # Ambient noise (fan, babble, room silence)
    "_silence_"               # Digital silence
]

NUM_CLASSES = len(COMMAND_CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(COMMAND_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(COMMAND_CLASSES)}

# --- Wake-Word & State Machine Parameters ---
WAKE_WORD_CLASS = "wake_word"
WAKE_TIMEOUT_SEC = 7.0        # Seconds assistant stays active waiting for a command
WAKE_CONFIDENCE = 0.60        # Confidence threshold specifically for wake word trigger

# --- Model & Training Hyperparameters ---
MODEL_NAME = "bc_resnet"      # Primary architecture: "bc_resnet" or "ds_cnn"
BC_RESNET_SCALE = 1           # BC-ResNet-1 (~65k parameters, ideal for RPi)
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 25

# --- Streaming Inference Engine Parameters ---
CONFIDENCE_THRESHOLD = 0.65   # Softmax probability threshold to trigger action
DETECTION_HOLDOFF_SEC = 1.0   # Debounce time (seconds) to prevent re-triggering
STREAM_HOP_SEC = 0.25         # Step size for sliding analysis window (250 ms)

# --- Raspberry Pi 4 / 5 Hardware GPIO Mapping (BCM Numbers) ---
# RGB LED
GPIO_LED_RED = 17             # Pin 11 (PWM capable)
GPIO_LED_GREEN = 27           # Pin 13 (PWM capable)
GPIO_LED_BLUE = 22            # Pin 15 (PWM capable)

# Active Buzzer
GPIO_BUZZER = 23              # Pin 16

# I2C OLED Display (SSD1306)
# SDA = GPIO 2 (Pin 3), SCL = GPIO 3 (Pin 5)
I2C_BUS = 1
OLED_I2C_ADDRESS = 0x3C
OLED_WIDTH = 128
OLED_HEIGHT = 64
