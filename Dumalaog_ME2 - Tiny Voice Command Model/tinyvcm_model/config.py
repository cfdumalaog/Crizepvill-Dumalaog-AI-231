"""Configuration for ME2 Spoken Command Dataset TinyDSCNN-48 training."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPTION_B_DATA = ROOT / 'data' / 'option_b'
MANIFEST_PATH = OPTION_B_DATA / 'manifest.csv'

# Audio & Spectrogram Parameters
SR = 16000
DURATION = 2.5  # 2.5 seconds window accommodates 89% of full spoken command sentences
SAMPLES = int(SR * DURATION)  # 40,000 samples
WINDOW = int(SR * 0.025)     # 400 samples (25 ms)
HOP = int(SR * 0.010)        # 160 samples (10 ms)
NFFT = 512
MELS = 40
TIME_STEPS = SAMPLES // HOP + 1  # 251 frames
FEATURE_SHAPE = (1, MELS, TIME_STEPS)  # (1, 40, 251)

# Architecture Hyperparameters
CHANNELS = 48
CLASSES = 31  # 31 fixed ME2 spoken command classes
DROPOUT = 0.15

# Live execution gates shared by runtime, release metadata and audit workbook.
# Offline classification scores use argmax without these rejection gates.
INTENT_CONFIDENCE_THRESHOLD = 0.68
INTENT_MARGIN_THRESHOLD = 0.15

# ME2 command labels (alphabetically sorted, 31 classes)
LABELS = [
    'ALARM_6_00AM',
    'ALARM_8_00AM',
    'ALARM_9_00PM',
    'BRIGHTNESS_100',
    'BRIGHTNESS_20',
    'BRIGHTNESS_60',
    'CALL',
    'COLOR_BLUE',
    'COLOR_GREEN',
    'COLOR_RED',
    'CREATE_REMINDER_DRINK_WATER',
    'CREATE_REMINDER_EXERCISE',
    'CREATE_REMINDER_STUDY',
    'LIGHT_OFF',
    'LIGHT_ON',
    'LIST_REMINDERS',
    'MESSAGE',
    'NEXT',
    'PAUSE',
    'PLAY_MUSIC',
    'STOP',
    'TEMPERATURE_18',
    'TEMPERATURE_22',
    'TEMPERATURE_26',
    'TIME',
    'TIMER_10s',
    'TIMER_1m',
    'TIMER_30s',
    'VOLUME_DOWN',
    'VOLUME_UP',
    'WEATHER'
]

# Experimental 32-class extension with custom wake word gating
WAKE_CLASSES = 32
WAKE_LABELS = LABELS + ['WAKE_WORD']
