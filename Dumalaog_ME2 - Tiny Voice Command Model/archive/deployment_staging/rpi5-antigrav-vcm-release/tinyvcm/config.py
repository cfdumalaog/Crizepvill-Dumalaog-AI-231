from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
SAMPLES = 24000
HOP = 160
NFFT = 512
WINDOW = 400
MELS = 40
LABELS = [
    'play_music', 'media_pause', 'media_resume', 'media_next', 'volume_up', 'volume_down',
    'question_weather', 'question_time', 'lights_on', 'lights_off',
    'dim_lights_25', 'dim_lights_50', 'dim_lights_75', 'dim_lights_100',
    'timer_1min', 'timer_5min', 'timer_10min', 'alarm_set',
    'temp_cooler', 'temp_warmer', 'temp_set_72', 'reminders_check', 'call_mom',
    'wake_word', '_background_noise_', '_silence_',
]
PHRASES = {
    'play_music': 'Play music', 'media_pause': 'Pause music', 'media_resume': 'Resume music',
    'media_next': 'Next song', 'volume_up': 'Volume up', 'volume_down': 'Volume down',
    'question_weather': "What is the weather", 'question_time': 'What time is it',
    'lights_on': 'Turn on the lights', 'lights_off': 'Turn off the lights',
    'dim_lights_25': 'Dim lights to twenty five percent',
    'dim_lights_50': 'Dim lights to fifty percent',
    'dim_lights_75': 'Dim lights to seventy five percent',
    'dim_lights_100': 'Lights to maximum',
    'timer_1min': 'Timer for one minute', 'timer_5min': 'Timer for five minutes',
    'timer_10min': 'Timer for ten minutes', 'alarm_set': 'Set alarm for seven AM',
    'temp_cooler': 'Make it cooler', 'temp_warmer': 'Make it warmer',
    'temp_set_72': 'Set temperature to seventy two degrees',
    'reminders_check': 'Check my reminders', 'call_mom': 'Call mom',
    'wake_word': 'Hi Dandan / Hello Dandan',
    '_background_noise_': 'Record room noise: fan, typing, TV (do not say a command)',
    '_silence_': 'Stay quiet',
    '_unknown_': 'Unrelated speech, e.g. I am reading a book / Hello Daniel / Good morning',
}
