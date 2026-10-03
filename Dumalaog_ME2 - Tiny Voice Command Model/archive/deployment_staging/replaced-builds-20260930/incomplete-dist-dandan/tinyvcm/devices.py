"""Low-voltage LED demo, real timers/clock, explicitly simulated appliance actions."""
from datetime import datetime, timedelta
import math
from pathlib import Path
import time

from tinyvcm.audio_output import SystemVolumeOutput

try:
    from tinyvcm_antigrav.media_engine import MediaEngine
except ImportError:
    class MediaEngine:
        def get_current_track(self):
            return {"title": "Lofi Chill Beats", "artist": "Lofi Girl", "url": "https://stream.zeno.fm/f3wvbbqmdg8uv", "youtube_id": "jfKfPfyJRdk", "cover": ""}
        def next_track(self): return self.get_current_track()
        def previous_track(self): return self.get_current_track()
        def play_query(self, q): return self.get_current_track()


def fetch_live_weather(lat=14.6537, lon=121.0685):
    """Fetch current Open-Meteo weather without an API key or cloud model."""
    import json
    import urllib.request
    try:
        url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=Asia%2FManila'
        req = urllib.request.Request(url, headers={'User-Agent': 'TinyVCM/1.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        curr = data['current']
        wcode_map = {
            0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
            45: 'Fog', 48: 'Depositing rime fog', 51: 'Light drizzle', 53: 'Moderate drizzle',
            61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain', 80: 'Rain showers',
            95: 'Thunderstorm'
        }
        code = int(curr['weather_code'])
        temp = float(curr['temperature_2m'])
        humidity = float(curr['relative_humidity_2m'])
        wind = float(curr['wind_speed_10m'])
        if not all(math.isfinite(value) for value in (temp, humidity, wind)):
            raise ValueError('Open-Meteo returned a non-finite weather value')
        desc = wcode_map.get(code, f'weather code {code}')
        return f"Diliman, QC: {temp:.1f}°C, {desc}, {humidity:g}% humidity, wind {wind:.1f} km/h (live Open-Meteo API)."
    except Exception:
        return 'Live weather is unavailable right now. Check the internet connection or weather service.'


class Devices:
    def __init__(self, gpio=False, live_weather=False):
        self.brightness = 0.0
        self.light_color = 'white'
        self.temperature_f = 72
        self.temperature_c = 22
        self.volume = 50
        self.volume_output = SystemVolumeOutput(enabled=gpio)
        self.media = False
        self.playback_state = 'stopped'
        self.ducked = False
        self.timer_end = None
        self.alarm = None
        self.alarm_ringing = False
        self.active_alarms = []
        self.media_engine = MediaEngine()
        self.live_weather = bool(live_weather)
        self.message = "Say 'Hi Dandan', wait for LISTENING, then a command."
        self.rgb = None
        self.buzzer = None
        if gpio:
            from gpiozero import RGBLED
            from gpiozero.pins.lgpio import LGPIOFactory
            factory = LGPIOFactory()
            self.rgb = RGBLED(17, 27, 22, pin_factory=factory)

    def set_led(self):
        if self.rgb:
            base = self._light_components(self.light_color)
            self.rgb.color = tuple(component * self.brightness for component in base)

    def set_volume(self, percent):
        self.volume = max(0, min(100, int(percent)))
        self.volume_output.set_percent(self.volume)
        return f'Volume set to {self.volume}%.'

    @staticmethod
    def _light_components(color):
        return {
            'white': (1.0, 1.0, 1.0),
            'red': (1.0, 0.0, 0.0),
            'green': (0.0, 1.0, 0.0),
            'blue': (0.0, 0.0, 1.0),
        }.get(color, (1.0, 1.0, 1.0))

    def event(self,event):
        kind=event['event']
        if kind=='WAKE':
            self.ducked=True
            self.message='Listening. Say a short command now.'
        elif kind=='TIMEOUT':
            self.ducked=False; self.message='Timed out. Say Hi Dandan to try again.'
        elif kind=='COMMAND':
            self.ducked=False
            self.message=self.execute(event['prediction']['label'])
        elif kind=='REJECT':
            self.message=event['message']
        return self.message

    def execute(self,label):
        # Lights & Brightness
        if label in ('lights_on', 'LIGHT_ON'):
            self.brightness=1.0; self.set_led()
            return 'Lights on.'
        if label in ('lights_off', 'LIGHT_OFF'):
            self.brightness=0.0; self.set_led()
            return 'Lights off.'
        if label in ('dim_lights_100', 'BRIGHTNESS_100'):
            self.brightness=1.0; self.set_led()
            return 'Lights at 100%.'
        if label in ('dim_lights_60', 'BRIGHTNESS_60'):
            self.brightness=0.6; self.set_led()
            return 'Lights at 60%.'
        if label in ('dim_lights_20', 'BRIGHTNESS_20'):
            self.brightness=0.2; self.set_led()
            return 'Lights at 20%.'
        if label.startswith('dim_lights_'):
            self.brightness=int(label.rsplit('_',1)[1])/100; self.set_led()
            return f'Lights at {self.brightness:.0%}.'
        if label in ('COLOR_RED', 'COLOR_GREEN', 'COLOR_BLUE'):
            c = label.split('_')[1].lower()
            self.light_color = c
            if self.brightness == 0.0:
                self.brightness = 1.0
            self.set_led()
            return f'LED color set to {c}.'

        # Timers & Alarms
        if label.startswith('timer_'):
            minutes=int(label.split('_')[1].replace('min',''))
            self.timer_end=time.monotonic()+minutes*60
            return f'{minutes}-minute timer started (real elapsed time).'
        if label.startswith('TIMER_'):
            s = label.split('_')[1]
            secs = 10 if s == '10s' else (30 if s == '30s' else 60)
            self.timer_end = time.monotonic() + secs
            return f'{secs}-second timer started.'
        if label in ('alarm_set', 'ALARM_6_00AM', 'ALARM_8_00AM', 'ALARM_9_00PM'):
            hour = 6 if '6_00' in label else (8 if '8_00' in label else (21 if '9_00' in label else 7))
            now = datetime.now()
            target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            self.alarm = target
            time_fmt = target.strftime('%I:%M %p')
            alarm_entry = {"time_str": time_fmt, "hour": hour, "minute": 0, "label": label, "enabled": True}
            if not any(a['time_str'] == time_fmt for a in self.active_alarms):
                self.active_alarms.append(alarm_entry)
            return f'Alarm set for {time_fmt}.'

        # The classifier stays local; weather lookup is opt-in so offline operation remains available.
        if label in ('question_time', 'TIME'):
            return datetime.now().strftime('It is %I:%M %p.')
        if label == 'question_weather':
            if self.live_weather:
                return fetch_live_weather()
            return 'Demo weather intent recognized. No live weather connected.'
        if label == 'WEATHER':
            if self.live_weather:
                return fetch_live_weather()
            return 'Weather intent recognized. Live weather is unavailable in offline mode.'

        # Thermostat
        if label.startswith('TEMPERATURE_'):
            self.temperature_c = int(label.split('_')[1])
            return f'Demo thermostat target {self.temperature_c} °C; no HVAC connected.'
        if label.startswith('temp_'):
            self.temperature_f=72 if label=='temp_set_72' else self.temperature_f+(-1 if label=='temp_cooler' else 1)
            return f'Demo thermostat target {self.temperature_f} F; no HVAC connected.'

        # Media controls (Actual music streaming)
        if label in ('play_music', 'PLAY_MUSIC', 'media_resume'):
            self.media = True
            self.playback_state = 'playing'
            track = self.media_engine.get_current_track()
            return f"Playing '{track['title']}' by {track['artist']}."
        if label in ('PAUSE', 'media_pause'):
            self.media = False
            self.playback_state = 'paused'
            return 'Music paused.'
        if label == 'STOP':
            self.media = False
            self.playback_state = 'stopped'
            return 'Music stopped.'
        if label in ('media_next', 'NEXT'):
            self.media = True
            self.playback_state = 'playing'
            track = self.media_engine.next_track()
            return f"Playing next track: '{track['title']}'."
        if label in ('volume_up', 'VOLUME_UP'):
            return self.set_volume(self.volume + 10)
        if label in ('volume_down', 'VOLUME_DOWN'):
            return self.set_volume(self.volume - 10)

        # Communication & Reminders
        if label in ('call_mom', 'CALL'):
            return 'Demo call intent recognized. No call placed.'
        if label == 'MESSAGE':
            return 'Demo message intent recognized.'
        if label.startswith('CREATE_REMINDER_'):
            item = label.replace('CREATE_REMINDER_', '').replace('_', ' ').title()
            return f'Reminder set: {item}.'
        if label in ('reminders_check', 'LIST_REMINDERS'):
            return 'Reminders: 1. Drink water, 2. Exercise, 3. Complete AI 231 ME2.'

        return 'Command rejected.'

    def dismiss_alarm(self):
        self.alarm_ringing = False
        self.alarm = None
        self.message = 'Alarm dismissed.'
        return self.message

    def snooze_alarm(self, minutes=5):
        self.alarm_ringing = False
        self.alarm = datetime.now() + timedelta(minutes=minutes)
        self.message = f'Alarm snoozed for {minutes} minutes.'
        return self.message

    def play_query(self, query):
        self.media = True
        self.playback_state = 'playing'
        track = self.media_engine.play_query(query)
        self.message = f"Playing '{track['title']}' by {track['artist']}."
        return track

    def tick(self):
        expired = self.timer_end is not None and time.monotonic() >= self.timer_end
        alarm_due = self.alarm is not None and datetime.now() >= self.alarm
        if expired or alarm_due:
            if expired:
                self.timer_end = None
                self.message = 'Timer expired.'
            if alarm_due:
                self.alarm = None
                self.alarm_ringing = True
                self.message = f"Alarm ringing! It is {datetime.now().strftime('%I:%M %p')}."
            if self.buzzer:
                self.buzzer.beep(.2, .2, n=3, background=True)
            return self.message

    def snapshot(self):
        sensor = None
        for p in Path('/sys/bus/w1/devices').glob('28-*/w1_slave'):
            try:
                value = p.read_text()
                if value.splitlines()[0].endswith('YES'):
                    sensor = int(value.split('t=')[-1]) / 1000
            except (OSError, ValueError):
                pass
        curr_vol = round(self.volume * 0.15) if self.ducked else self.volume
        light_rgb = tuple(round(component * self.brightness, 3) for component in self._light_components(self.light_color))
        light_hex = '#%02x%02x%02x' % tuple(round(component * 255) for component in light_rgb)
        return dict(
            lights_percent=round(self.brightness * 100),
            lights_color=self.light_color,
            lights_rgb=list(light_rgb),
            lights_hex=light_hex,
            rgb_hardware_enabled=bool(self.rgb),
            thermostat_demo_f=self.temperature_f,
            measured_temperature_c=sensor,
            media_playing=self.media,
            playback_state=self.playback_state,
            volume=self.volume,
            effective_volume=curr_vol,
            volume_hardware_enabled=self.volume_output.mode != 'browser',
            volume_hardware_mode=self.volume_output.mode,
            volume_hardware_status=self.volume_output.status,
            ducked=self.ducked,
            current_track=self.media_engine.get_current_track(),
            thermostat_demo_c=self.temperature_c,
            timer_seconds=max(0, round(self.timer_end - time.monotonic())) if self.timer_end else None,
            alarm_local=str(self.alarm) if self.alarm else None,
            alarm_ringing=self.alarm_ringing,
            active_alarms=self.active_alarms,
            clock_str=datetime.now().strftime('%I:%M:%S %p')
        )

    def close(self):
        if self.rgb: self.rgb.close()
        if self.buzzer: self.buzzer.close()
