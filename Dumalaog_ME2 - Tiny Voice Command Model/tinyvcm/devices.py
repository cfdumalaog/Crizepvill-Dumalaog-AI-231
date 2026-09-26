"""Low-voltage LED demo, real timers/clock, explicitly simulated appliance actions."""
from datetime import datetime, timedelta
from pathlib import Path
import time


class Devices:
    def __init__(self,gpio=False):
        self.brightness=0.; self.temperature_f=72; self.volume=50
        self.media=False; self.ducked=False; self.timer_end=None; self.alarm=None
        self.message='Say Hi Dandan, wait for LISTENING, then a short command.'
        self.rgb=None; self.buzzer=None
        if gpio:
            from gpiozero import RGBLED, Buzzer
            from gpiozero.pins.lgpio import LGPIOFactory
            factory=LGPIOFactory()
            self.rgb=RGBLED(17,27,22,pin_factory=factory)
            self.buzzer=Buzzer(23,pin_factory=factory)

    def set_led(self):
        if self.rgb: self.rgb.color=(self.brightness,)*3

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
        if label in ('lights_on','lights_off'):
            self.brightness=1. if label=='lights_on' else 0.; self.set_led()
            return 'Lights on.' if self.brightness else 'Lights off.'
        if label.startswith('dim_lights_'):
            self.brightness=int(label.rsplit('_',1)[1])/100; self.set_led()
            return f'Lights at {self.brightness:.0%}.'
        if label.startswith('timer_'):
            minutes=int(label.split('_')[1].replace('min',''))
            self.timer_end=time.monotonic()+minutes*60
            return f'{minutes}-minute timer started (real elapsed time).'
        if label=='alarm_set':
            now=datetime.now(); self.alarm=now.replace(hour=7,minute=0,second=0,microsecond=0)
            if self.alarm<=now: self.alarm+=timedelta(days=1)
            return f'Alarm set: {self.alarm:%Y-%m-%d %H:%M} local time.'
        if label=='question_time': return datetime.now().strftime('It is %I:%M %p.')
        if label=='question_weather': return 'Offline: no weather service connected. No live weather report available.'
        if label.startswith('temp_'):
            self.temperature_f=72 if label=='temp_set_72' else self.temperature_f+(-1 if label=='temp_cooler' else 1)
            return f'Demo thermostat target {self.temperature_f} F; no HVAC connected.'
        if label in ('play_music','media_resume','media_pause'):
            self.media=label!='media_pause'
            return 'Demo media playing.' if self.media else 'Demo media paused.'
        if label=='media_next': return 'Demo track restarted.'
        if label in ('volume_up','volume_down'):
            self.volume=max(0,min(100,self.volume+(10 if label=='volume_up' else -10)))
            return f'Demo volume {self.volume}%.'
        if label=='call_mom': return 'Demo call intent recognized. No call placed.'
        if label=='reminders_check': return 'Demo reminder: collect human recordings for AI 231 ME2.'
        return 'Command rejected.'

    def tick(self):
        expired=self.timer_end is not None and time.monotonic()>=self.timer_end
        alarm=self.alarm is not None and datetime.now()>=self.alarm
        if expired or alarm:
            if expired: self.timer_end=None
            if alarm: self.alarm=None
            self.message='Timer expired.' if expired else 'Alarm: it is seven AM.'
            if self.buzzer: self.buzzer.beep(.2,.2,n=3,background=True)
            return self.message

    def snapshot(self):
        sensor=None
        for p in Path('/sys/bus/w1/devices').glob('28-*/w1_slave'):
            try:
                value=p.read_text()
                if value.splitlines()[0].endswith('YES'): sensor=int(value.split('t=')[-1])/1000
            except (OSError,ValueError): pass
        return dict(lights_percent=round(self.brightness*100),thermostat_demo_f=self.temperature_f,
            measured_temperature_c=sensor,media_playing=self.media,volume=self.volume,ducked=self.ducked,
            timer_seconds=max(0,round(self.timer_end-time.monotonic())) if self.timer_end else None,
            alarm_local=str(self.alarm) if self.alarm else None)

    def close(self):
        if self.rgb: self.rgb.close()
        if self.buzzer: self.buzzer.close()
