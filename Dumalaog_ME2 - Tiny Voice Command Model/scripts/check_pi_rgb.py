"""Check the manually running Pi RGB API and pin levels; leave lights off."""
import json
from pathlib import Path
import subprocess
import re
import sys
import time
from urllib.request import Request, urlopen

BASE = 'http://127.0.0.1:7860'
results = []


def command(label, expected):
    request = Request(BASE + '/api/light/demo', data=json.dumps({'command':label}).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urlopen(request, timeout=5) as response:
        state = json.load(response)['devices']
    assert state['rgb_hardware_enabled'] is True
    assert state['lights_rgb'] == expected, (label,state)
    expected_levels = {17:int(expected[0]),27:int(expected[1]),22:int(expected[2])}
    consecutive = 0
    # lgpio applies PWM changes asynchronously. Require two settled readings.
    for _ in range(30):
        time.sleep(0.05)
        pins = subprocess.run(['pinctrl','get','17,27,22'],capture_output=True,text=True,check=True).stdout.strip()
        levels = {int(pin):int(level == 'hi') for pin,level in re.findall(r'^(\d+):.*\|\s+(hi|lo)',pins,re.MULTILINE)}
        consecutive = consecutive + 1 if levels == expected_levels else 0
        if consecutive == 2:
            break
    assert consecutive == 2, (label,expected_levels,pins)
    results.append(dict(command=label,expected_rgb=expected,actual_rgb=state['lights_rgb'],color=state['lights_color'],verified_pin_levels=levels,pins=pins))


try:
    for label, expected in [('COLOR_RED',[1.,0.,0.]),('COLOR_GREEN',[0.,1.,0.]),('COLOR_BLUE',[0.,0.,1.])]:
        command(label,expected)
        command('LIGHT_OFF',[0.,0.,0.])
        command('LIGHT_ON',[1.,1.,1.])
finally:
    command('LIGHT_OFF',[0.,0.,0.])
report = dict(status='passed',protocol='Direct GPIO API test, bypasses speech models; pin levels observed, optical output not measured.',results=results)
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
