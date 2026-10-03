# VCM benchmark - Dumalaog ME2 - 20261003-124806

Wake word: **Hi Dandan** - trials: 109 with the wake word + 16 without - shuffle seed: 231 - connection: ssh - holdout: huggingface

Pi log file: ~/vcm_benchmark/me2-20261003-125009-32514.log

Mic check (Pi input plughw:2,0): signal-to-noise 21.8 dB, laptop speech -23.2 dBFS, room noise -45.0 dBFS - ok

## At a glance

|                                   | overall       | real voice   | synthetic voice |
|-----------------------------------|---------------|--------------|-----------------|
| intent accuracy (19)              | 36.7%         | 16.3%        | 53.3%           |
| command accuracy (93)             | 33.9%         | 12.2%        | 51.7%           |
| false accept (out of scope fired) | 62.5% (10/16) | 70.0% (7/10) | 50.0% (3/6)     |
| false reject (command ignored)    | 36.6%         | 53.8%        | 24.1%           |
| false wake (no wake word, fired)  | 50.0% (8/16)  | 50.0% (3/6)  | 50.0% (5/10)    |
| slot exact                        | 88.0%         | 50.0%        | 95.2%           |
| latency p95                       | 6.99 s        | 7.82 s       | 5.86 s          |

**Pi:** real-time factor 0.003 (p95 0.004), inference 8 ms, CPU temp max 45.2 C, 42.1 MFLOP per inference

**Check:**

- the Pi answered only 63.3% of commands: check volume, distance, wake word and the log path
- 49 extra fire(s): more than one command for one utterance

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject)  | 93 commands (+reject) |
|---------------------------------------------------|-----------------------|-----------------------|
| accuracy                                          | 36.7%                 | 33.9%                 |
| balanced accuracy                                 | 30.8%                 | 33.4%                 |
| precision (macro)                                 | 52.8%                 | 31.8%                 |
| recall (macro)                                    | 30.8%                 | 33.4%                 |
| F1 (macro)                                        | 34.7%                 | 32.2%                 |
| F2 (macro)                                        | 31.5%                 | 32.7%                 |
| false accept rate (OOS fired)                     | 62.5%                 | 62.5%                 |
| false reject rate (in-scope silent/rejected)      | 36.6%                 | 36.6%                 |
| misfire rate (wrong command fired)                | 26.9%                 | 30.1%                 |
| accuracy 95% CI                                   | [28-46%]              | [26-43%]              |
| false accept 95% CI                               | [39-82%] (10/16)      | [39-82%]              |
| false wake rate (command without wake word fired) | 50.0% [28-72%] (8/16) | 50.0% [28-72%] (8/16) |

Responses: 63.3% of trials fired a command; no response: 40; extra fires: 49; wake detect rate: -

**False wakes (no wake word, Pi fired):** Brightness 20 percent -> BRIGHTNESS; Increase the volume -> VOLUME_UP; Make a phone call -> CALL; Pause the music -> PLAY_MUSIC; Play a song -> PLAY_MUSIC; Switch color to green -> COLOR; Timer 30 seconds -> CREATE_REMINDER; Wake me up at 8 AM -> ALARM

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice    | synthetic voice |
|--------------------------------|----------------|---------------|-----------------|
| clips (with wake word)         | 109            | 49            | 60              |
| **19 intents** accuracy        | 36.7% [28-46%] | 16.3% [9-29%] | 53.3% [41-65%]  |
| balanced accuracy              | 30.8%          | 14.7%         | 49.3%           |
| F1 (macro)                     | 34.7%          | 12.0%         | 49.9%           |
| F2 (macro)                     | 31.5%          | 13.4%         | 48.7%           |
| false accept rate              | 62.5% (10/16)  | 70.0% (7/10)  | 50.0% (3/6)     |
| false reject rate              | 36.6%          | 53.8%         | 24.1%           |
| misfire rate                   | 26.9%          | 33.3%         | 22.2%           |
| **93 commands** accuracy       | 33.9%          | 12.2%         | 51.7%           |
| balanced accuracy              | 33.4%          | 8.2%          | 51.8%           |
| F1 (macro)                     | 32.2%          | 7.9%          | 50.2%           |
| F2 (macro)                     | 32.7%          | 8.1%          | 51.0%           |
| misfire rate                   | 30.1%          | 38.5%         | 24.1%           |
| slot exact (intent right)      | 88.0% (n=25)   | 50.0% (n=4)   | 95.2% (n=21)    |
| latency p50 / p95              | 0.74 / 6.99 s  | 0.64 / 7.82 s | 0.75 / 5.86 s   |
| false wake rate (no wake word) | 50.0% (8/16)   | 50.0% (3/6)   | 50.0% (5/10)    |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 7  | 85.7%  | 17.1 min       | 0.019          | 0.071         | 0.071     |
| BRIGHTNESS      | 2  | 100.0% | 0.0 %          | 0.000          | 0.000         | 0.000     |
| COLOR           | 5  | 80.0%  | -              | -              | 0.200         | 0.160     |
| CREATE_REMINDER | 5  | 80.0%  | -              | -              | 0.175         | 0.200     |
| TEMPERATURE     | 3  | 100.0% | 0.0 deg        | 0.000          | 0.000         | 0.000     |
| TIMER           | 3  | 100.0% | 0.0 s          | 0.000          | 0.000         | 0.000     |
| ALL             | 25 | 88.0%  | -              | 0.009          | 0.095         | 0.092     |

## Raspberry Pi

- **Raspberry Pi 5 Model B Rev 1.0**, 4 cores  up to 2400.0 MHz, RAM 4045.1 MB, Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-2712, Python 3.13.5
- packages: onnxruntime 1.30.0, numpy 2.2.4, sounddevice 0.5.6

| metric                                      | mean / p95 / max            |
|---------------------------------------------|-----------------------------|
| response latency (command end -> Pi output) | 0.953 / 6.989 / 10.490 s    |
| latency p50 / p99                           | 0.739 / 9.187 s             |
| inference time (Pi-reported)                | 7.9 / 9.3 / 10.3 ms         |
| real-time factor (infer / audio window)     | 0.003 / 0.004 / 0.004       |
| CPU temperature                             | 41.0 / 43.0 / 45.2 C        |
| CPU use, whole Pi                           | 12.2 / 20.9 / 31.6 %        |
| CPU use, your runtime process               | -                           |
| RAM (RSS), your runtime process             | -                           |
| RAM used, whole Pi                          | 2164.7 / 2215.8 / 2285.2 MB |
| CPU clock                                   | 1734 / 2400 / 2400 MHz      |
| load average (1 min)                        | 0.80 / 1.96 / 2.41          |
| runtime CPU-seconds per second of speech    | -                           |
| runtime CPU share of wall time              | -                           |
| throttling flags seen                       | none                        |
| test wall time                              | 33.8 min                    |
| model parameters                            | 15,971                      |
| model size                                  | 0.04 MB                     |
| model FLOPs per inference                   | 42.1 MFLOP                  |
| effective GFLOP/s (FLOPs / mean infer time) | 5.33                        |

## Most frequent confusions

**intent level:** COLOR -> REJECT (4); CREATE_REMINDER -> REJECT (4); BRIGHTNESS -> REJECT (3); REJECT -> CREATE_REMINDER (3); TIMER -> REJECT (3); LIST_REMINDERS -> REJECT (2); TIMER -> CREATE_REMINDER (2); TEMPERATURE -> REJECT (2); PAUSE -> REJECT (2); REJECT -> TEMPERATURE (2)

**command level:** REJECT -> Reminder Drink water (3); REJECT -> Temperature 18 degrees (2); Brightness 100 percent -> REJECT (1); Skip song -> Reminder Drink water (1); Create a reminder to Study -> Reminder Drink water (1); Play some music -> Reminder Drink water (1); Adjust brightness to 100 percent -> Change color to Red (1); End playback -> Reminder Drink water (1); REJECT -> Stop (1); Lights on -> REJECT (1)

## Per-intent scores

| class           | n  | precision | recall | F1    | F2    |
|-----------------|----|-----------|--------|-------|-------|
| ALARM           | 9  | 77.8%     | 77.8%  | 77.8% | 77.8% |
| BRIGHTNESS      | 9  | 100.0%    | 22.2%  | 36.4% | 26.3% |
| CALL            | 3  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| COLOR           | 9  | 50.0%     | 55.6%  | 52.6% | 54.3% |
| CREATE_REMINDER | 9  | 27.8%     | 55.6%  | 37.0% | 46.3% |
| LIGHT_OFF       | 3  | 50.0%     | 33.3%  | 40.0% | 35.7% |
| LIGHT_ON        | 3  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| LIST_REMINDERS  | 3  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| MESSAGE         | 3  | 100.0%    | 33.3%  | 50.0% | 38.5% |
| NEXT            | 3  | 100.0%    | 33.3%  | 50.0% | 38.5% |
| PAUSE           | 3  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| PLAY_MUSIC      | 3  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| REJECT          | 16 | 15.0%     | 37.5%  | 21.4% | 28.8% |
| STOP            | 3  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| TEMPERATURE     | 9  | 60.0%     | 33.3%  | 42.9% | 36.6% |
| TIME            | 3  | 100.0%    | 33.3%  | 50.0% | 38.5% |
| TIMER           | 9  | 100.0%    | 33.3%  | 50.0% | 38.5% |
| VOLUME_DOWN     | 3  | 75.0%     | 100.0% | 85.7% | 93.8% |
| VOLUME_UP       | 3  | 100.0%    | 33.3%  | 50.0% | 38.5% |
| WEATHER         | 3  | 100.0%    | 33.3%  | 50.0% | 38.5% |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
