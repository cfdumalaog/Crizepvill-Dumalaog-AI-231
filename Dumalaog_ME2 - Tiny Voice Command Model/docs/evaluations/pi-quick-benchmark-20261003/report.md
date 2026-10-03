# Seeded local Pi ONNX evaluation

> File-based inference on the Pi using the class benchmark's prepared WAVs. This bypasses acoustic microphone capture and live speaker playback. It does not measure end-to-end acoustic response latency or physical VAD behavior. CPU/RAM/temperature samples describe the Pi during this local evaluator process, not the full live playback SOP.

# VCM benchmark - Dumalaog ME2 - 20261003-123618

Wake word: **Hi Dandan** - trials: 109 with the wake word + 16 without - shuffle seed: 231 - connection: seeded_local_onnx - holdout: airimonda/vcm-benchmark local WAV bundle

## At a glance

|                                   | overall      | real voice   | synthetic voice |
|-----------------------------------|--------------|--------------|-----------------|
| intent accuracy (19)              | 71.6%        | 49.0%        | 90.0%           |
| command accuracy (93)             | 70.6%        | 46.9%        | 90.0%           |
| false accept (out of scope fired) | 43.8% (7/16) | 40.0% (4/10) | 50.0% (3/6)     |
| false reject (command ignored)    | 18.3%        | 38.5%        | 3.7%            |
| false wake (no wake word, fired)  | 0.0% (0/16)  | 0.0% (0/6)   | 0.0% (0/10)     |
| slot exact                        | 97.7%        | 90.0%        | 100.0%          |
| latency p95                       | -            | -            | -               |

**Pi:** real-time factor 0.003 (p95 0.003), inference 6 ms, CPU temp max 49.0 C, runtime CPU 99% mean, runtime RAM 143 MB peak

**Check:**

- the Pi answered only 76.1% of commands: check volume, distance, wake word and the log path

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject) | 93 commands (+reject) |
|---------------------------------------------------|----------------------|-----------------------|
| accuracy                                          | 71.6%                | 70.6%                 |
| balanced accuracy                                 | 70.0%                | 72.9%                 |
| precision (macro)                                 | 81.2%                | 68.8%                 |
| recall (macro)                                    | 70.0%                | 72.9%                 |
| F1 (macro)                                        | 72.7%                | 70.1%                 |
| F2 (macro)                                        | 70.7%                | 71.5%                 |
| false accept rate (OOS fired)                     | 43.8%                | 43.8%                 |
| false reject rate (in-scope silent/rejected)      | 18.3%                | 18.3%                 |
| misfire rate (wrong command fired)                | 7.5%                 | 8.6%                  |
| accuracy 95% CI                                   | [62-79%]             | [62-78%]              |
| false accept 95% CI                               | [23-67%] (7/16)      | [23-67%]              |
| false wake rate (command without wake word fired) | 0.0% [0-19%] (0/16)  | 0.0% [0-19%] (0/16)   |

Responses: 76.1% of trials fired a command; no response: 26; extra fires: 0; wake detect rate: 100.0%

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 109            | 49             | 60              |
| **19 intents** accuracy        | 71.6% [62-79%] | 49.0% [36-63%] | 90.0% [80-95%]  |
| balanced accuracy              | 70.0%          | 40.0%          | 92.2%           |
| F1 (macro)                     | 72.7%          | 41.5%          | 90.8%           |
| F2 (macro)                     | 70.7%          | 40.0%          | 91.3%           |
| false accept rate              | 43.8% (7/16)   | 40.0% (4/10)   | 50.0% (3/6)     |
| false reject rate              | 18.3%          | 38.5%          | 3.7%            |
| misfire rate                   | 7.5%           | 15.4%          | 1.9%            |
| **93 commands** accuracy       | 70.6%          | 46.9%          | 90.0%           |
| balanced accuracy              | 72.9%          | 44.0%          | 93.6%           |
| F1 (macro)                     | 70.1%          | 41.8%          | 93.1%           |
| F2 (macro)                     | 71.5%          | 42.9%          | 93.4%           |
| misfire rate                   | 8.6%           | 17.9%          | 1.9%            |
| slot exact (intent right)      | 97.7% (n=43)   | 90.0% (n=10)   | 100.0% (n=33)   |
| latency p50 / p95              | -              | -              | -               |
| false wake rate (no wake word) | 0.0% (0/16)    | 0.0% (0/6)     | 0.0% (0/10)     |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 7  | 85.7%  | 17.1 min       | 0.019          | 0.071         | 0.071     |
| BRIGHTNESS      | 7  | 100.0% | 0.0 %          | 0.000          | 0.000         | 0.000     |
| COLOR           | 7  | 100.0% | -              | -              | 0.000         | 0.000     |
| CREATE_REMINDER | 9  | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 7  | 100.0% | 0.0 deg        | 0.000          | 0.000         | 0.000     |
| TIMER           | 6  | 100.0% | 0.0 s          | 0.000          | 0.000         | 0.000     |
| ALL             | 43 | 97.7%  | -              | 0.005          | 0.012         | 0.012     |

## Raspberry Pi

- **Raspberry Pi 5 Model B Rev 1.0**, 4 cores  up to 2400.0 MHz, RAM 4045.1 MB, Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-2712, Python 3.13.5
- packages: onnxruntime 1.30.0, numpy 2.2.4, sounddevice 0.5.6

| metric                                      | mean / p95 / max            |
|---------------------------------------------|-----------------------------|
| response latency (command end -> Pi output) | -                           |
| latency p50 / p99                           | -                           |
| inference time (Pi-reported)                | 6.3 / 7.3 / 9.3 ms          |
| real-time factor (infer / audio window)     | 0.003 / 0.003 / 0.004       |
| CPU temperature                             | 47.4 / 49.0 / 49.0 C        |
| CPU use, whole Pi                           | 47.1 / 48.9 / 49.0 %        |
| CPU use, your runtime process               | 99.2 / 111.9 / 119.0 %      |
| RAM (RSS), your runtime process             | 141.8 / 142.4 / 142.6 MB    |
| RAM used, whole Pi                          | 2271.3 / 2273.1 / 2273.1 MB |
| CPU clock                                   | 2400 / 2400 / 2400 MHz      |
| load average (1 min)                        | 1.58 / 1.60 / 1.61          |
| runtime CPU-seconds per second of speech    | 0.023                       |
| runtime CPU share of wall time              | 89.8%                       |
| throttling flags seen                       | none                        |
| test wall time                              | 0.1 min                     |

## Most frequent confusions

**intent level:** REJECT -> PLAY_MUSIC (3); TIMER -> REJECT (2); PLAY_MUSIC -> REJECT (2); TEMPERATURE -> REJECT (2); REJECT -> MESSAGE (2); WEATHER -> REJECT (2); ALARM -> REJECT (2); CALL -> REJECT (2); NEXT -> REJECT (1); COLOR -> REJECT (1)

**command level:** REJECT -> Play music (3); REJECT -> Message (2); Skip song -> REJECT (1); Change color to Blue -> REJECT (1); Timer 30 seconds -> REJECT (1); Shut off the lights -> Lights on (1); Play music -> REJECT (1); Change the temperature to 18 degrees -> REJECT (1); REJECT -> Brightness 60 percent (1); REJECT -> Temperature 18 degrees (1)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 9  | 100.0%    | 77.8%  | 87.5%  | 81.4%  |
| BRIGHTNESS      | 9  | 77.8%     | 77.8%  | 77.8%  | 77.8%  |
| CALL            | 3  | 100.0%    | 33.3%  | 50.0%  | 38.5%  |
| COLOR           | 9  | 100.0%    | 77.8%  | 87.5%  | 81.4%  |
| CREATE_REMINDER | 9  | 90.0%     | 100.0% | 94.7%  | 97.8%  |
| LIGHT_OFF       | 3  | 66.7%     | 66.7%  | 66.7%  | 66.7%  |
| LIGHT_ON        | 3  | 75.0%     | 100.0% | 85.7%  | 93.8%  |
| LIST_REMINDERS  | 3  | 100.0%    | 100.0% | 100.0% | 100.0% |
| MESSAGE         | 3  | 50.0%     | 66.7%  | 57.1%  | 62.5%  |
| NEXT            | 3  | 50.0%     | 66.7%  | 57.1%  | 62.5%  |
| PAUSE           | 3  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| PLAY_MUSIC      | 3  | 25.0%     | 33.3%  | 28.6%  | 31.2%  |
| REJECT          | 16 | 34.6%     | 56.2%  | 42.9%  | 50.0%  |
| STOP            | 3  | 66.7%     | 66.7%  | 66.7%  | 66.7%  |
| TEMPERATURE     | 9  | 87.5%     | 77.8%  | 82.4%  | 79.5%  |
| TIME            | 3  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| TIMER           | 9  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| VOLUME_DOWN     | 3  | 100.0%    | 100.0% | 100.0% | 100.0% |
| VOLUME_UP       | 3  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| WEATHER         | 3  | 100.0%    | 33.3%  | 50.0%  | 38.5%  |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
