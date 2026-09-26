# ME2 benchmark protocol

This is a proposed reproducible class benchmark. Classmates must agree on the
vocabulary, conditions, thresholds and held-out speakers before comparing models.
No collective agreement or human/Pi evaluation is claimed here.

## Dataset and split

Collect consented, anonymous speaker IDs from at least three real people. Prefer
five or more so training uses several speakers while validation and test each
hold out different people. Record 16 kHz mono PCM16 WAV, all command classes,
both wake variants, room noise, silence, unrelated speech and confusable phrases
such as "Hello Daniel". Aim for at least ten repetitions per command per person,
distributed across quiet-near, fan-near and quiet-far sessions. Keep original
recording/source IDs; all derived augmentations remain in the original split.
Do not put identifiable names in filenames. Check consent and listen to examples.

The collector currently accepts at most 1.5 seconds of detected speech per clip.
Do not rush or cut words to pass this limit: if natural commands exceed it,
increase the common training/runtime window and retrain. Fix the vocabulary and
window before the final benchmark. Use a separate silence/noise segment when
recording negative classes. Do not relabel TTS voices as human participants.

Human mode selects the last sorted speaker ID for test, penultimate for validation
and all others for training. With exactly three people this leaves just one
training speaker: a useful holdout, but weak coverage. More people are preferable.
Never select thresholds or architecture on the final test set. Store a manifest
with speaker, condition, class, source ID, SHA256 and split. The provided audit
rejects cross-split duplicates, overlapping source groups, missing classes and
insufficient speaker/condition coverage. Balance conditions within each class.

## Isolated and streaming tests

Report macro F1, per-class recall, confusion matrix and FP32/INT8 accuracy on
identical held-out recordings. Report noise levels separately; generated noise
mixes do not establish performance in a real noisy room.

For each held-out speaker, run at least five wake-then-command trials per intent
in each condition. Say the wake phrase, wait for its chime, then the command.
Count complete correct actions divided by all attempted trials, including missed
wakes and wrong actions. Record wake recall, command rejection, incorrect actions,
and time from end of speech to action separately. Do not infer microphone success
from state-machine tests fed predicted labels or from isolated file accuracy.

Run a negative stream for at least one hour per condition with room noise,
conversation, TV/radio and near-wake phrases. Log duration and actual false wake
counts; report false wakes/hour and unintended actions/hour. Zero observations
in a short test is not evidence of a zero underlying false-alarm rate.

## Actual Raspberry Pi 5, 4 GB

Record board, OS, CPU governor, cooling, microphone, gain, distance, sample rate,
runtime version and model SHA256. Warm up inference and measure at least 1,000
single-thread iterations. Use `python -m tinyvcm.runtime --output pi_benchmark.json`.
Report p50/p95/p99 neural inference and frontend-plus-model latency separately,
resident memory, CPU use and stream dropouts during a ten-minute live run. The
500,000-byte limit applies to the exported model, not Python dependencies.
The 10 ms inference target does not include the audio window or endpoint wait.

Demonstrate physical LED on/off/dimming, timed buzzer and optional temperature
sensor with the logged predicted intent and action. Label thermostat, weather,
phone and reminder behavior honestly. Use only low-voltage demo circuits.

## Current evidence (2026-09-26)

The current run is **development evidence only**, using Windows SAPI voices and
generated noise. INT8 isolated accuracy is 79.28% (1,221 files); wake-then-command
file replay succeeded in 77/216 sequences (35.65%), with 194/216 wakes detected.
Runtime behavior was revised while inspecting these synthetic results; they are
exploratory, not an untouched final human benchmark. No human false-alarm rate,
Pi latency, sustained Pi memory or physical actuation has been measured.
