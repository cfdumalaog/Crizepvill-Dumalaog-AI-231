# ME3 Class-group Voice Recorder

A browser tool so the class can record the **8 canonical ME3 benchmark
stimuli** (see [`data/DATASET.md`](../data/DATASET.md) §2) from a laptop or
phone on the lab LAN. Each recording is normalized to the dataset spec
(16 kHz mono int16, −20 dBFS peak, 0.30 s padding), saved under
`data/bench/speakers/<speaker_label>/`, and logged to
`data/bench/manifest.jsonl`.

## Teacher: start it

```bat
launchers\Start Class Recorder.bat
```

(or `scripts\Start-Recorder-ME3.ps1` directly). It prints a URL like
`http://192.168.x.x:7871`. **Read that URL out to the class.** The server
binds `0.0.0.0:7871` — lab LAN only, **no public tunnel**. Stop with Ctrl+C.

> Port 7871 is chosen to avoid ME2's range (7861–7865). It does not conflict
> with a running ME3 agent (`python -m me3.agent`).

## Students: record

1. Open the URL on a **laptop or phone** connected to the same LAN.
2. Fill in **Student ID**, **Speaker label** (e.g. `speaker_01`), and tick
   the **consent** box (audio is stored for the ME3 benchmark dataset only).
3. Pick a sentence from the dropdown, press **record**, say it **exactly** as
   shown, then **Save**. Phones that can't record in-browser: use the
   **upload** field instead.
4. Repeat for all 8. The status table shows which are saved for your label.

Re-saving the same speaker + sentence **overwrites the WAV** but appends a new
manifest row (history is preserved).

## The 8 sentences (say exactly)

| scenario_id | Say exactly |
|---|---|
| `hello` | Hello |
| `what_time_is_it` | What time is it? |
| `set_a_timer_5_minutes` | Set a timer for five minutes |
| `stop` | Stop |
| `what_is_2_plus_2` | What is 2 plus 2? |
| `tell_me_a_joke` | Tell me a joke |
| `play_some_music` | Play some music |
| `who_are_you` | Who are you? |

## Where files land

- WAV: `data/bench/speakers/<speaker_label>/<scenario_id>.wav`
- Manifest: `data/bench/manifest.jsonl` (one JSON line per save)
- The committed canonical set at `data/bench/<id>.wav` is **untouched**.

## Recording tips

- Quiet room, one mic, one gain, ~20–30 cm from the mouth.
- Speak the sentence once, naturally — don't pad with extra words.
- Keep it under 8 s; the recorder rejects longer clips.
