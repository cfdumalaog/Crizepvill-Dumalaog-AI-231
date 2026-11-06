import wave, math, struct

SR = 22050
notes = {
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
    'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25,
}

# Twinkle Twinkle Little Star
melody = [
    ('C4', 0.5), ('C4', 0.5), ('G4', 0.5), ('G4', 0.5),
    ('A4', 0.5), ('A4', 0.5), ('G4', 1.0),
    ('F4', 0.5), ('F4', 0.5), ('E4', 0.5), ('E4', 0.5),
    ('D4', 0.5), ('D4', 0.5), ('C4', 1.0),
]

frames = bytearray()
for name, dur in melody:
    f = notes[name]
    n = int(SR * dur)
    for i in range(n):
        t = i / SR
        # simple envelope to avoid clicks
        env = min(1.0, i / (SR * 0.02), (n - i) / (SR * 0.05))
        val = 0.35 * env * (math.sin(2 * math.pi * f * t) + 0.3 * math.sin(4 * math.pi * f * t))
        frames += struct.pack('<h', int(val * 32767))

with wave.open('tune.wav', 'wb') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(bytes(frames))
print('ok')
