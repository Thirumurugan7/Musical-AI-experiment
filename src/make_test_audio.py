"""Generate a synthetic guitar-like strummed progression with KNOWN ground truth.
Purpose: smoke-test the chord recognition pipeline. NOT a substitute for real audio."""
import os
import sys

import numpy as np, soundfile as sf

# Output directory. Defaults to work/ beside the repo; override with an
# argument or CHORDSTRUM_OUT. The two paths here were absolute sandbox paths
# from the machine this was first written on, so it could not run anywhere else.
OUT = (sys.argv[1] if len(sys.argv) > 1
       else os.environ.get("CHORDSTRUM_OUT")
       or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work"))
os.makedirs(OUT, exist_ok=True)

SR = 44100
BPM = 120
BEAT = 60.0 / BPM          # 0.5 s
BAR = 4 * BEAT             # 2.0 s

def note_hz(midi): return 440.0 * 2 ** ((midi - 69) / 12.0)

# Open-position guitar voicings (MIDI note numbers, low -> high)
VOICINGS = {
    "C":  [48, 52, 55, 60, 64],           # C3 E3 G3 C4 E4
    "G":  [43, 47, 50, 55, 59, 67],       # G2 B2 D3 G3 B3 G4
    "Am": [45, 52, 57, 60, 64],           # A2 E3 A3 C4 E4
    "F":  [41, 48, 53, 57, 60, 65],       # F2 C3 F3 A3 C4 F4
}
PROGRESSION = ["C", "G", "Am", "F"] * 2   # 8 bars = 16 s

def pluck(freq, dur, sr=SR, amp=1.0):
    """Karplus-Strong-ish plucked string."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    # harmonic stack with per-partial decay
    for h in range(1, 9):
        f = freq * h
        if f > sr / 2: break
        decay = np.exp(-t * (2.2 + 0.75 * h))
        out += (1.0 / h ** 1.4) * decay * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
    attack = np.minimum(1.0, t / 0.004)     # 4 ms attack
    return amp * attack * out

def strum(midis, dur, down=True, sr=SR):
    """Strum = staggered plucks. Downstroke sweeps low->high, upstroke high->low."""
    n = int(dur * sr)
    buf = np.zeros(n + sr)                  # headroom for ring-out
    order = midis if down else midis[::-1]
    spread = 0.018 if down else 0.012       # upstrokes are faster
    for i, m in enumerate(order):
        off = int(i * spread * sr)
        # upstrokes emphasise the higher strings
        amp = 1.0 if down else (0.55 + 0.1 * i)
        v = pluck(note_hz(m), 2.0, sr, amp)
        end = min(len(buf), off + len(v))
        buf[off:end] += v[: end - off]
    return buf

audio = np.zeros(int(len(PROGRESSION) * BAR * SR) + SR)
truth = []
pos = 0.0
# Strum pattern per bar: D  D  U  D  U   (on beats 1, 2, 2&, 3&, 4)
PATTERN = [(0.0, True), (1.0, True), (1.5, False), (2.5, False), (3.0, True)]
for bar, name in enumerate(PROGRESSION):
    t0 = bar * BAR
    truth.append((t0, t0 + BAR, name))
    for beat_off, down in PATTERN:
        s = strum(VOICINGS[name], BAR, down)
        start = int((t0 + beat_off * BEAT) * SR)
        end = min(len(audio), start + len(s))
        audio[start:end] += s[: end - start]
    pos = t0 + BAR

audio = audio[: int(pos * SR) + SR // 2]
audio /= (np.max(np.abs(audio)) + 1e-9)
audio *= 0.89
sf.write(os.path.join(OUT, "test_progression.wav"), audio.astype(np.float32), SR)

with open(os.path.join(OUT, "ground_truth.lab"), "w") as f:
    for s, e, c in truth:
        f.write(f"{s:.3f}\t{e:.3f}\t{c}\n")

print(f"wrote {len(audio)/SR:.2f}s @ {SR}Hz to {OUT}")
print("ground truth:", " ".join(c for _, _, c in truth))
