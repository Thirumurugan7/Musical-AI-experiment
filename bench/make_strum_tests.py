"""
Synthesise strumming test cases with exact ground truth.

Strumming is the one output nothing measures. There is no annotated dataset,
published patterns are teaching simplifications that disagree with each other,
and hand-annotation needs a person with headphones. So generate the audio
instead: if we decide the pattern, we know the answer exactly.

What this measures is the detector's mechanics — does it find the strokes that
are present, place them in the right slots, and mark the right ones accented.
What it cannot measure is behaviour on real recordings, which are quieter,
noisier and full of other instruments. Treat the numbers as an upper bound and
a regression net, not as accuracy on real music.

It would have caught the occupancy bug in one run: case `sparse` has four
strokes in a bar of eight, and the old detector reported eight.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

SR = 44100
OUT = os.path.dirname(os.path.abspath(__file__)) + "/../work/strumtests"

VOICINGS = {
    "C":  [48, 52, 55, 60, 64],
    "G":  [43, 47, 50, 55, 59, 67],
    "Am": [45, 52, 57, 60, 64],
    "F":  [41, 48, 53, 57, 60, 65],
    "D":  [50, 57, 62, 66],
    "Em": [40, 47, 52, 55, 59, 64],
}

# name: (bpm, beats_per_bar, subdiv, pattern)
# pattern is one entry per slot: "A" accent, "x" normal, "." ghost, " " rest
CASES = {
    # every subdivision struck, accents on the beat — Perfect's actual figure
    "compound_full":  (63, 4, 3, "A..A..A..A.."),
    # compound with genuine gaps
    "compound_gaps":  (70, 4, 3, "A.xA.xA.xA.x"),
    # straight eighths, accents on 1 and 3
    "simple_eighths": (100, 4, 2, "Ax xxAx x"[:8]),
    # the island strum: D DU UDU — four of eight slots silent
    "island":         (110, 4, 2, "A x A  xA"[:8]),
    # sparse: two strokes in the bar, six rests
    # strokes on beats only: subdivision is unobservable here by
    # construction, so the honest answer is "no evidence"
    "sparse":         (85, 4, 2, "A   A   "),
    # sixteenths
    "simple_16ths":   (95, 4, 4, "A.x.A.x.A.x.A.x."),
    # 6/8
    "six_eight":      (80, 2, 3, "A..x.."),
}
PROG = ["C", "G", "Am", "F"]
BARS = 8


def note_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def pluck(freq, dur, amp=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for h in range(1, 9):
        f = freq * h
        if f > SR / 2:
            break
        decay = np.exp(-t * (2.2 + 0.75 * h))
        out += (1.0 / h ** 1.4) * decay * np.sin(
            2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
    return amp * np.minimum(1.0, t / 0.004) * out


def strum(midis, down=True, amp=1.0):
    order = midis if down else midis[::-1]
    spread = 0.018 if down else 0.012
    buf = np.zeros(int(2.5 * SR))
    for i, m in enumerate(order):
        off = int(i * spread * SR)
        v = pluck(note_hz(m), 2.0, amp * (1.0 if down else 0.55 + 0.1 * i))
        buf[off:off + len(v)] += v
    return buf


def render(name, bpm, bpb, subdiv, pattern):
    beat = 60.0 / bpm
    bar_len = bpb * beat
    slot = beat / subdiv
    n_slots = bpb * subdiv
    assert len(pattern) == n_slots, f"{name}: {len(pattern)} marks, {n_slots} slots"

    total = BARS * bar_len
    audio = np.zeros(int((total + 2.5) * SR))
    chords, truth_bars = [], []

    for b in range(BARS):
        t0 = b * bar_len
        chord = PROG[b % len(PROG)]
        chords.append((t0, t0 + bar_len, chord))
        for s, mark in enumerate(pattern):
            if mark == " ":
                continue
            amp = {"A": 1.0, "x": 0.55, ".": 0.25}[mark]
            # alternate direction on offbeats in simple metre, all down in compound
            down = True if subdiv == 3 else (s % 2 == 0)
            sig = strum(VOICINGS[chord], down, amp)
            start = int((t0 + s * slot) * SR)
            audio[start:start + len(sig)] += sig
        truth_bars.append({"bar": b + 1, "start": round(t0, 4), "pattern": pattern})

    audio = audio[: int((total + 0.5) * SR)]
    audio /= (np.max(np.abs(audio)) + 1e-9)
    audio *= 0.89

    os.makedirs(OUT, exist_ok=True)
    sf.write(f"{OUT}/{name}.wav", audio.astype(np.float32), SR)
    with open(f"{OUT}/{name}.lab", "w") as f:
        for s, e, c in chords:
            f.write(f"{s:.3f}\t{e:.3f}\t{c}:{'min' if c.endswith('m') else 'maj'}\n")
    with open(f"{OUT}/{name}.truth.json", "w") as f:
        json.dump({"name": name, "bpm": bpm, "beats_per_bar": bpb,
                   "subdiv": subdiv, "n_slots": n_slots, "pattern": pattern,
                   "metre": f"{bpb*subdiv}/8" if subdiv == 3 else f"{bpb}/4",
                   "struck": sum(1 for m in pattern if m != " "),
                   "bars": truth_bars}, f, indent=2)
    return len(audio) / SR


def main():
    for name, (bpm, bpb, subdiv, pat) in CASES.items():
        dur = render(name, bpm, bpb, subdiv, pat)
        struck = sum(1 for m in pat if m != " ")
        print(f"  {name:16} {dur:5.1f}s  {bpm:3d}bpm  "
              f"{'12/8' if subdiv==3 and bpb==4 else ('6/8' if subdiv==3 else str(bpb)+'/4'):5} "
              f"|{pat}|  {struck}/{len(pat)} struck")
    print(f"\nwrote {len(CASES)} cases to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
