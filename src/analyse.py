"""
Full chord + rhythm chart.

Beat/downbeat grid: madmom DBN (real downbeats, handles compound metre).
Chords:             ChordMini's Chord-CNN-LSTM .lab output.
Rhythm:             librosa onsets snapped to the madmom grid.
"""
import sys, json, collections
import numpy as np
import scipy.signal

for _w in ("hann", "hamming"):
    if not hasattr(scipy.signal, _w):
        setattr(scipy.signal, _w, getattr(scipy.signal.windows, _w))

import librosa
from madmom.features.downbeats import (RNNDownBeatProcessor,
                                       DBNDownBeatTrackingProcessor)

PRETTY = {"maj": "", "min": "m", "dim": "dim", "aug": "aug", "maj7": "maj7",
          "min7": "m7", "7": "7", "hdim7": "m7b5", "dim7": "dim7",
          "sus2": "sus2", "sus4": "sus4", "maj6": "6", "min6": "m6"}
# Ab major spells its IV as Db, not C#
ENHARM = {"C#": "Db", "D#": "Eb", "G#": "Ab", "A#": "Bb", "F#": "Gb"}


def pretty(label, enharm=True):
    if label == "N":
        return "N.C."
    if ":" not in label:
        return label
    root, qual = label.split(":", 1)
    bass = ""
    if "/" in qual:
        qual, bass = qual.split("/", 1)
        bass = "/" + bass
    if enharm:
        root = ENHARM.get(root, root)
    return root + PRETTY.get(qual, ":" + qual) + bass


def load_lab(p):
    out = []
    for line in open(p):
        f = line.split()
        if len(f) >= 3:
            out.append((float(f[0]), float(f[1]), f[2]))
    return out


def dominant_chord(chords, t0, t1):
    """Chord occupying the most time in [t0, t1)."""
    acc = collections.Counter()
    for s, e, c in chords:
        ov = min(e, t1) - max(s, t0)
        if ov > 0:
            acc[c] += ov
    return acc.most_common(1)[0][0] if acc else "N"


def analyse(wav, lab, beats_per_bar=4, subdiv=3):
    chords = load_lab(lab)

    act = RNNDownBeatProcessor()(wav)
    beats = DBNDownBeatTrackingProcessor(
        beats_per_bar=[beats_per_bar], fps=100)(act)
    bt, bpos = beats[:, 0], beats[:, 1].astype(int)

    # subdivide each beat; subdiv=3 for compound metre, 2 for simple
    grid, meta = [], []
    for i in range(len(bt) - 1):
        t0, t1 = bt[i], bt[i + 1]
        for s in range(subdiv):
            grid.append(t0 + (t1 - t0) * s / subdiv)
            meta.append((bpos[i], s))
    grid = np.array(grid)
    step = np.median(np.diff(grid))

    y, sr = librosa.load(wav, sr=22050, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr)
    of = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                    backtrack=False, delta=0.05, wait=3)
    ot = librosa.frames_to_time(of, sr=sr)

    hits = {}
    for t in ot:
        i = int(np.argmin(np.abs(grid - t)))
        if abs(grid[i] - t) <= step * 0.5:
            hits.setdefault(i, []).append(t)
    err = [abs(grid[i] - t) for i, ts in hits.items() for t in ts]

    # group grid slots into bars
    downbeat_idx = [i for i, (b, s) in enumerate(meta) if b == 1 and s == 0]
    bars = []
    for n, start in enumerate(downbeat_idx[:-1]):
        end = downbeat_idx[n + 1]
        t0, t1 = grid[start], grid[end]
        slots = ["·"] * (beats_per_bar * subdiv)
        for i in range(start, end):
            if i in hits:
                pos = i - start
                if pos < len(slots):
                    slots[pos] = "x"
        bars.append({
            "bar": n + 1, "start": round(float(t0), 3),
            "chord": pretty(dominant_chord(chords, t0, t1)),
            "slots": slots,
        })

    ibi = np.median(np.diff(bt))
    return {
        "tempo_bpm": round(60.0 / ibi, 1),
        "beats_per_bar": beats_per_bar,
        "subdiv": subdiv,
        "metre": f"{beats_per_bar*subdiv}/8" if subdiv == 3 else f"{beats_per_bar}/4",
        "bar_seconds": round(float(ibi * beats_per_bar), 3),
        "n_bars": len(bars),
        "n_onsets": int(len(ot)),
        "onsets_on_grid": int(sum(len(v) for v in hits.values())),
        "median_grid_error_ms": round(float(np.median(err)) * 1000, 1) if err else None,
        "bars": bars,
    }


def render(r, max_bars=40):
    L = []
    L.append(f"tempo {r['tempo_bpm']} BPM   metre {r['metre']}   "
             f"bar = {r['bar_seconds']}s   {r['n_bars']} bars")
    L.append(f"onsets {r['n_onsets']}, {r['onsets_on_grid']} landed on grid "
             f"(median error {r['median_grid_error_ms']} ms)")
    L.append("")
    count = []
    for b in range(r["beats_per_bar"]):
        count += [str(b + 1)] + ["."] * (r["subdiv"] - 1)
    L.append(f"      {'':<8}  " + " ".join(count))
    L.append("      " + "-" * (10 + 2 * len(count)))
    for b in r["bars"][:max_bars]:
        L.append(f"  {b['bar']:>3} {b['chord']:<8}  " + " ".join(b["slots"]))
    if r["n_bars"] > max_bars:
        L.append(f"  ... {r['n_bars'] - max_bars} more bars")
    L.append("")
    L.append("  x = attack   · = silence")
    return "\n".join(L)


if __name__ == "__main__":
    res = analyse(sys.argv[1], sys.argv[2])
    print(render(res))
    if len(sys.argv) > 3:
        json.dump(res, open(sys.argv[3], "w"), indent=2)
