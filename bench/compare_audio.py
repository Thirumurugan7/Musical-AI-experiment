"""
Compare a recording against audio rendered from our own transcription.

The idea is analysis-by-synthesis: take the chords and strum pattern the
pipeline produced, render them back to audio, and ask how well that render
explains the original. It needs no ground truth, which is the whole appeal —
every other measurement in bench/ depends on a reference somebody had to write
down, and those references were wrong ten times out of eleven when checked.

Three things are measured, in descending order of trustworthiness.

  chroma      Per bar, cosine similarity between the pitch-class profile of the
              original and of the render. A wrong chord produces a visibly
              wrong profile. This is the strong one: it asks whether our chord
              labels actually explain the harmony on the record.

  onsets      Stroke times in the render against stroke times detected in a
              reference signal, matched within a tolerance. Only meaningful
              against a signal that did NOT produce our marks — an isolated
              guitar stem — otherwise it measures the renderer and nothing
              else. Refuses to report when no independent signal is available.

  downbeats   Rendered bar starts against the original's tracked downbeats.
              Both come from the same source so this should be near-perfect;
              it is a canary, not a score.

CALIBRATION — read this before trusting a number.

Measured on 12 benchmark songs, chroma similarity against known chord F1:

    matched song and transcription      0.774 - 0.897
    deliberately mismatched pairs       0.514 - 0.720
    correlation with actual chord F1    -0.044

So it separates right from grossly wrong with a clear margin, and has NO
resolving power inside the right band. Ho Hey scores 0.848 with a chord F1 of
0.75; Let It Be scores 0.837 with a perfect 1.00. The ranking is inverted.

The reason is structural: chroma averages pitch classes over a bar, and every
chord in a diatonic song draws on the same seven notes. Swapping C for Em moves
the profile barely at all. What this measures is really harmonic-field
agreement — roughly, "is this the right key and roughly the right harmony" —
not chord-level correctness.

Use it as a gross-error alarm and a regression detector. Below about 0.75 on a
matched pair, something is badly wrong. Between 0.78 and 0.90, it cannot tell
you which of two transcriptions is better, and bench/score.py against published
charts remains the only thing that can.

What it cannot do at all: validate stroke direction (never measured from audio
in the first place), distinguish a wrong chord from a deliberate simplification
(Perfect's Em7 folded to Em costs similarity while being the requested chart),
or catch consistently repeated error.
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.signal
for _w in ("hann", "hamming"):
    if not hasattr(scipy.signal, _w):
        setattr(scipy.signal, _w, getattr(scipy.signal.windows, _w))
import librosa
import soundfile as sf

SR = 22050
PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
      "F": 5, "E#": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
      "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11}

# interval stacks by chord quality, semitones from the root
QUALITIES = [
    ("maj7", (0, 4, 7, 11)), ("min7", (0, 3, 7, 10)), ("m7", (0, 3, 7, 10)),
    ("dim7", (0, 3, 6, 9)), ("dim", (0, 3, 6)), ("aug", (0, 4, 8)),
    ("sus4", (0, 5, 7)), ("sus2", (0, 2, 7)), ("add9", (0, 4, 7, 14)),
    ("m6", (0, 3, 7, 9)), ("6", (0, 4, 7, 9)), ("9", (0, 4, 7, 10, 14)),
    ("7", (0, 4, 7, 10)), ("m", (0, 3, 7)),
]


def parse_chord(name):
    """'F#m7' -> (6, (0,3,7,10)).  Returns None for N.C. and unparseable."""
    if not name or name in ("N.C.", "N", "X"):
        return None
    name = name.split("/")[0].strip()
    root = name[:2] if len(name) > 1 and name[1] in "#b" else name[:1]
    if root not in PC:
        return None
    rest = name[len(root):]
    for suffix, iv in QUALITIES:
        if rest.startswith(suffix):
            return PC[root], iv
    return PC[root], (0, 4, 7)


def voicing(root_pc, intervals, low=40, high=76):
    """
    Lay the chord out across a guitar's range, lowest note the root.

    Not an attempt at real fingerings — those depend on the capo and shape, and
    the point here is the pitch-class content, which is what chroma compares.
    """
    root = low + ((root_pc - low) % 12)
    notes = []
    for iv in intervals:
        n = root + iv
        while n > high:
            n -= 12
        notes.append(n)
    # double the root an octave up so the render has some upper-register energy
    if root + 12 <= high:
        notes.append(root + 12)
    return sorted(set(notes))


def pluck(freq, dur, amp=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for h in range(1, 9):
        f = freq * h
        if f > SR / 2:
            break
        out += (1.0 / h ** 1.4) * np.exp(-t * (2.2 + 0.75 * h)) * \
            np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
    return amp * np.minimum(1.0, t / 0.004) * out


def strum(midis, down, amp):
    order = midis if down else midis[::-1]
    spread = 0.018 if down else 0.012
    buf = np.zeros(int(2.2 * SR))
    for i, m in enumerate(order):
        off = int(i * spread * SR)
        v = pluck(440.0 * 2 ** ((m - 69) / 12.0), 1.8,
                  amp * (1.0 if down else 0.6))
        buf[off:off + len(v)] += v
    return buf


def render(tr, duration):
    """Render the transcription's chords and strum marks back to audio."""
    audio = np.zeros(int((duration + 2.5) * SR))
    bars = tr["bars"]
    stroke_times = []

    for i, bar in enumerate(bars):
        toks = bar["pattern"].split()
        if not toks:
            continue
        end = bars[i + 1]["start"] if i + 1 < len(bars) else bar["start"] + \
            (bars[i]["start"] - bars[i - 1]["start"] if i else 2.0)
        if end <= bar["start"]:
            continue
        step = (end - bar["start"]) / len(toks)
        ch = parse_chord(bar.get("sounding") or bar.get("shape"))
        if ch is None:
            continue
        midis = voicing(*ch)
        for k, tok in enumerate(toks):
            if tok == "·":
                continue
            bare = tok.strip("()")
            amp = 1.0 if bare.isupper() else (0.3 if tok.startswith("(") else 0.6)
            down = bare.lower() != "u"
            t = bar["start"] + k * step
            a = int(t * SR)
            if a >= len(audio):
                continue                       # bar extends past the audio
            sig = strum(midis, down, amp)
            e = min(len(audio), a + len(sig))
            audio[a:e] += sig[: e - a]
            stroke_times.append(t)

    audio = audio[: int(duration * SR)]
    peak = np.max(np.abs(audio)) or 1.0
    return (audio / peak * 0.89).astype(np.float32), sorted(stroke_times)


def bar_chroma(y, bars, duration):
    """Mean chroma per bar."""
    ch = librosa.feature.chroma_cqt(y=y, sr=SR, bins_per_octave=36)
    times = librosa.times_like(ch, sr=SR)
    out = []
    for i, bar in enumerate(bars):
        end = bars[i + 1]["start"] if i + 1 < len(bars) else duration
        w = (times >= bar["start"]) & (times < end)
        out.append(ch[:, w].mean(axis=1) if w.any() else np.zeros(12))
    return out


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def match(a, b, tol):
    used = [False] * len(b)
    tp = 0
    for x in a:
        best, bi = tol + 1, -1
        for j, y in enumerate(b):
            if used[j]:
                continue
            d = abs(x - y)
            if d < best:
                best, bi = d, j
        if bi >= 0 and best <= tol:
            used[bi] = True
            tp += 1
    return tp, len(a) - tp, len(b) - tp


def main():
    ap = argparse.ArgumentParser(
        description="compare a recording against audio rendered from our transcription")
    ap.add_argument("audio", help="the original recording")
    ap.add_argument("json", help="transcribe.py -o output for it")
    ap.add_argument("-s", "--stem",
                    help="isolated instrument stem, for the onset comparison. "
                         "Without it the rhythm score is circular and is skipped.")
    ap.add_argument("-o", "--out", help="write a stereo file: original left, render right")
    ap.add_argument("--worst", type=int, default=8, help="how many weak bars to list")
    ap.add_argument("--tol", type=float, default=0.06, help="onset match window, seconds")
    a = ap.parse_args()

    tr = json.load(open(a.json))
    y, _ = librosa.load(a.audio, sr=SR, mono=True)
    dur = len(y) / SR

    rendered, stroke_times = render(tr, dur)

    # ---- chroma, per bar -------------------------------------------------
    bars = tr["bars"]
    co = bar_chroma(y, bars, dur)
    cr = bar_chroma(rendered, bars, dur)
    sims = [cosine(o, r) for o, r in zip(co, cr)]
    sims_v = [s for s in sims if s > 0]

    print(f"{tr.get('title', a.audio)}")
    print(f"  {len(bars)} bars, {dur:.1f}s, key {tr.get('sounding_key')}, "
          f"{tr.get('metre')}, {tr.get('tempo_bpm')} BPM")
    print()
    print("CHROMA  does the render's harmony explain the recording?")
    if sims_v:
        arr = np.array(sims_v)
        print(f"  mean similarity   {arr.mean():.3f}")
        print(f"  median            {np.median(arr):.3f}")
        print(f"  bars below 0.50   {int((arr < 0.5).sum())}/{len(arr)}")
        worst = sorted(range(len(sims)), key=lambda i: sims[i])[: a.worst]
        print(f"\n  weakest bars:")
        for i in sorted(worst):
            b = bars[i]
            print(f"    bar {b['bar']:>3}  {b['start']:7.2f}s  "
                  f"{(b.get('sounding') or b.get('shape')):<6} sim {sims[i]:.2f}")
    else:
        print("  no usable bars")

    # ---- onsets ----------------------------------------------------------
    print()
    print("ONSETS  do the strokes land where the instrument actually plays?")
    if not a.stem:
        print("  skipped — needs an independent signal (--stem). Comparing against")
        print("  the same mix the marks came from measures the renderer, not the marks.")
    elif not os.path.isfile(a.stem):
        print(f"  skipped — stem not found: {a.stem}")
    else:
        ys, _ = librosa.load(a.stem, sr=SR, mono=True)
        rel = 20 * np.log10(max(float(np.sqrt(np.mean(ys ** 2))), 1e-12) /
                            (float(np.sqrt(np.mean(y ** 2))) or 1e-9))
        if rel < -25:
            print(f"  refused — stem is {rel:.0f} dB below the mix, i.e. separation")
            print("  failed. Any score here would be computed on silence.")
        else:
            env = librosa.onset.onset_strength(y=ys, sr=SR)
            ref = sorted(librosa.onset.onset_detect(
                onset_envelope=env, sr=SR, units="time", backtrack=True).tolist())
            tp, fp, fn = match(stroke_times, ref, a.tol)
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            print(f"  stem level        {rel:+.0f} dB vs mix")
            print(f"  strokes rendered  {len(stroke_times)}")
            print(f"  onsets in stem    {len(ref)}")
            print(f"  precision         {p:.3f}   ({fp} rendered strokes with nothing there)")
            print(f"  recall            {r:.3f}   ({fn} played onsets we never strike)")
            print(f"  F1                {f1:.3f}   (within {a.tol*1000:.0f} ms)")

    # ---- downbeat canary --------------------------------------------------
    print()
    print("DOWNBEATS  canary: both come from the same tracker, so this should be ~1.0")
    if len(bars) > 2:
        env = librosa.onset.onset_strength(y=y, sr=SR)
        pk = sorted(librosa.onset.onset_detect(
            onset_envelope=env, sr=SR, units="time", backtrack=True).tolist())
        starts = [b["start"] for b in bars]
        tp, fp, fn = match(starts, pk, 0.10)
        print(f"  {tp}/{len(starts)} bar starts have an onset within 100 ms")

    if a.out:
        n = min(len(y), len(rendered))
        stereo = np.stack([y[:n], rendered[:n]], axis=1)
        sf.write(a.out, stereo, SR)
        print()
        print(f"wrote {a.out} — original hard left, render hard right.")
        print("Play it in headphones: correct chords fuse into a doubled part,")
        print("wrong ones beat audibly against the record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
