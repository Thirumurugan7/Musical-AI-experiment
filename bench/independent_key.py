#!/usr/bin/env python3
"""
Independent key detection, for auditing the reference set.

Krumhansl-Schmuckler: correlate a track's average chroma against the twelve
rotations of an empirically derived major and minor key profile. It shares no
code with the pipeline — no chord model, no capo logic, no diatonic scoring —
so when it agrees with the pipeline against a reference row, that row is very
likely wrong.

This exists because the reference set was written from memory and kept being
wrong in one direction: guitar sites publish the key that is easy to play, not
the key on the record. Nine of the first twelve rows checked were mistakes.
Rather than keep trusting my own recall, ask the audio twice, independently.
"""
import json
import sys

import numpy as np
import scipy.signal
for _w in ("hann", "hamming"):
    if not hasattr(scipy.signal, _w):
        setattr(scipy.signal, _w, getattr(scipy.signal.windows, _w))
import librosa

# Krumhansl & Kessler (1982) probe-tone profiles
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']


def detect(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    # CQT chroma follows pitch better than STFT chroma on full mixes
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
    v = ch.mean(axis=1)
    v = (v - v.mean()) / (v.std() or 1.0)

    best = None
    for tonic in range(12):
        for prof, is_min in ((MAJOR, False), (MINOR, True)):
            p = np.roll(prof, tonic)
            p = (p - p.mean()) / (p.std() or 1.0)
            r = float(np.dot(v, p) / len(v))
            if best is None or r > best[0]:
                best = (r, tonic, is_min)
    r, tonic, is_min = best
    return NAMES[tonic] + ("m" if is_min else ""), round(r, 3)


def main():
    out = {}
    for path in sys.argv[1:]:
        ident = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        try:
            key, conf = detect(path)
            out[ident] = {"key": key, "confidence": conf}
            print(f"{ident:20} {key:5} (r={conf})", flush=True)
        except Exception as e:
            out[ident] = {"key": None, "error": str(e)}
            print(f"{ident:20} ERROR {e}", flush=True)
    with open("/Users/thirumurugansivalingam/Desktop/personal/music-ai/"
              "work/bench/independent_keys.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
