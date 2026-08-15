"""
Strum pattern detection — the layer ChordMini does not ship.

Pipeline:
  1. beat / downbeat grid   (librosa here; swap in ChordMini's madmom or Beat-Transformer)
  2. onset detection        (spectral flux on the harmonic component)
  3. snap onsets to an 8th-note grid derived from the beats
  4. assign down/up strokes from metric position, refined by spectral centroid

Rationale for (4): in normal strumming the hand moves in continuous 8th-note
motion. On-beat 8ths are struck on the way down, off-beat 8ths on the way up.
Upstrokes also start from the treble strings, so they read brighter — a higher
spectral centroid at the onset is corroborating evidence.
"""
import sys, json
import numpy as np
import scipy.signal

# librosa 0.10.1 still calls scipy.signal.hann, removed in scipy >= 1.13.
# ChordMini ships the same shim in python_backend/compat/scipy_patch.py.
for _w in ("hann", "hamming", "blackman", "bartlett"):
    if not hasattr(scipy.signal, _w):
        setattr(scipy.signal, _w, getattr(scipy.signal.windows, _w))

import librosa


def detect_strums(path, sr=22050, subdiv=2, beats_per_bar=4):
    y, sr = librosa.load(path, sr=sr, mono=True)
    duration = len(y) / sr

    # --- 1. beat grid -------------------------------------------------------
    # One shared onset envelope for both beat tracking and onset picking, so the
    # grid and the events cannot disagree about where the energy is.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beat_times) < 2:
        raise RuntimeError("could not establish a beat grid")

    # subdivide into 8th notes (subdiv=2) or 16ths (subdiv=4)
    grid = []
    for i in range(len(beat_times) - 1):
        t0, t1 = beat_times[i], beat_times[i + 1]
        for s in range(subdiv):
            grid.append(t0 + (t1 - t0) * s / subdiv)
    step = np.median(np.diff(beat_times)) / subdiv
    grid.append(beat_times[-1])
    grid = np.array(grid)

    # --- 2. onsets ----------------------------------------------------------
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, backtrack=False, delta=0.05, wait=3,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_str = onset_env[onset_frames] if len(onset_frames) else np.array([])

    # --- 3. snap to grid ----------------------------------------------------
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    cent_times = librosa.times_like(cent, sr=sr)

    events = []
    used = set()
    for t, stren in zip(onset_times, onset_str):
        idx = int(np.argmin(np.abs(grid - t)))
        if abs(grid[idx] - t) > step * 0.5:      # too far from any subdivision
            continue
        if idx in used:                           # keep the strongest per slot
            prev = next(e for e in events if e["grid_idx"] == idx)
            if stren <= prev["strength"]:
                continue
            events.remove(prev)
        used.add(idx)
        # brightness in a 60 ms window at the attack
        w = (cent_times >= t) & (cent_times < t + 0.06)
        brightness = float(np.mean(cent[w])) if w.any() else float("nan")
        events.append({
            "time": float(t),
            "grid_idx": idx,
            "strength": float(stren),
            "brightness": brightness,
        })
    events.sort(key=lambda e: e["grid_idx"])
    if not events:
        raise RuntimeError("no onsets detected")

    # --- 4. direction -------------------------------------------------------
    # metric prior: even subdivision index = on-beat = downstroke
    med_bright = np.nanmedian([e["brightness"] for e in events])
    for e in events:
        on_beat = (e["grid_idx"] % subdiv) == 0
        prior_down = 1.0 if on_beat else 0.0
        # brightness evidence: brighter than median -> leans upstroke
        b = e["brightness"]
        eb = 0.5 if np.isnan(b) else float(np.clip(
            0.5 - (b - med_bright) / (med_bright + 1e-9) * 1.5, 0.0, 1.0))
        score = 0.75 * prior_down + 0.25 * eb        # metric position dominates
        e["direction"] = "D" if score >= 0.5 else "U"
        e["confidence"] = round(float(abs(score - 0.5) * 2), 3)
        e["beat"] = round(e["grid_idx"] / subdiv + 1, 3)
        e["bar"] = int(e["grid_idx"] // (subdiv * beats_per_bar)) + 1
        e["beat_in_bar"] = round(
            (e["grid_idx"] % (subdiv * beats_per_bar)) / subdiv + 1, 3)

    bars = {}
    for e in events:
        bars.setdefault(e["bar"], []).append(e)

    return {
        "duration": round(duration, 3),
        "tempo": round(tempo, 2),
        "subdivision": "8th" if subdiv == 2 else f"1/{subdiv*4}",
        "beats_per_bar": beats_per_bar,
        "events": events,
        "bars": {
            str(b): {
                "pattern": " ".join(e["direction"] for e in evs),
                "positions": [e["beat_in_bar"] for e in evs],
            } for b, evs in sorted(bars.items())
        },
    }


if __name__ == "__main__":
    res = detect_strums(sys.argv[1])
    print(f"tempo ~{res['tempo']} BPM   duration {res['duration']}s   "
          f"{len(res['events'])} strums")
    print()
    for bar, d in res["bars"].items():
        pos = " ".join(f"{p:g}" for p in d["positions"])
        print(f"  bar {bar:>2}   {d['pattern']:<16}  at beats  {pos}")
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {sys.argv[2]}")
