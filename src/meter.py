"""
Tatum-first metrical analysis.

librosa.beat.beat_track assumes a duple feel and silently mis-groups compound
metre (6/8, 12/8) into pairs of eighths. This module instead:

  1. finds the tatum (fastest consistent pulse) from the onset envelope
  2. tests whether tatums group in 2s or 3s, and at which phase
  3. reports the metre, the true beat period, and a grid to snap onsets to

Needs no pretrained weights.
"""
import numpy as np
import scipy.signal

for _w in ("hann", "hamming"):
    if not hasattr(scipy.signal, _w):
        setattr(scipy.signal, _w, getattr(scipy.signal.windows, _w))

import librosa


def analyse(y, sr, hop=512):
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env = env - env.mean()
    frame_t = hop / sr

    # --- 1. tatum via autocorrelation of the onset envelope -----------------
    ac = librosa.autocorrelate(env, max_size=int(4.0 / frame_t))
    ac[0] = 0
    lo, hi = int(0.10 / frame_t), int(0.60 / frame_t)   # 100-600 ms tatum range
    lag = lo + int(np.argmax(ac[lo:hi]))
    tatum = lag * frame_t

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop, backtrack=False,
        delta=0.05, wait=3)
    onset_t = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    onset_s = env[onset_frames]

    # refine tatum to the median inter-onset interval near the ac estimate
    ioi = np.diff(onset_t)
    near = ioi[(ioi > tatum * 0.7) & (ioi < tatum * 1.3)]
    if len(near) > 20:
        tatum = float(np.median(near))

    # --- 2. grouping: 2s or 3s, and phase -----------------------------------
    n_tatums = int((onset_t[-1] - onset_t[0]) / tatum) + 1
    grid0 = onset_t[0]
    idx = np.round((onset_t - grid0) / tatum).astype(int)

    strength = np.zeros(n_tatums + 1)
    for i, s in zip(idx, onset_s):
        if 0 <= i <= n_tatums:
            strength[i] = max(strength[i], s)

    best = None
    for group in (2, 3, 4):
        for phase in range(group):
            on = strength[phase::group]
            off = np.array([strength[i] for i in range(len(strength))
                            if (i - phase) % group != 0])
            if not len(on) or not len(off):
                continue
            contrast = on.mean() / (off.mean() + 1e-9)
            if best is None or contrast > best["contrast"]:
                best = {"group": group, "phase": phase,
                        "contrast": float(contrast)}

    group, phase = best["group"], best["phase"]
    beat_period = tatum * group

    # --- 3. quality of fit --------------------------------------------------
    residual = np.abs((onset_t - grid0) / tatum - np.round((onset_t - grid0) / tatum))
    align_ms = float(np.median(residual) * tatum * 1000)

    metre = {2: "duple (eighths in 2s -> 2/4 or 4/4)",
             3: "compound (eighths in 3s -> 6/8 or 12/8)",
             4: "duple (sixteenths in 4s)"}[group]

    return {
        "tatum_s": round(tatum, 4),
        "tatum_bpm": round(60.0 / tatum, 1),
        "group": group,
        "phase": phase,
        "accent_contrast": round(best["contrast"], 3),
        "metre": metre,
        "beat_period_s": round(beat_period, 4),
        "beat_bpm": round(60.0 / beat_period, 1),
        "n_onsets": int(len(onset_t)),
        "median_grid_error_ms": round(align_ms, 1),
        "onset_times": onset_t,
        "onset_strength": onset_s,
        "grid_origin": float(grid0),
    }


if __name__ == "__main__":
    import sys
    y, sr = librosa.load(sys.argv[1], sr=22050, mono=True)
    r = analyse(y, sr)
    print(f"  tatum            {r['tatum_s']*1000:6.1f} ms  ({r['tatum_bpm']} per min)")
    print(f"  grouping         {r['group']} (phase {r['phase']}, accent contrast {r['accent_contrast']})")
    print(f"  metre            {r['metre']}")
    print(f"  beat             {r['beat_period_s']*1000:6.1f} ms  ({r['beat_bpm']} BPM)")
    print(f"  onsets           {r['n_onsets']}")
    print(f"  median grid err  {r['median_grid_error_ms']} ms")
