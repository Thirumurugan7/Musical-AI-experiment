"""
Full transcription: readable chords (capo shapes) + accent-based rhythm.

    python src/transcribe.py song.wav song.lab [out.json]
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

import capo as capo_mod
import rhythm as rhythm_mod


def load_lab(p):
    out = []
    for line in open(p):
        f = line.split()
        if len(f) >= 3:
            out.append((float(f[0]), float(f[1]), f[2]))
    return out


def dominant(chords, t0, t1):
    acc = collections.Counter()
    for s, e, c in chords:
        ov = min(e, t1) - max(s, t0)
        if ov > 0:
            acc[c] += ov
    return acc.most_common(1)[0][0] if acc else "N"


def detect_subdiv(beats, env, env_times):
    """Compound (3) or simple (2)? Test which grid the onsets prefer."""
    bt = beats[:, 0]
    best, scores = 2, {}
    for sd in (2, 3, 4):
        vals = []
        for i in range(len(bt) - 1):
            t0, t1 = bt[i], bt[i + 1]
            for s in range(sd):
                t = t0 + (t1 - t0) * s / sd
                w = (env_times >= t - 0.04) & (env_times < t + 0.08)
                vals.append(float(env[w].max()) if w.any() else 0.0)
        v = np.array(vals).reshape(-1, sd)
        # a real subdivision is one where the *offbeats* also carry energy
        occupancy = float((v > np.median(v[v > 0]) * 0.35).mean())
        contrast = float(v[:, 0].mean() / (v[:, 1:].mean() + 1e-9))
        scores[sd] = {"occupancy": round(occupancy, 3),
                      "beat_contrast": round(contrast, 3)}
    # prefer the finest grid whose slots are mostly occupied
    for sd in (3, 4, 2):
        if scores[sd]["occupancy"] >= 0.75:
            best = sd
            break
    return best, scores


def transcribe(wav, lab, title=None, stem=None):
    """
    wav  : the full mix — used for beat/downbeat tracking, which is more
           reliable with drums present
    stem : optional isolated guitar track — used for onset detection, so the
           rhythm reflects the guitar rather than the whole arrangement
    """
    chords = load_lab(lab)
    onset_src = stem or wav
    y, sr = librosa.load(onset_src, sr=22050, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr)
    et = librosa.times_like(env, sr=sr)

    act = RNNDownBeatProcessor()(wav)
    beats = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)(act)
    bt, bpos = beats[:, 0], beats[:, 1].astype(int)

    subdiv, subdiv_scores = detect_subdiv(beats, env, et)
    bars_raw, bpb = rhythm_mod.slot_strengths(env, et, beats, bpos, subdiv)
    if not bars_raw:
        raise RuntimeError("no complete bars found")
    bars_raw = rhythm_mod.classify(bars_raw)
    med, canon_marks = rhythm_mod.canonical(bars_raw, bpb * subdiv)

    beat_period = float(np.median(np.diff(bt)))
    stroke_rate = subdiv / beat_period
    canon_dirs, dir_rule = rhythm_mod.directions(canon_marks, subdiv, stroke_rate)

    # capo
    weighted = [(c, e - s) for s, e, c in chords if c != "N"]
    ranked = capo_mod.choose_capo(weighted)
    best = ranked[0]
    sounding, sounding_flats = capo_mod.sounding_key(weighted)

    def shape(label):
        p = capo_mod.parse(label)
        if not p:
            return "N.C."
        return capo_mod.render((p[0] - best["capo"]) % 12, p[1], None,
                               best["use_flats"])

    def sound(label):
        p = capo_mod.parse(label)
        if not p:
            return "N.C."
        return capo_mod.render(p[0], p[1], None, sounding_flats)

    # bars
    bar_list = []
    for i, b in enumerate(bars_raw):
        t0 = b["start"]
        t1 = bars_raw[i + 1]["start"] if i + 1 < len(bars_raw) else t0 + beat_period * bpb
        lab_c = dominant(chords, t0, t1)
        dirs, _ = rhythm_mod.directions(b["marks"], subdiv, stroke_rate)
        bar_list.append({
            "bar": i + 1, "start": round(t0, 3),
            "shape": shape(lab_c), "sounding": sound(lab_c),
            "pattern": rhythm_mod.render_pattern(b["marks"], dirs, subdiv, bpb),
        })

    uniq_shapes, uniq_sound, seen = [], [], set()
    for label, _ in sorted(weighted, key=lambda x: -x[1]):
        s = shape(label)
        if s not in seen:
            seen.add(s)
            uniq_shapes.append(s)
            uniq_sound.append(sound(label))

    # how densely occupied the grid is — on a full mix this is meaningless
    # (everything has energy); on an isolated stem it is the actual strum rate
    occupancy = float(np.mean([sum(1 for m in b["marks"] if m != " ") /
                               len(b["marks"]) for b in bars_raw]))

    return {
        "title": title or wav,
        "onset_source": "isolated stem" if stem else "full mix",
        "onset_source_file": onset_src,
        "grid_occupancy": round(occupancy, 3),
        "sounding_key": sounding,
        "capo": best["capo"],
        "shape_key": best["key_shape"],
        "capo_ranking": ranked[:4],
        "tempo_bpm": round(60.0 / beat_period, 1),
        "beats_per_bar": bpb,
        "subdiv": subdiv,
        "metre": f"{bpb*subdiv}/8" if subdiv == 3 else f"{bpb}/4",
        "subdiv_evidence": subdiv_scores,
        "stroke_rate_hz": round(stroke_rate, 2),
        "direction_rule": dir_rule,
        "chords_shapes": uniq_shapes[:8],
        "chords_sounding": uniq_sound[:8],
        "canonical_pattern": rhythm_mod.render_pattern(
            canon_marks, canon_dirs, subdiv, bpb),
        "canonical_strength": [round(float(v), 3) for v in med],
        "count_line": rhythm_mod.count_line(subdiv, bpb),
        "n_bars": len(bar_list),
        "bars": bar_list,
    }


def render(r, max_bars=32):
    L = []
    L.append(r["title"])
    L.append("=" * len(r["title"]))
    capo = f"CAPO {r['capo']} — play in {r['shape_key']}" if r["capo"] else \
           f"no capo — play in {r['shape_key']}"
    L.append(f"sounds in {r['sounding_key']}   |   {capo}   |   "
             f"{r['tempo_bpm']:.0f} BPM   |   {r['metre']}")
    L.append("")
    L.append("  you play :  " + "   ".join(r["chords_shapes"][:6]))
    L.append("  it sounds:  " + "   ".join(r["chords_sounding"][:6]))
    L.append("")
    L.append("STRUMMING PATTERN  (one bar)")
    pad = " " * 6
    L.append(pad + r["count_line"])
    L.append(pad + r["canonical_pattern"])
    L.append("")
    L.append("  D = accented stroke    d = lighter stroke")
    L.append("  (d) = ghost stroke     ·  = no stroke")
    L.append(f"  direction: {r['direction_rule']}")
    L.append("             — derived from metre and stroke rate, not measured")
    L.append("")
    L.append("CHART")
    L.append(f"  {'':>3} {'':<7}   {r['count_line']}")
    L.append("  " + "-" * (13 + len(r["count_line"])))
    for b in r["bars"][:max_bars]:
        L.append(f"  {b['bar']:>3} {b['shape']:<7}   {b['pattern']}")
    if r["n_bars"] > max_bars:
        L.append(f"  ... {r['n_bars'] - max_bars} more bars")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Chord + rhythm chart with capo-relative chord names.")
    ap.add_argument("audio", help="the full mix (wav/mp3)")
    ap.add_argument("lab", help=".lab chord file from Chord-CNN-LSTM")
    ap.add_argument("-o", "--out", help="write full analysis as JSON")
    ap.add_argument("-t", "--title", help="title for the chart header")
    ap.add_argument("-s", "--stem",
                    help="isolated guitar track; onsets are read from this "
                         "instead of the full mix")
    ap.add_argument("-n", "--bars", type=int, default=32,
                    help="how many bars to print (default 32)")
    a = ap.parse_args()

    res = transcribe(a.audio, a.lab, title=a.title, stem=a.stem)
    print(render(res, max_bars=a.bars))
    print(f"\n  onsets read from: {res['onset_source']}"
          f"   grid occupancy: {res['grid_occupancy']:.0%}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"  wrote {a.out}")
