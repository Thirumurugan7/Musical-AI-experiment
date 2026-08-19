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

import os
USE_BASS = os.environ.get('CHORDSTRUM_BASS', '0') == '1'

import capo as capo_mod
import rhythm as rhythm_mod
import simplify as simplify_mod


def load_lab(p):
    out = []
    for line in open(p):
        f = line.split()
        if len(f) >= 3:
            out.append((float(f[0]), float(f[1]), f[2]))
    return out


def dominant(chords, t0, t1):
    """
    The chord a bar is in.

    'N' (no chord) wins on overlap far too often: the model emits it wherever
    the texture is thin, which on a sparse intro can be most of a bar even
    though a chord is plainly being played. So a real chord beats N whenever
    one sounds at all during the bar; N is returned only when nothing else does.
    """
    acc = collections.Counter()
    for s, e, c in chords:
        ov = min(e, t1) - max(s, t0)
        if ov > 0:
            acc[c] += ov
    if not acc:
        return "N"
    real = {c: v for c, v in acc.items() if c not in ("N", "X")}
    if real:
        return max(real, key=real.get)
    return acc.most_common(1)[0][0]


def fill_no_chord(labels):
    """
    Carry a chord across bars the model refused to label.

    A bar with no chord at all is nearly always the model under-committing on a
    sparse passage, not silence — the intro riff of a song is still harmonic.
    Carry the previous chord forward; for leading bars, borrow the first real
    chord that follows.
    """
    out = list(labels)
    last = None
    for i, c in enumerate(out):
        if c not in ("N", "X"):
            last = c
        elif last is not None:
            out[i] = last
    nxt = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] not in ("N", "X"):
            nxt = out[i]
        elif nxt is not None:
            out[i] = nxt
    return out


def grid_offset(env, env_times, bt, sr=22050, max_frac=0.25, min_ms=12.0):
    """
    How far the tracked beat grid sits from where notes actually land.

    Beat trackers report a perceptual pulse, which is not obliged to coincide
    with the attacks — mp3 decoder padding, a laid-back performance, or the
    tracker locking to a smoothed envelope all shift it. On Riptide the offset
    is -82 ms, 14% of a beat: every attack falls outside the sampling window,
    so the beats read as rests and the offbeats read as strums. Beat 2 and 4,
    where the snare is loudest, came back marked as pauses.

    Measured as the median signed distance from each detected onset to its
    nearest beat. The median resists the onsets that genuinely fall between
    beats; only a consistent shift of the whole grid survives it.

    DEFAULT OFF, and the reasoning behind that is worth keeping.

    For most of this work the two benchmarks disagreed about this function:
    turning it off cost real-song metre (55/55 -> 52/55) while gaining a lot of
    synthetic stroke accuracy. That looked like a genuine trade-off between
    real and synthetic evidence, and it was documented here twice as one.

    It was not. It was a symptom. detect_subdiv used to sample the onset
    envelope on a lattice, which is fragile enough that nudging the grid
    underneath it changed the answer — so this correction was propping up a
    weak method rather than fixing anything. Once detect_subdiv started working
    from onset phases with a phase-invariant test, metre held at 55/55 with the
    correction off, and the "trade-off" disappeared:

        alignment on   metre 55/55, chord F1 0.922, synthetic stroke F1 0.514
        alignment off  metre 55/55, chord F1 0.920, synthetic stroke F1 0.774

    Same metre, chord F1 inside noise, and a quarter of stroke accuracy back.
    Set CHORDSTRUM_ALIGN=1 to re-enable for a source that genuinely needs it.
    """
    # backtrack=False on purpose, and it is not an oversight.
    #
    # This function aligns the grid that detect_subdiv uses, and detect_subdiv
    # samples the onset ENVELOPE, whose peaks lag the physical attack. Aligning
    # to peaks is therefore right *for this consumer*. Stroke detection is the
    # other consumer and wants the opposite — it matches attacks, so it uses
    # backtrack=True where the onsets are computed in transcribe().
    #
    # Measured: backtracking here costs real-song metre 55/55 -> 52/55 and
    # chord F1 0.920 -> 0.914, buying 0.059 of synthetic stroke F1. The two
    # signals want different definitions of "onset" and get them separately.
    on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="time",
                                    backtrack=False)
    if len(on) < 12 or len(bt) < 4:
        return 0.0, {"applied": False, "reason": "too few onsets or beats"}
    period = float(np.median(np.diff(bt)))
    d = []
    for t in on:
        i = int(np.argmin(np.abs(bt - t)))
        frac = (t - bt[i]) / period
        if abs(frac) < max_frac:
            d.append(frac)
    if len(d) < 8:
        return 0.0, {"applied": False, "reason": "no onsets near the grid"}
    med = float(np.median(d))
    shift = med * period
    spread = float(np.median(np.abs(np.array(d) - med)))
    info = {"offset_ms": round(shift * 1000, 1),
            "offset_fraction_of_beat": round(med, 4),
            "spread": round(spread, 4), "onsets_used": len(d)}
    # only correct a shift that is both large enough to matter and consistent
    if abs(shift) * 1000 < min_ms or spread > 0.12:
        info["applied"] = False
        return 0.0, info
    info["applied"] = True
    return shift, info


def detect_subdiv(beats, env, env_times, res=12, onset_times=None):
    """
    How many subdivisions per beat: 2, 3 or 4?

    Two earlier versions sampled the onset envelope on a lattice and compared
    positions against each other, then against an off-grid floor. Both were
    fragile for the same reason: the envelope is continuous, so a ringing chord
    puts energy at every lattice point, and the answer came down to how the
    sampling window and the lattice spacing happened to interact. Threshold
    nudging moved the answer around without converging — 4/7 to 5/7 to 4/7
    across three attempts on the synthetic suite.

    This works from discrete onsets instead. Each detected attack gets a phase
    within its beat, and a candidate subdivision is judged on how tightly the
    attacks cluster on its grid positions relative to what uniformly scattered
    onsets would give. That last part is what makes the comparison fair: a
    finer grid has more positions and would always capture more onsets by
    chance, so the score is concentration over chance, not raw hit rate.

    Falls back to the envelope method when no onsets are supplied.
    """
    bt = beats[:, 0]
    if onset_times is not None and len(onset_times) >= 8 and len(bt) >= 4:
        period = float(np.median(np.diff(bt)))
        phases = []
        for t in onset_times:
            i = int(np.searchsorted(bt, t) - 1)
            if 0 <= i < len(bt) - 1:
                span = bt[i + 1] - bt[i]
                if 0.5 * period < span < 1.5 * period:
                    ph = (t - bt[i]) / span
                    if 0.0 <= ph < 1.0:
                        phases.append(ph)
        if len(phases) >= 8:
            ph = np.array(phases)
            tol = 0.07                     # +/-7% of a beat around each position

            def concentration(d, offset):
                near = np.zeros(len(ph), dtype=bool)
                for k in range(d):
                    c = (k / d + offset) % 1.0
                    near |= (np.abs(ph - c) < tol) | \
                            (np.abs(ph - c - 1.0) < tol) | \
                            (np.abs(ph - c + 1.0) < tol)
                return float(near.mean()) / min(1.0, d * 2 * tol)

            # The grid's absolute phase is not the question — its spacing is.
            # Beat trackers land tens of milliseconds off the attacks, which at
            # 95 BPM is ~6% of a beat, right at the tolerance edge; sixteenths
            # that were dead on the grid scored 0.277 purely from that shift.
            # Give each candidate its own best offset so they compete on
            # spacing alone.
            scores, offsets = {}, {}
            for d in (2, 3, 4):
                best_off, best_c = 0.0, -1.0
                for o in np.arange(0.0, 1.0 / d, 0.01):
                    c = concentration(d, o)
                    if c > best_c:
                        best_off, best_c = float(o), c
                scores[d] = best_c
                offsets[d] = round(best_off, 3)
            best = max(scores, key=lambda k: scores[k])
            ev = {"method": "onset-phase", "n_onsets": len(ph),
                  "concentration": {str(k): round(v, 3) for k, v in scores.items()},
                  "best_phase": {str(k): v for k, v in offsets.items()}}
            # a finer grid must clearly beat a coarser one, not edge it
            if best == 4 and scores[4] < scores[2] * 1.10:
                best = 2
            if best == 3 and scores[3] < scores[2] * 1.05:
                best = 2
            ev["chosen"] = best
            return best, ev

    # ---- fallback: envelope lattice --------------------------------------
    rows = []
    for i in range(len(bt) - 1):
        t0, t1 = bt[i], bt[i + 1]
        if not (0.2 < t1 - t0 < 2.5):
            continue
        row = []
        for k in range(res):
            t = t0 + (t1 - t0) * k / res
            w = (env_times >= t - 0.03) & (env_times < t + 0.06)
            row.append(float(env[w].max()) if w.any() else 0.0)
        rows.append(row)
    if not rows:
        return 2, {"error": "no usable beats"}

    m = np.array(rows).mean(axis=0)
    off_grid = [1, 2, 5, 7, 10, 11]
    noise = float(np.mean([m[i] for i in off_grid])) or 1e-9
    binary = float(m[res // 2])
    ternary = float(m[res // 3] + m[2 * res // 3]) / 2
    sixteenth = float(m[res // 4] + m[3 * res // 4]) / 2
    T = 1.35
    has_bin, has_ter = binary > noise * T, ternary > noise * T
    has_16 = sixteenth > noise * T
    scores = {"method": "envelope-lattice",
              "binary_offbeat": round(binary / noise, 3),
              "ternary_offbeat": round(ternary / noise, 3),
              "sixteenth_offbeat": round(sixteenth / noise, 3)}
    if has_16 and has_bin and sixteenth >= binary * 0.5:
        best = 4
    elif has_ter and (not has_bin or ternary > binary):
        best = 3
    elif has_bin:
        best = 2
    elif has_ter:
        best = 3
    else:
        best = 2
        scores["no_evidence"] = True
    scores["chosen"] = best
    return best, scores


def transcribe(wav, lab, title=None, stem=None, simplify=True):
    """
    wav  : the full mix — used for beat/downbeat tracking, which is more
           reliable with drums present
    stem : optional isolated guitar track — used for onset detection, so the
           rhythm reflects the guitar rather than the whole arrangement
    """
    chords = load_lab(lab)

    # The mix drives metre and beats: it has the drums, and a stem that failed
    # to separate carries no metrical information at all. Only the strum grid
    # reads the stem.
    y_mix, sr = librosa.load(wav, sr=22050, mono=True)
    env_mix = librosa.onset.onset_strength(y=y_mix, sr=sr)
    et_mix = librosa.times_like(env_mix, sr=sr)

    onset_src = stem or wav
    stem_warning = None
    if stem:
        y, _ = librosa.load(stem, sr=22050, mono=True)
        mix_rms = float(np.sqrt(np.mean(y_mix ** 2))) or 1e-9
        stem_rms = float(np.sqrt(np.mean(y ** 2)))
        rel_db = 20 * np.log10(max(stem_rms, 1e-12) / mix_rms)
        if rel_db < -25:
            # separation produced nothing: htdemucs_6s returns a near-empty
            # stem when it fails to recognise the instrument at all, and the
            # residue still has enough structure to fool a self-normalised
            # threshold. Refuse it rather than transcribing noise.
            stem_warning = (f"stem is {rel_db:.0f} dB below the mix — "
                            f"separation failed; falling back to the full mix")
            onset_src = wav
            env, et = env_mix, et_mix
        else:
            env = librosa.onset.onset_strength(y=y, sr=sr)
            et = librosa.times_like(env, sr=sr)
    else:
        env, et = env_mix, et_mix

    act = RNNDownBeatProcessor()(wav)
    # [4] only, and this was tested rather than assumed. Allowing [3, 4] so the
    # tracker could find non-four metres did not rescue the 6/8 case (still 3 of
    # 8 bars) and broke a 12/8 one outright: it was re-read as 3/4 at 94 BPM
    # instead of 12/8 at 63, taking subdivision, metre and tempo with it.
    # Compound metre is better served by keeping four beats and letting
    # detect_subdiv find the threes inside them.
    beats = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)(act)

    # Align the grid to where notes actually land before anything samples it.
    # Everything downstream — slot strengths, metre, bar timestamps, the chord
    # each bar gets — reads these times, so a shift here corrects all of them.
    shift, align = grid_offset(env_mix, et_mix, beats[:, 0]) \
        if os.environ.get('CHORDSTRUM_ALIGN', '0') == '1' else (0.0, {'applied': False, 'reason': 'disabled'})
    if shift:
        beats = beats.copy()
        beats[:, 0] = beats[:, 0] + shift
    bt, bpos = beats[:, 0], beats[:, 1].astype(int)

    # Beat trackers lock to whichever pulse is strongest, often the eighth
    # rather than the quarter: Wonderwall comes back at 176 BPM for a song
    # every player counts at 88. Fold the grid here, before anything consumes
    # it, so the bars the chart shows are the bars it counted. Folding only the
    # displayed number produced a chart claiming 88 BPM over 197 bars of a
    # 277-second song, where 4/4 at 88 gives 101.
    raw_beat_period = float(np.median(np.diff(bt))) if len(bt) > 1 else 0.5
    fold = 1
    while 60.0 / (raw_beat_period * fold) > 155.0:
        fold *= 2
    if fold > 1 and len(bt) > 4 * fold:
        bt = bt[::fold]
        bpos = np.array([(i % 4) + 1 for i in range(len(bt))])
        beats = np.stack([bt, bpos], axis=1)
    beat_period = float(np.median(np.diff(bt))) if len(bt) > 1 else raw_beat_period
    while 60.0 / beat_period < 55.0:
        beat_period /= 2.0

    # discrete attacks, from the same signal the slots are sampled from.
    # delta is how far above the local mean a peak must rise to count. Lowering
    # it looked like the obvious way to recover the missed quiet strokes, and it
    # does nothing: swept 0.07 / 0.05 / 0.03 / 0.015, recall sat at exactly
    # 0.388 (or 0.498 unaligned) at every setting while precision fell. The
    # missed strokes are not being rejected by this threshold — they never
    # reach it. Recall is capped by subdivision instead: a stroke cannot be
    # matched to a slot the grid does not have. Left at librosa's default.
    onset_delta = float(os.environ.get("CHORDSTRUM_ONSET_DELTA", "0.07"))
    onset_times = sorted(librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, units="time", backtrack=True,
        delta=onset_delta).tolist())
    subdiv, subdiv_scores = detect_subdiv(beats, env_mix, et_mix,
                                          onset_times=onset_times)
    bars_raw, bpb = rhythm_mod.slot_strengths(env, et, beats, bpos, subdiv,
                                              onset_times=onset_times)
    if not bars_raw:
        raise RuntimeError("no complete bars found")
    # thresholds are set per section further down, once the chord cycle tells
    # us where the sections are; a single track-wide level cannot serve both a
    # quiet verse and a loud chorus. The noise floor stays global so a dead
    # stem is still recognised as dead.
    ref, rest_at = rhythm_mod.thresholds(bars_raw)

    stroke_rate = subdiv / beat_period

    # Beat trackers lock onto whichever pulse is strongest, which is often the
    # eighth rather than the quarter — Wonderwall came back 176.5 BPM for a
    # song every player counts at 88. The octave is genuinely ambiguous (song
    # databases list Wonderwall at 175 too), but a chart should print the
    # tempo a player counts. Fold into 55-155, which spans nearly all popular
    # music while leaving a real 150 BPM song alone.
    raw_bpm = 60.0 / raw_beat_period
    bpm = 60.0 / beat_period

    # capo
    weighted = [(c, e - s) for s, e, c in chords if c != "N"]
    ranked = capo_mod.choose_capo(weighted)
    best = ranked[0]
    bass_pcs = bass_profile(y_mix, sr) if USE_BASS else None
    sounding, sounding_flats = capo_mod.sounding_key(
        weighted,
        order=[c for _, _, c in chords if c not in ('N', 'X')],
        bass=bass_pcs)

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

    bar_labels = []
    for i, b in enumerate(bars_raw):
        t0 = b["start"]
        t1 = bars_raw[i + 1]["start"] if i + 1 < len(bars_raw) else t0 + beat_period * bpb
        bar_labels.append(dominant(chords, t0, t1))
    n_unlabelled = sum(1 for c in bar_labels if c in ("N", "X"))
    bar_labels = fill_no_chord(bar_labels)

    # A chart is a simplification on purpose: a few easy shapes, looping.
    # Rare chords are far more often detection errors than real harmony.
    simplification = None
    if simplify:
        bar_labels, simplification = simplify_mod.reduce_chords(
            bar_labels, capo_mod.parse)

    shapes_seq = [shape(c) for c in bar_labels]
    spans = rhythm_mod.sections(bars_raw, shapes_seq)

    # classify each section against its own level, with the global noise floor
    global_floor = rhythm_mod.noise_floor(bars_raw)
    for a, b in spans:
        s_ref, s_rest = rhythm_mod.thresholds(bars_raw[a:b])
        rhythm_mod.classify_span(bars_raw[a:b], s_ref,
                                 max(s_rest, global_floor))

    # safety net: no bar may leave this stage unclassified, whatever the
    # section logic decides. A missing mark is a crash three stages later.
    stray = [b for b in bars_raw if "marks" not in b]
    if stray:
        rhythm_mod.classify_span(stray, ref, max(rest_at, global_floor))

    med, canon_marks = rhythm_mod.canonical(bars_raw, bpb * subdiv, ref, rest_at)
    canon_dirs, dir_rule = rhythm_mod.directions(canon_marks, subdiv, stroke_rate)

    bar_list = []
    for i, b in enumerate(bars_raw):
        t0 = b["start"]
        lab_c = bar_labels[i]
        dirs, _ = rhythm_mod.directions(b["marks"], subdiv, stroke_rate)
        bar_list.append({
            "bar": i + 1, "start": round(t0, 3),
            "shape": shape(lab_c), "sounding": sound(lab_c),
            "pattern": rhythm_mod.render_pattern(b["marks"], dirs, subdiv, bpb),
        })

    # one repeating pattern per section, which is how charts are actually read
    section_list = []
    for a, b in spans:
        s_ref, s_rest = rhythm_mod.thresholds(bars_raw[a:b])
        _, marks = rhythm_mod.canonical(bars_raw[a:b], bpb * subdiv,
                                        s_ref, max(s_rest, global_floor))
        dirs, _ = rhythm_mod.directions(marks, subdiv, stroke_rate)
        cycle = []
        for sh in shapes_seq[a:b]:
            if not cycle or cycle[-1] != sh:
                cycle.append(sh)
        section_list.append({
            "from_bar": a + 1, "to_bar": b,
            "start": bar_list[a]["start"],
            "chords": cycle[:8],
            "pattern": rhythm_mod.render_pattern(marks, dirs, subdiv, bpb),
            "occupancy": round(sum(1 for m in marks if m != " ") / len(marks), 3),
        })

    if simplify:
        section_list = simplify_mod.collapse_sections(section_list)

    uniq_shapes, uniq_sound, seen = [], [], set()
    bar_order = [c for c, _ in collections.Counter(bar_labels).most_common()]
    for label in bar_order:
        s = shape(label)
        if s not in seen and s != "N.C.":
            seen.add(s)
            uniq_shapes.append(s)
            uniq_sound.append(sound(label))

    # how densely occupied the grid is — on a full mix this is meaningless
    # (everything has energy); on an isolated stem it is the actual strum rate
    occupancy = float(np.mean([sum(1 for m in b["marks"] if m != " ") /
                               len(b["marks"]) for b in bars_raw]))

    return {
        "title": title or wav,
        "onset_source": ("isolated stem" if onset_src != wav else "full mix"),
        "stem_warning": stem_warning,
        "onset_source_file": onset_src,
        "grid_occupancy": round(occupancy, 3),
        "sounding_key": sounding,
        "capo": best["capo"],
        "shape_key": best["key_shape"],
        "capo_ranking": ranked[:4],
        "tempo_bpm": round(bpm, 1),
        "tempo_bpm_tracked": round(raw_bpm, 1),
        "beats_per_bar": bpb,
        "subdiv": subdiv,
        "metre": f"{bpb*subdiv}/8" if subdiv == 3 else f"{bpb}/4",
        "subdiv_evidence": subdiv_scores,
        "grid_alignment": align,
        "onsets_detected": len(onset_times),
        "stroke_rate_hz": round(stroke_rate, 2),
        "direction_rule": dir_rule,
        "chords_shapes": uniq_shapes[:8],
        "chords_sounding": uniq_sound[:8],
        "canonical_pattern": rhythm_mod.render_pattern(
            canon_marks, canon_dirs, subdiv, bpb),
        "canonical_strength": [round(float(v), 3) for v in med],
        "count_line": rhythm_mod.count_line(subdiv, bpb),
        "sections": section_list,
        "simplification": simplification,
        "n_bars": len(bar_list),
        "bars_unlabelled_before_fill": n_unlabelled,
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
    ap.add_argument("--no-simplify", action="store_true",
                    help="keep every detected chord and section pattern")
    ap.add_argument("-s", "--stem",
                    help="isolated guitar track; onsets are read from this "
                         "instead of the full mix")
    ap.add_argument("-n", "--bars", type=int, default=32,
                    help="how many bars to print (default 32)")
    a = ap.parse_args()

    res = transcribe(a.audio, a.lab, title=a.title, stem=a.stem,
                     simplify=not a.no_simplify)
    print(render(res, max_bars=a.bars))
    print(f"\n  onsets read from: {res['onset_source']}"
          f"   grid occupancy: {res['grid_occupancy']:.0%}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"  wrote {a.out}")
