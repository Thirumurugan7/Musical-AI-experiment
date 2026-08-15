"""
Accent-based rhythm extraction.

Earlier versions emitted down/up arrows from a metric prior. That prior was
tested against "Perfect" and scored 0 % — in compound metre players use
continuous downstrokes, so alternating D-U is simply wrong. See COMPARISON.md.

What *is* recoverable from a full mix is accent structure: on "Perfect" the
strokes Fender tells you to accent (1, 4, 7, 10) show 2.12x the onset strength
of their neighbours. So the rhythm output is now accents, which are measured,
plus a stroke direction that is *derived from metre and stroke rate* and
labelled as such rather than pretended to be a measurement.
"""
import numpy as np


def slot_strengths(env, env_times, beats, beat_pos, subdiv):
    """Onset strength at every subdivision, grouped into bars."""
    bt = beats[:, 0] if beats.ndim > 1 else beats
    downs = [i for i in range(len(beat_pos)) if beat_pos[i] == 1]
    bpb = None
    for a, b in zip(downs, downs[1:]):
        if bpb is None:
            bpb = b - a
        elif b - a != bpb:
            continue
    bpb = bpb or 4

    bars = []
    for a, b in zip(downs, downs[1:]):
        if b - a != bpb:
            continue
        vals, times = [], []
        for k in range(bpb):
            t0, t1 = bt[a + k], bt[a + k + 1]
            for s in range(subdiv):
                t = t0 + (t1 - t0) * s / subdiv
                w = (env_times >= t - 0.04) & (env_times < t + 0.08)
                vals.append(float(env[w].max()) if w.any() else 0.0)
                times.append(float(t))
        bars.append({"start": float(bt[a]), "strength": vals, "times": times})
    return bars, bpb


def classify(bars, accent_ratio=1.30, rest_ratio=0.30, ghost_ratio=0.60):
    """
    Per slot: 'A' accented hit, 'x' normal hit, '.' ghost/light, ' ' rest.
    Thresholds are relative to each bar's own median, so the classification
    survives level changes between verse and chorus.
    """
    for b in bars:
        s = np.array(b["strength"])
        med = np.median(s[s > 0]) if (s > 0).any() else 1.0
        marks = []
        for v in s:
            if v < med * rest_ratio:
                marks.append(" ")
            elif v >= med * accent_ratio:
                marks.append("A")
            elif v >= med * ghost_ratio:
                marks.append("x")
            else:
                marks.append(".")
        b["marks"] = marks
    return bars


def canonical(bars, n_slots):
    """The song's representative bar: median strength per slot position."""
    M = np.array([b["strength"] for b in bars])
    med = np.median(M, axis=0)
    base = np.median(med[med > 0]) if (med > 0).any() else 1.0
    marks = []
    for v in med:
        if v < base * 0.30:
            marks.append(" ")
        elif v >= base * 1.30:
            marks.append("A")
        elif v >= base * 0.60:
            marks.append("x")
        else:
            marks.append(".")
    return med, marks


def directions(marks, subdiv, stroke_rate_hz):
    """
    Stroke direction, DERIVED (not measured) from metre and stroke rate.

    - Compound metre (3 subdivisions per beat): continuous alternation would
      flip the hand's orientation every beat, which players do not do. Below
      roughly 5 strokes/sec they play all downstrokes; above that they must
      alternate to keep up.
    - Simple metre (2 per beat): the hand moves down on beats, up on offbeats.

    Rests do not consume a stroke of the alternation, because the hand keeps
    moving through them.
    """
    if subdiv == 3:
        if stroke_rate_hz <= 5.0:
            rule = "compound, moderate tempo -> all downstrokes"
            return ["D" if m != " " else " " for m in marks], rule
        rule = "compound, fast -> down on beats, alternating between"
        return ([("D" if i % 3 == 0 else ("U" if i % 3 == 2 else "D"))
                 if m != " " else " " for i, m in enumerate(marks)], rule)
    rule = "simple metre -> down on beats, up on offbeats"
    return ([("D" if i % 2 == 0 else "U") if m != " " else " "
             for i, m in enumerate(marks)], rule)


def render_pattern(marks, dirs, subdiv, bpb):
    """'D d d D d d ...' — uppercase means accented."""
    out = []
    for m, d in zip(marks, dirs):
        if m == " ":
            out.append("·")
        elif m == "A":
            out.append(d.upper())
        elif m == "x":
            out.append(d.lower())
        else:
            out.append("(" + d.lower() + ")")
    return " ".join(out)


def count_line(subdiv, bpb):
    row = []
    for b in range(bpb):
        row.append(str(b + 1))
        row += ["." if subdiv == 3 else "&"] * (subdiv - 1)
    return " ".join(row)
