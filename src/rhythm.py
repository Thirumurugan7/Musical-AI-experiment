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


def slot_strengths(env, env_times, beats, beat_pos, subdiv, onset_times=None):
    """
    Onset strength at every subdivision, grouped into bars.

    When `onset_times` is given, each slot also records whether a detected
    attack actually lands in it. That distinction is the whole game: sampling
    the envelope asks "is there energy here", and a ringing chord answers yes
    everywhere, which is why sustain was being read as strumming. Peak-picked
    onsets ask "is there an attack here", which is the question a strum chart
    is actually asking.
    """
    bt = beats[:, 0] if beats.ndim > 1 else beats
    downs = [i for i in range(len(beat_pos)) if beat_pos[i] == 1]
    bpb = None
    for a, b in zip(downs, downs[1:]):
        if bpb is None:
            bpb = b - a
        elif b - a != bpb:
            continue
    bpb = bpb or 4

    # The final downbeat has no successor, so a naive pass over consecutive
    # pairs silently drops the last bar of every song — one bar in eight on the
    # synthetic suite, which capped stroke recall at 0.875 before anything else
    # got a chance to. Give it a synthetic end at the median bar length.
    spans = list(zip(downs, downs[1:]))
    if downs and len(bt) > downs[-1] + bpb:
        spans.append((downs[-1], downs[-1] + bpb))

    bars = []
    for a, b in spans:
        if b - a != bpb or a + bpb >= len(bt):
            continue
        vals, times, hits = [], [], []
        for k in range(bpb):
            t0, t1 = bt[a + k], bt[a + k + 1]
            slot_dur = (t1 - t0) / subdiv
            for s in range(subdiv):
                t = t0 + (t1 - t0) * s / subdiv
                if onset_times is not None:
                    tol = min(0.5 * slot_dur, 0.09)
                    hits.append(bool(_near(onset_times, t, tol)))
                # symmetric: the old [-40,+80] ms window biased every
                # reading 20 ms late, which showed up as a systematic
                # +40 ms stroke offset on synthetic tests
                w = (env_times >= t - 0.05) & (env_times < t + 0.05)
                vals.append(float(env[w].max()) if w.any() else 0.0)
                times.append(float(t))
        bar = {"start": float(bt[a]), "strength": vals, "times": times}
        if onset_times is not None:
            bar["hit"] = hits
        bars.append(bar)
    return bars, bpb


def _near(sorted_times, t, tol):
    """Is any detected onset within tol of t? (sorted_times must be sorted)"""
    import bisect
    i = bisect.bisect_left(sorted_times, t - tol)
    return i < len(sorted_times) and sorted_times[i] <= t + tol


def noise_floor(bars, pct=10, mult=1.25):
    """
    Absolute quiet level for the whole track, from its quietest slots.

    Kept separate from the per-section reference on purpose. A floor derived
    from the track's *median* would scale with the loud parts and silence any
    quiet section outright; a floor derived from its quietest slots only ever
    rejects material that is genuinely near-silent — a failed stem, a gap.
    """
    allv = np.concatenate([np.asarray(b["strength"], dtype=float) for b in bars])
    return float(np.percentile(allv, pct)) * mult if allv.size else 0.0


def thresholds(bars, floor_pct=10, floor_mult=1.25, rest_ratio=0.32,
               flat_range=2.5):
    """
    Track-wide reference level and rest threshold.

    A rest means the player stopped, so it has to be anchored to silence — the
    quietest material on the track — and not to a fraction of the median.

    An earlier version used `max(noise, median * 0.32)`. That second term
    invents rests in any song strummed continuously: Riptide's eight slot
    positions span 3.06 to 4.59, a range of 1.5:1 with no gaps at all, yet a
    third of the median cut straight through the middle of it and marked beats
    2 and 4 — the quietest slots, but still plainly played — as pauses.

    So the median term applies only when the track has the dynamic range to
    justify it. If the loud and quiet slots are within `flat_range` of each
    other the playing is continuous, and the pattern lives in the accents
    rather than in any silence.
    """
    allv = np.concatenate([np.asarray(b["strength"], dtype=float) for b in bars])
    pos = allv[allv > 0]
    ref = float(np.median(pos)) if pos.size else 1.0
    if not allv.size:
        return ref, 0.0
    lo = float(np.percentile(allv, floor_pct))
    hi = float(np.percentile(allv, 90))
    rest_at = lo * floor_mult
    if lo > 0 and hi / lo > flat_range:
        # genuinely dynamic: quiet slots really are gaps, so the median-relative
        # threshold is meaningful and catches them
        rest_at = max(rest_at, ref * rest_ratio)
    return ref, rest_at


def _mark(v, ref, rest_at, accent_ratio=1.30, ghost_ratio=0.75):
    if v < rest_at:
        return " "
    if v >= ref * accent_ratio:
        return "A"
    if v >= ref * ghost_ratio:
        return "x"
    return "."


def classify(bars, accent_ratio=1.30, rest_ratio=0.45, ghost_ratio=0.75,
             floor_pct=15, floor_mult=2.0):
    """
    Per slot: 'A' accented hit, 'x' normal hit, '.' ghost/light, ' ' rest.

    Thresholds are global to the track, plus an absolute noise floor.

    The previous version compared each slot against *its own bar's* median.
    That threshold was computed from the twelve values being thresholded, so
    roughly half of any bar sat above it by construction and almost nothing
    ever fell below 0.30x of it. The result: ~98 % of slots marked struck for
    any input whatsoever — including a stem attenuated to -40 dB, and audio
    with the guitar removed entirely.

    A track-wide reference fixes the real failure, which was that a bar with
    no strumming in it could not be recognised as such: its own quiet slots
    still straddled its own quiet median. The noise floor, taken from the
    quietest slots on the track, catches near-silent input outright.
    """
    allv = np.concatenate([np.asarray(b["strength"], dtype=float) for b in bars])
    pos = allv[allv > 0]
    ref = float(np.median(pos)) if pos.size else 1.0
    noise = float(np.percentile(allv, floor_pct)) if allv.size else 0.0
    rest_at = max(noise * floor_mult, ref * rest_ratio)

    for b in bars:
        marks = []
        for v in b["strength"]:
            if v < rest_at:
                marks.append(" ")
            elif v >= ref * accent_ratio:
                marks.append("A")
            elif v >= ref * ghost_ratio:
                marks.append("x")
            else:
                marks.append(".")
        b["marks"] = marks
    return bars


def classify_span(bars, ref, rest_at):
    """
    Mark one section's bars.

    Where onset evidence exists it decides struck-versus-rest outright, and the
    envelope level only chooses among accent, normal and ghost. Level alone
    cannot tell a struck slot from the tail of the previous one; an attack can.
    """
    for b in bars:
        hits = b.get("hit")
        if hits is None:
            b["marks"] = [_mark(v, ref, rest_at) for v in b["strength"]]
        else:
            b["marks"] = [
                (_mark_hit(v, ref) if h else " ")
                for v, h in zip(b["strength"], hits)
            ]
    return bars


def _mark_hit(v, ref, accent_ratio=1.30, ghost_ratio=0.70):
    """Loudness class for a slot already known to carry an attack."""
    if v >= ref * accent_ratio:
        return "A"
    if v >= ref * ghost_ratio:
        return "x"
    return "."


def canonical(bars, n_slots, ref=None, rest_at=None):
    """
    The song's representative bar: median strength per slot position.

    Classified against the same track-wide thresholds as the individual bars,
    so a rest in the summary pattern means what a rest means everywhere else.
    Taking the median across bars first is what makes this usable at all — it
    averages away the per-bar noise, leaving the pattern that actually recurs.
    """
    M = np.array([b["strength"] for b in bars])
    med = np.median(M, axis=0)
    if ref is None or rest_at is None:
        ref, rest_at = thresholds(bars)
    return med, [_mark(v, ref, rest_at) for v in med]


def sections(bars, shapes, min_bars=4):
    """
    Split the song where the chord cycle changes, so each part gets its own
    pattern instead of one average smeared over verse and chorus alike.

    Finds the cycle length that best explains the chord sequence, then cuts
    where that cycle stops holding. Falls back to a single section when no
    cycle is evident — better one honest pattern than invented structure.
    """
    n = len(shapes)
    if n < min_bars * 2:
        return [(0, n)]

    best_len, best_score = None, 0.0
    for L in range(2, min(9, n // 2 + 1)):
        hits = sum(1 for i in range(n - L) if shapes[i] == shapes[i + L])
        score = hits / (n - L)
        if score > best_score:
            best_len, best_score = L, score
    if not best_len or best_score < 0.5:
        return [(0, n)]

    cuts, i = [0], best_len
    while i < n - 1:
        if shapes[i] != shapes[i - best_len] and \
           shapes[min(i + 1, n - 1)] != shapes[min(i + 1 - best_len, n - 1)]:
            if i - cuts[-1] >= min_bars:
                cuts.append(i)
            i += best_len
        else:
            i += 1
    cuts.append(n)

    # Every bar must land in exactly one span. Dropping short spans instead of
    # merging them left those bars unclassified, and the next stage died on a
    # missing key — "I'm Yours" crashed on a four-bar tail.
    spans = []
    for a, b in zip(cuts, cuts[1:]):
        if b <= a:
            continue
        if spans and (b - a) < min_bars:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    if len(spans) > 1 and (spans[0][1] - spans[0][0]) < min_bars:
        spans[1] = (spans[0][0], spans[1][1])
        spans.pop(0)
    return spans or [(0, n)]


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
