"""
Match a measured bar against real strumming patterns.

Deciding each slot independently can only ever produce something noise-shaped:
eight separate yes/no calls on a continuous signal give you a grid with holes,
not a rhythm. Compare what we produced for Riptide against what the Strum app
publishes for the same song:

    Strum   down  -  down  -  -  down  up  down
    ours    d  u  -  u  d  u  -  u

Theirs is a figure a hand plays. Ours is a threshold applied eight times.

Players do not make eight independent decisions per bar — they play one of a
small number of learned figures. So score the measured slot strengths against
a library of those figures and take the best fit. This is what a transcriber
actually does: recognise the pattern, not classify each eighth.

The cost is honest and worth stating: this cannot discover a pattern outside
the library, and it will force an unusual rhythm into the nearest familiar one.
In exchange the output is always something playable, and a near-miss stays
musical instead of degrading into alternation.
"""

# Each entry: name, subdivisions per beat, per-slot marks.
#   D = accented down, d = down, u = up, . = ghost, ' ' = no stroke
# Drawn from the standard beginner/intermediate repertoire that guitar
# teaching sites converge on.
LIBRARY = [
    # --- 4/4, eighths ---
    ("all downs",        2, "D d D d D d D d"),
    ("D DU UDU (island)",2, "D   D u   u D u"),
    ("D DU D DU",        2, "D   D u D   D u"),
    ("DD DU UDU",        2, "D d D u   u D u"),
    ("D D DU D",         2, "D   D   D u D  "),
    ("DU DU DU DU",      2, "D u d u D u d u"),
    ("D  D  DU D",       2, "D     D   D u D"),
    ("on the beat",      2, "D   D   D   D  "),
    ("D DU  UD",         2, "D   D u     u D"),
    ("folk D DU DU",     2, "D   D u D u D u"),
    # --- 4/4, sixteenths (only the common ones) ---
    ("16th gallop",      4, "D  d D  d D  d D  d"[:16]),
    # --- 12/8 / 6/8 compound ---
    ("compound all down",3, "D d d D d d D d d D d d"),
    ("compound 1 & 4",   3, "D     D     D     D    "),
    ("compound skip 2",  3, "D   d D   d D   d D   d"),
    ("6/8 two groups",   3, "D d d D d d"),
    ("6/8 accented",     3, "D     D d d"),
]


def _slots(mark_string, n_slots):
    toks = [mark_string[i] for i in range(len(mark_string)) if i % 2 == 0]
    toks = (toks + [" "] * n_slots)[:n_slots]
    return toks


def _weight(tok):
    """Expected relative strength of a slot, for correlation."""
    return {"D": 1.0, "d": 0.62, "u": 0.55, ".": 0.28, " ": 0.0}.get(tok, 0.0)


def candidates(subdiv, beats_per_bar):
    n = subdiv * beats_per_bar
    for name, sd, marks in LIBRARY:
        if sd != subdiv:
            continue
        toks = _slots(marks, n)
        if len(toks) == n:
            yield name, toks


def best_match(strengths, subdiv, beats_per_bar, min_score=0.30):
    """
    Best-fitting library pattern for one bar's slot strengths.

    Correlation on normalised strengths, so it compares shape rather than
    level. Returns (name, tokens, score) or None when nothing fits well enough
    — in which case the caller should keep its own per-slot result rather than
    force a pattern that is not there.
    """
    n = subdiv * beats_per_bar
    if len(strengths) != n:
        return None
    lo, hi = min(strengths), max(strengths)
    if hi <= lo:
        return None
    norm = [(v - lo) / (hi - lo) for v in strengths]

    best = None
    for name, toks in candidates(subdiv, beats_per_bar):
        w = [_weight(t) for t in toks]
        wl, wh = min(w), max(w)
        if wh <= wl:
            continue
        wn = [(x - wl) / (wh - wl) for x in w]
        ma = sum(norm) / n
        mb = sum(wn) / n
        num = sum((a - ma) * (b - mb) for a, b in zip(norm, wn))
        da = sum((a - ma) ** 2 for a in norm) ** 0.5
        db = sum((b - mb) ** 2 for b in wn) ** 0.5
        if da == 0 or db == 0:
            continue
        r = num / (da * db)
        if best is None or r > best[2]:
            best = (name, toks, r)
    if best and best[2] >= min_score:
        return best
    return None
