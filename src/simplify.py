"""
Reduce a transcription to something a person would actually play.

A chord model reports what it hears frame by frame, which is not what a guitar
chart is. A chart is a deliberate simplification: three or four open shapes,
looping, with one strumming pattern. Ultimate Guitar's Riptide page says
"capo 1, Am G C" — not because those are the only sonorities in the recording,
but because they are what you need to play it.

So this stage does on purpose what a chart does on purpose:

  1. keep the few chords that carry the song
  2. fold every rare chord into the nearest one that was kept
  3. prefer shapes that are easy in open position

Rare chords are overwhelmingly errors rather than discoveries. On Riptide the
model emitted a B major — not in the key at all — for six bars, traceable to
segments labelled C:maj sitting between neighbours labelled C#:maj. A semitone
slip, not a borrowed chord.

The trade is explicit: harmonic detail for playability. Anything folded away is
reported in `simplification` so the original is never silently lost.
"""
import collections

# chord-tone sets by quality, as semitone offsets from the root
TONES = {
    "maj": (0, 4, 7), "min": (0, 3, 7),
    "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "7": (0, 4, 7, 10),
    "dim": (0, 3, 6), "aug": (0, 4, 8), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
    "maj6": (0, 4, 7, 9), "min6": (0, 3, 7, 9), "dim7": (0, 3, 6, 9),
}

# open-position shapes, easiest first — used to break ties toward playability
EASY = ["G", "Em", "C", "D", "Am", "A", "E", "Dm", "Em7", "Am7",
        "G7", "D7", "A7", "E7", "Cadd9", "Dsus4", "Asus2", "F"]


def tones(root, quality):
    base = TONES.get(quality)
    if base is None:                       # unknown colouring: treat as triad
        base = TONES["min"] if "min" in quality else TONES["maj"]
    return {(root + i) % 12 for i in base}


def root_distance(a, b):
    d = abs(a - b) % 12
    return min(d, 12 - d)


def substitute(label, core, parse):
    """
    The kept chord that best replaces a rare one.

    Score is `2 * shared chord tones - root distance`. Shared tones matter most
    — a substitute should sound like what it replaces — but the root distance
    term is what catches the common failure, a chord detected a semitone off:
    it shares almost no tones with its true neighbour yet sits one fret away.
    """
    p = parse(label)
    if not p:
        return None
    best, best_score = None, None
    for cand in core:
        q = parse(cand)
        if not q:
            continue
        shared = len(tones(p[0], p[1]) & tones(q[0], q[1]))
        score = 2 * shared - root_distance(p[0], q[0])
        if best_score is None or score > best_score:
            best, best_score = cand, score
    return best


def reduce_chords(bar_labels, parse, keep_share=0.85, max_core=4, min_share=0.04):
    """
    Pick the core vocabulary and map everything else onto it.

    Chords are taken in order of how many bars they hold until either
    `keep_share` of the song is covered or `max_core` chords are kept; anything
    holding less than `min_share` is never core, however the ordering falls.

    Returns (new_labels, report).
    """
    counts = collections.Counter(c for c in bar_labels if c not in ("N", "X"))
    total = sum(counts.values())
    if not total:
        return list(bar_labels), {"core": [], "folded": {}, "coverage": 0.0}

    core, covered = [], 0
    for label, n in counts.most_common():
        share = n / total
        if core and (covered >= keep_share or len(core) >= max_core
                     or share < min_share):
            break
        core.append(label)
        covered += share

    mapping = {}
    for label in counts:
        if label not in core:
            sub = substitute(label, core, parse)
            if sub:
                mapping[label] = sub

    out = [mapping.get(c, c) for c in bar_labels]
    folded = {k: {"to": v, "bars": counts[k]} for k, v in mapping.items()}
    return out, {
        "core": core,
        "core_coverage": round(covered, 3),
        "folded": folded,
        "bars_changed": sum(counts[k] for k in mapping),
    }


def patterns_agree(a, b, tol=0.25):
    """Do two section patterns differ enough to be worth printing separately?"""
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    struck = lambda t: t != "·"
    diff = sum(1 for x, y in zip(ta, tb) if struck(x) != struck(y))
    return diff / len(ta) <= tol


def collapse_sections(sections, tol=0.25):
    """
    Merge neighbouring sections whose strumming is effectively the same.

    A chart that prints six near-identical patterns is claiming a distinction
    the player cannot hear. Merge until what remains is genuinely different.
    """
    if not sections:
        return sections
    out = [dict(sections[0])]
    for s in sections[1:]:
        prev = out[-1]
        if patterns_agree(prev["pattern"], s["pattern"], tol):
            prev["to_bar"] = s["to_bar"]
            prev["chords"] = list(dict.fromkeys(prev["chords"] + s["chords"]))[:8]
            prev["merged"] = prev.get("merged", 1) + 1
        else:
            out.append(dict(s))
    return out
