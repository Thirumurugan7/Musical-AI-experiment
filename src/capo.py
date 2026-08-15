"""
Capo selection and readable chord spelling.

Chord models emit absolute pitch ("Ab:maj"). Guitarists want shapes ("G" with
a capo on 1). This picks the capo position that turns the song into the
easiest open shapes, then spells the result correctly for the resulting key.
"""

PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
      "F": 5, "E#": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
      "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11}

SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Keys conventionally written with flats (by tonic pitch class, major)
FLAT_KEYS = {5, 10, 3, 8, 1, 6}          # F Bb Eb Ab Db Gb

QUALITY = {"maj": "", "min": "m", "dim": "dim", "aug": "aug", "maj7": "maj7",
           "min7": "m7", "7": "7", "hdim7": "m7b5", "dim7": "dim7",
           "sus2": "sus2", "sus4": "sus4", "maj6": "6", "min6": "m6",
           "maj9": "maj9", "min9": "m9", "9": "9", "11": "11", "13": "13"}

# How comfortable each open shape is. 1.0 = first-position open chord.
OPEN_MAJ = {"G": 1.0, "C": 1.0, "D": 1.0, "A": 1.0, "E": 1.0,
            "F": 0.35, "B": 0.20}
OPEN_MIN = {"Em": 1.0, "Am": 1.0, "Dm": 1.0,
            "Bm": 0.35, "F#m": 0.30, "Cm": 0.15}
HARD = 0.12


def parse(label):
    """'Ab:min7/b3' -> (8, 'min7', 'b3')  |  'N' -> None"""
    if label in ("N", "X"):
        return None
    if ":" in label:
        root, rest = label.split(":", 1)
    else:
        root, rest = label, "maj"
    bass = None
    if "/" in rest:
        rest, bass = rest.split("/", 1)
    if root not in PC:
        return None
    return PC[root], rest, bass


def spell(pc, use_flats):
    return (FLAT if use_flats else SHARP)[pc % 12]


def render(pc, quality, bass, use_flats):
    name = spell(pc, use_flats) + QUALITY.get(quality, ":" + quality)
    return name + ("/" + bass if bass else "")


def playability(pc, quality, use_flats):
    name = spell(pc, use_flats)
    if quality in ("min", "min7", "min6", "min9"):
        base = OPEN_MIN.get(name + "m", HARD)
        # m7 is usually easier than the plain minor barre (Bm7 < Bm)
        if quality == "min7" and base < 0.5:
            base += 0.10
    elif quality in ("maj", "maj7", "7", "maj6", "sus2", "sus4", "maj9", "9"):
        base = OPEN_MAJ.get(name, HARD)
        if quality in ("7", "maj7") and base < 0.5:
            base += 0.12          # F7 / B7 easier than F / B
    else:
        base = OPEN_MAJ.get(name, HARD) * 0.8
    return base


def choose_capo(weighted_chords, max_capo=7):
    """
    weighted_chords: list of (label, seconds)
    Returns a ranked list of {capo, score, key, chords}
    """
    parsed = []
    for label, dur in weighted_chords:
        p = parse(label)
        if p:
            parsed.append((p[0], p[1], p[2], dur))
    if not parsed:
        return []

    # tonic = most-played root, good enough for diatonic pop
    tally = {}
    for pc, q, _, d in parsed:
        tally[pc] = tally.get(pc, 0) + d
    tonic = max(tally, key=tally.get)

    total = sum(d for *_, d in parsed)
    out = []
    for capo in range(0, max_capo + 1):
        shaped_tonic = (tonic - capo) % 12
        use_flats = shaped_tonic in FLAT_KEYS
        score = sum(playability((pc - capo) % 12, q, use_flats) * d
                    for pc, q, _, d in parsed) / total
        score -= capo * 0.012            # mild preference for a low capo
        seen, chords = set(), []
        for pc, q, _, d in sorted(parsed, key=lambda x: -x[3]):
            nm = render((pc - capo) % 12, q, None, use_flats)
            if nm not in seen:
                seen.add(nm)
                chords.append(nm)
        out.append({"capo": capo, "score": round(score, 4),
                    "key_shape": spell(shaped_tonic, use_flats),
                    "use_flats": use_flats, "chords": chords[:8]})
    out.sort(key=lambda r: -r["score"])
    return out


def sounding_key(weighted_chords):
    tally = {}
    for label, dur in weighted_chords:
        p = parse(label)
        if p:
            tally[p[0]] = tally.get(p[0], 0) + dur
    if not tally:
        return None, False
    tonic = max(tally, key=tally.get)
    use_flats = tonic in FLAT_KEYS
    return spell(tonic, use_flats), use_flats
