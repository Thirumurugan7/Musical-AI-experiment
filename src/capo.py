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


# diatonic triad quality at each scale degree of a major key
MAJOR_DEGREES = {0: "maj", 2: "min", 4: "min", 5: "maj",
                 7: "maj", 9: "min", 11: "dim"}


def sounding_key(weighted_chords, order=None):
    """
    The key the recording sounds in.

    This used to return whichever root was held longest, which is wrong often
    enough to matter: the IV chord frequently occupies more of a pop song than
    the I. Across a five-song benchmark it named D for A, E for B and F for C —
    every error a fourth above the true tonic.

    Instead, score all twelve major keys by how well the song's chords fit their
    diatonic triads, then decide between the winning key and its relative minor
    using the chords the song opens and closes on. Those two positions carry
    more tonal weight than duration anywhere else: "Wonderwall" spends little
    time on F#m but begins and ends there, and every chart calls it F# minor.

    `order` is the chord labels in playing order, used only for that decision.
    """
    if not weighted_chords:
        return None, False

    total_dur = sum(d for _, d in weighted_chords) or 1.0

    # chord roots in playing order, for cadence counting
    seq = []
    for label in (order or []):
        p = parse(label)
        if p and (not seq or seq[-1] != p):
            seq.append(p)

    def fit(tonic):
        s = 0.0
        for label, dur in weighted_chords:
            p = parse(label)
            if not p:
                continue
            deg = (p[0] - tonic) % 12
            want = MAJOR_DEGREES.get(deg)
            is_min = p[1].startswith("min")
            if want is None:
                s -= dur * 0.5                      # chromatic to this key
            elif want == "dim" or (want == "min") == is_min:
                s += dur                            # fits, quality and all
            else:
                s += dur * 0.25                     # right root, wrong quality
        s /= total_dur

        # Scale membership alone cannot separate a key from its IV or its V —
        # they share six of seven notes, which is why an earlier version called
        # Creep C instead of G and Sweet Home Alabama G instead of D. What marks
        # a tonic is arrival: the chord a dominant resolves to, and the chord a
        # song opens and closes on.
        if seq:
            dom = (tonic + 7) % 12
            sub = (tonic + 5) % 12
            cad = sum(1 for a, b in zip(seq, seq[1:])
                      if b[0] == tonic and a[0] in (dom, sub))
            s += 0.55 * cad / max(len(seq) - 1, 1)
            if seq[0][0] == tonic:
                s += 0.22
            if seq[-1][0] == tonic:
                s += 0.28
        return s

    scale = max(range(12), key=fit)

    # the tonic is either that major key or its relative minor; the chords the
    # song starts and ends on decide which
    minor = (scale + 9) % 12
    edge = 0.0
    if order:
        firsts = [c for c in order[:2] if parse(c)]
        lasts = [c for c in order[-2:] if parse(c)]
        for label in firsts + lasts:
            p = parse(label)
            if p[0] == minor and p[1].startswith("min"):
                edge += 1
            elif p[0] == scale:
                edge -= 1

    total = {}
    for label, dur in weighted_chords:
        p = parse(label)
        if p:
            total[(p[0], p[1].startswith("min"))] = \
                total.get((p[0], p[1].startswith("min")), 0) + dur
    maj_w = total.get((scale, False), 0)
    min_w = total.get((minor, True), 0)

    is_minor = edge > 0 or (edge == 0 and min_w > maj_w * 1.3)

    # A song that finishes on the major tonic or its dominant is in the major
    # key, whatever the durations say. Hallelujah spends more time on Am than
    # on C and is detected opening on Am, but it closes on G — the V that
    # resolves to C — and every published chart calls it C major.
    if order:
        for label in reversed(order):
            p = parse(label)
            if not p:
                continue
            if p[0] in (scale, (scale + 7) % 12) and not p[1].startswith("min"):
                is_minor = False
            break
    tonic = minor if is_minor else scale
    use_flats = tonic in FLAT_KEYS
    name = spell(tonic, use_flats) + ("m" if is_minor else "")
    return name, use_flats
