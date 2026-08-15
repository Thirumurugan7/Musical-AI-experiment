"""Merge Chord-CNN-LSTM chord labels with detected strums into a playable chart."""
import sys, json
import numpy as np

PRETTY = {"maj": "", "min": "m", "dim": "dim", "aug": "aug",
          "maj7": "maj7", "min7": "m7", "7": "7", "hdim7": "m7b5",
          "dim7": "dim7", "sus2": "sus2", "sus4": "sus4",
          "maj6": "6", "min6": "m6", "maj9": "maj9", "min9": "m9", "9": "9"}


def pretty(label):
    if label == "N":
        return "N.C."
    if ":" not in label:
        return label
    root, qual = label.split(":", 1)
    bass = ""
    if "/" in qual:
        qual, bass = qual.split("/", 1)
        bass = "/" + bass
    return root + PRETTY.get(qual, ":" + qual) + bass


def load_lab(path):
    out = []
    for line in open(path):
        parts = line.split()
        if len(parts) >= 3:
            out.append((float(parts[0]), float(parts[1]), parts[2]))
    return out


def chord_at(chords, t):
    for s, e, c in chords:
        if s <= t < e:
            return c
    return chords[-1][2] if chords else "N"


def build(lab_path, strum_path):
    chords = load_lab(lab_path)
    strums = json.load(open(strum_path))
    bpb = strums["beats_per_bar"]
    events = strums["events"]

    bars = {}
    for e in events:
        bars.setdefault(e["bar"], []).append(e)

    lines = []
    header = (f"tempo ~{strums['tempo']:.0f} BPM   |   {bpb}/4   |   "
              f"{strums['duration']:.1f}s   |   {len(events)} strums")
    lines.append(header)
    lines.append("=" * len(header))
    lines.append("")

    for bar in sorted(bars):
        evs = sorted(bars[bar], key=lambda e: e["grid_idx"])
        # chords sounding in this bar
        names, seen = [], None
        for e in evs:
            c = pretty(chord_at(chords, e["time"]))
            if c != seen:
                names.append(c)
                seen = c
        chord_str = " ".join(names)

        # lay the strums out on a 1 & 2 & 3 & 4 & ruler
        slots = ["·"] * (bpb * 2)
        for e in evs:
            pos = int(round((e["beat_in_bar"] - 1) * 2))
            if 0 <= pos < len(slots):
                slots[pos] = e["direction"]
        ruler = " ".join(slots)
        lines.append(f"| {chord_str:<12} |  {ruler}")

    lines.append("")
    lines.append("  count:        " + " ".join(
        (str(i // 2 + 1) if i % 2 == 0 else "&") for i in range(bpb * 2)))
    lines.append("  D = downstroke   U = upstroke   · = no strum")

    conf = np.mean([e["confidence"] for e in events]) if events else 0
    lines.append(f"  mean stroke-direction confidence: {conf:.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(build(sys.argv[1], sys.argv[2]))
