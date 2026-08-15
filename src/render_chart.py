"""
Render a transcription the way guitar sites actually present one.

Conventions borrowed from Ultimate Guitar, Fender and the Hal Leonard strum
key, which all agree on the same idiom:

  - arrows for direction, not letters             (UG: down / up arrows)
  - '>' above the arrow for an accent             (UG)
  - a blank column for a rest — you let it ring   (UG: "empty spaces")
  - compound time counted in threes: 1-2-3 2-2-3  (Hal Leonard, Guitar Pro)
  - ONE repeating bar plus a chord progression, never a per-bar strum row

That last one is the substantive change. Printing 66 individual strum rows
implies per-bar measurement we cannot back up: the per-bar marks come from a
detector whose rest threshold normalises against the bar it is measuring, so
it marks ~98 % of subdivisions struck no matter what audio it is given. The
song-wide median survives that, because averaging 66 bars cancels the noise.
"""
import json
import sys

# open-position shapes, low E -> high E; None = string not played
SHAPES = {
    "G":   [3, 2, 0, 0, 0, 3],
    "Em":  [0, 2, 2, 0, 0, 0],
    "Em7": [0, 2, 2, 0, 3, 0],
    "C":   [None, 3, 2, 0, 1, 0],
    "D":   [None, None, 0, 2, 3, 2],
    "D7":  [None, None, 0, 2, 1, 2],
    "Am":  [None, 0, 2, 2, 1, 0],
    "A":   [None, 0, 2, 2, 2, 0],
    "E":   [0, 2, 2, 1, 0, 0],
    "F":   [1, 3, 3, 2, 1, 1],
}


def tokens(pattern):
    return pattern.strip().split()


def kind(tok):
    """accent | normal | ghost | rest"""
    if tok == "·":
        return "rest"
    if tok.startswith("("):
        return "ghost"
    return "accent" if tok.strip("()") .isupper() else "normal"


def count_row(subdiv, bpb):
    """12/8 -> '1 2 3   2 2 3   3 2 3   4 2 3'; 4/4 -> '1 + 2 + ...'"""
    cells = []
    for b in range(bpb):
        if subdiv == 3:
            cells += [str(b + 1), "2", "3"]
        elif subdiv == 2:
            cells += [str(b + 1), "+"]
        else:
            cells += [str(b + 1)] + ["e", "+", "a"][: subdiv - 1]
    return cells


def strum_block(pattern, subdiv, bpb, upstrokes=False):
    """Three aligned rows: accents, count, arrows. Groups spaced by beat."""
    tk = tokens(pattern)
    cnt = count_row(subdiv, bpb)
    n = min(len(tk), len(cnt))

    acc, num, arw = [], [], []
    for i in range(n):
        k = kind(tk[i])
        acc.append(">" if k == "accent" else " ")
        num.append(cnt[i])
        if k == "rest":
            arw.append(" ")          # blank column: let it ring
        else:
            # direction comes from the token the detector emitted (d/D down,
            # u/U up) rather than being re-derived from the slot index — in
            # simple metre the two disagree wherever a rest shifts the hand
            arw.append("↑" if tk[i].strip("()").lower() == "u" else "↓")
    # ghosts get a lighter glyph, the way UG draws a smaller arrow for a
    # brushed stroke — the hand keeps moving but barely catches the strings
    for i in range(n):
        if kind(tk[i]) == "ghost" and arw[i] != " ":
            arw[i] = "˅" if arw[i] == "↓" else "˄"

    def join(cells):
        out = []
        for b in range(bpb):
            grp = cells[b * subdiv:(b + 1) * subdiv]
            out.append("  ".join(c.ljust(1) for c in grp))
        return "     ".join(out)

    return join(acc), join(num), join(arw)


def diagram(name):
    """Two-line compact fingering: 'G  3 2 0 0 0 3'."""
    sh = SHAPES.get(name)
    if not sh:
        return None
    frets = " ".join("x" if f is None else str(f) for f in sh)
    return f"{name:<4} {frets}"


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def progression(bars, per_line=4):
    """
    '0:41  | G | Em7 | C | D |' — one chord per bar, stamped with the bar's
    real start time from the beat tracker.

    Identical lines are NOT collapsed into 'x3' here: once each line carries a
    timestamp it is no longer a repeat, and collapsing would throw away the
    only thing that lets you find the bar in the recording.
    """
    out = []
    for i in range(0, len(bars), per_line):
        chunk = bars[i:i + per_line]
        line = "| " + " | ".join(b["shape"].ljust(3) for b in chunk) + " |"
        out.append(f"{mmss(chunk[0]['start']):>5}  {line}")
    return out


def changes(bars, bar_len):
    """Chord changes with start time and how long each is held."""
    out, i = [], 0
    while i < len(bars):
        j = i
        while j + 1 < len(bars) and bars[j + 1]["shape"] == bars[i]["shape"]:
            j += 1
        n = j - i + 1
        start = bars[i]["start"]
        end = (bars[j + 1]["start"] if j + 1 < len(bars)
               else bars[j]["start"] + bar_len)
        out.append(f"{mmss(start):>5} - {mmss(end):>5}   "
                   f"{bars[i]['shape']:<4} {n} bar{'s' if n > 1 else ''}"
                   f"   ({end - start:.1f}s)")
        i = j + 1
    return out


def read_lyrics(path):
    """
    Read a lyric file and return [(time_or_None, text)].

    Accepts LRC ('[01:23.45] line') and plain text. LRC is preferred because
    its timestamps let chords be placed at the moment they actually change;
    plain text falls back to one line per bar, which is only approximate.
    """
    import re
    lines = []
    for raw in open(path, encoding="utf-8"):
        raw = raw.rstrip("\n")
        if not raw.strip():
            continue
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)", raw)
        if m:
            t = int(m.group(1)) * 60 + float(m.group(2))
            lines.append((t, m.group(3)))
        else:
            lines.append((None, raw))
    return lines


def chords_over_lyrics(bars, lyrics, bar_len):
    """
    The layout every guitar site uses: a chord row sitting above the lyric,
    each symbol over the syllable where the change lands.

    Placement is by time. A line spanning t0..t1 is mapped across its own
    width, and any bar starting inside that window is written at the column
    matching its offset. Chords never overwrite one another; if two would
    collide the later one shifts right, so nothing is silently dropped.
    """
    timed = [l for l in lyrics if l[0] is not None]
    out = []

    if not timed:
        # no timestamps: fall back to one lyric line per bar, in order
        for i, (_, text) in enumerate(lyrics):
            if i < len(bars):
                out.append(bars[i]["shape"])
                out.append(text)
                out.append("")
        return out, False

    for i, (t0, text) in enumerate(timed):
        t1 = timed[i + 1][0] if i + 1 < len(timed) else t0 + bar_len
        span = max(t1 - t0, 1e-6)
        row = [" "] * max(len(text), 1)
        for b in bars:
            if t0 <= b["start"] < t1:
                col = int((b["start"] - t0) / span * len(row))
                col = max(0, min(col, len(row) - 1))
                while col < len(row) and row[col] != " ":
                    col += 1
                if col + len(b["shape"]) > len(row):
                    row += [" "] * (col + len(b["shape"]) - len(row))
                for k, ch in enumerate(b["shape"]):
                    row[col + k] = ch
        chord_line = "".join(row).rstrip()
        if chord_line:
            out.append(chord_line)
        out.append(text)
    return out, True


def render(d, lyrics_path=None):
    L = []
    title = d.get("title") or "untitled"
    L.append(title)
    L.append("=" * len(title))

    capo = d.get("capo", 0)
    key_line = f"Key {d['sounding_key']}"
    if capo:
        key_line += f"  ·  capo {capo} (play in {d['shape_key']})"
    beat = "dotted quarter" if d.get("subdiv", 3) == 3 else "quarter"
    key_line += f"  ·  {d['metre']}  ·  {d['tempo_bpm']:.0f} BPM ({beat})"
    L.append(key_line)
    L.append("")

    shapes = []
    for b in d["bars"]:
        if b["shape"] not in shapes:
            shapes.append(b["shape"])
    L.append("CHORDS   " + "   ".join(shapes))
    L.append("")
    for s in shapes:
        dg = diagram(s)
        if dg:
            L.append("   " + dg)
    L.append("   " + " " * 5 + "E A D G B e")
    L.append("")

    subdiv = d.get("subdiv", 3)
    bpb = d.get("beats_per_bar", 4)
    acc, num, arw = strum_block(d["canonical_pattern"], subdiv, bpb)
    L.append(f"STRUMMING   one bar of {d['metre']}, repeats")
    L.append("")
    L.append("   " + acc)
    L.append("   " + num)
    L.append("   " + arw)
    L.append("")
    L.append("   >  accent (measured)      ↓  downstroke (derived, see below)")
    L.append("   blank column = no stroke, let it ring")
    L.append("")

    bar_len = d.get("bar_len") or (
        d["bars"][1]["start"] - d["bars"][0]["start"] if len(d["bars"]) > 1 else 0)
    first, last = d["bars"][0]["start"], d["bars"][-1]["start"] + bar_len
    L.append(f"PROGRESSION   one chord per bar  ·  bar = {bar_len:.2f}s  ·  "
             f"first downbeat {mmss(first)}, last bar ends {mmss(last)}")
    L.append("")
    for line in progression(d["bars"]):
        L.append("   " + line)
    L.append("")

    L.append("CHORD CHANGES   cue times against the recording")
    L.append("")
    for line in changes(d["bars"], bar_len):
        L.append("   " + line)
    L.append("")

    if lyrics_path:
        lyrics = read_lyrics(lyrics_path)
        block, timed = chords_over_lyrics(d["bars"], lyrics, bar_len)
        L.append("CHORDS OVER LYRICS" +
                 ("   placed by timestamp" if timed else
                  "   NO timestamps in file — one line per bar, approximate"))
        L.append("")
        for line in block:
            L.append("   " + line)
        L.append("")

    simp = d.get("simplification")
    if simp and simp.get("folded"):
        L.append("SIMPLIFIED FOR PLAYING")
        L.append("")
        L.append(f"   kept the {len(simp['core'])} chords covering "
                 f"{simp.get('core_coverage', 0):.0%} of the song and folded "
                 f"{simp.get('bars_changed', 0)} bars into them:")
        L.append("")
        for k, v in simp["folded"].items():
            L.append(f"     {k:<12} -> {v['to']:<12} {v['bars']} bar"
                     f"{'s' if v['bars'] > 1 else ''}")
        L.append("")
        L.append("   rare chords are usually detection errors, not harmony —")
        L.append("   run with --no-simplify to see every chord detected")
        L.append("")

    L.append("HOW MUCH OF THIS IS MEASURED")
    L.append("")
    L.append("   measured   chords, key, capo, tempo, metre, accent placement")
    L.append("   derived    stroke direction — from metre and stroke rate,")
    L.append(f"              not from audio ({d.get('direction_rule','')})")
    L.append("   averaged   the strum row is the median across "
             f"{d.get('n_bars', len(d['bars']))} bars, not any single bar")
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="render a transcription as a guitar chart")
    ap.add_argument("json", help="output of transcribe.py -o")
    ap.add_argument("-l", "--lyrics",
                    help="lyric file to lay chords over: .lrc (timestamped, "
                         "placed exactly) or plain text (one line per bar)")
    ap.add_argument("-k", "--key", default="fullmix",
                    help="which variant, when the file holds several")
    a = ap.parse_args()

    d = json.load(open(a.json))
    if "fullmix" in d:            # simulator payload holds both variants
        d = d[a.key]
    print(render(d, lyrics_path=a.lyrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
