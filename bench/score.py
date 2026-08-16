#!/usr/bin/env python3
"""
Score pipeline output against the published-chart reference set.

Two decisions worth stating, because they change the numbers:

Chords are compared as *sounding* pitch classes, not shapes. A shape depends
on the capo, so two charts that produce identical music score zero against
each other if you compare shapes. Comparing shapes is what made an earlier
run of this benchmark report 0.68 when the real figure was 0.98.

Capo is reported but not scored. Capo 5 with G shapes and capo 0 with C
shapes are the same music; preferring one that avoids a barre chord is a
feature, not an error.
"""
import json
import os
import sys

ROOT = "/Users/thirumurugansivalingam/Desktop/personal/music-ai"
BENCH = f"{ROOT}/work/bench"

PC = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
      'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10,
      'B': 11}


def parse(ch):
    """'F#m7' -> (6, True). Quality reduced to major/minor: charts disagree
    constantly about 7ths and adds, and a player reads them off the same shape."""
    ch = ch.strip()
    if not ch or ch in ("N.C.", "N"):
        return None
    root = ch[:2] if len(ch) > 1 and ch[1] in "#b" else ch[:1]
    if root not in PC:
        return None
    rest = ch[len(root):]
    minor = rest.startswith("m") and not rest.startswith("maj")
    return PC[root], minor


def same_key(a, b):
    """Exact match on the detector's first choice."""
    pa, pb = parse(a.split(" or ")[0]), parse(b)
    return pa is not None and pa == pb


def key_offered(a, b):
    """
    Match if the reference is among the candidates the detector offered.

    When two keys score within a hair of each other the detector says so
    ("C or Em") instead of guessing. That is a weaker claim than a single
    answer, so it is scored separately rather than folded into one number.
    """
    pb = parse(b)
    return any(parse(x) == pb for x in a.split(" or ") if parse(x))


def compound(m):
    return m in ("6/8", "12/8", "9/8", "3/8")


def main():
    ref = json.load(open(f"{ROOT}/bench/songs.json"))["songs"]
    rows, missing = [], []
    for s in ref:
        p = f"{BENCH}/{s['id']}.json"
        if not os.path.isfile(p):
            missing.append(s["id"])
            continue
        rows.append((s, json.load(open(p))))

    print(f"{'SONG':30} {'KEY':>14}  {'METRE':>12}  CHORD-F1  CAPO")
    print("-" * 96)

    k_ok = m_ok = k_any = amb = 0
    f1s = []
    for s, g in rows:
        kg = same_key(g["sounding_key"], s["key"])
        ka = key_offered(g["sounding_key"], s["key"])
        k_any += ka
        amb += (" or " in g["sounding_key"])
        mg = compound(g["metre"]) == compound(s["metre"])
        R = {parse(c) for c in s["chords"]} - {None}
        O = {parse(c) for c in g["chords_sounding"]} - {None}
        hit = len(R & O)
        prec = hit / max(len(O), 1)
        rec = hit / max(len(R), 1)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        k_ok += kg
        m_ok += mg
        f1s.append(f1)
        name = f"{s['artist']} — {s['title']}"
        print(f"{name[:30]:30} "
              f"{'OK' if kg else 'XX'} {g['sounding_key']:>5}/{s['key']:<5} "
              f"{'OK' if mg else 'XX'} {g['metre']:>5}/{s['metre']:<5} "
              f"  {f1:.2f}     {g['capo']}/{s['capo']}")

    n = len(rows) or 1
    print("-" * 96)
    print(f"songs scored          {len(rows)}"
          + (f"   (missing: {', '.join(missing)})" if missing else ""))
    print(f"key, first choice     {k_ok}/{n}  = {k_ok/n:.1%}")
    print(f"key, among candidates  {k_any}/{n}  = {k_any/n:.1%}")
    print(f"flagged ambiguous     {amb}/{n}")
    print(f"metre correct         {m_ok}/{n}  = {m_ok/n:.1%}")
    print(f"mean chord F1         {sum(f1s)/n:.3f}")
    exact = sum(1 for f in f1s if f >= 0.999)
    print(f"chords exactly right  {exact}/{n}  = {exact/n:.1%}")
    weak = sorted(((f, s['artist'] + ' — ' + s['title'])
                   for f, (s, _) in zip(f1s, rows)))[:8]
    print("\nweakest chord results:")
    for f, name in weak:
        print(f"   {f:.2f}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
