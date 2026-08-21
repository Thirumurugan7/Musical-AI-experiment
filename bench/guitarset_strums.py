"""
Derive strum ground truth from GuitarSet's per-string note annotations.

GuitarSet was recorded through a hexaphonic pickup -- one isolated signal per
string -- so its note onsets were tracked on monophonic audio, the one case
where onset detection is genuinely reliable. That gives us something this repo
has never had: real stroke times, played by six people who never saw our grid.

A strum is not annotated directly, but it does not need to be. It is a cluster
of note onsets across strings within a few tens of milliseconds, and the order
the strings fire in gives the direction. Annotation index 0 is the lowest
string and 5 the highest -- verified against median pitch, which runs
monotonically from around E2 to E4 -- so time rising with string index is a
down stroke and falling is an up stroke.

The stroke time is the cluster's *first* onset, the leading edge of the sweep.
That matters: it is the convention Murgul et al. score against, and it is what
a player hears as the beat. Taking the centre of the sweep instead would
introduce exactly the systematic lateness documented in PRIOR_ART.md.

Direction is the less certain half. A cluster of two notes barely constrains a
slope, so `confidence` reports how strongly the ordering actually holds, and
callers should filter on it rather than trusting every label.
"""
import json
import glob
import os
import sys

import numpy as np

# Notes further apart than this are separate events rather than one sweep.
# Set from the data: see the spread percentiles printed by --stats.
MAX_SPREAD = 0.070


def onsets_by_string(jams_path):
    d = json.load(open(jams_path))
    notes = [a for a in d["annotations"] if a["namespace"] == "note_midi"]
    out = []
    for idx, a in enumerate(notes):
        for o in a["data"]:
            out.append((float(o["time"]), idx, float(o["value"])))
    out.sort()
    return out, float(d["file_metadata"]["duration"])


def cluster(events, max_spread=MAX_SPREAD):
    """Group onsets into sweeps. Returns a list of lists of (t, string, midi)."""
    groups, cur = [], []
    for e in events:
        if cur and e[0] - cur[0][0] > max_spread:
            groups.append(cur)
            cur = []
        cur.append(e)
    if cur:
        groups.append(cur)
    return groups


def describe(group):
    ts = np.array([g[0] for g in group])
    ss = np.array([g[1] for g in group], dtype=float)
    n = len(group)
    d = {"time": float(ts.min()), "n_strings": n,
         "spread_ms": float((ts.max() - ts.min()) * 1000)}
    if n >= 2 and ts.max() > ts.min():
        # sign of the string-index/time relationship gives the sweep direction
        r = float(np.corrcoef(ss, ts)[0, 1]) if np.std(ss) > 0 else 0.0
        d["direction"] = "D" if r > 0 else "U"
        d["confidence"] = abs(r)
    else:
        d["direction"] = None
        d["confidence"] = 0.0
    return d


def strums(jams_path, min_strings=2, max_spread=MAX_SPREAD):
    events, dur = onsets_by_string(jams_path)
    out = [describe(g) for g in cluster(events, max_spread)]
    return [s for s in out if s["n_strings"] >= min_strings], dur


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stats = "--stats" in sys.argv
    pat = args[0] if args else "work/guitarset/annotation/*_comp.jams"
    files = sorted(glob.glob(pat))
    if not files:
        print(f"no annotations matching {pat}")
        return 1

    if stats:
        allspread, allsizes, dirs, confs, rates = [], [], [], [], []
        for f in files:
            ss, dur = strums(f)
            for s in ss:
                allspread.append(s["spread_ms"])
                allsizes.append(s["n_strings"])
                if s["direction"]:
                    dirs.append(s["direction"])
                    confs.append(s["confidence"])
            if dur:
                rates.append(len(ss) / dur)
        a = np.array(allspread)
        print(f"files                {len(files)}")
        print(f"strums (>=2 strings) {len(allspread)}")
        print(f"strums per second    {np.mean(rates):.2f}")
        print(f"strings per strum    mean {np.mean(allsizes):.2f}  "
              f"median {int(np.median(allsizes))}  max {max(allsizes)}")
        print(f"\nsweep spread, first onset to last, milliseconds:")
        for p in (50, 75, 90, 95, 99):
            print(f"  p{p:<3} {np.percentile(a, p):6.1f} ms")
        print(f"  max  {a.max():6.1f} ms")
        nd = dirs.count("D")
        print(f"\ndirection            {nd} down / {len(dirs)-nd} up "
              f"({100*nd/max(len(dirs),1):.0f}% down)")
        print(f"mean confidence      {np.mean(confs):.2f}")
        return 0

    for f in files:
        ss, dur = strums(f)
        print(f"\n{os.path.basename(f)}  ({dur:.1f}s, {len(ss)} strums)")
        for s in ss[:12]:
            print(f"  {s['time']:7.3f}  {s['n_strings']} strings  "
                  f"{s['spread_ms']:5.1f} ms  {s['direction'] or '-'} "
                  f"({s['confidence']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
