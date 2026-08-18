"""
Score strum detection against synthetic ground truth, by stroke TIME.

Slot-by-slot comparison is only meaningful when both sides agree on the grid,
and they often do not: a bar of eight eighths can be detected as sixteen
sixteenths, at which point every slot index is wrong while the audible strokes
may be perfectly placed. That conflates two very different failures.

So strokes are compared as times with a tolerance window, the standard method
for onset evaluation. Grid choice is scored separately, as its own line, and
so is the phase offset — a pattern right in shape but shifted is a different
defect from a pattern with the wrong strokes in it.
"""
import json
import glob
import os
import sys

WORK = os.path.dirname(os.path.abspath(__file__)) + "/../work/strumtests"
TOL = 0.060          # +/-60 ms, a little wider than the 50 ms onset standard,
                     # because a strum is a sweep rather than a point event


def detected_times(g):
    out = []
    bars = g["bars"]
    for i, bar in enumerate(bars):
        toks = bar["pattern"].split()
        if not toks:
            continue
        end = bars[i + 1]["start"] if i + 1 < len(bars) else bar["start"] + g.get("bar_len", 0)
        if end <= bar["start"]:
            continue
        step = (end - bar["start"]) / len(toks)
        for k, t in enumerate(toks):
            if t != "·":
                out.append(bar["start"] + k * step)
    return sorted(out)


def truth_times(t):
    beat = 60.0 / t["bpm"]
    slot = beat / t["subdiv"]
    out = []
    for bar in t["bars"]:
        for k, m in enumerate(t["pattern"]):
            if m != " ":
                out.append(bar["start"] + k * slot)
    return sorted(out)


def match(a, b, tol=TOL):
    """Greedy one-to-one matching within tol. Returns (tp, fp, fn)."""
    used = [False] * len(b)
    tp = 0
    for x in a:
        best, bi = tol + 1, -1
        for j, y in enumerate(b):
            if used[j]:
                continue
            d = abs(x - y)
            if d < best:
                best, bi = d, j
        if bi >= 0 and best <= tol:
            used[bi] = True
            tp += 1
    return tp, len(a) - tp, len(b) - tp


def main():
    cases = sorted(glob.glob(f"{WORK}/*.truth.json"))
    if not cases:
        print("no test cases — run bench/make_strum_tests.py first")
        return 1

    print(f"{'CASE':16} {'SUBDIV':>9} {'METRE':>10} {'TEMPO':>10}  "
          f"{'P':>5} {'R':>5} {'F1':>5}  {'OFFSET':>7}")
    print("-" * 80)
    TP = FP = FN = 0
    sub_ok = met_ok = tem_ok = n = 0

    for tf in cases:
        t = json.load(open(tf))
        gf = f"{WORK}/{t['name']}.json"
        if not os.path.isfile(gf):
            print(f"{t['name']:16} -- not transcribed --")
            continue
        g = json.load(open(gf))
        n += 1

        s_ok = g.get("subdiv") == t["subdiv"]
        # 6/8 detected as 12/8 is the same feel barred differently
        m_ok = (g["metre"] == t["metre"]
                or {g["metre"], t["metre"]} == {"6/8", "12/8"})
        tm_ok = abs(g["tempo_bpm"] - t["bpm"]) <= max(2.0, t["bpm"] * 0.03)
        sub_ok += s_ok; met_ok += m_ok; tem_ok += tm_ok

        gt, dt = truth_times(t), detected_times(g)
        tp, fp, fn = match(dt, gt)
        TP += tp; FP += fp; FN += fn
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / (p + r) if p + r else 0.0

        # median signed offset of matched strokes, to separate a phase error
        # from a wrong-strokes error
        offs = []
        for x in dt:
            near = min(gt, key=lambda y: abs(y - x), default=None)
            if near is not None and abs(near - x) <= TOL:
                offs.append(x - near)
        offs.sort()
        off = offs[len(offs) // 2] * 1000 if offs else float("nan")

        print(f"{t['name']:16} {'OK' if s_ok else 'XX'} {g.get('subdiv')}/{t['subdiv']:<4} "
              f"{'OK' if m_ok else 'XX'} {g['metre']:>6} "
              f"{'OK' if tm_ok else 'XX'} {g['tempo_bpm']:>5.0f}  "
              f"{p:5.2f} {r:5.2f} {f1:5.2f}  {off:6.0f}ms")

    P = TP / max(TP + FP, 1); R = TP / max(TP + FN, 1)
    F = 2 * P * R / (P + R) if P + R else 0.0
    print("-" * 80)
    print(f"cases                {n}")
    print(f"subdivision correct  {sub_ok}/{n}")
    print(f"metre correct        {met_ok}/{n}   (6/8 and 12/8 treated as equivalent)")
    print(f"tempo correct        {tem_ok}/{n}")
    print(f"stroke precision     {P:.3f}   ({FP} claimed strokes with no stroke there)")
    print(f"stroke recall        {R:.3f}   ({FN} real strokes missed)")
    print(f"stroke F1            {F:.3f}   (matched within {TOL*1000:.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
