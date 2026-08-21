"""
Score our strum detection against real playing.

Every strum number in this repo has been measured against seven synthetic cases
written by the same person who wrote the detector. PRIOR_ART.md names that as
the problem it cannot escape: the synthesiser and the detector share the same
assumptions about what a strum is, so agreement between them proves nothing.

GuitarSet breaks the circle. Six players, five styles, recorded through a
hexaphonic pickup, none of whom knew about our grid. Stroke times come from
clustering per-string note onsets -- see guitarset_strums.py -- and the audio
is the reference microphone, which is what our pipeline is built to read.

Scored at 50 ms, the mir_eval onset standard, matching bench/score_strum.py.

Note what this does and does not isolate. GuitarSet is solo guitar, so there is
no separation involved and no drums to confuse the onset envelope. A score here
is the strum detector's ceiling on clean input, not its performance on a mix.
That is the right first measurement: if it cannot find strokes in isolated
guitar, nothing downstream will save it.
"""
import os
import sys
import json
import glob
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guitarset_strums import strums                       # noqa: E402
from score_strum import detected_times, match             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = f"{ROOT}/work/guitarset/audio_mono-mic"
ANNO = f"{ROOT}/work/guitarset/annotation"
LAB = f"{ROOT}/work/guitarset/lab"
OUT = f"{ROOT}/work/guitarset/transcribed"
BE = f"{ROOT}/work/ChordMiniApp/python_backend"
TOL = 0.050


def transcribe(name):
    """Run the pipeline on one excerpt; returns parsed json or None."""
    out = f"{OUT}/{name}.json"
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        return json.load(open(out))
    wav = f"{AUDIO}/{name}_mic.wav"
    lab = f"{LAB}/{name}.lab"
    if not (os.path.isfile(wav) and os.path.isfile(lab)):
        return None
    os.makedirs(OUT, exist_ok=True)
    tmp = out + ".tmp"
    r = subprocess.run(
        [f"{BE}/.venv/bin/python", "transcribe.py", wav, lab, "-t", name,
         "-o", tmp],
        cwd=f"{BE}/chordstrum", capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
        if os.path.isfile(tmp):
            os.remove(tmp)
        return None
    os.replace(tmp, out)
    return json.load(open(out))


def style_of(name):
    tok = name.split("_")[1].split("-")[0]
    return "".join(c for c in tok if not c.isdigit())


def true_tempo(name):
    """
    The tempo the excerpt was actually played at.

    GuitarSet carries it twice -- as a jams `tempo` annotation and in the
    filename, which encodes style-take-tempo-key. Prefer the annotation and
    fall back to the filename. Nothing in this repo has ever scored tempo
    against a real recording; PRIOR_ART.md notes it was silently wrong twice
    for exactly that reason.
    """
    try:
        d = json.load(open(f"{ANNO}/{name}.jams"))
        for a in d["annotations"]:
            if a["namespace"] == "tempo" and a["data"]:
                return float(a["data"][0]["value"])
    except Exception:
        pass
    try:
        return float(name.split("_")[1].split("-")[1])
    except Exception:
        return None


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    only = sys.argv[2] if len(sys.argv) > 2 else None

    names = [os.path.basename(p)[:-len(".jams")]
             for p in sorted(glob.glob(f"{ANNO}/*_comp.jams"))]
    if only:
        names = [n for n in names if style_of(n) == only]
    names = names[:limit]

    per_style = {}
    tempo_rows = []
    TP = FP = FN = 0
    done = failed = 0
    print(f"{'excerpt':26} {'style':5} {'bpm':>5} {'got':>6} {'sub':>4} "
          f"{'truth':>6} {'found':>6} {'P':>5} {'R':>5} {'F1':>5}")
    print("-" * 88)
    for name in names:
        g = transcribe(name)
        if g is None:
            failed += 1
            print(f"{name[:28]:28} -- transcribe failed --")
            continue
        done += 1
        truth = [s["time"] for s in strums(f"{ANNO}/{name}.jams")[0]]
        det = detected_times(g)
        tp, fp, fn = match(det, truth, tol=TOL)
        TP += tp; FP += fp; FN += fn
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        st = style_of(name)
        b = per_style.setdefault(st, [0, 0, 0])
        b[0] += tp; b[1] += fp; b[2] += fn

        bt = true_tempo(name)
        got = g.get("tempo_bpm")
        ok = bt is not None and got is not None and abs(got - bt) <= max(2.0, bt * 0.03)
        tempo_rows.append((name, bt, got, ok, r))
        print(f"{name[:26]:26} {st:5} {bt if bt else 0:5.0f} "
              f"{got if got else 0:6.1f}{'' if ok else '!'} "
              f"{g.get('subdiv', 0):3d} "
              f"{len(truth):6d} {len(det):6d} {p:5.2f} {r:5.2f} {f1:5.2f}")

    print("-" * 88)
    for st in sorted(per_style):
        tp, fp, fn = per_style[st]
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        print(f"  {st:20} P {p:.3f}  R {r:.3f}  "
              f"F1 {2*p*r/(p+r) if p+r else 0:.3f}")
    if tempo_rows:
        good = [t for t in tempo_rows if t[3]]
        bad = [t for t in tempo_rows if not t[3]]
        print(f"\ntempo correct        {len(good)}/{len(tempo_rows)} "
              f"(within 3%, marked ! when wrong)")
        if good and bad:
            print(f"  recall when tempo right {np.mean([t[4] for t in good]):.3f}")
            print(f"  recall when tempo wrong {np.mean([t[4] for t in bad]):.3f}")
            print("  a wrong tempo is a wrong grid, so strokes land outside the")
            print("  50 ms window even where the stroke itself was found")

    P = TP / max(TP + FP, 1); R = TP / max(TP + FN, 1)
    print(f"\nexcerpts scored      {done}  ({failed} failed)")
    print(f"stroke precision     {P:.3f}   ({FP} claimed with nothing there)")
    print(f"stroke recall        {R:.3f}   ({FN} real strokes missed)")
    print(f"stroke F1            {2*P*R/(P+R) if P+R else 0:.3f}   "
          f"(matched within {TOL*1000:.0f} ms, real playing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
