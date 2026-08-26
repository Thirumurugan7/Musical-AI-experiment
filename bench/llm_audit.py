"""
Ask a local LLM whether a transcription is musically plausible -- then check
whether its opinion is worth anything.

The second half is the point. This repo has twice built a proxy that looked
sensible and measured nothing: bench/compare_audio.py renders our output back
to audio and correlates -0.044 with chord F1, and HPSS cost 0.092 of F1 while
seeming obviously right. An LLM auditor is exactly that shape of idea, so it
arrives with its own calibration attached.

What the model can and cannot do is worth stating plainly. It cannot hear the
recording, so it can never tell us whether our chords match the song. All it
can judge is internal coherence: whether the chords sit in the claimed key,
whether the progression resembles music, whether capo and key agree. That is a
real signal or it is nothing, and which one is an empirical question.

The test: run the audit on every scored song, then correlate the model's
plausibility score against the chord F1 we already know. A useful auditor gives
low scores to the songs we got wrong. An auditor that correlates near zero is
compare_audio.py again and should be deleted rather than kept "just in case".

Usage:
    python bench/llm_audit.py            # audit + calibrate against F1
    python bench/llm_audit.py --model X  # default gemma4
"""
import os
import sys
import json
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = f"{ROOT}/work/bench"
OLLAMA = "http://127.0.0.1:11434/api/generate"
CACHE = f"{ROOT}/work/llm_audit_cache"

SYSTEM = (
    "You are a musicologist auditing an automatic guitar transcription. "
    "You cannot hear the song. Judge only internal musical coherence."
)

PROMPT = """Audit this automatic guitar transcription for internal musical coherence.

Detected key: {key}
Capo: {capo}
Chords found (sounding pitch): {chords}
Bar-by-bar progression: {prog}

You cannot hear the recording, so do NOT guess what the song is. Judge only:
1. Do the chords belong to the stated key? Count any that do not.
2. Does the progression look like real music, or like noise (chords changing
   with no pattern, a chord appearing once for one bar, random semitone moves)?
3. Is the chord vocabulary a plausible size for a guitar song (2-7 typical)?

Reply with ONLY a JSON object, no other text:
{{"plausibility": <integer 0-10>, "out_of_key": <integer>, "problems": ["short phrase", ...]}}

plausibility 10 = a coherent chart a guitarist would play.
plausibility 0 = incoherent, almost certainly a failed transcription."""


def ask(model, prompt, timeout=180):
    body = json.dumps({
        "model": model, "prompt": prompt, "system": SYSTEM,
        "stream": False, "format": "json",
        "options": {"temperature": 0, "seed": 7, "num_predict": 300},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"]


def condense(bars, limit=48):
    """Bar labels with consecutive repeats collapsed: the shape, not the length."""
    out = []
    for b in bars:
        lab = b.get("sounding") or b.get("shape") or "?"
        if not out or out[-1] != lab:
            out.append(lab)
    return " ".join(out[:limit]) + (" ..." if len(out) > limit else "")


def audit(model, g):
    prompt = PROMPT.format(
        key=g.get("sounding_key", "?"), capo=g.get("capo", "?"),
        chords=", ".join(g.get("chords_sounding", [])) or "none",
        prog=condense(g.get("bars", [])))
    raw = ask(model, prompt)
    try:
        d = json.loads(raw)
    except Exception:
        return None
    try:
        return {"plausibility": max(0, min(10, int(d.get("plausibility", -1)))),
                "out_of_key": int(d.get("out_of_key", 0)),
                "problems": [str(x)[:70] for x in (d.get("problems") or [])][:4]}
    except Exception:
        return None


def main():
    model = "gemma4"
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0

    sys.path.insert(0, f"{ROOT}/bench")
    import score as S

    ref = json.load(open(f"{ROOT}/bench/songs.json"))["songs"]
    os.makedirs(CACHE, exist_ok=True)

    rows = []
    for s in ref:
        p = f"{BENCH}/{s['id']}.json"
        if not os.path.isfile(p):
            continue
        g = json.load(open(p))
        R = {S.parse(c) for c in s["chords"]} - {None}
        O = {S.parse(c) for c in g.get("chords_sounding", [])} - {None}
        hit = len(R & O)
        prec = hit / max(len(O), 1)
        rec = hit / max(len(R), 1)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append((s, g, f1))
    if limit:
        rows = rows[:limit]

    print(f"model: {model}   songs: {len(rows)}")
    print(f"{'song':26} {'F1':>5} {'plaus':>6} {'OOK':>4}  problems")
    print("-" * 88)
    got = []
    t0 = time.time()
    for s, g, f1 in rows:
        cf = f"{CACHE}/{s['id']}.{model.replace(':','_')}.json"
        a = json.load(open(cf)) if os.path.isfile(cf) else None
        if a is None:
            try:
                a = audit(model, g)
            except Exception as e:
                a = None
                print(f"{s['id'][:26]:26} -- error: {str(e)[:40]}")
            if a:
                json.dump(a, open(cf, "w"))
        if not a:
            print(f"{s['id'][:26]:26} {f1:5.2f} -- no parseable verdict --")
            continue
        got.append((s["id"], f1, a))
        print(f"{s['id'][:26]:26} {f1:5.2f} {a['plausibility']:6d} "
              f"{a['out_of_key']:4d}  {'; '.join(a['problems'])[:34]}")

    print("-" * 88)
    if len(got) < 3:
        print("too few verdicts to calibrate")
        return 1

    import numpy as np
    f1s = np.array([r[1] for r in got])
    pl = np.array([float(r[2]["plausibility"]) for r in got])
    ook = np.array([float(r[2]["out_of_key"]) for r in got])

    def corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print(f"audited              {len(got)} songs in {time.time()-t0:.0f}s")
    print(f"plausibility spread  min {pl.min():.0f}  max {pl.max():.0f}  "
          f"mean {pl.mean():.2f}  sd {pl.std():.2f}")
    print()
    print(f"corr(plausibility, chord F1)   {corr(pl, f1s):+.3f}")
    print(f"corr(out_of_key,   chord F1)   {corr(ook, f1s):+.3f}   "
          f"(expected negative if useful)")
    print()
    lo = f1s[pl <= np.median(pl)]
    hi = f1s[pl > np.median(pl)]
    if len(lo) and len(hi):
        print(f"mean F1 where model is pessimistic  {lo.mean():.3f}  (n={len(lo)})")
        print(f"mean F1 where model is optimistic   {hi.mean():.3f}  (n={len(hi)})")
        print(f"separation                          {hi.mean()-lo.mean():+.3f}")
    print()
    print("A correlation near zero means the auditor is measuring nothing, the")
    print("same verdict bench/compare_audio.py earned at -0.044. Delete rather")
    print("than keep on the grounds that it looks reasonable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
