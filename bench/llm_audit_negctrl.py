"""
Negative control for the LLM auditor.

Correlating against 55 real transcriptions is a weak test: their chord F1 runs
0.80-1.00, so there is almost no badness present to detect. A flat response
there could mean the auditor is blind, or merely that everything it saw was
fine.

So corrupt the input deliberately and see whether it notices. If a chart of
random chords in no key scores the same 8 as a correct one, the auditor is
measuring nothing and no amount of real-song correlation will rescue it.
"""
import json, os, sys, random
sys.path.insert(0, "bench")
from llm_audit import audit

MODEL = "gemma4"
NOTES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
rng = random.Random(11)

def corrupt(g, mode):
    h = json.loads(json.dumps(g))
    bars = h.get("bars", [])
    if mode == "shuffled":                       # right chords, no structure
        labs = [b.get("sounding") for b in bars]
        rng.shuffle(labs)
        for b, l in zip(bars, labs): b["sounding"] = l
    elif mode == "random":                       # no key, no structure
        pool = [f"{n}{q}" for n in NOTES for q in ("", "m")]
        for b in bars: b["sounding"] = rng.choice(pool)
        h["chords_sounding"] = sorted({b["sounding"] for b in bars})[:12]
    elif mode == "semitone_noise":               # the real failure mode
        for b in bars:
            if rng.random() < 0.4:
                lab = b.get("sounding") or "C"
                root = lab[:2] if len(lab) > 1 and lab[1] == "#" else lab[:1]
                rest = lab[len(root):]
                if root in NOTES:
                    b["sounding"] = NOTES[(NOTES.index(root) + rng.choice([-1,1])) % 12] + rest
        h["chords_sounding"] = sorted({b["sounding"] for b in bars})[:12]
    return h

songs = ["wonderwall", "letitbe", "hohey", "fastcar", "hallelujah"]
print(f"{'song':14} {'variant':16} {'plaus':>6} {'OOK':>4}  problems")
print("-"*74)
agg = {}
for sid in songs:
    p = f"work/bench/{sid}.json"
    if not os.path.isfile(p): continue
    g = json.load(open(p))
    for mode in ("original", "shuffled", "semitone_noise", "random"):
        h = g if mode == "original" else corrupt(g, mode)
        try:
            a = audit(MODEL, h)
        except Exception as e:
            print(f"{sid:14} {mode:16} error {str(e)[:30]}"); continue
        if not a:
            print(f"{sid:14} {mode:16} unparseable"); continue
        agg.setdefault(mode, []).append(a["plausibility"])
        print(f"{sid:14} {mode:16} {a['plausibility']:6d} {a['out_of_key']:4d}  "
              f"{'; '.join(a['problems'])[:28]}")
print("-"*74)
for mode in ("original","shuffled","semitone_noise","random"):
    v = agg.get(mode, [])
    if v: print(f"{mode:16} mean plausibility {sum(v)/len(v):5.2f}  (n={len(v)})")
print()
print("If 'random' scores near 'original', the auditor is blind.")
