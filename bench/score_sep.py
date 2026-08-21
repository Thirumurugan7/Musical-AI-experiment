"""
Score a guitar separation against a known guitar track.

Every separation number in this repo so far has been a leak detector -- level
against the mix, percussive fraction, spectral centroid. Useful for catching a
stem that failed outright, useless for ranking two stems that both look
plausible. Ranking needs a reference, and there is no isolated guitar for any
commercial recording.

So the reference is built rather than found: a real guitar recording is mixed
with real accompaniment at a known level, and the separator is asked to recover
the part we put in. The mixture is artificial -- no shared room, no bus
compression, no mastering -- so absolute scores will flatter every model. What
it does support is the comparison, because every model is handed the identical
signal and the identical answer.

Two metrics, because they fail differently:

  SDR      mir_eval's bss_eval_sources. The field standard, and what every
           published number in bench/SEPARATION.md is quoted in, so ours are
           comparable. Allows a distorting filter, which is generous.
  SI-SDR   scale-invariant SDR. Allows only a gain change, so it punishes the
           spectral smearing that bss_eval's filter forgives. Cheap, and it
           does not blow up on near-silent references the way bss_eval can.

A separator that wins on SDR and loses on SI-SDR is filtering the guitar into
shape rather than isolating it.
"""
import os
import sys
import glob
import json

import numpy as np
import soundfile as sf

SR = 22050


def load(path, n=None):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y[:n] if n is not None else y


def si_sdr(est, ref, eps=1e-12):
    """Scale-invariant SDR: the best gain is solved for, not assumed."""
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    ref = ref - ref.mean()
    est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    proj = a * ref
    noise = est - proj
    return 10 * np.log10((np.dot(proj, proj) + eps) / (np.dot(noise, noise) + eps))


def bss_sdr(est_guitar, ref_guitar, mix):
    """
    bss_eval SDR/SIR/SAR for the guitar, with the rest of the mix as the second
    source so the metric can account for what leaked in from where.
    """
    from mir_eval.separation import bss_eval_sources
    n = min(len(est_guitar), len(ref_guitar), len(mix))
    e, r, m = est_guitar[:n], ref_guitar[:n], mix[:n]
    refs = np.vstack([r, m - r])
    ests = np.vstack([e, m - e])
    # bss_eval is undefined if any row is silent, and one legitimate variant
    # makes that happen: a separator that returns the mix unchanged -- which is
    # what our own full-mix fallback does -- leaves `mix - est` exactly zero.
    # Report nan rather than crashing, and let SI-SDR carry those cases.
    for row in (refs[0], refs[1], ests[0], ests[1]):
        if not np.any(np.abs(row) > 1e-8):
            return float("nan"), float("nan"), float("nan")
    sdr, sir, sar, _ = bss_eval_sources(refs, ests, compute_permutation=False)
    return float(sdr[0]), float(sir[0]), float(sar[0])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: score_sep.py TESTDIR")
        print("  expects <name>.mix.wav, <name>.guitar.wav, <name>.<variant>.wav")
        return 1
    d = sys.argv[1].rstrip("/")

    cases = sorted(glob.glob(f"{d}/*.guitar.wav"))
    if not cases:
        print(f"no cases in {d} (need <name>.guitar.wav)")
        return 1

    rows = {}
    for gt in cases:
        name = os.path.basename(gt)[:-len(".guitar.wav")]
        mixp = f"{d}/{name}.mix.wav"
        if not os.path.isfile(mixp):
            print(f"  skip {name}: no mix")
            continue
        ref = load(gt)
        mix = load(mixp, n=len(ref))
        for est in sorted(glob.glob(f"{d}/{name}.*.wav")):
            variant = os.path.basename(est)[len(name) + 1:-len(".wav")]
            if variant in ("guitar", "mix"):
                continue
            e = load(est, n=len(ref))
            if len(e) < len(ref):
                e = np.pad(e, (0, len(ref) - len(e)))
            sdr, sir, sar = bss_sdr(e, ref, mix)
            rows.setdefault(variant, []).append(
                {"case": name, "sdr": sdr, "sir": sir, "sar": sar,
                 "si_sdr": si_sdr(e, ref)})

    if not rows:
        print("no estimates found")
        return 1

    print(f"{'variant':30} {'n':>3} {'SDR':>7} {'SI-SDR':>8} {'SIR':>7} {'SAR':>7}")
    print("-" * 68)
    order = sorted(rows, key=lambda v: -np.nanmean([r["si_sdr"] for r in rows[v]]))
    for v in order:
        rs = rows[v]
        print(f"{v:30} {len(rs):3d} "
              f"{np.nanmean([r['sdr'] for r in rs]):7.2f} "
              f"{np.nanmean([r['si_sdr'] for r in rs]):8.2f} "
              f"{np.nanmean([r['sir'] for r in rs]):7.2f} "
              f"{np.nanmean([r['sar'] for r in rs]):7.2f}")
    print("-" * 68)
    print("SDR/SIR/SAR in dB, higher is better. A variant that beats another on")
    print("SDR but loses on SI-SDR is shaping the guitar with a filter rather")
    print("than isolating it.")
    print(json.dumps(rows), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
