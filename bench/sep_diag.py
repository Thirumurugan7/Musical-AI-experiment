"""
Diagnostics for a separated stem, without ground truth.

We have no isolated-guitar reference for any real song, so SDR is not
available. These are the properties we can measure directly, chosen because
each one corresponds to a way separation is known to fail here:

  rel_db      level against the mix. htdemucs_6s returns a near-empty stem when
              it does not recognise the instrument, and the residue still has
              enough structure to fool a self-normalised threshold. transcribe.py
              already refuses a stem below -25 dB; this reports the same number.
  perc_frac   fraction of energy in the percussive component after HPSS. Drum
              bleed is the failure that matters most for us: the strum grid
              reads onsets from this stem, and a leaked snare is indistinguishable
              from a strum.
  onset_pk    peak-to-median ratio of the onset envelope. A clean guitar track
              has sharp, isolated attacks; a smeared or bleeding one does not.
  centroid    spectral centroid in Hz. Guitar sits well below cymbals and well
              above bass, so a centroid far outside ~200-3000 Hz suggests the
              stem is mostly something else.
  hf_frac     fraction of energy above 6 kHz, where cymbals live and guitar
              largely does not.

None of these is a quality score. They are leak detectors.
"""
import sys, json
import numpy as np
import librosa

SR = 22050


def diag(path, mix_rms=None):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if y.size == 0:
        return {"error": "empty"}
    rms = float(np.sqrt(np.mean(y ** 2)))
    out = {"rms": rms, "dur_s": round(len(y) / SR, 1)}
    if mix_rms:
        out["rel_db"] = round(20 * np.log10(max(rms, 1e-12) / mix_rms), 1)

    h, p = librosa.effects.hpss(y)
    he, pe = float(np.sum(h ** 2)), float(np.sum(p ** 2))
    out["perc_frac"] = round(pe / max(he + pe, 1e-12), 3)

    env = librosa.onset.onset_strength(y=y, sr=SR)
    med = float(np.median(env))
    out["onset_pk"] = round(float(np.max(env)) / max(med, 1e-9), 1)

    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=SR)
    p_spec = S.sum(axis=1)
    out["centroid"] = round(float((freqs * p_spec).sum() / max(p_spec.sum(), 1e-12)))
    out["hf_frac"] = round(float(p_spec[freqs > 6000].sum() / max(p_spec.sum(), 1e-12)), 3)
    return out


def main():
    if len(sys.argv) < 3:
        print("usage: sep_diag.py MIX.wav STEM.wav [STEM2.wav ...]")
        return 1
    mix = sys.argv[1]
    ym, _ = librosa.load(mix, sr=SR, mono=True)
    mix_rms = float(np.sqrt(np.mean(ym ** 2))) or 1e-9

    rows = {}
    for p in sys.argv[2:]:
        rows[p] = diag(p, mix_rms)

    hdr = ("stem", "rel_db", "perc_frac", "onset_pk", "centroid", "hf_frac")
    print(f"{hdr[0]:34} {hdr[1]:>7} {hdr[2]:>10} {hdr[3]:>9} {hdr[4]:>9} {hdr[5]:>8}")
    print("-" * 82)
    for p, d in rows.items():
        if "error" in d:
            print(f"{p.split('/')[-1][:34]:34}  {d['error']}")
            continue
        print(f"{p.split('/')[-1][:34]:34} {d.get('rel_db', 0):7.1f} "
              f"{d['perc_frac']:10.3f} {d['onset_pk']:9.1f} "
              f"{d['centroid']:9d} {d['hf_frac']:8.3f}")
    print(json.dumps(rows), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
