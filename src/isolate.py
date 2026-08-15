"""
Guitar-forward reduction WITHOUT pretrained weights.

This is not true source separation. A real guitar stem needs a learned model —
Demucs `htdemucs_6s` has a dedicated guitar source and is the right tool. Its
weights could not be downloaded in the environment this was written in, so
this module does what classical DSP can:

  1. mid/side — vocals in a pop mix are centred, so the side channel loses them
  2. HPSS    — drop the percussive component, which removes drums and pick clack
  3. REPET-style nn_filter — the accompaniment repeats, the vocal does not, so
     a median filter over spectrally-similar frames keeps the repeating part

What survives is "harmonic accompaniment": guitar plus piano, strings and any
other sustained instrument. On a sparse acoustic recording that is close to a
guitar stem. On a dense production it is not, and the honest description is a
reduced backing track that is easier to play along to.
"""
import sys
import numpy as np
import scipy.signal

for _w in ("hann", "hamming"):
    if not hasattr(scipy.signal, _w):
        setattr(scipy.signal, _w, getattr(scipy.signal.windows, _w))

import librosa
import soundfile as sf


def reduce_to_accompaniment(path, sr=22050, margin_v=3.0, margin_b=2.0):
    y, _ = librosa.load(path, sr=sr, mono=False)
    stereo = y.ndim > 1 and y.shape[0] == 2
    mono = librosa.to_mono(y) if stereo else np.atleast_1d(y)

    # --- 1. side channel: cancels anything panned dead centre --------------
    side = ((y[0] - y[1]) / 2.0) if stereo else None

    # --- 2. REPET-style: repeating accompaniment vs non-repeating vocal ----
    S_full, phase = librosa.magphase(librosa.stft(mono))
    S_filter = librosa.decompose.nn_filter(
        S_full, aggregate=np.median, metric="cosine",
        width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)

    mask_bg = librosa.util.softmask(S_filter, margin_b * (S_full - S_filter),
                                    power=2)
    mask_fg = librosa.util.softmask(S_full - S_filter, margin_v * S_filter,
                                    power=2)
    background = librosa.istft(mask_bg * S_full * phase, length=len(mono))
    vocal = librosa.istft(mask_fg * S_full * phase, length=len(mono))

    # --- 3. drop percussion ------------------------------------------------
    harm, perc = librosa.effects.hpss(background, margin=3.0)

    def norm(x):
        p = np.percentile(np.abs(x), 99.9)
        return (x / p * 0.89).astype(np.float32) if p > 0 else x.astype(np.float32)

    return {
        "sr": sr,
        "accompaniment": norm(harm),        # the play-along track
        "background": norm(background),     # vocals removed, drums kept
        "vocal": norm(vocal),               # what was taken out, for checking
        "side": norm(side) if side is not None else None,
        "percussive": norm(perc),
    }


if __name__ == "__main__":
    src = sys.argv[1]
    stem = sys.argv[2] if len(sys.argv) > 2 else "out"
    r = reduce_to_accompaniment(src)
    for name in ("accompaniment", "background", "vocal", "side"):
        if r[name] is not None:
            p = f"{stem}_{name}.wav"
            sf.write(p, r[name], r["sr"])
            print(f"  wrote {p}  ({len(r[name])/r['sr']:.1f}s)")
