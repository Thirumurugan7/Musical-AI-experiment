"""
Build guitar-separation test cases with exact ground truth.

There is no isolated guitar for any commercial recording, so the reference is
constructed instead of found:

  guitar        a real GuitarSet comping excerpt. Recorded with a reference
                microphone, six players, five styles. Real playing, not
                synthesis -- which matters, because a separator trained on real
                music has no reason to recognise a synthesised pluck as guitar.
  accompaniment drums + bass + vocals lifted from a real recording by
                htdemucs_6s.
  mix           the two summed at a known ratio.

The separator is then asked to recover exactly what we put in.

Two honest limitations, neither of which breaks the comparison:

  The mixture is artificial. No shared room, no bleed between microphones, no
  bus compression, no mastering. Absolute scores will flatter every model.

  Guitar and accompaniment come from different songs, so they do not agree on
  key or tempo. Separation models are spectro-temporal and do not track
  harmony, so this costs less than it appears to, but it is not a real mix.

What the construction does support is ranking, because every model is handed
the identical signal and graded against the identical answer.

Accompaniment deliberately excludes `other`, `piano` and `guitar` from the
source song: those stems can contain guitar, and any guitar in the
accompaniment would be scored as an error the separator did not make.
"""
import os
import sys
import glob
import json
import random

import numpy as np
import soundfile as sf
import librosa

SR = 44100
OUT = "work/septests"
GS_AUDIO = "work/guitarset/audio_mono-mic"
GS_ANNO = "work/guitarset/annotation"
ACCOMP_ROOT = "work/sep/demucs6/htdemucs_6s"
ACCOMP_STEMS = ("drums", "bass", "vocals")

# guitar level relative to the accompaniment, in dB. -6 is a guitar sitting
# under a band; 0 is a guitar-led arrangement.
RATIOS = (0.0, -6.0)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) or 1e-9


def load_accomp(song_dir, n):
    """Sum the non-guitar stems of one song to n samples."""
    acc = np.zeros(n)
    used = []
    for s in ACCOMP_STEMS:
        for ext in (".mp3", ".wav"):
            p = os.path.join(song_dir, s + ext)
            if os.path.isfile(p):
                y, _ = librosa.load(p, sr=SR, mono=True)
                if y.size < n:                    # loop short stems
                    y = np.tile(y, int(np.ceil(n / max(y.size, 1))))
                acc += y[:n]
                used.append(s)
                break
    return acc, used


def main():
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    os.makedirs(OUT, exist_ok=True)

    comps = sorted(glob.glob(f"{GS_AUDIO}/*_comp_mic.wav"))
    if not comps:
        print(f"no GuitarSet comping audio in {GS_AUDIO}")
        return 1
    songs = sorted(d for d in glob.glob(f"{ACCOMP_ROOT}/*") if os.path.isdir(d))
    if not songs:
        print(f"no accompaniment stems in {ACCOMP_ROOT}")
        return 1

    # spread across players and styles rather than taking the first N, which
    # would be six takes by player 00 of the same three progressions
    rng = random.Random(0)
    by_style = {}
    for c in comps:
        # strip the take number: "Rock1" and "Rock3" are the same style, and
        # keeping them apart made alphabetical spreading stop at Jazz, so the
        # two styles that are actually strummed -- Rock and Singer-Songwriter
        # -- never made it into the set at all.
        tok = os.path.basename(c).split("_")[1].split("-")[0]
        style = "".join(ch for ch in tok if not ch.isdigit())
        by_style.setdefault(style, []).append(c)
    picked = []
    while len(picked) < n_cases and any(by_style.values()):
        for st in sorted(by_style):
            if by_style[st] and len(picked) < n_cases:
                picked.append(by_style[st].pop(rng.randrange(len(by_style[st]))))

    manifest = []
    for i, gpath in enumerate(picked):
        name = os.path.basename(gpath)[:-len("_mic.wav")]
        g, _ = librosa.load(gpath, sr=SR, mono=True)
        if g.size < SR * 5:
            continue
        song = songs[i % len(songs)]
        acc, used = load_accomp(song, g.size)
        if not used:
            continue

        acc = acc / rms(acc)
        gn = g / rms(g)
        for ratio in RATIOS:
            tag = f"{name}_{int(ratio):+d}dB".replace("+0", "0")
            gg = gn * (10 ** (ratio / 20.0))
            mix = gg + acc
            peak = np.max(np.abs(mix)) or 1.0
            scale = 0.89 / peak                    # leave headroom, no clipping
            sf.write(f"{OUT}/{tag}.guitar.wav", gg * scale, SR)
            sf.write(f"{OUT}/{tag}.mix.wav", mix * scale, SR)
            manifest.append({
                "case": tag, "guitarset": os.path.basename(gpath),
                "accompaniment_song": os.path.basename(song),
                "accompaniment_stems": used, "guitar_rel_db": ratio,
                "dur_s": round(g.size / SR, 2),
            })

    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {len(manifest)} cases to {OUT}/")
    styles = sorted({m['guitarset'].split('_')[1].split('-')[0] for m in manifest})
    print(f"  styles:  {', '.join(styles)}")
    print(f"  players: {sorted({m['guitarset'][:2] for m in manifest})}")
    print(f"  ratios:  {RATIOS} dB guitar-to-accompaniment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
