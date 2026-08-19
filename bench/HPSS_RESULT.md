# HPSS before chord recognition — tested, rejected

Idea, taken from Guitariz: percussion is broadband, so every drum hit smears
energy across all twelve pitch classes in the chroma, worst exactly at chord
changes. Split it off with librosa HPSS and feed the chord model only the
harmonic component.

Tested on a 12-song subset, `margin=3.0`, harmonic component to the chord
model, original audio still used for beats.

```
                  base    HPSS
mean chord F1     0.903   0.811    -0.092
key correct       12/12   11/12

riptide           1.00 ->  0.00    key Db -> C   (broke completely)
hohey             0.86 ->  0.75
10 others         unchanged
```

## Why it fails

JointChordNet is trained on full-mix CQT. Harmonic-only audio is a
train/test distribution shift, and the model has evidently learned to use
whatever the percussive component contributes. The reasoning behind the idea
is sound for a *template matcher* — Guitariz uses HPSS and cosine-matches
against hand-weighted templates, where a cleaner chroma is unambiguously
better. It does not transfer to a trained classifier.

Would only be worth revisiting alongside finetuning the model on HPSS'd
audio, which needs a labelled corpus we do not have.

## Reproduce

    bash bench/hpss_ab.sh          # HPSS_MARGIN=3.0 SONGS="..." to vary
