# Track separation: what is actually available, and what it is worth

## The finding that decides everything else

Across the two largest open catalogues — `audio-separator`'s registry (166
models) and ZFTurbo's `Music-Source-Separation-Training` — **exactly one open
model emits a guitar stem: `htdemucs_6s`.** The one we already use.

No RoFormer, no SCNet, no MDX23C variant produces guitar. They are two-stem
(vocals / instrumental) or four-stem (vocals / drums / bass / other). Nobody
publishes a guitar SDR for `htdemucs_6s` either, its authors included; the
column is literally `guitar: ---`. The ensemble survey (arXiv:2410.20773) names
the gap outright: further work is needed on "niche stem separations such as
guitar and piano".

So the guitar stem cannot be upgraded by swapping models. There is nothing to
swap to.

`htdemucs_6s` also paid for its two extra stems with quality everywhere else:

| model | vocals | drums | bass | guitar |
|---|---|---|---|---|
| htdemucs_ft | 10.8 | 10.0 | 12.0 | — |
| htdemucs_6s | 9.6 | 8.5 | 10.1 | unmeasured |

For contrast, the best RoFormer instrumental models score 16.1–16.5 SDR. The
modern architectures are dramatically better — at a task that does not give us
a guitar.

## How reliable is the guitar head?

Level of each stem relative to the mix, nine songs:

| song | guitar | other | verdict |
|---|---|---|---|
| hallelujah | −5.1 | −35.0 | guitar dominant |
| hohey | −6.4 | −19.7 | guitar dominant |
| wagonwheel | −6.9 | −8.6 | guitar dominant |
| wonderwall | −7.5 | −21.5 | guitar dominant |
| canthelp | −9.4 | −19.3 | guitar dominant |
| fastcar | −9.4 | −37.9 | guitar dominant |
| letitbe | −9.6 | −10.5 | guitar dominant |
| imyours | −12.1 | −15.5 | guitar dominant |
| **riptide** | **−31.5** | −14.5 | **failed** |

Eight of nine are fine. This corrects an impression formed from a sample of
two: the guitar head is not generally unstable.

Riptide is the exception and it is a bad one — 31.5 dB below the mix, past the
−25 dB bar `transcribe.py` uses to declare separation failed, in a song that is
almost nothing but strummed guitar. The guitar was filed under `other`, 17 dB
louder. Riptide being this project's running example is how the wrong
impression formed in the first place.

Fixed: when the guitar stem fails the −25 dB test, try the sibling `other`
before falling back to the full mix. See `src/transcribe.py`.

## Two-stage separation: tried, no benefit found

Premise: `htdemucs_6s` cannot be replaced, but it can be given a cleaner input.
Strip the vocals first with a RoFormer (instrumental SDR ~16, far past demucs),
then run `htdemucs_6s` on the residue.

Measured on a 30 s excerpt of Wonderwall, vocal content in the resulting guitar
stem, against the RoFormer's own vocal output as reference:

| guitar stem | env_corr | spec_proj | rms dB |
|---|---|---|---|
| direct `htdemucs_6s` | −0.400 | 0.083 | −31.7 |
| two-stage | −0.228 | 0.126 | −31.5 |
| *(demucs vocals stem, as a control)* | *0.930* | *0.846* | *−23.2* |

The control confirms the metric works: two independently trained models agree
0.930 on what the vocal is. But both guitar stems already sit near 0.1 — there
was essentially no vocal in the guitar stem for stage 1 to remove. The premise
does not hold, at least here, and two-stage costs an extra ~4x-realtime pass.

Caveat: one excerpt of one song, and the reference vocal comes from the same
model used in stage 1, so residual correlation reflects spectral overlap rather
than true leakage. This is suggestive, not settled.

## Measured, at last

20 mixtures: 10 GuitarSet comping excerpts across all five styles and five
players, each at 0 dB and -6 dB guitar-to-accompaniment, with drums, bass and
vocals from real recordings.

| variant | n | SDR | SI-SDR | SIR | SAR |
|---|---|---|---|---|---|
| htdemucs_6s `guitar` | 20 | **13.36** | **12.83** | 25.97 | 13.77 |
| the mix, untouched | 20 | — | -2.99 | — | — |
| htdemucs_6s `other` | 20 | -22.01 | -30.06 | -0.83 | -15.50 |

The guitar head is worth **+15.8 dB of SI-SDR over doing nothing**, and SDR
sits within half a decibel of SI-SDR with SIR near 26, so it is genuinely
isolating the guitar rather than filtering the mix into its shape. That is the
first number this project has ever had for the stem it depends on.

`other` is 27 dB worse than doing nothing. That is expected -- when the guitar
head works, `other` is by construction everything the guitar is not -- but it
is worth stating plainly, because we ship a fallback that reaches for it.

## The fallback shipped in 43ce40d is still untested

It fires only when the guitar stem comes back below -25 dB. Across all 20
mixtures the guitar stem ranged from -2.9 dB to -8.6 dB. It never fired:

  guitar head fell below -25 dB on 0/20 cases

So this benchmark says nothing about whether `other` beats the full mix in the
one situation the fallback exists for. The table above is not evidence for it,
and reading it as such would be backwards -- it measures `other` precisely when
the guitar head is healthy, which is when the fallback does not run.

What the change rests on remains a level argument: on Riptide the guitar stem
is -31.5 dB and `other` is -14.5 dB, and a song that is almost entirely
strummed guitar must have that guitar somewhere. Reasonable, unmeasured.

That test was then run, by burying the guitar 18 dB under the accompaniment.
It did not work, and failing to force the failure taught us more than forcing
it would have:

| guitar level in mix | guitar stem SDR | SI-SDR | mix control SI-SDR |
|---|---|---|---|
| 0 dB | 14.51 | 14.01 | 0.01 |
| -6 dB | 12.20 | 11.64 | -5.98 |
| -18 dB | 6.22 | 5.71 | -17.96 |

At -18 dB the guitar stem comes back at about -19 dB -- still above the -25 dB
bar -- and separation is *working*, 23.7 dB better than the untouched mix. The
head does not go silent when the guitar is quiet; it returns a proportionally
quiet guitar, correctly separated. Across all 30 mixtures the guitar head never
once fell below -25 dB.

So the -25 dB check does not detect what it claims to. It conflates "separation
failed" with "the guitar is low in this mix", and those need opposite responses:
the first wants a different stem, the second wants the stem it already has.
Riptide's failure is a misclassification, and misclassification cannot be
manufactured by turning the guitar down.

What actually distinguishes Riptide is the *gap*: its `other` stem is seventeen
decibels louder than its guitar stem. The guitar went somewhere, and that is
where it went. So the fallback now requires that gap -- `other` at least 8 dB
above the guitar stem -- rather than firing on a quiet stem alone. Riptide
still triggers it; a quiet but correctly separated guitar no longer can. Given
`other` measures -30.1 dB SI-SDR against -3.0 for doing nothing, a false alarm
costs far more than the fallback can win, so the guard is deliberately strict.

The firing case is still unvalidated. Nothing here shows `other` beats the full
mix on a genuinely misclassified song, because we have no such song with ground
truth. It remains a judgement call, now a narrower one.

## Why the leak detectors are still not a ranking

There is no isolated-guitar reference for any real song in this repo, and
`bench/songs.json` carries no strum annotations. So SDR cannot be computed and
strum accuracy cannot be scored on real audio. `bench/sep_diag.py` therefore
reports leak detectors, not quality scores:

- `rel_db` — level against the mix; catches the Riptide failure
- `perc_frac` — energy in the percussive HPSS component; drum bleed is the
  failure that matters most, since a leaked snare is indistinguishable from a
  strum to the grid
- `onset_pk` — peak-to-median of the onset envelope; clean guitar has sharp
  isolated attacks
- `centroid`, `hf_frac` — is this stem even in the right frequency range

To actually rank separators for guitar we need labelled guitar stems.
**MoisesDB** (240 tracks, 45 artists, guitar as a labelled category, built for
exactly the beyond-4-stems case) is the dataset for it:
zenodo.org/records/10265363.

## Practical notes for running this on the Mac

- `audio-separator` (`nomadkaraoke/python-audio-separator`) is the right tool
  for trying many models: one CLI, 166 pretrained models, MPS + CoreML both
  detected automatically on Apple Silicon.
- Its `[cpu]` extra does not install `audioread`; add it manually.
- It calls `librosa.get_duration(filename=...)`, removed in librosa 1.0. One
  call site in `separator/common_separator.py`; patched in `work/.sep-venv`
  (original kept as `.orig`). Without it every run completes inference and then
  dies at the write step.
- `melband_roformer_inst_v2.ckpt` is 1.57 GB and runs at roughly 4x realtime on
  an M-series Air with 16 GB. A whole-song pass held the process in
  uninterruptible I/O wait; chunking or excerpts are the safer route.
- Model weights live in `work/.sep-models`, not `/tmp`, so they survive.
