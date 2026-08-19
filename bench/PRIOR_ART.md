# What the published work says about strum detection

Written after searching for the thing I had repeatedly claimed did not exist:
ground truth for guitar strumming. It exists. Two sources matter.

## 1. Murgul, Schimper & Heizmann, ISMIR 2025

"Joint Transcription of Acoustic Guitar Strumming Directions and Chords",
arXiv:2508.07973, Klangio GmbH + Karlsruhe Institute of Technology.
This is our exact problem — strum onsets, up/down direction, and chords, from
one audio signal — done properly and measured.

### Their setup

Ground truth comes from an ESP32 smartwatch accelerometer strapped to the
strumming hand. The sign of the y-acceleration derivative at an onset gives the
stroke direction: negative is an up-stroke, positive a down-stroke. That is the
trick we could not find — direction is not recoverable from a recording alone
without a model, but it *is* recoverable from the playing hand, and once you
have it labelled you can train a model to hear it.

- 90 min of real playing, 3 guitarists, 28 patterns in 4/4, 60/80/100 BPM,
  finger and pick, three volumes. Recorded simultaneously through a pickup and
  an iPhone 15 Pro microphone.
- ~4 h of synthetic audio: 51 chord progressions x 36 strumming patterns on a
  16th grid, exported as GuitarPro and rendered through DAWDreamer + Ample Sound
  virtual instruments, then augmented with Pedalboard (distortion, filtering,
  compression, convolutional reverb, ambient noise, fret and tap noise). The
  last note of a strum is dropped 50% of the time on purpose, to imitate how
  amateurs actually play.
- CRNN: 16 kHz log-mel (2048 window, 160 hop, 229 bins from 30 Hz), 10 s clips
  at 1 s hop, conv stack + biGRU(256), two heads. Trained 20k steps, ~2 h on one
  V100.
- **Evaluated at 50 ms with mir_eval.**

### Their numbers

Strum detection on clean solo pickup audio (Table 2):

| Method | F1 | P | R |
|---|---|---|---|
| Spectral flux (librosa) | 79.49% | 78.53% | 81.86% |
| Super flux | 74.36% | 77.04% | 73.36% |
| CD-ODF | 79.32% | 68.50% | 98.15% |
| Their CRNN | **97.60%** | 96.54% | 98.73% |

On real microphone audio, by what they trained on (Table 4):

| Training data | F1 any | F1 down | F1 up |
|---|---|---|---|
| Synthetic only | 89.77% | 73.92% | 52.64% |
| Real phone only | 85.06% | 79.90% | 66.81% |
| Real phone + pickup | 89.45% | 82.94% | 75.10% |
| All three | **92.75%** | **85.51%** | **79.02%** |

Chords, 24 major/minor classes, assigned at detected strum times (Table 6):
Deep Chroma 80.37%, BTC transformer 89.21%, theirs 90.06%.

### The three things this changes for us

**Our method is at its ceiling, and the ceiling is not where I was looking.**
Spectral flux with a threshold — which is what `src/rhythm.py` is — tops out at
79.5% F1 on *clean, solo, close-miked pickup audio*. Every remaining point from
there to 97.6% came from replacing the threshold with a trained model, not from
tuning the threshold. I have spent this session tuning the threshold. Further
tuning is not going to move it much.

**We are at a badly chosen operating point.** At 50 ms our synthetic set gives
precision 1.000, recall 0.681. Their spectral-flux baseline sits at 0.785/0.819.
We are not worse at the task so much as tuned to almost never claim a stroke we
are unsure of — we miss 148 real strokes and invent zero. That was a deliberate
choice (the `flat_range` guard, the absolute noise floor) made to stop the
detector inventing rests, and it overshot. There is a third of the strokes
sitting behind a conservative threshold.

**Timing is not the problem; detection is.** Sweeping our tolerance:

| tol | 20 ms | 30 ms | 40 ms | 50 ms | 60 ms | 80 ms | 100 ms |
|---|---|---|---|---|---|---|---|
| F1 | 0.154 | 0.585 | 0.782 | 0.810 | 0.810 | 0.810 | 0.810 |

Flat from 50 ms up. The strokes we miss are not near-misses that a wider window
would catch — they are not detected at all. Scoring was moved from 60 ms to
50 ms on this evidence: it costs nothing and makes our number comparable.

**Synthetic training transfers for onsets but not for direction.** Their
synthetic-only model gets 89.77% on real audio for onsets and 52.64% for up
strums — barely above chance. So a synthetic benchmark is legitimate evidence
about *when* a stroke happened and near-worthless about *which way the hand
moved*. Our synthetic suite is the right tool for the job we use it for, and
would be the wrong tool if we ever claim direction.

## 2. GuitarSet (Xi, Bittner et al., ISMIR 2018)

360 excerpts of ~30 s, 6 players, 5 styles (rock, singer-songwriter, bossa
nova, jazz, funk), 3 progressions, 2 tempi. Half of every player's set is
*comping* — chordal accompaniment — so roughly 90 minutes of real strumming,
the same scale as the paper's real corpus, already public under CC BY 4.0.

The point is the recording method: a **hexaphonic pickup**, one isolated signal
per string. Annotations are 6 pitch contours + 6 MIDI note tracks (one per
string), beat positions, tempo, and two chord annotations (instructed and
performed). The per-string note onsets were derived automatically from isolated
monophonic audio, which is the one case where onset detection is genuinely
reliable.

That gives us real strum ground truth by construction, without a smartwatch:

- a **strum** is a cluster of note onsets across strings within a few tens of ms
- its **time** is the cluster's first onset
- its **direction** is the order the strings fire in — low-to-high is a down
  stroke, high-to-low an up stroke

Direction is the speculative half; stroke times are not speculative at all.

Downloads (zenodo.org/records/3371780): `annotation.zip` 39.1 MB,
`audio_mono-mic.zip` 656.9 MB, `audio_hex-pickup_debleeded.zip` 3.6 GB.
For scoring our pipeline we need the annotations plus the mono microphone
audio — under 700 MB — because the mic mix is the input our pipeline is
actually built for.

## Why this matters more than another threshold

Every strum number in this repo is measured against seven synthetic cases I
wrote myself. That is circular in a way I have not been able to escape: the
synthesiser and the detector share my assumptions about what a strum looks
like. GuitarSet breaks the circle with 90 minutes of six real players, none of
whom knew about our grid.

## What we tested as a result

### Tolerance moved 60 ms -> 50 ms (kept)

No change to any score. See the sweep above.

### Snapping stroke times to measured onsets (rejected)

Our output has no per-stroke times in it. Every stroke time is reconstructed
downstream as `bar.start + k * (bar_len / n_slots)` — a perfectly uniform grid.
Nothing in the chart records when a stroke was actually heard. Combined with a
median matched-stroke offset of +18 to +30 ms on six of seven cases, that looked
like a clear defect: we were reporting where the grid says the stroke should be,
roughly 20 ms before where the sound actually is.

A constant -20 ms shift confirms the bias is real and systematic —
F1@20ms goes 0.154 -> 0.759 — but that number is not usable. The shift is
calibrated on the sweep width of our own synthesiser (`strum()` spreads six
strings over 18 ms each, so the energy of a down-stroke centres ~45 ms after
its nominal time). Hard-coding it would fit our detector to our generator,
which is exactly the circularity this file exists to point out.

The principled version is to snap each struck slot to the nearest measured
onset. It was tried at four window widths and both backtrack settings:

| variant | F1@20 | F1@30 | F1@50 |
|---|---|---|---|
| grid only (current) | 0.154 | 0.585 | **0.810** |
| snap ±30 ms, backtrack=False | 0.259 | 0.582 | 0.764 |
| snap ±40 ms, backtrack=False | 0.256 | 0.538 | 0.713 |
| snap ±30 ms, backtrack=True | 0.233 | 0.423 | 0.705 |
| snap ±40 ms, backtrack=True | 0.269 | 0.377 | 0.628 |

Every variant is worse at the tolerance we score at. The uniform grid places
strokes more accurately than librosa's onset times do — which is consistent
with the paper: plain spectral-flux onset detection is a 79% method even on
clean solo pickup audio, and our grid is beating it because the grid carries
information about the whole bar that a single peak does not.

So the +20 ms bias stands, unfixed and now documented. It costs nothing at
50 ms and it is not correctable by any means we currently have.

### Where the missing 32% actually is

Recall is 0.681 with precision 1.000, and the tolerance sweep is flat above
50 ms, so the missed strokes are not near-misses. Per case:

- `simple_16ths` — subdivision detected as 2 against a truth of 4. Half the
  strokes are not merely missed, they are *unrepresentable*: there is no slot
  for them. Recall 0.44. This is the whole of that case's failure.
- `six_eight` — subdivision correct, recall 0.48. Strokes are being classified
  as rests.

Both are detection failures, not timing failures. Neither is helped by anything
in the timing work above, and neither is helped by tuning the tolerance.
