# Our pipeline vs. human transcriptions — Ed Sheeran, "Perfect"

## The scoreboard

| Source | Key | Tempo | Metre | Chords | Strumming |
|---|---|---|---|---|---|
| **Our pipeline** | Ab | **63.2** BPM dotted-qtr | **12/8** | Ab, Fm, Fm7, Db, Eb, Eb7 | 12 attacks/bar, accents on 1·4·7·10 — **direction wrong** |
| Hooktheory | Ab major | 190 BPM (eighths) = **63.3** dotted-qtr | 12/8 verse, 6/8 chorus | I–vi–IV–I–V | — |
| Ultimate Guitar | Ab (capo 1, play G) | — | — | G Em C D @ capo 1 | not given |
| Fender | capo 1 | — | **12/8** | G Em7 Cadd9 D | all downstrokes, eighths, **accent 1·4·7·10** |
| Lauren Bateman | — | — | 6/8 | G Em C D | `DDD D D D` — "all downs" |
| PickUpTheGuitar | capo 1 | — | triplets | G Em C D | `DDD DDD DDD DDD`, first of each triplet harder |
| Chordify | Ab | 112 BPM | not stated | **Fm7**, Db, Ab, Eb | — |
| SongBPM | **G major** | **97** BPM | **3 beats/bar** | — | — |

Capo 1 with G-shapes sounds Ab – Fm – Db – Eb, so every guitar source above is
describing the same progression our model returned.

## Where we matched

**Key — exact.** Ab, agreeing with Ultimate Guitar, Hooktheory and Chordify.
We spell the IV as C# rather than Db; enharmonically identical, cosmetically
wrong for a flat key, fixed with a lookup table.

**Tempo — exact.** Hooktheory says 190 BPM counted in eighths. 190 / 3 = 63.33
dotted-quarter. We measured 63.2. That is 0.2 % error against a
human-curated source.

**Metre — exact.** We reported 12/8. Fender says 12/8. Hooktheory says 12/8
verse, 6/8 chorus (the same feel, barred differently).

**Chords — exact, including the colouring.** Everyone gives Ab–Fm–Db–Eb. We
returned that plus Fm7, and Chordify independently lists **Fm7** rather than
plain Fm — matching a detail we found rather than contradicting it. We also
caught an Eb7 secondary dominant and two first-inversion slash chords that none
of the guitar sources bother to notate.

**Accent structure — confirmed, and this one is worth dwelling on.** Fender
instructs players to accent strokes 1, 4, 7 and 10. That is a falsifiable
prediction about the recording, so we tested it against our onset strengths
across all 66 bars:

```
  slot   mean onset strength
    1    4.201  ##################################################   <- accent
    2    2.048  ########################
    3    2.290  ###########################
    4    6.179  ##########################################################################   <- accent
    5    2.054  ########################
    6    3.386  ########################################
    7    4.741  ########################################################   <- accent
    8    1.865  ######################
    9    2.971  ###################################
   10    7.041  ####################################################################################   <- accent
   11    2.523  ##############################
   12    3.763  #############################################

  accented slots 5.541  vs  others 2.613     ratio 2.12x
  triplet groups where the first eighth is strongest: 4/4
```

Every predicted accent position is a local maximum. The accents are 2.12x
stronger than the surrounding eighths. Fender's instruction and our signal
analysis describe the same recording.

## Where we were wrong

**Stroke direction: 0 % correct.** Every human source says the same thing —
"all downs." Fender: downstrokes throughout. Lauren Bateman: "Strumming is
going to be kept super simple on this. All downs." PickUpTheGuitar:
`DDD DDD DDD DDD`.

Our metric prior produced alternating D-U-D-U and got the direction wrong on
every single stroke.

This kills the prior as a general method. The assumption — on-beat subdivisions
are struck downward, off-beat ones upward — is a *duple* strumming heuristic. In
compound metre with three eighths per beat, continuous alternation would put the
hand in the wrong place every other beat, so players simply don't do it; they
either play all downstrokes or reset each triplet. Metric position cannot
determine direction. It has to come from acoustic evidence or from a
metre-conditioned model of hand motion.

**We under-read our own output.** I earlier called "10 of 12 slots filled per
bar" a failure — the mix being uniformly dense. Every human source says the
guitar plays **12 eighth-note strums per bar**. The density was approximately
right and I dismissed it. The honest position is that dense mix and dense guitar
part are indistinguishable without stem separation, so the density agreement
does not prove the detector works — but it was wrong to call it a failure.

## Where the commercial metadata is wrong

**SongBPM: 97 BPM, key G major, 3 beats per bar.** All three wrong. The key is
Ab — G is what the *capo-1 shapes* are called, so it looks like shape names
leaked into a key field. And 97 BPM is the compound metre grouped into pairs of
eighths.

That last error is the interesting one: **librosa made the identical mistake**,
reporting 95.7 BPM. Two independent systems, same duple bias, same wrong answer.
It is a systematic failure mode of beat trackers on 6/8 and 12/8, not a quirk of
our setup — and swapping in madmom's DBN with `beats_per_bar=[4]` fixed it.

**Chordify: 112 BPM**, which corresponds to nothing in this song — not 63.3,
not 95, not 190. Chordify got the chords right (including Fm7) and the key
right; its tempo figure is unexplained.

## What this establishes

On the parts we can currently do, the pipeline is **at or above commercial
quality**: it matched a human-curated source (Hooktheory) exactly on key, tempo
and metre, matched every guitar source on chords, and beat two commercial
services (SongBPM, Chordify) on tempo and metre.

The gap is entirely in rhythm output, and the gap is now precisely characterised
rather than vague:

1. Direction cannot come from metric position — proven wrong on this song.
2. Onset detection needs an isolated guitar stem before its density means
   anything.
3. Accent detection, by contrast, **already works** — 2.12x contrast at exactly
   the predicted positions, straight off the full mix, with no separation.

That third point was not expected and is the most useful thing to come out of
this comparison. Accent structure is recoverable from a dense mix. If accents
are recoverable and direction is not, then the honest deliverable for a song
like this is a **rhythm chart with accents** rather than a down/up pattern —
which is exactly how the human sources notate it anyway.

## Sources

- Ultimate Guitar — https://tabs.ultimate-guitar.com/tab/ed-sheeran/perfect-chords-1956589
- Hooktheory — https://www.hooktheory.com/theorytab/view/ed-sheeran/perfect
- Fender — https://www.fender.com/articles/songs/learn-perfect-ed-sheeran-guitar
- Lauren Bateman — https://www.laurenbateman.com/perfect-chord-chart/
- PickUpTheGuitar — https://pickuptheguitar.com/how-to-play-perfect-by-ed-sheeran/
- Chordify — https://chordify.net/chords/ed-sheeran-songs/perfect-chords
- SongBPM — https://songbpm.com/@ed-sheeran/perfect
