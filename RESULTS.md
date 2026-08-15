# Test on a real song — Ed Sheeran, "Perfect" (4:23, 320 kbps)

## Chord recognition: works

Chord-CNN-LSTM, 5-fold ensemble, CPU only, **31 seconds** for 263 s of audio
(~8.5x realtime — model loading is most of the fixed cost).

Time share across the whole song:

| label | time | share |
|---|---|---|
| Ab:maj  | 71.9 s | 27.3 % |
| C#:maj (=Db) | 63.9 s | 24.3 % |
| Eb:maj  | 59.4 s | 22.5 % |
| F:min   | 30.8 s | 11.7 % |
| F:min7  | 27.9 s | 10.6 % |
| Eb:7    |  2.8 s |  1.1 % |
| inversions + N | 6.8 s | 2.6 % |

The song is I–vi–IV–V in Ab major: **Ab – Fm – Db – Eb**. The model spends
**96.4 %** of the song inside exactly that set, produces no spurious chords,
and additionally picks up the Fm7 colouring, the Eb7 secondary dominant, and
two first-inversion slash chords. It spells Db as C# — enharmonically correct,
cosmetically wrong for the key, fixed with a lookup table.

Verse (bars 1–16) came out as:

```
| Ab | Fm  | Db | Eb |
| Ab | Fm7 | Db | Eb |
| Ab | Fm7 | Db | Eb |
| Ab | Fm7 | Db | Eb |
```

That is the correct chart.

## Metre: librosa was wrong, madmom is right

"Perfect" is in 12/8 at ~63 BPM. librosa's `beat_track` reported **95.7 BPM in
4/4** — it grouped compound eighths into pairs, a classic duple bias.

madmom's `DBNDownBeatTrackingProcessor` with `beats_per_bar=[4]`:
**63.2 BPM, bar = 3.79 s, 12/8.** Correct.

Independent confirmation: the chord segments Chord-CNN-LSTM produced are
3.85 / 3.74 / 3.78 / 3.81 s long — one chord per bar, matching madmom's 3.79 s
bar to within 2 %. Two unrelated models agreeing on the bar length is a strong
signal both are right.

Onset alignment improved from a median grid error of 69.7 ms (librosa) to
**39.3 ms** (madmom), with 837 of 843 onsets landing on the grid.

Building madmom needs the git master; PyPI's 0.16.1 fails to compile on
Python 3.10 (Cython-generated C calls removed CPython APIs):

```bash
uv pip install --python .venv/bin/python cython'<3'
uv pip install --python .venv/bin/python "madmom @ git+https://github.com/CPJKU/madmom" --no-build-isolation
```

## Strum detection: fails, and the reason is clear

Mean **10.0 of 12** eighth-note slots per bar register an attack; 42 of 65 bars
are ≥10/12 filled. That is not a rhythm pattern, it is "the mix has energy
everywhere." Onset detection on a full commercial production picks up drums,
piano, strings and vocal consonants indiscriminately. Nothing about the
guitar's rhythm survives.

The direction output is equally void: 328 D vs 323 U is the metric prior
alternating, not a measurement of anything acoustic.

**Prerequisite is source separation.** Isolate the guitar stem, then detect
onsets on that alone. This is not a hard problem — Demucs does it well — but it
could not be tested here: the sandbox reaches PyPI and `git clone` only.
Blocked: `dl.fbaipublicfiles.com` (Demucs weights), `huggingface.co`,
GitHub release assets, `zenodo.org`. Demucs installs fine; its weights refuse
to download. On a normal network this is a non-issue.

Secondary point specific to this song: "Perfect" is fingerpicked arpeggios, not
strummed. Down/up is arguably the wrong abstraction for it — a picking pattern
or per-string tab would be the honest output. A test on a genuinely strummed
acoustic track would isolate the algorithm's quality from the song's idiom.

## Where this leaves the build

| component | status |
|---|---|
| Chord recognition | works, accurate, fast enough |
| Beat / downbeat / metre | works via madmom, compound metre handled |
| Bar-aligned chord chart | works |
| Strum onsets | blocked on stem separation |
| Strum direction | unsolved; needs acoustic evidence, not a metric prior |
| Notation output | not started |

## Next

1. Run Demucs locally, feed the guitar stem to the existing onset code. This is
   the single highest-value step and it is mostly plumbing.
2. Retest on a strummed acoustic song rather than a fingerpicked ballad.
3. Direction from sweep micro-timing: within one strum the string attacks are
   staggered 10–30 ms; low→high ordering means a downstroke. Only measurable on
   an isolated stem.
4. Notation: music21 → MusicXML/MIDI, PyGuitarPro → .gp5, MuseScore CLI → PDF.
5. Source BTC-SL / BTC-PL checkpoints to enable model cross-checking.
