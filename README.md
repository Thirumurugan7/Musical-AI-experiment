# chord-strum

Automatic chord, metre and rhythm transcription for guitar, built on top of
[ChordMini](https://github.com/ptnghia-j/ChordMiniApp) (MIT).

The goal is the thing no open-source tool ships: a chart with **chords and a
playable rhythm**, derived from the recording rather than from a crowd-sourced
transcription.

## Status

| Component | State |
|---|---|
| Chord recognition | **works** — Chord-CNN-LSTM, matches human transcriptions |
| Beat / downbeat / metre | **works** — madmom DBN, handles compound metre |
| Bar-aligned chord chart | **works** |
| Accent detection | **works** — 2.12x contrast on a full mix, no separation needed |
| Strum onsets | needs an isolated guitar stem to be meaningful |
| Strum direction (D/U) | **unsolved** — the metric prior is disproven, see below |
| Notation output (MusicXML / GP5 / PDF) | not started |

## What we learned

Validated against Ed Sheeran's "Perfect" and cross-checked with Ultimate
Guitar, Hooktheory, Fender, Chordify and SongBPM — see `COMPARISON.md`.

- Key, tempo and metre matched a human-curated source to within 0.2 %.
- Chords matched every guitar source, including the Fm7 colouring that
  Chordify also lists.
- **Beat trackers systematically mis-read compound metre.** librosa reported
  95.7 BPM in 4/4 for a song that is 63.2 BPM in 12/8, grouping eighths into
  pairs. SongBPM makes the identical error (97 BPM). madmom's
  `DBNDownBeatTrackingProcessor(beats_per_bar=[4])` gets it right.
- **Stroke direction cannot be inferred from metric position.** The
  on-beat-down / off-beat-up heuristic is duple-specific. In compound metre
  players use continuous downstrokes; our prior scored 0 % on direction while
  every human source says "all downs."
- **Accents survive a dense mix.** Fender's instruction to accent strokes
  1, 4, 7 and 10 is directly visible in the onset envelope at 2.12x contrast,
  with the first eighth strongest in 4/4 triplet groups. This was not expected.

The practical consequence: for compound-metre songs the honest output is a
**rhythm chart with accents**, not a down/up arrow pattern — which is how the
human sources notate it anyway.

## Layout

```
src/
  analyse.py         chords + madmom grid -> bar-aligned chart   (main entry)
  meter.py           tatum + duple/triple grouping, no pretrained weights
  strum_detect.py    librosa onsets -> strum events (superseded by analyse.py)
  build_chart.py     merge a .lab chord file with strum JSON
  make_test_audio.py synthesise a strummed progression with known ground truth
data/
  perfect.lab            Chord-CNN-LSTM output for the test song
  perfect_analysis.json  full bar/slot analysis
  perfect_chart.txt      rendered 65-bar chart
  ground_truth.lab       ground truth for the synthetic test
```

Audio is gitignored — not redistributable, and large.

## Setup

See `NOTES.md` for the full working recipe and the four gotchas that cost
time. Short version: Python 3.10 (not 3.11), and build madmom from git master
because the PyPI release will not compile.

```bash
uv pip install --python .venv/bin/python cython'<3'
uv pip install --python .venv/bin/python \
  "madmom @ git+https://github.com/CPJKU/madmom" --no-build-isolation
```

## Usage

```bash
# chords
cd ChordMiniApp/python_backend/models/Chord-CNN-LSTM
python chord_recognition.py song.wav song.lab submission

# bar-aligned chart with metre detection
python src/analyse.py song.wav song.lab out.json
```

## Next

1. Rewrite rhythm output as accent-based, since that demonstrably works today.
2. Test on a genuinely strummed song, where up/down actually matters.
3. Demucs guitar stem -> onset detection, for real strum isolation.
4. Notation: music21 -> MusicXML/MIDI, PyGuitarPro -> .gp5, MuseScore CLI -> PDF.
5. Source BTC-SL / BTC-PL checkpoints to enable model cross-checking.
