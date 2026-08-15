# ChordMini + strum layer — working setup notes

State as of this session. Everything below was actually run, not planned.

## Environment that works

The README asks for Python 3.10.16; 3.10.20 is fine. Do **not** use 3.11 —
`numba==0.59.1` and the pinned numeric stack will fight you.

```bash
git clone --depth 1 https://github.com/ptnghia-j/ChordMiniApp.git
cd ChordMiniApp
git submodule update --init --depth 1 \
  python_backend/models/ChordMini \
  python_backend/models/Chord-CNN-LSTM

cd python_backend
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python \
  numpy==1.26.4 scipy==1.13.1 numba==0.59.1 torch==2.6.0 \
  librosa==0.10.1 soundfile==0.12.1 audioread==3.0.1 resampy==0.4.3 \
  soxr==0.5.0.post1 pydub==0.25.1 pretty_midi==0.2.10 mir_eval==0.8.2 \
  mido==1.3.3 h5py==3.13.0 joblib==1.4.2 scikit-learn==1.6.1 \
  PyYAML==6.0.2 tqdm==4.67.1 requests==2.31.0 \
  matplotlib pandas jams==0.3.4 pumpp==0.6.0 figures \
  setuptools==79.0.1
```

### Gotchas hit along the way

- **`setuptools` is not optional.** `pretty_midi` imports `pkg_resources`;
  without setuptools the whole import chain dies. Not in requirements.txt.
- **No `download.pytorch.org`.** Sandbox only reaches PyPI, so torch arrives
  with ~6 GB of unused `nvidia-*` CUDA deps. Harmless, just fat.
- **`scipy.signal.hann` is gone** in scipy >= 1.13, and librosa 0.10.1 still
  calls it. ChordMini ships a shim at `python_backend/compat/scipy_patch.py`;
  standalone scripts need their own (see top of `strum_detect.py`).
- **TensorFlow and Spleeter are skippable.** They are only needed for
  Beat-Transformer. Chord-CNN-LSTM is pure torch, and its checkpoints
  (5 folds x 5.5 MB) ship inside the submodule — no separate download.
- **BTC-SL / BTC-PL checkpoints are NOT in the repo.** The detector looks for
  `models/ChordMini/checkpoints/SL/btc_model_large_voca.pt`, which is absent.
  Sourcing those is an open item.
- The Flask backend is already a clean REST API — `/api/recognize-chords`,
  `/api/detect-beats`, `/api/chord-model-info`. Firebase only appears in
  separate `-firebase` endpoint variants, so a headless deployment does not
  need Firebase or Gemini at all. That contradicts the README's setup list.

## Running it

```bash
# chords (writes a .lab file)
cd python_backend/models/Chord-CNN-LSTM
../../.venv/bin/python chord_recognition.py song.wav out.lab submission

# strums
.venv/bin/python strum_detect.py song.wav strums.json

# merged chart
.venv/bin/python build_chart.py out.lab strums.json
```

## Measured so far

Synthetic 16.5 s strummed progression (C G Am F x2), 2 vCPU, no GPU:

- Chord-CNN-LSTM: **8/8 chords correct**, boundaries within ~20 ms, 19.6 s wall
  clock (~1.2x realtime, so a 4 min song ≈ 5 min).
- Strum detection: **40/40 onsets**, correct grid positions, correct directions.

Caveat that matters: the synthetic signal is the easy case, and the
up/down assignment leans on a metric prior that matches how the test file was
generated. This validates the plumbing, **not** the acoustic discrimination.
Real recordings are the actual test.

## Known issues / next up

1. **Chord boundaries are not beat-synchronous.** librosa estimated 117.45 BPM
   against a true 120, so the grid drifts and bar 5 of the test picks up a
   stray `F` before the `C`. Fix: snap chord label boundaries to the detected
   beat grid before rendering bars.
2. **Beat tracking is the weak link.** Swap librosa for madmom's DBN tracker or
   ChordMini's Beat-Transformer — both give real downbeats, which librosa does
   not. Bar numbering is currently assumed, not detected.
3. **No stem separation yet.** Demucs on the guitar stem before onset detection
   should sharply improve strum detection on full-band mixes.
4. **Notation output not started.** Plan: music21 for MusicXML + MIDI,
   PyGuitarPro for .gp5, MuseScore CLI or LilyPond for PDF.
5. **Direction refinement is naive.** Current model is 75% metric prior, 25%
   spectral centroid. Better signal: analyse the micro-timing sweep of
   individual string attacks within one strum (low->high = down).
