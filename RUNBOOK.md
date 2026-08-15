# Runbook — from zero to a chord + strumming chart

Written for macOS (Apple Silicon). Every command is copy-pasteable. Steps 1–5
are one-time setup; steps 6–9 are what you run per song.

Two separate virtualenvs on purpose. The analysis stack is pinned to
`numpy==1.26.4` because `numba` and `librosa` need it; Demucs will happily pull
a newer numpy and break them. Keep them apart.

---

## 1. System prerequisites

```bash
xcode-select --install          # C compiler — madmom is built from source
brew install ffmpeg uv git-lfs
```

`xcode-select` is a no-op if you already have the tools. Skip anything already
installed.

---

## 2. Get ChordMini and its chord model

```bash
cd ~/cowork
git clone --depth 1 https://github.com/ptnghia-j/ChordMiniApp.git
cd ChordMiniApp
git submodule update --init --depth 1 \
  python_backend/models/ChordMini \
  python_backend/models/Chord-CNN-LSTM
```

The Chord-CNN-LSTM checkpoints (5 folds, 5.5 MB each) ship inside that
submodule — nothing else to download. BTC-SL / BTC-PL checkpoints are *not*
in the repo; only Chord-CNN-LSTM works out of the box.

---

## 3. Build the analysis environment

```bash
cd ~/cowork/ChordMiniApp/python_backend
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

`setuptools` is not optional — `pretty_midi` imports `pkg_resources`, and the
whole import chain dies without it. It is missing from the project's own
`requirements.txt`.

Do **not** install `tensorflow` or `spleeter`. They are only needed for
Beat-Transformer, which we don't use.

---

## 4. Build madmom from source

The PyPI release (0.16.1) will not compile on Python 3.10 — its
Cython-generated C calls CPython APIs that no longer exist. Git master is fine.

```bash
cd ~/cowork/ChordMiniApp/python_backend
uv pip install --python .venv/bin/python "cython<3"
uv pip install --python .venv/bin/python \
  "madmom @ git+https://github.com/CPJKU/madmom" --no-build-isolation
```

Verify:

```bash
.venv/bin/python -c "import madmom; print(madmom.__version__)"
```

---

## 5. Drop this project's code in

```bash
cp -r ~/cowork/chord-strum/src ~/cowork/ChordMiniApp/python_backend/chordstrum
```

---

## 6. Per song — convert the audio

```bash
cd ~/cowork
ffmpeg -i ~/Downloads/YOUR_SONG.mp3 -ac 1 -ar 44100 song.wav
```

Mono 44.1 kHz. The chord model resamples internally, but feeding it a clean
wav avoids mp3 decoder quirks.

---

## 7. Per song — chord recognition

```bash
cd ~/cowork/ChordMiniApp/python_backend/models/Chord-CNN-LSTM
../../.venv/bin/python chord_recognition.py \
  ~/cowork/song.wav ~/cowork/song.lab submission
```

Roughly 30 seconds for a 4-minute song. `submission` is the chord dictionary;
alternatives are `full`, `ismir2017`, `extended`.

Output is a `.lab` file: `start_seconds  end_seconds  chord`.

---

## 8. Per song — chart with capo and strumming

```bash
cd ~/cowork/ChordMiniApp/python_backend/chordstrum
../.venv/bin/python transcribe.py \
  ~/cowork/song.wav ~/cowork/song.lab \
  -t "Artist — Title" -o ~/cowork/song.json -n 40
```

Options:

| flag | meaning |
|---|---|
| `-t` | title in the chart header |
| `-o` | write the full analysis as JSON |
| `-n` | how many bars to print (default 32) |
| `-s` | isolated guitar track — see step 9 |

Reading the output: `D` is an accented stroke, `d` lighter, `(d)` a ghost
stroke, `·` no stroke. Stroke *direction* is derived from metre and stroke
rate, not measured — it is labelled as such in the output and you should
trust the accents far more than the arrows.

---

## 9. The guitar stem, and the rhythm that depends on it

Separate venv, because Demucs and the analysis stack disagree about numpy.

```bash
python3 -m venv ~/cowork/.demucs-venv
~/cowork/.demucs-venv/bin/pip install -U pip demucs

~/cowork/.demucs-venv/bin/python -m demucs \
  --two-stems=guitar -n htdemucs_6s --mp3 \
  -o ~/cowork/stems ~/Downloads/YOUR_SONG.mp3
```

`htdemucs_6s` is the six-source model (`drums bass other vocals guitar piano`)
and is the only one with a dedicated guitar output — the default four-source
model buries guitar in `other`. First run downloads ~300 MB of weights.

Then re-run the chart, reading beats from the full mix but onsets from the
guitar alone:

```bash
cd ~/cowork/ChordMiniApp/python_backend/chordstrum
../.venv/bin/python transcribe.py \
  ~/cowork/song.wav ~/cowork/song.lab \
  -s ~/cowork/stems/htdemucs_6s/YOUR_SONG/guitar.mp3 \
  -t "Artist — Title" -o ~/cowork/song_stem.json
```

**Watch the `grid occupancy` figure at the bottom.** On the full mix it reads
98 % — every subdivision has energy, because drums, piano, strings and vocals
all land somewhere. That number is the whole reason rhythm detection failed on
a full mix. On a real guitar stem it should drop to roughly the fraction of
subdivisions the guitar actually strikes. If it stays near 98 %, either the
song genuinely strums every subdivision (true for "Perfect", which is 12
eighths per bar) or the separation didn't work — compare against the isolated
stem by ear before concluding.

Useful Demucs flags: `--shifts 5` for a slower but cleaner split, `-d cpu` to
force CPU, `-j 2` for parallel jobs.

---

## Shortcut

`setup.sh` runs steps 1–5 in one go:

```bash
bash ~/cowork/chord-strum/setup.sh
```

`get_guitar_stem.sh` runs step 9's separation:

```bash
bash ~/cowork/chord-strum/get_guitar_stem.sh ~/Downloads/YOUR_SONG.mp3
```

---

## If something breaks

| symptom | cause |
|---|---|
| `No module named 'pkg_resources'` | `setuptools` missing — step 3 |
| `scipy.signal has no attribute 'hann'` | scipy ≥ 1.13 vs librosa 0.10.1; the shim is at the top of each script, so you're running an unpatched file |
| madmom compile fails with `PyUnicode_FromUnicode` | you installed the PyPI release instead of git master — step 4 |
| `numba` import error | numpy got upgraded past 1.26.4, probably by installing Demucs into the same venv |
| Chord output is all `N` | audio failed to load, or the track has no clear harmonic content |
| Tempo looks doubled or halved | compound metre mis-grouped; check the `metre` line, and see COMPARISON.md |
