# Getting a real guitar stem

`src/isolate.py` does what classical DSP can with no pretrained weights. It is
not source separation and should not be described as such. For an actual
guitar track, use Demucs' 6-source model, which has a dedicated `guitar`
source (the standard 4-source model does not — guitar lands in `other`
alongside piano and strings).

## On your own machine

```bash
pip install -U demucs
demucs --two-stems=guitar -n htdemucs_6s "song.mp3"
# -> separated/htdemucs_6s/song/guitar.wav  and  no_guitar.wav
```

Drop `--two-stems` to get all six (`drums bass other vocals guitar piano`):

```bash
demucs -n htdemucs_6s "song.mp3"
```

Useful flags:

- `--mp3` write mp3 instead of wav
- `-d cpu` force CPU if the GPU path misbehaves
- `--shifts 5` slower, noticeably cleaner (averages 5 random offsets)
- `-j 2` parallel jobs

First run downloads ~300 MB of weights from `dl.fbaipublicfiles.com`.

## Why not here

The sandbox this was built in reaches PyPI and `git clone` only. Confirmed
blocked: `dl.fbaipublicfiles.com` (Demucs), `huggingface.co`, GitHub release
assets, `codeload.github.com`, `zenodo.org`, `download.pytorch.org`
(torchaudio's bundled HDEMUCS pipeline). Candidate GitHub mirrors were checked
with treeless clones — none carry the weights as committed files.

Demucs itself installs fine from PyPI. Only the weights are unreachable.

## What isolate.py actually produces

| output | what it is |
|---|---|
| `*_background.wav` | vocals suppressed, drums kept — usually the best play-along |
| `*_accompaniment.wav` | vocals suppressed *and* percussion removed; bass-heavy |
| `*_vocal.wav` | what was removed, for checking the separation worked |
| `*_side.wav` | L−R, cancels anything panned dead centre |

Measured on "Perfect": energy in the 250 Hz–4 kHz vocal band drops from 0.518
of the mix to 0.425 (`background`) and 0.356 (`accompaniment`). The extracted
`vocal` track holds only 0.067 of its energy below 250 Hz, which is what a real
voice looks like — evidence the split is doing something genuine rather than
smearing everything.

`accompaniment` carries 62 % of its energy below 250 Hz, so expect it to sound
muddy; HPSS pulls the mix bass-ward. Judge by ear — these numbers say the
processing did something, not that it sounds good.
