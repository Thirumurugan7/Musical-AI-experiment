#!/usr/bin/env bash
# One-shot setup for the chord + strumming pipeline (macOS / Linux).
# Mirrors steps 1-5 of RUNBOOK.md. Safe to re-run.
set -euo pipefail

ROOT="${CHORDSTRUM_ROOT:-$HOME/cowork}"
APP="$ROOT/ChordMiniApp"
BE="$APP/python_backend"
HERE="$(cd "$(dirname "$0")" && pwd)"

say() { printf '\n\033[1m>> %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

say "checking prerequisites"
for t in git ffmpeg; do
  have "$t" || { echo "!! missing '$t' — brew install $t"; exit 1; }
done
if ! have uv; then
  echo "   installing uv"
  if have brew; then brew install uv; else pip3 install --user uv; fi
fi
have uv || { echo "!! uv not on PATH; add ~/.local/bin"; exit 1; }
if [ "$(uname)" = "Darwin" ] && ! xcode-select -p >/dev/null 2>&1; then
  echo "!! Xcode command line tools required (madmom builds from source)"
  echo "   run: xcode-select --install"
  exit 1
fi

say "fetching ChordMini"
mkdir -p "$ROOT"
# ChordMiniApp tracks SongFormer's 1.3GB MuQ weights in LFS, and that repo's LFS
# budget is exhausted upstream — with git-lfs installed the smudge filter fails
# and aborts the checkout. Nothing in this pipeline reads those weights, so skip
# them. The chord checkpoints below are plain git objects and arrive regardless.
export GIT_LFS_SKIP_SMUDGE=1
if [ ! -d "$APP/.git" ]; then
  git clone --depth 1 https://github.com/ptnghia-j/ChordMiniApp.git "$APP"
fi
cd "$APP"
git submodule update --init --depth 1 \
  python_backend/models/ChordMini \
  python_backend/models/Chord-CNN-LSTM

CKPT="$BE/models/Chord-CNN-LSTM/cache_data"
[ -d "$CKPT" ] || { echo "!! chord checkpoints missing at $CKPT"; exit 1; }
FOLDS=$(ls "$CKPT" | grep -c best)
[ "$FOLDS" -gt 0 ] || { echo "!! no checkpoint files in $CKPT"; exit 1; }
# an LFS pointer is ~130 bytes; a real checkpoint is ~5.7MB
SMALLEST=$(find "$CKPT" -name '*best*' -exec stat -f%z {} \; 2>/dev/null | sort -n | head -1)
[ "${SMALLEST:-0}" -gt 100000 ] || {
  echo "!! checkpoints are LFS pointers, not real weights ($SMALLEST bytes)"
  echo "   run: git -C $BE/models/Chord-CNN-LSTM lfs pull"; exit 1; }
echo "   checkpoints: $FOLDS folds found"

say "building python 3.10 environment"
cd "$BE"
[ -x .venv/bin/python ] || uv venv --python 3.10 .venv

uv pip install --python .venv/bin/python \
  numpy==1.26.4 scipy==1.13.1 numba==0.59.1 torch==2.6.0 \
  librosa==0.10.1 soundfile==0.12.1 audioread==3.0.1 resampy==0.4.3 \
  soxr==0.5.0.post1 pydub==0.25.1 pretty_midi==0.2.10 mir_eval==0.8.2 \
  mido==1.3.3 h5py==3.13.0 joblib==1.4.2 scikit-learn==1.6.1 \
  PyYAML==6.0.2 tqdm==4.67.1 requests==2.31.0 \
  matplotlib pandas jams==0.3.4 pumpp==0.6.0 figures \
  setuptools==79.0.1

say "building madmom from git (the PyPI release will not compile on 3.10)"
if ! .venv/bin/python -c "import madmom" 2>/dev/null; then
  uv pip install --python .venv/bin/python "cython<3"
  uv pip install --python .venv/bin/python \
    "madmom @ git+https://github.com/CPJKU/madmom" --no-build-isolation
fi

say "installing project code"
rm -rf "$BE/chordstrum"
cp -r "$HERE/src" "$BE/chordstrum"

say "verifying"
.venv/bin/python - <<'PY'
import importlib, sys
ok = True
for m in ("numpy","scipy","numba","torch","librosa","madmom","pretty_midi"):
    try:
        mod = importlib.import_module(m)
        print(f"   {m:12s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        ok = False; print(f"   {m:12s} FAILED: {e}")
sys.exit(0 if ok else 1)
PY
cd "$BE/models/Chord-CNN-LSTM"
../../.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from chord_recognition import chord_recognition
print('   chord model import OK')
"

cat <<EOF

>> setup complete

next, per song:

  ffmpeg -i ~/Downloads/YOUR_SONG.mp3 -ac 1 -ar 44100 $ROOT/song.wav

  cd $BE/models/Chord-CNN-LSTM
  ../../.venv/bin/python chord_recognition.py $ROOT/song.wav $ROOT/song.lab submission

  cd $BE/chordstrum
  ../.venv/bin/python transcribe.py $ROOT/song.wav $ROOT/song.lab -t "Artist — Title"

see RUNBOOK.md for the guitar stem step.
EOF
