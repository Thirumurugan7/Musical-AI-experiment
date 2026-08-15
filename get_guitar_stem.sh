#!/usr/bin/env bash
# Separate the guitar onto its own track with Demucs' 6-source model.
#
#   bash get_guitar_stem.sh ~/Downloads/YOUR_SONG.mp3
#
# Uses its own virtualenv on purpose: Demucs pulls a newer numpy than the
# analysis stack's pinned 1.26.4, and mixing them breaks numba and librosa.
set -euo pipefail

SONG="${1:-}"
ROOT="${CHORDSTRUM_ROOT:-$HOME/cowork}"
OUT="${2:-$ROOT/stems}"
VENV="$ROOT/.demucs-venv"

if [ -z "$SONG" ]; then
  echo "usage: bash get_guitar_stem.sh /path/to/song.mp3 [output_dir]"; exit 1
fi
[ -f "$SONG" ] || { echo "!! not found: $SONG"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "!! python3 not installed"; exit 1; }

say() { printf '\n\033[1m>> %s\033[0m\n' "$*"; }

if [ ! -x "$VENV/bin/python" ]; then
  say "creating demucs virtualenv at $VENV"
  python3 -m venv "$VENV"
fi

say "installing demucs (first run only)"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q demucs

mkdir -p "$OUT"
say "separating — first run downloads ~300MB of weights, then several minutes of CPU"
# htdemucs_6s is the only model with a dedicated guitar source; the default
# 4-source model buries guitar inside "other" with piano and strings.
"$VENV/bin/python" -m demucs \
  --two-stems=guitar -n htdemucs_6s --mp3 -o "$OUT" "$SONG"

say "done"
find "$OUT" -name "guitar.mp3" | sed 's/^/  /'
cat <<EOF

next — rebuild the chart using the guitar for onsets:

  cd \$ROOT/ChordMiniApp/python_backend/chordstrum
  ../.venv/bin/python transcribe.py \\
    \$ROOT/song.wav \$ROOT/song.lab \\
    -s "\$(find "$OUT" -name guitar.mp3 | head -1)" \\
    -t "Artist — Title"

watch the 'grid occupancy' line: 98% on a full mix, and it should drop on a
real stem to whatever fraction of subdivisions the guitar actually strikes.
EOF
