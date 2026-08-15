#!/usr/bin/env bash
# Transcribe an isolated vocal stem to a timestamped .lrc.
#
#   bash get_lyrics.sh work/voice_only.mp3 [output.lrc]
#
# Third virtualenv on purpose. The analysis stack is pinned to numpy 1.26.4,
# demucs needs its own numpy, and faster-whisper wants ctranslate2 — keeping
# all three apart is what stops one install from breaking the other two.
set -euo pipefail

AUDIO="${1:-}"
ROOT="${CHORDSTRUM_ROOT:-$HOME/cowork}"
OUT="${2:-$ROOT/lyrics.lrc}"
MODEL="${WHISPER_MODEL:-medium}"
VENV="$ROOT/.whisper-venv"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$AUDIO" ]; then
  echo "usage: bash get_lyrics.sh /path/to/vocal_stem.mp3 [out.lrc]"; exit 1
fi
[ -f "$AUDIO" ] || { echo "!! not found: $AUDIO"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "!! python3 not installed"; exit 1; }

say() { printf '\n\033[1m>> %s\033[0m\n' "$*"; }

if [ ! -x "$VENV/bin/python" ]; then
  say "creating whisper virtualenv at $VENV"
  python3 -m venv "$VENV"
fi

say "installing faster-whisper (first run only)"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q faster-whisper

say "transcribing $(basename "$AUDIO") with model '$MODEL'"
"$VENV/bin/python" "$HERE/src/lyrics.py" "$AUDIO" -o "$OUT" -m "$MODEL"

cat <<EOF

next — lay the chords over them:

  python3 $HERE/src/render_chart.py <chart.json> -l "$OUT"

EOF
