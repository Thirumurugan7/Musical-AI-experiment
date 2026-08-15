#!/usr/bin/env bash
# Clean-room benchmark run.
#
# Every derived artifact is deleted before it is rebuilt, so nothing from an
# earlier run can survive into the scores. The source mp3s are the only inputs
# kept, and their checksums are recorded so a changed input cannot pass unseen.
#
# Two failure modes this guards against, both of which have already bitten:
#   - a failed transcribe leaving the PREVIOUS json in place to be scored,
#     which made three different inputs look byte-identical
#   - stale __pycache__ shadowing freshly copied source
#
# Every stage writes to a temp file and moves it into place only on success,
# and every failure is recorded rather than hidden.
set -uo pipefail

ROOT=/Users/thirumurugansivalingam/Desktop/personal/music-ai
BENCH=$ROOT/work/bench
BE=$ROOT/work/ChordMiniApp/python_backend
STEMS=${USE_STEMS:-0}

echo "=== clean-room benchmark  $(date '+%H:%M:%S') ==="
echo "harmonic-stem chord recognition: $([ "$STEMS" = 1 ] && echo ON || echo OFF)"

# ---------- 1. wipe every derived artifact ----------
echo; echo ">>> clearing derived artifacts"
rm -f "$BENCH"/*.wav "$BENCH"/*.lab "$BENCH"/*.json "$BENCH"/*.tmp.wav
rm -rf "$BENCH"/harmonic
rm -rf "$BE/chordstrum/__pycache__" "$BE/models/Chord-CNN-LSTM/__pycache__"
find "$ROOT/src" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
echo "    wav/lab/json removed, __pycache__ cleared"

# ---------- 2. record the inputs ----------
echo; echo ">>> input manifest"
: > "$BENCH/inputs.sha256"
for f in "$BENCH"/*.mp3; do
  shasum -a 256 "$f" >> "$BENCH/inputs.sha256"
done
echo "    $(wc -l < "$BENCH/inputs.sha256" | tr -d ' ') mp3s checksummed"

# ---------- 3. fresh copy of the source ----------
echo; echo ">>> syncing source"
rm -f "$BE/chordstrum"/*.py
cp "$ROOT"/src/*.py "$BE/chordstrum/"
echo "    $(ls "$BE/chordstrum"/*.py | wc -l | tr -d ' ') modules copied"

# ---------- 4. process every song ----------
echo; echo ">>> processing"
ok=0; fail=0
for f in "$BENCH"/*.mp3; do
  id=$(basename "$f" .mp3)

  # -f wav is required: ffmpeg picks the muxer from the extension, and a
  # ".wav.tmp" name has no format it recognises, so every decode aborts
  if ! ffmpeg -y -loglevel error -i "$f" -ac 1 -ar 44100 -f wav \
       "$BENCH/$id.tmp.wav" 2>/dev/null; then
    echo "    FAIL $id (decode)"; fail=$((fail+1)); rm -f "$BENCH/$id.tmp.wav"; continue
  fi
  mv "$BENCH/$id.tmp.wav" "$BENCH/$id.wav"

  # optional: recognise chords on a harmonic stem, vocals and drums removed
  CHORD_SRC="$BENCH/$id.wav"
  if [ "$STEMS" = 1 ]; then
    mkdir -p "$BENCH/harmonic"
    if [ ! -s "$BENCH/harmonic/$id.wav" ]; then
      "$ROOT/work/.demucs-venv/bin/python" -m demucs -n htdemucs_6s --mp3 \
        -o "$BENCH/harmonic/raw" "$f" >/dev/null 2>&1
      d="$BENCH/harmonic/raw/htdemucs_6s/$id"
      if [ -f "$d/other.mp3" ]; then
        ffmpeg -y -loglevel error \
          -i "$d/other.mp3" -i "$d/guitar.mp3" -i "$d/piano.mp3" -i "$d/bass.mp3" \
          -filter_complex "[0:a][1:a][2:a][3:a]amix=inputs=4:normalize=0[a]" \
          -map "[a]" -ac 1 -ar 44100 "$BENCH/harmonic/$id.wav" 2>/dev/null
      fi
    fi
    [ -s "$BENCH/harmonic/$id.wav" ] && CHORD_SRC="$BENCH/harmonic/$id.wav"
  fi

  ( cd "$BE/models/Chord-CNN-LSTM" && \
    ../../.venv/bin/python chord_recognition.py \
      "$CHORD_SRC" "$BENCH/$id.lab.tmp" submission >/dev/null 2>&1 )
  if [ ! -s "$BENCH/$id.lab.tmp" ]; then
    echo "    FAIL $id (chords)"; fail=$((fail+1)); rm -f "$BENCH/$id.lab.tmp"; continue
  fi
  mv "$BENCH/$id.lab.tmp" "$BENCH/$id.lab"

  if ( cd "$BE/chordstrum" && \
       ../.venv/bin/python transcribe.py "$BENCH/$id.wav" "$BENCH/$id.lab" \
         -t "$id" -o "$BENCH/$id.json.tmp" >/dev/null 2>&1 ) \
     && [ -s "$BENCH/$id.json.tmp" ]; then
    mv "$BENCH/$id.json.tmp" "$BENCH/$id.json"
    ok=$((ok+1)); echo "    ok   $id"
  else
    rm -f "$BENCH/$id.json.tmp"
    echo "    FAIL $id (transcribe)"; fail=$((fail+1))
  fi
done

echo; echo ">>> processed ok=$ok fail=$fail"

# ---------- 5. score ----------
echo; echo ">>> scoring"
python3 "$ROOT/bench/score.py"
echo; echo "=== done  $(date '+%H:%M:%S') ==="
