#!/usr/bin/env bash
# A/B chord recognition with and without HPSS pre-processing, on a subset.
set -uo pipefail
R=/Users/thirumurugansivalingam/Desktop/personal/music-ai
B=$R/work/bench; BE=$R/work/ChordMiniApp/python_backend
V=$BE/.venv/bin/python
MARGIN=${HPSS_MARGIN:-3.0}
mkdir -p "$B/hpss"

SONGS="${SONGS:-riptide letitbe perfect wonderwall boulevard hotelcalifornia hohey imagine nothingelse tearsinheaven freefallin knockinheaven}"

for id in $SONGS; do
  [ -s "$B/$id.wav" ] || continue
  # harmonic component
  if [ ! -s "$B/hpss/$id.wav" ]; then
    $V "$R/bench/hpss_prep.py" "$B/$id.wav" "$B/hpss/$id.wav" "$MARGIN" >/dev/null 2>&1
  fi
  # chords from the harmonic component
  ( cd "$BE/models/Chord-CNN-LSTM" && \
    ../../.venv/bin/python chord_recognition.py "$B/hpss/$id.wav" "$B/hpss/$id.lab" submission >/dev/null 2>&1 )
  # transcribe using the HPSS lab, but the ORIGINAL wav for beats
  ( cd "$BE/chordstrum" && \
    ../.venv/bin/python transcribe.py "$B/$id.wav" "$B/hpss/$id.lab" -t "$id" \
      -o "$B/hpss/$id.json" >/dev/null 2>&1 ) && echo "ok $id" || echo "FAIL $id"
done
