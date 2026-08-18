#!/usr/bin/env bash
# Run every tool in the project over a handful of songs and print one report.
set -uo pipefail
R=/Users/thirumurugansivalingam/Desktop/personal/music-ai
B=$R/work/bench
V=$R/work/ChordMiniApp/python_backend/.venv/bin/python
OUT=$R/work/toolrun
mkdir -p "$OUT"

SONGS="${*:-perfect riptide letitbe wonderwall heysoulsister}"

for id in $SONGS; do
  [ -s "$B/$id.json" ] || { echo "skip $id (no transcription)"; continue; }
  echo "==================================================================="
  $V - "$B/$id.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{d.get('title', '?')}")
print(f"  key {d['sounding_key']}  |  capo {d['capo']} play {d['shape_key']}  |  "
      f"{d['tempo_bpm']} BPM  |  {d['metre']}  |  {d['n_bars']} bars")
print(f"  chords   {' '.join(d['chords_shapes'])}")
print(f"           sounds {' '.join(d['chords_sounding'])}")
print(f"  strum    {d['canonical_pattern']}")
print(f"  sections {len(d.get('sections') or [])}   occupancy {d['grid_occupancy']:.0%}"
      f"   onsets {d.get('onsets_detected','?')}")
simp = d.get('simplification') or {}
if simp.get('folded'):
    print(f"  folded   " + ", ".join(f"{k}->{v['to']}" for k, v in list(simp['folded'].items())[:4]))
PY

  # text chart
  python3 "$R/src/render_chart.py" "$B/$id.json" > "$OUT/$id.chart.txt" 2>/dev/null \
    && echo "  chart    $OUT/$id.chart.txt ($(wc -l < "$OUT/$id.chart.txt" | tr -d ' ') lines)" \
    || echo "  chart    FAILED"

  # interactive page
  python3 "$R/src/build_page.py" -o "$OUT/$id.html" \
    -c "Full-mix detection=$B/$id.json" -a "Full mix=../bench/$id.mp3" >/dev/null 2>&1 \
    && echo "  page     $OUT/$id.html" || echo "  page     FAILED"

  # audio comparison against our own render
  $V "$R/bench/compare_audio.py" "$B/$id.wav" "$B/$id.json" 2>/dev/null \
    | grep -E "mean similarity|bars below|bar starts" | sed 's/^/  /'
done
echo "==================================================================="
echo "scoring all against published charts:"
python3 "$R/bench/score.py" 2>/dev/null | sed -n '/songs scored/,/chords exactly/p' | sed 's/^/  /'
