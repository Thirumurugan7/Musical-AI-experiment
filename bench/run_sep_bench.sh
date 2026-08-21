#!/usr/bin/env bash
# Apply every separation variant to the constructed test mixtures.
#
#   bash bench/run_sep_bench.sh          # demucs variants only (fast)
#   WITH_ROFORMER=1 bash bench/run_sep_bench.sh
#
# Each variant writes work/septests/<case>.<variant>.wav, which is what
# bench/score_sep.py looks for. The mix itself is emitted as a variant on
# purpose: it is the floor, the score you get for doing nothing, and any
# variant that fails to beat it is worse than useless.
set -uo pipefail
ROOT=/Users/thirumurugansivalingam/Desktop/personal/music-ai
T=$ROOT/work/septests
DM=$ROOT/work/.demucs-venv/bin/python
SEP=$ROOT/work/.sep-venv/bin/audio-separator
SCRATCH=$ROOT/work/septests/_scratch
WITH_ROFORMER=${WITH_ROFORMER:-0}
WITH_FT=${WITH_FT:-0}

ls "$T"/*.mix.wav >/dev/null 2>&1 || {
  echo "no test mixtures — run bench/make_sep_tests.py first"; exit 1; }
mkdir -p "$SCRATCH"

demucs_variant () {           # model, stem, tag, input, case
  local model=$1 stem=$2 tag=$3 in=$4 case=$5
  local o="$SCRATCH/$tag/$case"
  [ -f "$T/$case.$tag.wav" ] && return 0
  rm -rf "$o"; mkdir -p "$o"
  "$DM" -m demucs -n "$model" -o "$o" "$in" >/dev/null 2>&1 || return 1
  local f
  f=$(find "$o" -name "$stem.wav" | head -1)
  [ -n "$f" ] && mv "$f" "$T/$case.$tag.wav"
}

n=0
for mix in "$T"/*.mix.wav; do
  case=$(basename "$mix" .mix.wav)
  n=$((n+1))
  echo ">>> [$n] $case"

  # control: doing nothing at all
  [ -f "$T/$case.mix_control.wav" ] || cp "$mix" "$T/$case.mix_control.wav"

  demucs_variant htdemucs_6s guitar demucs6_guitar "$mix" "$case" \
    && echo "    demucs6_guitar" || echo "    FAIL demucs6_guitar"
  demucs_variant htdemucs_6s other  demucs6_other  "$mix" "$case" \
    && echo "    demucs6_other"  || echo "    FAIL demucs6_other"
  # htdemucs_ft is an ensemble of four models: four checkpoints to fetch and
  # four forward passes per case. On a slow link that alone took longer than
  # every other variant combined, so it is opt-in.
  if [ "$WITH_FT" = 1 ]; then
    demucs_variant htdemucs_ft  other  demucsft_other "$mix" "$case" \
      && echo "    demucsft_other" || echo "    FAIL demucsft_other"
  fi

  if [ "$WITH_ROFORMER" = 1 ]; then
    if [ ! -f "$T/$case.twostage_guitar.wav" ]; then
      s1="$SCRATCH/roformer/$case"; rm -rf "$s1"; mkdir -p "$s1"
      if "$SEP" "$mix" --model_filename melband_roformer_inst_v2.ckpt \
           --model_file_dir "$ROOT/work/.sep-models" \
           --output_dir "$s1" --output_format WAV >/dev/null 2>&1; then
        inst=$(ls "$s1"/*[Ii]nstrumental*.wav 2>/dev/null | head -1)
        [ -n "$inst" ] && demucs_variant htdemucs_6s guitar twostage_guitar \
          "$inst" "$case" && echo "    twostage_guitar" \
          || echo "    FAIL twostage_guitar (stage 2)"
      else
        echo "    FAIL twostage_guitar (stage 1)"
      fi
    fi
  fi
done

echo; echo ">>> scoring"
"$ROOT/work/ChordMiniApp/python_backend/.venv/bin/python" \
  "$ROOT/bench/score_sep.py" "$T" 2>/dev/null
