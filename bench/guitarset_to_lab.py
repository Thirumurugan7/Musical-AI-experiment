"""
Convert GuitarSet chord annotations to .lab files.

The pipeline reads chords from a .lab -- start, end, label, whitespace
separated -- and GuitarSet already stores chords in Harte notation, so this is
a format change rather than an interpretation.

Each excerpt carries two chord annotations and they are not interchangeable:

  instructed  the lead sheet handed to the player. Plain triads, no inversions.
              This is the closest thing to what a chart would print, and so the
              fairer target for a system whose output is a chart.
  performed   semi-automatic transcription with manual verification, including
              inversions and the substitutions players actually made
              (D:sus2/2 where the sheet said D:maj). This is what is audible.

Default is `performed`, because scoring a recognition system against a sheet
the player departed from measures the player, not the system. Pass
--instructed to get the other.

Inversions are dropped by default: a chart does not print them, and our chord
model does not emit them, so keeping them would score a difference neither side
is trying to express. --keep-bass retains them.
"""
import json
import glob
import os
import sys

PERFORMED = "Semi-automatic chord transcription with manual verification"


def to_lab(jams_path, performed=True, keep_bass=False):
    d = json.load(open(jams_path))
    ch = [a for a in d["annotations"] if a["namespace"] == "chord"]
    if not ch:
        return []
    want = PERFORMED if performed else ""
    picked = None
    for a in ch:
        if a["annotation_metadata"].get("data_source", "") == want:
            picked = a
            break
    if picked is None:                      # fall back rather than emit nothing
        picked = ch[-1] if performed else ch[0]

    rows = []
    for o in picked["data"]:
        label = o["value"]
        if not keep_bass and "/" in label:
            label = label.split("/")[0]
        rows.append((float(o["time"]),
                     float(o["time"]) + float(o["duration"]), label))
    return rows


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    performed = "--instructed" not in sys.argv
    keep_bass = "--keep-bass" in sys.argv
    src = argv[0] if argv else "work/guitarset/annotation/*_comp.jams"
    out = argv[1] if len(argv) > 1 else "work/guitarset/lab"

    files = sorted(glob.glob(src))
    if not files:
        print(f"no annotations matching {src}")
        return 1
    os.makedirs(out, exist_ok=True)

    n_rows = 0
    for f in files:
        rows = to_lab(f, performed, keep_bass)
        if not rows:
            continue
        name = os.path.basename(f)[:-len(".jams")]
        with open(os.path.join(out, name + ".lab"), "w") as fh:
            for s, e, lab in rows:
                fh.write(f"{s:.6f} {e:.6f} {lab}\n")
        n_rows += len(rows)

    kind = "performed" if performed else "instructed"
    print(f"wrote {len(files)} .lab files to {out}/ ({kind}, "
          f"{'with' if keep_bass else 'without'} inversions)")
    print(f"  {n_rows} chord segments, {n_rows/max(len(files),1):.1f} per excerpt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
