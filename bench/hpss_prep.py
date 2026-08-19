"""
Write the harmonic component of a track, for feeding to chord recognition.

Chords live in the harmonic part of a mix. Percussion contributes broadband
energy at every attack, which lands in the chroma as noise across all twelve
pitch classes — worst exactly at the moment a chord changes, which is where
the model most needs a clean reading.

librosa's HPSS separates the two by median-filtering the spectrogram along
time (percussion is broadband and brief, so it survives filtering across
frequency) and along frequency (harmonics are narrow and sustained, so they
survive filtering across time). `margin` controls how aggressively: 1.0 is a
soft split where both parts keep shared energy, higher values force a harder
separation and discard the ambiguous middle.
"""
import sys
import librosa
import soundfile as sf

def main():
    src, dst = sys.argv[1], sys.argv[2]
    margin = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    y, sr = librosa.load(src, sr=44100, mono=True)
    harm, _ = librosa.effects.hpss(y, margin=margin)
    sf.write(dst, harm, sr)
    print(f"{dst}  margin={margin}  {len(harm)/sr:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
