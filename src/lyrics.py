"""
Transcribe an isolated vocal stem to a timestamped .lrc file.

Runs on the separated vocal rather than the full mix: the backing instruments
are what wreck ASR on music, and removing them is most of the battle. Uses
faster-whisper (CTranslate2) rather than openai-whisper because it needs no
torch, so it installs in its own small virtualenv without touching either of
the other two environments.

Singing is still much harder than speech. Expect the timings to be good and
the words to need correcting by hand — .lrc is plain text, so that is easy.
"""
import argparse
import os
import sys


def fmt_lrc(t):
    m, s = divmod(max(t, 0.0), 60)
    return f"[{int(m):02d}:{s:05.2f}]"


def main():
    ap = argparse.ArgumentParser(description="vocal stem -> timestamped .lrc")
    ap.add_argument("audio", help="isolated vocal stem (mp3/wav)")
    ap.add_argument("-o", "--out", required=True, help="output .lrc path")
    ap.add_argument("-m", "--model", default="medium",
                    help="tiny|base|small|medium|large-v3 (default medium)")
    ap.add_argument("--language", default="en")
    a = ap.parse_args()

    if not os.path.isfile(a.audio):
        print(f"!! not found: {a.audio}", file=sys.stderr)
        return 1

    from faster_whisper import WhisperModel

    print(f">> loading {a.model} (first run downloads the weights)")
    model = WhisperModel(a.model, device="cpu", compute_type="int8")

    print(">> transcribing")
    segments, info = model.transcribe(
        a.audio,
        language=a.language,
        word_timestamps=True,
        # singing has long held vowels and long gaps; the speech defaults cut
        # lines in the wrong places and hallucinate through instrumental rests
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 700},
        condition_on_previous_text=False,
    )

    lines, n = [], 0
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        n += 1
        lines.append(f"{fmt_lrc(seg.start)} {text}")
        print(f"   {fmt_lrc(seg.start)} {text}")

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n>> {n} lines -> {a.out}")
    print(f"   detected language: {info.language} "
          f"(confidence {info.language_probability:.2f})")
    print("   check the words by ear — ASR on singing gets timing right more"
          " often than wording")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
