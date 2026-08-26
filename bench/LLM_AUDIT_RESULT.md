# A local LLM cannot audit this output

Rejected on measurement. Recorded so it is not tried again.

## The idea

`gemma4` runs offline through Ollama. Give it a finished transcription -- key,
capo, chord vocabulary, bar-by-bar progression -- and ask whether the result is
musically coherent. Cheap, private, and it needs no ground truth, so it could
in principle flag bad transcriptions on songs we have never scored.

The model cannot hear the recording, so it can never say whether our chords
match the song. It can only judge internal coherence. That is either a real
signal or nothing.

## Calibration against 55 scored songs

`bench/llm_audit.py`, temperature 0, fixed seed, JSON output, 55 songs in 761 s
(13.8 s per song).

| metric | value |
|---|---|
| plausibility range | 7 to 9 |
| mean / sd | 8.04 / **0.33** |
| corr(plausibility, chord F1) | **+0.148** |
| corr(out_of_key, chord F1) | -0.014 |

51 of 55 songs received the same score. At n=55 the standard error on a
correlation is about 0.139, so +0.148 is roughly one standard error from zero.

That alone was not conclusive. Chord F1 across these songs runs 0.80 to 1.00,
so there is very little badness present to detect, and a flat response could
mean the auditor is blind or merely that everything it saw was fine.

## The negative control settles it

`bench/llm_audit_negctrl.py` feeds the same model four versions of five songs:
the real chart, the chords shuffled into random order, 40% of chords displaced
by a semitone (our actual failure mode -- the one that put a spurious B major
into Riptide for six bars), and a chart of entirely random chords belonging to
no key.

| variant | mean plausibility |
|---|---|
| original | 7.80 |
| **shuffled** | **8.00** |
| semitone_noise | 6.00 |
| random | 5.80 |

**Shuffling scores higher than the original.** Not once across five songs was a
shuffled progression penalised. Destroying every trace of musical structure
while keeping the chord vocabulary intact costs nothing, which means the model
is reading the chord list and ignoring the progression -- and the progression is
most of what a chart is.

**Random chords in no key score 5.80 out of 10.** On any honest scale that is a
zero. The entire dynamic range between a correct transcription and arbitrary
noise is two points, the same distance that separates two correct
transcriptions from each other.

**Semitone noise and pure randomness are indistinguishable** at 6.00 and 5.80,
though one is a mostly-correct chart with some slipped chords and the other is
gibberish. A grader that cannot tell those apart cannot rank anything.

The `out_of_key` count is the one weakly responsive signal -- 0 to 2 on real
charts, 3 to 4 on random ones -- but it correlates -0.014 with actual chord F1,
so it does not predict the thing we care about. Counting out-of-key chords does
not need a 9.6 GB model; `src/simplify.py` already does it against the detected
key, deterministically and instantly.

## Verdict

Blind to structure, and its response to corruption is too compressed to act on.
This is the third proxy in this repo to look reasonable and measure nothing:

  bench/compare_audio.py   renders output back to audio     corr -0.044 with F1
  bench/HPSS_RESULT.md     harmonic/percussive separation   -0.092 F1
  this file                local LLM plausibility audit     corr +0.148, blind
                                                            to shuffled input

The pattern is consistent enough to be worth naming: a check that never sees
ground truth tends to measure its own priors. The things that have actually
moved numbers here all compare against something external -- GuitarSet's
per-string onsets, the 55-song chord reference, mir_eval's SDR.

Both scripts are kept, as compare_audio.py was, so the result stays
reproducible. Neither belongs in the pipeline.
