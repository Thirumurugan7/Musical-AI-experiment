"""
Build the "detected grid vs real audio" page for any song.

    python3 src/build_page.py -o work/riptide.html \
        -c "Full-mix detection=work/riptide.json" \
        -c "Stem detection=work/riptide_stem.json" \
        -a "Full mix=riptide.mp3" \
        -a "Guitar stem=riptide_stems/htdemucs_6s/riptide/guitar.mp3" \
        -l work/lyrics_riptide.lrc

Audio paths are written into the page as-is, so they must be relative to the
output file's own directory. Chart JSONs are the output of `transcribe.py -o`.

The point of the page is that audio source and marks source are independent:
you can play the real recording while watching what a stem-based detection
claimed, which is what makes a wrong mark visible instead of merely numeric.
"""
import argparse
import json
import os
import re


def load_chart(path):
    d = json.load(open(path))
    bars = d["bars"]
    if not d.get("bar_len"):
        # transcribe.py does not always set it; derive from the bar spacing
        gaps = [bars[i + 1]["start"] - bars[i]["start"] for i in range(len(bars) - 1)]
        gaps.sort()
        d["bar_len"] = round(gaps[len(gaps) // 2], 3) if gaps else 0.0
    d.setdefault("subdiv", 3)
    d.setdefault("beats_per_bar", 4)
    return d


def load_lyrics(path):
    out = []
    if not path or not os.path.isfile(path):
        return out
    for raw in open(path, encoding="utf-8"):
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)", raw)
        if m and m.group(3).strip():
            out.append({"t": int(m.group(1)) * 60 + float(m.group(2)),
                        "x": m.group(3).strip()})
    out.sort(key=lambda r: r["t"])
    return out


def pair(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected LABEL=VALUE, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), v.strip()


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — grid vs audio</title>
<style>
  :root{--bg:#14110e;--card:#1e1a16;--line:#332c25;--ink:#f0e9e0;--dim:#a2968a;
    --gold:#d9a441;--blue:#7fb3d5;--green:#8fbf7f;--red:#d97a6c}
  *{box-sizing:border-box}
  body{margin:0;padding:28px 20px 60px;background:var(--bg);color:var(--ink);
    font:16px/1.5 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif}
  .wrap{max-width:1100px;margin:0 auto}
  h1{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}
  .sub{color:var(--dim);margin:0 0 22px;font-size:15px}
  .hdr{display:flex;gap:26px;flex-wrap:wrap;margin-bottom:22px}
  .hdr div{font-size:13px;color:var(--dim)}
  .hdr b{display:block;font-size:19px;color:var(--ink);font-weight:600}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
    padding:20px 22px;margin-bottom:16px}
  .ctl{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .lbl{font:600 11px/1 ui-monospace,Menlo,monospace;letter-spacing:.08em;
    text-transform:uppercase;color:var(--dim);margin-right:2px}
  button{font:inherit;font-size:14px;padding:9px 15px;border-radius:8px;cursor:pointer;
    border:1px solid var(--line);background:#26211c;color:var(--ink)}
  button:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
  button.on{background:var(--blue);color:#06202e;border-color:var(--blue);font-weight:600}
  button.play{background:var(--gold);color:#201703;border-color:var(--gold);
    font-weight:600;min-width:104px}
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;
    flex-wrap:wrap}
  .seg button{border:0;border-radius:0;border-right:1px solid var(--line)}
  .seg button:last-child{border-right:0}
  #seek{width:100%;margin:16px 0 4px;accent-color:var(--gold)}
  .times{display:flex;justify-content:space-between;
    font:13px ui-monospace,Menlo,monospace;color:var(--dim)}
  .now{display:flex;align-items:baseline;gap:16px;margin-bottom:4px}
  .chord{font-size:44px;font-weight:700;letter-spacing:-.02em;line-height:1}
  .snd{color:var(--dim);font-size:15px}
  .barno{margin-left:auto;color:var(--dim);font:13px ui-monospace,Menlo,monospace}
  .grid,.cnt{display:grid;gap:6px}
  .cnt{font:12px ui-monospace,Menlo,monospace;color:var(--dim);margin-bottom:6px}
  .cnt span{text-align:center}
  .cnt span.beat{color:var(--ink);font-weight:700}
  .grid{margin-top:14px}
  .cell{aspect-ratio:1/1.15;border-radius:8px;display:flex;align-items:center;
    justify-content:center;font:700 17px ui-monospace,Menlo,monospace;
    border:1px solid var(--line);background:#191512;color:var(--dim);
    transition:transform .06s linear}
  .cell.accent{background:var(--gold);color:#201703;border-color:var(--gold)}
  .cell.normal{background:#4a3f33;color:var(--ink);border-color:#5c4d3e}
  .cell.ghost{background:#26211c;color:var(--dim)}
  .cell.rest{background:#131110;color:#4a423a}
  .cell.hit{outline:3px solid var(--green);outline-offset:2px;transform:scale(1.08)}
  .next{margin-top:16px;font:13px/1.9 ui-monospace,Menlo,monospace;color:var(--dim)}
  .next b{color:var(--ink)}
  .chip{font:12px ui-monospace,Menlo,monospace;background:#26211c;
    border:1px solid var(--line);padding:5px 10px;border-radius:20px;color:var(--dim)}
  .note{color:var(--dim);font-size:14px}
  .note strong{color:var(--ink)}
  .warn{border-left:4px solid var(--red)}
  .lyr{font-size:17px;line-height:1.7}
  .lyr.prev,.lyr.next2{color:#6d6259;font-size:15px}
  .lyr.cur{color:var(--ink);font-weight:600;font-size:21px}
  @media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="wrap">
  <h1 id="title">—</h1>
  <p class="sub">The chart runs in sync with the recording. Watch whether the marks
     land where you hear the strums.</p>

  <div class="hdr">
    <div>SOUNDS IN<b id="h-key"></b></div>
    <div>CAPO<b id="h-capo"></b></div>
    <div>BPM<b id="h-bpm"></b></div>
    <div>METRE<b id="h-metre"></b></div>
    <div>OCCUPANCY<b id="h-occ"></b></div>
  </div>

  <div class="card">
    <div class="ctl" style="margin-bottom:14px">
      <span class="lbl">Audio</span><span class="seg" id="src"></span>
    </div>
    <div class="ctl">
      <span class="lbl">Marks from</span><span class="seg" id="marks"></span>
      <button id="click">Click on accents: off</button>
    </div>
    <input type="range" id="seek" min="0" max="100" step="0.05" value="0" aria-label="seek">
    <div class="times"><span id="t-now">0:00</span><span id="t-end">0:00</span></div>
    <div class="ctl" style="margin-top:12px">
      <button class="play" id="play">&#9654; Play</button>
      <button id="back">&larr; 1 bar</button>
      <button id="fwd">1 bar &rarr;</button>
      <span class="chip" id="chip"></span>
    </div>
  </div>

  <div class="card">
    <div class="now">
      <span class="chord" id="chord">&mdash;</span>
      <span class="snd" id="snd"></span>
      <span class="barno" id="barno"></span>
    </div>
    <div class="cnt" id="cnt"></div>
    <div class="grid" id="grid"></div>
    <div class="next" id="next"></div>
  </div>

  <div class="card" id="strumCard">
    <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:12px;flex-wrap:wrap">
      <span class="lbl">Strumming pattern</span>
      <span class="chip" id="strum-meta"></span>
    </div>
    <div class="ctl" style="margin-bottom:14px">
      <span class="seg" id="sect"></span>
    </div>
    <div class="cnt" id="s-cnt"></div>
    <div class="grid" id="s-grid"></div>
    <div class="ctl" style="margin-top:16px">
      <button class="play" id="s-play">&#9654; Play pattern</button>
      <label class="lbl" for="s-tempo">Tempo</label>
      <input type="range" id="s-tempo" min="40" max="140" value="100" step="1"
             style="width:150px;accent-color:var(--gold)">
      <span class="chip" id="s-tempo-v">100%</span>
      <span class="chip" id="s-chords"></span>
    </div>
    <p class="note" style="margin:14px 0 0">Loops one bar. Accents are louder and
      brighter; ghost strokes are the quiet brushes between them. This is the
      detected pattern alone &mdash; no recording underneath.</p>
  </div>

  <div class="card" id="lyrCard" style="display:none">
    <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:10px">
      <span class="lbl">Lyrics</span><span class="chip" id="lyr-meta"></span>
    </div>
    <div class="lyr prev" id="ly-p"></div>
    <div class="lyr cur" id="ly-c">&mdash;</div>
    <div class="lyr next2" id="ly-n"></div>
  </div>

  <div class="card warn">
    <p class="note" style="margin:0 0 10px"><strong>What you are looking for.</strong>
      The green outline steps through the grid in time with the audio. Turn on
      <strong>Click on accents</strong> and listen for whether the clicks coincide with
      real strums or land on silence.</p>
    <p class="note" style="margin:0"><strong>Trustworthy:</strong> chord changes, bar
      boundaries, tempo. <strong>Not trustworthy:</strong> the metre, and which
      subdivisions are marked struck &mdash; both come from a threshold that normalises
      against its own data. The arrows are stroke <em>direction</em>, never measured.</p>
  </div>
</div>

<script id="chartdata" type="application/json">__CHARTS__</script>
<script id="lyricdata" type="application/json">__LYRICS__</script>
<script id="audiodata" type="application/json">__AUDIO__</script>
<script>
const DATA   = JSON.parse(document.getElementById("chartdata").textContent);
const LYRICS = JSON.parse(document.getElementById("lyricdata").textContent);
const SRCS   = JSON.parse(document.getElementById("audiodata").textContent);
const $ = id => document.getElementById(id);
const audio = new Audio(); audio.preload = "metadata";
let markKey = Object.keys(DATA)[0], clickOn = false, actx = null, lastSlot = -2;

const fmt = s => isFinite(s)
  ? Math.floor(s/60) + ":" + String(Math.floor(s%60)).padStart(2,"0") : "0:00";

function segButtons(host, entries, onPick){
  entries.forEach(([label, val], i) => {
    const b = document.createElement("button");
    b.textContent = label; b.dataset.v = val;
    if (i === 0) b.classList.add("on");
    b.addEventListener("click", () => {
      [...host.children].forEach(x => x.classList.remove("on"));
      b.classList.add("on"); onPick(val);
    });
    host.appendChild(b);
  });
}

function nSlots(d){ return d.subdiv * d.beats_per_bar; }

function buildGrid(){
  const d = DATA[markKey], n = nSlots(d);
  const cnt = $("cnt"), grid = $("grid");
  cnt.innerHTML = ""; grid.innerHTML = "";
  cnt.style.gridTemplateColumns = `repeat(${n},1fr)`;
  grid.style.gridTemplateColumns = `repeat(${n},1fr)`;
  for (let i = 0; i < n; i++){
    const s = document.createElement("span");
    const inBeat = i % d.subdiv;
    s.textContent = inBeat === 0 ? String(i / d.subdiv + 1)
                  : (d.subdiv === 3 ? "." : "+");
    if (inBeat === 0) s.className = "beat";
    cnt.appendChild(s);
    const c = document.createElement("div");
    c.className = "cell"; c.id = "c" + i;
    grid.appendChild(c);
  }
}

function meta(){
  const d = DATA[markKey];
  $("title").textContent = d.title || "untitled";
  document.title = (d.title || "chart") + " — grid vs audio";
  $("h-key").textContent = d.sounding_key;
  $("h-capo").textContent = d.capo + " → " + d.shape_key;
  $("h-bpm").textContent = d.tempo_bpm;
  $("h-metre").textContent = d.metre;
  $("h-occ").textContent = Math.round(d.grid_occupancy * 100) + "%";
  $("chip").textContent = d.onset_source + " · " + d.n_bars + " bars";
  buildGrid(); lastSlot = -2;
}

const tokens = p => p.trim().split(/\s+/);
function kind(t){
  if (t === "·") return "rest";
  if (t.startsWith("(")) return "ghost";
  return t.replace(/[()]/g,"") === t.replace(/[()]/g,"").toUpperCase() ? "accent" : "normal";
}
function barAt(t){
  const b = DATA[markKey].bars;
  for (let i = b.length - 1; i >= 0; i--) if (t >= b[i].start) return i;
  return -1;
}

function paint(i){
  const d = DATA[markKey], b = d.bars[i], n = nSlots(d);
  if (!b) return;
  $("chord").textContent = b.shape;
  $("snd").textContent = b.sounding && b.sounding !== b.shape ? "sounds " + b.sounding : "";
  $("barno").textContent = "bar " + (i+1) + " / " + d.bars.length;
  const tk = tokens(b.pattern);
  for (let k = 0; k < n; k++){
    const el = $("c"+k); if (!el) continue;
    const t = tk[k] || "·";
    el.className = "cell " + kind(t);
    el.textContent = t === "·" ? "·" : t.replace(/[()]/g,"");
  }
  $("next").innerHTML = d.bars.slice(i+1, i+4)
    .map(x => "<b>" + x.shape + "</b> " + x.pattern).join("<br>");
}

function click(accent){
  if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
  const o = actx.createOscillator(), g = actx.createGain(), t = actx.currentTime;
  o.frequency.value = accent ? 1600 : 1050;
  g.gain.setValueAtTime(accent ? 0.5 : 0.22, t);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.04);
  o.connect(g); g.connect(actx.destination); o.start(t); o.stop(t + 0.05);
}

let lastLyric = -2;
function paintLyrics(t){
  if (!LYRICS.length) return;
  let i = -1;
  for (let k = 0; k < LYRICS.length; k++){ if (t >= LYRICS[k].t) i = k; else break; }
  if (i === lastLyric) return;
  lastLyric = i;
  $("ly-p").textContent = i > 0 ? LYRICS[i-1].x : "";
  $("ly-c").textContent = i >= 0 ? LYRICS[i].x : "—";
  $("ly-n").textContent = LYRICS[i+1] ? LYRICS[i+1].x : "";
}

function update(){
  const t = audio.currentTime, d = DATA[markKey], n = nSlots(d);
  paintLyrics(t);
  $("t-now").textContent = fmt(t);
  if (audio.duration){
    $("t-end").textContent = fmt(audio.duration);
    $("seek").value = t / audio.duration * 100;
  }
  const i = barAt(t);
  if (i < 0) return;
  paint(i);
  const b = d.bars[i];
  const nx = d.bars[i+1] ? d.bars[i+1].start : b.start + d.bar_len;
  const slot = Math.max(0, Math.min(n-1, Math.floor((t - b.start) / ((nx - b.start)/n))));
  for (let k = 0; k < n; k++){
    const el = $("c"+k); if (el) el.classList.toggle("hit", k === slot);
  }
  const key = i * n + slot;
  if (key !== lastSlot){
    lastSlot = key;
    const tk = tokens(b.pattern)[slot] || "·";
    if (clickOn && !audio.paused && tk !== "·") click(kind(tk) === "accent");
  }
}

// rAF gets zero ticks in a hidden tab, so back it with media events
function frame(){ update(); requestAnimationFrame(frame); }
["timeupdate","seeked","loadeddata"].forEach(e => audio.addEventListener(e, update));

function setSrc(url){
  const was = audio.currentTime, playing = !audio.paused;
  audio.src = url;
  audio.addEventListener("loadedmetadata", function h(){
    audio.currentTime = was; if (playing) audio.play();
    audio.removeEventListener("loadedmetadata", h);
  });
}

/* ---------- strumming pattern player ---------- */
/* A strum is a burst of broadband noise shaped by a bandpass — closer to what
   a pick across six strings actually sounds like than a tone would be. Accent,
   normal and ghost differ in gain, brightness and decay, which is what makes
   the pattern legible by ear rather than a flat row of ticks. */
let sectIdx = 0, loopTimer = null, loopStart = 0, nextSlot = 0, playing = false;
let noiseBuf = null;

function patterns(){
  const d = DATA[markKey];
  const list = (d.sections || []).map((s, i) => ({
    label: "Section " + (i+1), pattern: s.pattern,
    chords: (s.chords || []).join(" "), from: s.from_bar, to: s.to_bar
  }));
  list.push({ label: "Whole song", pattern: d.canonical_pattern,
              chords: (d.chords_shapes || []).slice(0,6).join(" "),
              from: 1, to: d.n_bars });
  return list;
}

function ensureNoise(){
  if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
  if (noiseBuf) return;
  const n = Math.floor(actx.sampleRate * 0.4);
  noiseBuf = actx.createBuffer(1, n, actx.sampleRate);
  const ch = noiseBuf.getChannelData(0);
  for (let i = 0; i < n; i++) ch[i] = Math.random() * 2 - 1;
}

function strum(at, k, up){
  ensureNoise();
  const gains  = { accent: 0.5,  normal: 0.26, ghost: 0.1 };
  const cutoff = { accent: 2400, normal: 1700, ghost: 1100 };
  const decay  = { accent: 0.34, normal: 0.24, ghost: 0.11 };
  const src = actx.createBufferSource(); src.buffer = noiseBuf;
  const bp = actx.createBiquadFilter();
  bp.type = "bandpass"; bp.frequency.value = cutoff[k]; bp.Q.value = 0.7;
  // an upstroke catches the thin strings first — a touch brighter
  if (up) bp.frequency.value *= 1.35;
  const g = actx.createGain();
  g.gain.setValueAtTime(0.0001, at);
  g.gain.exponentialRampToValueAtTime(gains[k], at + 0.006);
  g.gain.exponentialRampToValueAtTime(0.0001, at + decay[k]);
  src.connect(bp); bp.connect(g); g.connect(actx.destination);
  src.start(at); src.stop(at + decay[k] + 0.02);
}

function drawPattern(){
  const d = DATA[markKey], p = patterns()[sectIdx], n = nSlots(d);
  const tk = tokens(p.pattern);
  const cnt = $("s-cnt"), grid = $("s-grid");
  cnt.innerHTML = ""; grid.innerHTML = "";
  cnt.style.gridTemplateColumns = `repeat(${n},1fr)`;
  grid.style.gridTemplateColumns = `repeat(${n},1fr)`;
  for (let i = 0; i < n; i++){
    const s = document.createElement("span");
    const ib = i % d.subdiv;
    s.textContent = ib === 0 ? String(i / d.subdiv + 1) : (d.subdiv === 3 ? "." : "+");
    if (ib === 0) s.className = "beat";
    cnt.appendChild(s);
    const t = tk[i] || "·";
    const c = document.createElement("div");
    c.className = "cell " + kind(t); c.id = "s" + i;
    c.textContent = t === "·" ? "·" : t.replace(/[()]/g, "");
    grid.appendChild(c);
  }
  $("s-chords").textContent = p.chords ? "chords: " + p.chords : "";
  $("strum-meta").textContent =
    `${d.metre} · bars ${p.from}–${p.to} · ${d.tempo_bpm} BPM`;
}

function slotSeconds(){
  const d = DATA[markKey];
  const rate = parseInt($("s-tempo").value, 10) / 100;
  return (d.bar_len / nSlots(d)) / rate;
}

function tick(){
  const d = DATA[markKey], n = nSlots(d);
  const tk = tokens(patterns()[sectIdx].pattern);
  const dur = slotSeconds();
  while (loopStart + nextSlot * dur < actx.currentTime + 0.15){
    const i = nextSlot % n;
    const t = tk[i] || "·", k = kind(t);
    const at = loopStart + nextSlot * dur;
    if (k !== "rest") strum(at, k, t.replace(/[()]/g,"").toLowerCase() === "u");
    const delay = Math.max(0, (at - actx.currentTime) * 1000);
    setTimeout(() => {
      for (let j = 0; j < n; j++){
        const el = $("s"+j); if (el) el.classList.toggle("hit", j === i);
      }
    }, delay);
    nextSlot++;
  }
}

$("s-play").addEventListener("click", () => {
  ensureNoise();
  if (playing){
    playing = false; clearInterval(loopTimer); loopTimer = null;
    $("s-play").innerHTML = "&#9654; Play pattern";
    document.querySelectorAll("#s-grid .cell").forEach(c => c.classList.remove("hit"));
    return;
  }
  if (actx.state === "suspended") actx.resume();
  playing = true; nextSlot = 0; loopStart = actx.currentTime + 0.1;
  $("s-play").innerHTML = "&#9632; Stop";
  tick(); loopTimer = setInterval(tick, 25);
});
$("s-tempo").addEventListener("input", e => {
  $("s-tempo-v").textContent = e.target.value + "%";
  if (playing){ nextSlot = 0; loopStart = actx.currentTime + 0.05; }
});

segButtons($("src"), SRCS, setSrc);
segButtons($("marks"), Object.keys(DATA).map(k => [k, k]),
           k => { markKey = k; meta(); buildSections(); });

function buildSections(){
  const host = $("sect"); host.innerHTML = ""; sectIdx = 0;
  segButtons(host, patterns().map((p, i) => [p.label, String(i)]), v => {
    sectIdx = parseInt(v, 10); drawPattern();
    if (playing){ nextSlot = 0; loopStart = actx.currentTime + 0.05; }
  });
  drawPattern();
}

$("click").addEventListener("click", e => {
  clickOn = !clickOn;
  e.target.textContent = "Click on accents: " + (clickOn ? "on" : "off");
  e.target.classList.toggle("on", clickOn);
});
$("play").addEventListener("click", () => {
  if (audio.paused){ audio.play(); $("play").innerHTML = "&#10074;&#10074; Pause"; }
  else { audio.pause(); $("play").innerHTML = "&#9654; Play"; }
});
$("seek").addEventListener("input", e => {
  if (audio.duration) audio.currentTime = e.target.value / 100 * audio.duration;
});
$("back").addEventListener("click", () => {
  const i = barAt(audio.currentTime), b = DATA[markKey].bars;
  if (i > 0) audio.currentTime = b[i-1].start;
});
$("fwd").addEventListener("click", () => {
  const i = barAt(audio.currentTime), b = DATA[markKey].bars;
  if (i >= 0 && b[i+1]) audio.currentTime = b[i+1].start;
});
audio.addEventListener("ended", () => { $("play").innerHTML = "&#9654; Play"; });

if (LYRICS.length){
  $("lyrCard").style.display = "";
  $("lyr-meta").textContent = LYRICS.length + " lines · from the vocal stem";
}
meta(); buildSections(); if (SRCS.length) setSrc(SRCS[0][1]); frame();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="build the grid-vs-audio page")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-c", "--chart", action="append", required=True, type=pair,
                    metavar="LABEL=path.json",
                    help="a transcribe.py -o JSON; repeat for several detections")
    ap.add_argument("-a", "--audio", action="append", required=True, type=pair,
                    metavar="LABEL=relative/path.mp3",
                    help="audio source, path relative to the output file")
    ap.add_argument("-l", "--lyrics", help="optional .lrc")
    a = ap.parse_args()

    charts = {label: load_chart(path) for label, path in a.chart}
    lyrics = load_lyrics(a.lyrics)
    title = next(iter(charts.values())).get("title", "chart")

    html = (TEMPLATE
            .replace("__CHARTS__", json.dumps(charts, ensure_ascii=False))
            .replace("__LYRICS__", json.dumps(lyrics, ensure_ascii=False))
            .replace("__AUDIO__", json.dumps(a.audio, ensure_ascii=False))
            .replace("__TITLE__", title))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"wrote {a.out}")
    print(f"  charts : {', '.join(charts)}")
    print(f"  audio  : {', '.join(l for l, _ in a.audio)}")
    print(f"  lyrics : {len(lyrics)} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
