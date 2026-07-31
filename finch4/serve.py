"""Live reporting server for the agentic substrate: agents tell the
engine directly instead of relaying results through the orchestrator.

    python3 -m finch4.serve --run benchmarks/agentic/runs/r2 \
        --tasks binpack tsp            # creates state.json if absent

One process holds the ONE AgenticGA instance; a lock serializes every
request, so concurrent agents can report the moment they finish without
racing each other or a shared state file (the file-relay pattern is only
safe because the orchestrator is the single writer — this server is the
single writer with the relay removed). State is saved to the run's
state.json after every mutating call, so a crash loses nothing.

The server binds localhost only and writes `server.json` (port, pid)
into the run directory so agents can discover it. JSON in, JSON out:

    GET  /summary               engine summary
    GET  /due                   {"due": bool} — consolidation due?
    GET  /batch                 per-task best-evers (consolidation input)
    GET  /stale                 individuals whose score predates the base
    GET  /contradictions        contradiction report per task
    POST /ask                   {"n": optional} -> jobs for one round
    POST /tell                  {job_id, variation, score, artifact?,
                                 contradicts_base?, log?} -> {"id": ...}
    POST /abandon               {job_id}
    POST /consolidated          {} -> survivors owing rewrites
                                (call AFTER editing the base playbook)
    POST /rewrite               {id, variation, contradicts_base?}
    POST /rescore               {id, score, artifact?}
    POST /audit                 {id, passed}
    POST /media                 {name, kind: image|svg|text, data,
                                 epoch?, evaluations?} — see below

An agent reports its own result with one line:

    curl -s -X POST localhost:PORT/tell -d '{"job_id": "j0004", ...}'

MEDIA is the dashboard's second channel: alongside the fitness curves, a
run may post what it is evolving — latest-wins per name, images as PNG
data URIs (png_data_uri encodes any numpy image array with stdlib only),
SVG and text inline. The latest item per name is mirrored to
run_dir/media/, so finished runs keep their media and the hub shows it
on the run's card. Producers: live_progress(images="auto") posts each
function's best-ever phenotype automatically when it is image-shaped;
progress.report_media / media_client(run_dir) post anything else.
Media is telemetry for the eyes, NEVER evidence: a picture is not a
score, and the canonical scorer and the audit path remain the only
sources of truth.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import struct
import threading
import time
import zlib
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .agentic import AgenticGA

PALETTE = ["#7ac", "#c96", "#9c7", "#b8a", "#8cc", "#ca8"]


# ----------------------------------------------------------------- media
#
# The dashboard's second channel: what a run is evolving, not just how
# well. Latest-wins per name, bounded, mirrored to run_dir/media/ so
# finished runs keep it. Telemetry for the eyes, never evidence.

MEDIA_EXT = {"image": ".png", "svg": ".svg", "text": ".txt"}
MEDIA_MAX_BYTES = 256 * 1024      # per item, encoded
MEDIA_MAX_NAMES = 16              # per run
FILMSTRIP_FRAMES = 10             # in-memory trajectory, images only


def _as_rgb8(array) -> np.ndarray:
    """Any image-ish numpy array -> (H, W, C) uint8. Accepts grayscale
    (H, W), channels-last (H, W, C<=4), channels-first (C<=4, H, W);
    floats are read in [0, 1], uint8 passes through."""
    a = np.asarray(array)
    if a.ndim == 3 and a.shape[0] <= 4 and a.shape[-1] > 4:
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = np.rint(np.clip(a.astype(np.float64), 0.0, 1.0)
                    * 255).astype(np.uint8)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3 or a.shape[-1] not in (1, 3, 4):
        raise ValueError(f"cannot render shape {np.asarray(array).shape} "
                         "as an image")
    return np.ascontiguousarray(a)


def _png_bytes(rgb8: np.ndarray) -> bytes:
    """Minimal PNG encoder, stdlib only (zlib + struct): 8-bit
    grayscale/RGB/RGBA, no filtering — keeps the no-dependency rule
    (PIL never enters the library)."""
    h, w, c = rgb8.shape
    color_type = {1: 0, 3: 2, 4: 6}[c]

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data)))

    header = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    raw = b"".join(b"\x00" + rgb8[y].tobytes() for y in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def png_data_uri(array) -> str:
    """Encode a numpy image array as a PNG data URI (the wire format of
    media kind "image"). Layouts and value ranges as _as_rgb8."""
    return ("data:image/png;base64,"
            + base64.b64encode(_png_bytes(_as_rgb8(array))).decode())


def looks_like_image(shape) -> bool:
    """The images="auto" test: could a phenotype of this shape be
    rendered? 2-D grids and 3-D arrays with a small channel dim."""
    if len(shape) == 2:
        return min(shape) >= 2
    if len(shape) == 3:
        return shape[0] <= 4 or shape[-1] <= 4
    return False


def _media_body(name, image=None, svg=None, text=None, epoch=None,
                evaluations=None):
    """One media item as a /media request body; exactly one of image=
    (numpy array), svg= (inline markup) or text= must be given."""
    if image is not None:
        kind, data = "image", png_data_uri(image)
    elif svg is not None:
        kind, data = "svg", str(svg)
    elif text is not None:
        kind, data = "text", str(text)
    else:
        raise ValueError("pass image=, svg= or text=")
    return {"name": str(name), "kind": kind, "data": data,
            "epoch": epoch, "evaluations": evaluations}


def registry_path():
    """Global registry of every run this machine has served — one JSON
    line per server start. The hub (hub.py) reads it to show ALL
    evolution jobs, live and finished, on one page."""
    return (os.environ.get("FINCH4_REGISTRY")
            or os.path.expanduser("~/.finch4/registry.jsonl"))


def register_run(run_dir, port):
    path = registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"run_dir": os.path.abspath(run_dir),
                            "port": port, "pid": os.getpid(),
                            "started": time.time()}) + "\n")


def curve_svg(series, points=None, xlabel="evaluations", w=720, h=260,
              labels=True):
    """Fitness-over-time chart, dependency-free. series maps a name to
    its best-so-far [(x, y), ...] step curve — one line per task or
    fitness function, same rendering for every kind of run. points are
    optional (x, y, label) markers (agentic individuals)."""
    series = {k: v for k, v in series.items() if len(v) >= 2}
    if not series:
        return "<p>curve appears after two scored points</p>"
    allx = [x for c in series.values() for x, _ in c]
    ally = [y for c in series.values() for _, y in c]
    if points:
        ally += [p[1] for p in points]
    lo, hi = min(ally), max(ally)
    pad = max((hi - lo) * 0.15, 1e-9)
    lo, hi = lo - pad, hi + pad
    ML, MB = (62, 28) if labels else (8, 8)
    xmax = max(allx) * 1.05 + 1e-9

    def X(x):
        return ML + (w - ML - 14) * x / xmax

    def Y(y):
        return 12 + (h - 12 - MB) * (hi - y) / (hi - lo)

    s = [f'<svg width="{w}" height="{h}" style="background:#161616;'
         'border:1px solid #333">']
    if labels:
        s.append(f'<text x="{ML}" y="{h-8}" font-size="10" fill="#888">'
                 f'x &#8212; {xlabel} &#183; y &#8212; best score so far '
                 '(higher is better)</text>')
    for i, (name, curve) in enumerate(sorted(series.items())):
        color = PALETTE[i % len(PALETTE)]
        path = ""
        for j, (x, y) in enumerate(curve):
            path += (f"M {X(x):.1f} {Y(y):.1f} " if j == 0
                     else f"L {X(x):.1f} {Y(curve[j-1][1]):.1f} "
                          f"L {X(x):.1f} {Y(y):.1f} ")
        s.append(f'<path d="{path}" fill="none" stroke="{color}" '
                 'stroke-width="2"/>')
        if labels:
            lx, ly = curve[-1]
            s.append(f'<text x="{X(lx)-4:.1f}" y="{Y(ly)-6:.1f}" '
                     f'font-size="10" fill="{color}" text-anchor="end">'
                     f'{html.escape(str(name))} {ly:.5g}</text>')
    if labels:
        for x, y, lab in (points or []):
            s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" '
                     'fill="#666"/>')
            s.append(f'<text x="{X(x)+5:.1f}" y="{Y(y)+11:.1f}" '
                     f'font-size="8.5" fill="#777">{lab}</text>')
        for gy in (lo + pad, hi - pad):
            s.append(f'<text x="{ML-6}" y="{Y(gy)+3:.1f}" font-size="9" '
                     f'fill="#888" text-anchor="end">{gy:.5g}</text>')
    s.append("</svg>")
    return "".join(s)


def agentic_curves(individuals):
    """Per-task best-ever step curves over tell order; disqualified
    scores (<= -90) count on the x axis but not in the curves."""
    series, points, best, n = {}, [], {}, 0
    for ind in sorted(individuals, key=lambda i: i["id"]):
        n += 1
        if ind["score"] <= -90:
            continue
        t = ind["task"]
        points.append((n, ind["score"], ind["id"]))
        if t not in best or ind["score"] > best[t]:
            best[t] = ind["score"]
        series.setdefault(t, []).append((n, best[t]))
    return series, points


def telemetry_curves(telemetry):
    series = {}
    for point in telemetry:
        x = point.get("evaluations") or point.get("epoch") or 0
        for name, score in (point.get("best") or {}).items():
            if score is None:
                continue
            cur = series.setdefault(name, [])
            y = max(score, cur[-1][1]) if cur else score
            cur.append((x, y))
    return series


class GAService:
    """The engine plus the lock and the save-after-every-mutation rule.

    Also the ONE telemetry sink for every kind of run: agentic runs
    feed it through tell/audit/consolidated, and the tensor solver
    feeds it through POST /telemetry (see live_progress below), so the
    /progress dashboard is the same page for every evolutionary problem
    this library runs. ga may be None (telemetry-only mode)."""

    def __init__(self, ga, state_path, run_dir=None):
        self.ga = ga
        self.state_path = state_path
        self.run_dir = run_dir
        self.lock = threading.Lock()
        self.events = deque(maxlen=300)
        self.telemetry = []
        self.media = {}          # name -> latest item (kind, data, epoch)
        self.filmstrips = {}     # name -> deque of recent image frames
        self.started = time.time()

    def event(self, text):
        self.events.append((time.strftime("%H:%M:%S"), text))

    def call(self, name, body):
        ga = self.ga
        with self.lock:
            if name == "telemetry":
                point = {"epoch": body.get("epoch"),
                         "evaluations": body.get("evaluations"),
                         "best": body.get("best", {})}
                self.telemetry.append(point)
                if self.run_dir:
                    with open(os.path.join(self.run_dir,
                                           "telemetry.jsonl"), "a") as f:
                        f.write(json.dumps(point) + "\n")
                return {"ok": True}, False
            if name == "media":
                return self._store_media(body), False
            if name == "page.json":
                return self._page_data(), False
            if ga is None:
                if name == "summary":
                    last = self.telemetry[-1] if self.telemetry else {}
                    return {"telemetry_points": len(self.telemetry),
                            "best": last.get("best", {})}, False
                raise KeyError(f"{name} needs an engine (telemetry-only "
                               "server)")
            if name == "summary":
                return ga.summary(), False
            if name == "due":
                return {"due": ga.consolidation_due()}, False
            if name == "batch":
                return ga.consolidation_batch(), False
            if name == "stale":
                return ga.stale(), False
            if name == "contradictions":
                return ga.contradiction_report(), False
            if name == "ask":
                return ga.ask(), True
            if name == "tell":
                ind = ga.tell(body["job_id"], body["variation"],
                              body["score"], artifact=body.get("artifact"),
                              contradicts_base=body.get(
                                  "contradicts_base", False),
                              log=body.get("log"),
                              fresh_start=body.get("fresh_start", False))
                return {"id": ind}, True
            if name == "abandon":
                ga.abandon(body["job_id"])
                return {"ok": True}, True
            if name == "consolidated":
                return ga.record_consolidation(), True
            if name == "rewrite":
                ga.tell_rewrite(body["id"], body["variation"],
                                contradicts_base=body.get(
                                    "contradicts_base"))
                return {"ok": True}, True
            if name == "rescore":
                ga.retell_score(body["id"], body["score"],
                                artifact=body.get("artifact"))
                return {"ok": True}, True
            if name == "audit":
                ga.mark_audited(body["id"], body.get("passed", True))
                return {"ok": True}, True
            raise KeyError(name)

    def handle(self, name, body):
        result, mutated = self.call(name, body)
        if mutated:
            self.ga.save(self.state_path)
            if name == "tell":
                self.event(f"tell {result['id']} score={body['score']:.5f}"
                           f" ({body.get('variation', '')[:70]}...)")
            elif name == "ask":
                self.event(f"ask: {len(result)} jobs "
                           f"({', '.join(j['kind'] for j in result)})")
            elif name == "consolidated":
                self.event(f"CONSOLIDATED -> base v"
                           f"{self.ga.base_version}; "
                           f"{len(result)} rewrites owed")
            elif name == "rewrite":
                self.event(f"rewrite {body['id']}")
            elif name == "audit":
                self.event(f"audit {body['id']}: "
                           f"{'pass' if body.get('passed', True) else 'FAIL'}")
            elif name == "abandon":
                self.event(f"abandon {body['job_id']}")
        return result

    def _store_media(self, body):
        """Latest-wins media telemetry: display only, never evidence — a
        picture is not a score. Bounded: MEDIA_MAX_NAMES names,
        MEDIA_MAX_BYTES per item; images also feed a short in-memory
        filmstrip (the trajectory), and the latest item per name is
        mirrored to run_dir/media/ (what finished runs and the hub
        read)."""
        name = str(body.get("name") or "media")
        kind = body.get("kind")
        data = body.get("data")
        if kind not in MEDIA_EXT:
            raise ValueError(f"unknown media kind {kind!r}; "
                             f"kinds: {sorted(MEDIA_EXT)}")
        if not isinstance(data, str):
            raise ValueError("media data must be a string (image: a PNG "
                             "data URI — see png_data_uri; svg/text: the "
                             "content itself)")
        if len(data) > MEDIA_MAX_BYTES:
            raise ValueError(f"media item {name!r} over "
                             f"{MEDIA_MAX_BYTES} bytes")
        if name not in self.media and len(self.media) >= MEDIA_MAX_NAMES:
            raise ValueError(f"over {MEDIA_MAX_NAMES} media names; "
                             "reuse a name (latest wins)")
        self.media[name] = {"kind": kind, "data": data,
                            "epoch": body.get("epoch"),
                            "evaluations": body.get("evaluations")}
        if kind == "image":
            strip = self.filmstrips.setdefault(
                name, deque(maxlen=FILMSTRIP_FRAMES))
            strip.append(data)
        if self.run_dir:
            media_dir = os.path.join(self.run_dir, "media")
            os.makedirs(media_dir, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
            content = (base64.b64decode(data.split(",", 1)[1])
                       if kind == "image" else data.encode())
            with open(os.path.join(media_dir, safe + MEDIA_EXT[kind]),
                      "wb") as f:
                f.write(content)
        return {"ok": True}

    # ------------------------------------------------------ progress page

    def curve_svg(self, series, points=None, xlabel="evaluations"):
        return curve_svg(series, points, xlabel)

    def _agentic_curves(self, individuals):
        return agentic_curves(individuals)

    def _telemetry_curves(self):
        return telemetry_curves(self.telemetry)

    def _page_data(self):
        """Everything the dashboard needs, one JSON payload — the page
        polls this instead of reloading itself."""
        ga = self.ga
        data = {"name": os.path.basename((self.run_dir or "run")
                                         .rstrip("/")),
                "up_min": round((time.time() - self.started) / 60, 1),
                "events": [list(e) for e in self.events][-120:],
                "media": {k: dict(v) for k, v in self.media.items()},
                "filmstrips": {k: list(v)
                               for k, v in self.filmstrips.items()}}
        if ga is None:
            last = self.telemetry[-1] if self.telemetry else {}
            if not data["events"]:
                data["events"] = [
                    ["", f"epoch {p.get('epoch')} · "
                         f"evals {p.get('evaluations')} · "
                         f"best {p.get('best')}"]
                    for p in self.telemetry[-40:]]
            data.update(
                mode="solver",
                series=telemetry_curves(self.telemetry), points=[],
                summary={"reports": len(self.telemetry),
                         "evaluations": last.get("evaluations") or 0},
                best={k: {"score": v} for k, v in
                      (last.get("best") or {}).items()}, living=[])
            return data
        series, points = agentic_curves(list(ga.individuals.values()))
        series.update(telemetry_curves(self.telemetry))
        living = sorted((i for i in ga.individuals.values()
                         if i["alive"]),
                        key=lambda i: (i["task"], -i["score"]))
        data.update(
            mode="agentic", summary=ga.summary(), series=series,
            points=points,
            best={t: {"score": b["score"], "id": b["id"],
                      "variation": (b["variation"] or "")[:200]}
                  for t, b in ga.best.items() if b is not None},
            living=[{"id": i["id"], "task": i["task"],
                     "score": i["score"], "origin": i["origin"],
                     "stale": i["scored_on_base"] < ga.base_version,
                     "contra": i["contradicts_base"],
                     "exhausted": bool(i.get("exhausted")),
                     "audited": i["audited"],
                     "variation": (i["variation"] or "")[:200]}
                    for i in living])
        return data

    def progress_html(self):
        from .ui import page
        return page("Finch 4 run", PROGRESS_BODY, PROGRESS_JS)


PROGRESS_BODY = """
<div class="hdr"><h1 id="title">run</h1>
<span class="pill live"><span class="dot"></span><span id="mode">live</span></span>
<span class="sub" id="up"></span></div>
<div class="tiles" id="tiles"></div>
<div class="panel"><h2>fitness over time</h2>
<div class="chartwrap"><div id="chart"></div><div class="tip"></div></div></div>
<div class="panel" id="mediapanel" style="display:none"><h2>evolved media</h2>
<div class="mediagrid" id="media"></div></div>
<div class="panel" id="poppanel" style="display:none"><h2>living population</h2>
<div style="overflow-x:auto"><table><thead><tr><th>id</th><th>task</th>
<th>score</th><th>origin</th><th>flags</th><th>variation</th></tr></thead>
<tbody id="pop"></tbody></table></div></div>
<div class="panel"><h2>event stream</h2><div class="events" id="events"></div></div>
"""

PROGRESS_JS = """
async function tick(){
  let d; try{d=await (await fetch('page.json')).json();}catch(e){return;}
  document.title=d.name;
  document.getElementById('title').textContent=d.name;
  document.getElementById('mode').textContent=d.mode;
  document.getElementById('up').textContent=d.up_min+' min up';
  const s=d.summary||{}, tiles=[];
  for(const [task,b] of Object.entries(d.best||{}))
    tiles.push(['hot', fmt(b.score), 'best · '+task]);
  if(d.mode==='agentic'){
    tiles.push(['', s.population, 'population'],
      ['', s.open_jobs, 'jobs in flight'],
      ['', 'v'+s.base_version, 'base playbook'],
      ['', s.stale, 'stale scores'],
      ['', (d.points||[]).length, 'evaluations']);
  } else {
    tiles.push(['', s.reports, 'progress reports'],
      ['', s.evaluations, 'evaluations']);
  }
  document.getElementById('tiles').innerHTML=tiles.map(t=>
    '<div class="tile '+t[0]+'"><div class="v">'+esc(t[1])+
    '</div><div class="k">'+esc(t[2])+'</div></div>').join('');
  drawChart(document.getElementById('chart'), d.series||{}, d.points||[],
    {xlabel: d.mode==='agentic'?'evaluation order':'evaluations'});
  const media=d.media||{}, strips=d.filmstrips||{},
        mkeys=Object.keys(media).sort();
  if(mkeys.length){
    document.getElementById('mediapanel').style.display='';
    document.getElementById('media').innerHTML=mkeys.map(k=>{
      const m=media[k]; let body;
      if(m.kind==='image') body='<img class="pix big" src="'+m.data+'">';
      else if(m.kind==='svg') body='<div class="svgwrap">'+m.data+'</div>';
      else body='<pre class="mediatext">'+esc(m.data.slice(0,2000))+'</pre>';
      const strip=(strips[k]||[]).slice(0,-1).map(s=>
        '<img class="pix" src="'+s+'">').join('');
      return '<figure class="mediacard">'+body+
        (strip?'<div class="filmstrip">'+strip+'</div>':'')+
        '<figcaption>'+esc(k)+
        (m.epoch!=null?' · epoch '+m.epoch:'')+'</figcaption></figure>';
    }).join('');
  }
  if(d.mode==='agentic'&&(d.living||[]).length){
    document.getElementById('poppanel').style.display='';
    const max=Math.max(...d.living.map(i=>i.score));
    const min=Math.min(...d.living.map(i=>i.score));
    document.getElementById('pop').innerHTML=d.living.map(i=>{
      const w=max>min?4+86*(i.score-min)/(max-min):45;
      const flags=[i.stale?'stale':'',i.contra?'⚡contra':'',
        i.exhausted?'exhausted':'',i.audited?'✓audited':''].filter(Boolean);
      return '<tr><td class="mono2">'+esc(i.id)+'</td><td>'+esc(i.task)+
      '</td><td><span class="scorebar"><i style="width:'+w+'px"></i></span>'+
      fmt(i.score)+'</td><td><span class="badge '+esc(i.origin)+'">'+
      esc(i.origin)+'</span></td><td class="dim">'+flags.join(' ')+
      '</td><td class="dim">'+esc(i.variation.slice(0,140))+'…</td></tr>';
    }).join('');
  }
  document.getElementById('events').innerHTML=(d.events||[]).slice()
    .reverse().map(e=>'<div><span class="t">'+esc(e[0])+'</span>'+
    esc(e[1])+'</div>').join('');
}
tick(); setInterval(tick, 2000);
"""


GETS = {"summary", "due", "batch", "stale", "contradictions",
        "page.json"}
POSTS = {"ask", "tell", "abandon", "consolidated", "rewrite", "rescore",
         "audit", "telemetry", "media"}


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self, allowed):
            name = self.path.lstrip("/").split("?")[0]
            if name in ("", "progress") and allowed is GETS:
                page = service.progress_html().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if name not in allowed:
                self._reply(404, {"error": f"unknown route {name!r}"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                self._reply(200, service.handle(name, body))
            except KeyError as e:
                self._reply(404, {"error": f"unknown id/job {e}"})
            except Exception as e:
                self._reply(400, {"error": repr(e)})

        def do_GET(self):
            self._route(GETS)

        def do_POST(self):
            self._route(POSTS)

        def log_message(self, fmt, *args):
            print(f"[serve] {args[0]}", flush=True)
    return Handler


def serve(run_dir, port=0, tasks=None, telemetry_only=False, **ga_kwargs):
    """Build (or load) the run's engine and return a ready server.
    port=0 picks a free port. Caller runs .serve_forever().
    telemetry_only=True starts the same server with no engine — the
    dashboard for tensor solve() runs (see live_progress)."""
    os.makedirs(run_dir, exist_ok=True)
    if telemetry_only:
        ga, state_path = None, None
    else:
        state_path = os.path.join(run_dir, "state.json")
        if os.path.exists(state_path):
            ga = AgenticGA.load(state_path)
        else:
            if not tasks:
                raise SystemExit("no state.json — pass --tasks to "
                                 "create a run")
            ga = AgenticGA(tasks=tasks, **ga_kwargs)
            ga.save(state_path)
    service = GAService(ga, state_path, run_dir=run_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(service))
    server.service = service
    with open(os.path.join(run_dir, "server.json"), "w") as f:
        json.dump({"port": server.server_address[1], "pid": os.getpid()}, f)
    register_run(run_dir, server.server_address[1])
    return server


def live_progress(run_dir=None, port=0, names=None, images="auto"):
    """One dashboard for every run: pass the result as solve()'s
    progress= callback and open the printed URL.

        from finch4 import solve, live_progress
        solve(fitness_fns, output_shape=(64, 64), epochs=10_000,
              progress=live_progress())

    Starts a telemetry-only reporting server (same /progress page the
    agentic substrate uses) in a daemon thread and returns a callback
    with solve()'s progress signature. names labels the fitness
    functions on the chart (default fn0, fn1, ...). The default run
    directory lives under ~/.finch4/runs/ — deliberately persistent, so
    the hub keeps showing the run after it finishes.

    images="auto" additionally posts each function's best-ever phenotype
    to the dashboard as a PNG whenever it is image-shaped ((H, W),
    (H, W, C<=4) or (C<=4, H, W)); "off" disables. Anything else goes
    through progress.report_media(name, image=|svg=|text=) — or, from a
    separate process, media_client(run_dir). Media is telemetry for the
    eyes, never evidence."""
    run_dir = run_dir or os.path.expanduser(
        f"~/.finch4/runs/live-{os.getpid()}-{int(time.time())}")
    server = serve(run_dir, port=port, telemetry_only=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/progress"
    print(f"[finch4] live progress: {url}", flush=True)

    def progress(epoch, epochs, evaluations, best_pheno, best_score):
        best = {}
        for i, score in enumerate(best_score):
            label = names[i] if names and i < len(names) else f"fn{i}"
            value = float(score)
            if value == value and abs(value) != float("inf"):
                best[label] = value
        server.service.handle("telemetry", {
            "epoch": int(epoch), "evaluations": int(evaluations),
            "best": best})
        server.service.event(f"epoch {epoch}/{epochs} "
                             f"evals={evaluations} best={best}")
        if images == "auto":
            for i, pheno in enumerate(best_pheno):
                if pheno is None or not looks_like_image(pheno.shape):
                    continue
                label = names[i] if names and i < len(names) else f"fn{i}"
                server.service.handle("media", {
                    "name": f"best {label}", "kind": "image",
                    "data": png_data_uri(pheno),
                    "epoch": int(epoch),
                    "evaluations": int(evaluations)})

    def report_media(name, image=None, svg=None, text=None, epoch=None,
                     evaluations=None):
        return server.service.handle("media", _media_body(
            name, image=image, svg=svg, text=text, epoch=epoch,
            evaluations=evaluations))

    progress.url = url
    progress.server = server
    progress.run_dir = run_dir
    progress.report_media = report_media
    return progress


def media_client(run_dir):
    """Media POSTer for a separate process (an agent, a wrapper around a
    scorer) — discovers the run's live server via server.json, the same
    discovery agents already use to report scores. Returns
    send(name, image=|svg=|text=, epoch=None, evaluations=None)."""
    import urllib.request
    port = json.load(open(os.path.join(run_dir, "server.json")))["port"]

    def send(name, image=None, svg=None, text=None, epoch=None,
             evaluations=None):
        body = json.dumps(_media_body(
            name, image=image, svg=svg, text=text, epoch=epoch,
            evaluations=evaluations)).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/media", data=body, method="POST")
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    return send


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--founders", type=int, default=2)
    p.add_argument("--children", type=int, default=4)
    p.add_argument("--population-cap", type=int, default=12)
    p.add_argument("--consolidate-every", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--telemetry", action="store_true",
                   help="no engine: dashboard-only server for solver runs")
    a = p.parse_args()
    server = serve(a.run, a.port, a.tasks, telemetry_only=a.telemetry,
                   founders=a.founders,
                   children=a.children, population_cap=a.population_cap,
                   consolidate_every=a.consolidate_every, seed=a.seed)
    print(f"[serve] run={a.run} port={server.server_address[1]}",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
