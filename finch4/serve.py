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

An agent reports its own result with one line:

    curl -s -X POST localhost:PORT/tell -d '{"job_id": "j0004", ...}'
"""
from __future__ import annotations

import argparse
import html
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agentic import AgenticGA

PALETTE = ["#7ac", "#c96", "#9c7", "#b8a", "#8cc", "#ca8"]


def registry_path():
    """Global registry of every run this machine has served — one JSON
    line per server start. The hub (hub.py) reads it to show ALL
    evolution jobs, live and finished, on one page."""
    return (os.environ.get("FINCH4_REGISTRY")
            or os.environ.get("FINCH4_REGISTRY")
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
                "events": [list(e) for e in self.events][-120:]}
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
         "audit", "telemetry"}


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


def live_progress(run_dir=None, port=0, names=None):
    """One dashboard for every run: pass the result as solve()'s
    progress= callback and open the printed URL.

        from finch4 import solve, live_progress
        solve(fitness_fns, output_shape=(64, 64), epochs=10_000,
              progress=live_progress())

    Starts a telemetry-only reporting server (same /progress page the
    agentic substrate uses) in a daemon thread and returns a callback
    with solve()'s progress signature. names labels the fitness
    functions on the chart (default fn0, fn1, ...)."""
    import tempfile
    run_dir = run_dir or tempfile.mkdtemp(prefix="finch4-live-")
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

    progress.url = url
    progress.server = server
    return progress


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
