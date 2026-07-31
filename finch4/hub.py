"""The super-evolution hub: every evolution job on one page.

    python3 -m finch4.hub            # port 8800

Every reporting server registers itself in a global registry
(~/.finch4/registry.jsonl, override with FINCH4_REGISTRY) the
moment it starts — agentic runs and tensor solve() runs alike. The hub
reads that registry and renders one self-refreshing page with a card
per run: LIVE or FINISHED, the best score per task or fitness function,
a mini fitness curve, and a link to the run's own full dashboard while
it is alive.

The hub needs nothing from the runs to be running: agentic state
(state.json) and solver telemetry (telemetry.jsonl) are persisted in
each run directory, so finished runs stay on the board with their final
curves. However many problems, algorithms, agents, and machines are
evolving at once, this is the one page to watch.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .serve import agentic_curves, curve_svg, registry_path, \
    telemetry_curves


def load_registry():
    """Latest entry per run directory, newest first. At the default
    location the pre-rename registry (~/.latentspace/registry.jsonl) is
    read too — the campaign's runs registered there, and the board keeps
    showing them (this fallback existed in the latentspace hub and was
    dropped in the Finch 4 port; restored 2026-07-31)."""
    paths = [registry_path()]
    if not os.environ.get("FINCH4_REGISTRY"):
        paths.append(os.path.expanduser("~/.latentspace/registry.jsonl"))
    latest = {}
    for path in dict.fromkeys(p for p in paths if os.path.exists(p)):
        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if os.path.isdir(entry["run_dir"]):  # vanished tmp dirs
                        latest[entry["run_dir"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
    return sorted(latest.values(), key=lambda e: -e.get("started", 0))


def run_status(entry):
    """One run's card data, read from disk plus a liveness probe."""
    run_dir = entry["run_dir"]
    info = {"run_dir": run_dir, "name": os.path.basename(run_dir.rstrip("/")),
            "port": entry.get("port"), "live": False, "kind": "solver",
            "best": {}, "evaluations": 0, "series": {}, "points": [],
            "media": []}
    media_dir = os.path.join(run_dir, "media")
    if os.path.isdir(media_dir):
        # card thumbnails: the run's evolved images, latest per name,
        # straight from the persisted media mirror (works for live and
        # finished runs alike)
        for fname in sorted(os.listdir(media_dir))[:4]:
            if not fname.endswith(".png"):
                continue
            try:
                with open(os.path.join(media_dir, fname), "rb") as f:
                    info["media"].append({
                        "name": fname[:-4],
                        "src": ("data:image/png;base64,"
                                + base64.b64encode(f.read()).decode())})
            except OSError:
                continue
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{entry['port']}/summary",
                timeout=0.5) as r:
            json.loads(r.read())
        info["live"] = True
    except Exception:
        pass
    state_path = os.path.join(run_dir, "state.json")
    telem_path = os.path.join(run_dir, "telemetry.jsonl")
    try:
        if os.path.exists(state_path):
            state = json.load(open(state_path))
            info["kind"] = "agentic"
            inds = list(state.get("individuals", {}).values())
            info["evaluations"] = len(inds)
            info["series"], info["points"] = agentic_curves(inds)
            info["best"] = {t: (None if b is None else round(b["score"], 5))
                            for t, b in state.get("best", {}).items()}
        if os.path.exists(telem_path):
            telemetry = [json.loads(l) for l in open(telem_path)]
            info["series"].update(telemetry_curves(telemetry))
            if telemetry:
                info["evaluations"] = max(
                    info["evaluations"],
                    telemetry[-1].get("evaluations") or 0)
                if not info["best"]:
                    info["best"] = {k: round(v, 5) for k, v in
                                    (telemetry[-1].get("best") or {}).items()}
    except (json.JSONDecodeError, OSError):
        info["error"] = "unreadable run data"
    return info


def _downsample(curve, n=70):
    if len(curve) <= n:
        return curve
    step = len(curve) / n
    out = [curve[int(i * step)] for i in range(n)]
    if out[-1] != curve[-1]:
        out.append(curve[-1])
    return out


def hub_data():
    """Every registered run as one JSON payload for the board."""
    cards = []
    for entry in load_registry():
        info = run_status(entry)
        cards.append({
            "name": info["name"], "live": info["live"],
            "port": info["port"], "kind": info["kind"],
            "evaluations": info["evaluations"],
            "best": info["best"], "media": info["media"],
            "series": {k: _downsample(v)
                       for k, v in info["series"].items()}})
    cards.sort(key=lambda c: (not c["live"]))
    return {"cards": cards}


HUB_BODY = """
<div class="hdr"><h1>Finch 4 Dashboard</h1>
<span class="pill" id="count"></span></div>
<div class="grid" id="grid"></div>
"""

HUB_JS = """
async function tick(){
  let d; try{d=await (await fetch('data.json')).json();}catch(e){return;}
  const live=d.cards.filter(c=>c.live).length;
  document.getElementById('count').textContent=
    d.cards.length+' runs · '+live+' live';
  const grid=document.getElementById('grid');
  grid.innerHTML=d.cards.map((c,idx)=>{
    const pill=c.live?
      '<span class="pill live"><span class="dot"></span>LIVE</span>':
      '<span class="pill"><span class="dot"></span>finished</span>';
    const name=c.live?
      '<a href="http://127.0.0.1:'+c.port+'/progress">'+esc(c.name)+'</a>':
      esc(c.name);
    const best=Object.entries(c.best||{}).slice(0,3).map(([k,v])=>
      esc(k)+' <b style="color:var(--ink)">'+fmt(typeof v==='object'?v.score:v)+
      '</b>').join(' · ')||'—';
    const thumbs=(c.media||[]).map(m=>
      '<img class="pix" src="'+m.src+'" title="'+esc(m.name)+'">').join('');
    return '<div class="card">'+
      '<div style="display:flex;justify-content:space-between;'+
      'align-items:center"><span class="name">'+name+'</span>'+pill+'</div>'+
      '<div class="meta">'+esc(c.kind)+' · '+c.evaluations+' evaluations · '+
      best+'</div>'+(thumbs?'<div class="thumbrow">'+thumbs+'</div>':'')+
      '<div id="mc'+idx+'"></div></div>';
  }).join('')||'<p class="dim">no runs registered yet</p>';
  d.cards.forEach((c,idx)=>{
    const el=document.getElementById('mc'+idx);
    drawChart(el, c.series||{}, null,
      {mini:true, w:Math.max((el.clientWidth||330)-2,120), h:96});
  });
}
tick(); setInterval(tick, 3000);
"""


def hub_html():
    from .ui import page
    return page("Finch 4 Dashboard", HUB_BODY, HUB_JS)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=8800)
    a = p.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/data.json"):
                payload = json.dumps(hub_data()).encode()
                ctype = "application/json"
            else:
                payload = hub_html().encode()
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"[hub] http://127.0.0.1:{server.server_address[1]}/  "
          f"(registry: {registry_path()})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
