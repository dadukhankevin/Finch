"""The shared design system for every Finch 4 page — the dashboard
hub and each run's page wear the same skin: warm paper background,
earthy accents, live JSON polling (no full-page refresh), labeled
multi-series fitness charts with hover readout. Self-contained: inline
CSS/JS, no external assets, served by stdlib HTTP.

The categorical chart palette is validated (lightness band, chroma
floor, colorblind separation, contrast on the paper surface) — keep the
ORDER fixed; series colors attach to sorted series identity, never to
rank. Status colors (good/warn/bad) are reserved for states and never
used as series identity."""

# Fixed categorical order — terracotta, teal, ochre, plum, leaf, slate.
CHART_PALETTE = ["#b0480f", "#0089a1", "#ab7a0c", "#8d4a9e", "#41803a",
                 "#3b6fb0"]

CSS = """
:root{
  --bg:#f6f2e9; --panel:#fffdf8; --panel2:#faf7ef;
  --line:#e5dcc8; --line2:#f0ead9;
  --ink:#2b2519; --ink2:#6c6252; --ink3:#a49a82;
  --accent:#b0480f; --teal:#0089a1; --gold:#ab7a0c; --plum:#8d4a9e;
  --leaf:#41803a; --blue:#3b6fb0;
  --good:#41803a; --warn:#a8842c; --bad:#a84434;
  --mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
  --shadow:0 1px 2px rgba(94,80,63,.06), 0 4px 16px rgba(94,80,63,.05);
}
*{box-sizing:border-box;margin:0;padding:0}
svg{max-width:100%}
body{background:
  radial-gradient(1100px 480px at 75% -12%, rgba(176,72,15,.045), transparent),
  var(--bg);
  color:var(--ink);
  font:13.5px/1.5 -apple-system,'SF Pro Text','Segoe UI',system-ui,sans-serif;
  padding:20px 26px 40px;min-height:100vh}
code,pre,.mono{font-family:var(--mono)}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
h1{font-size:18px;letter-spacing:-.01em;font-weight:700}
.small{font-size:11px;color:var(--ink2)}
.dim{color:var(--ink3)} .mono2{font-family:var(--mono);color:var(--ink2)}
.hdr{display:flex;align-items:center;gap:12px;margin-bottom:14px;
  border-bottom:1px solid var(--line);padding-bottom:12px}
.hdr .sub{color:var(--ink3);font-size:12px;margin-left:auto}
.pill{display:inline-flex;align-items:center;gap:6px;font-size:11px;
  font-weight:600;padding:3px 11px;border-radius:999px;
  border:1px solid var(--line);color:var(--ink2);background:var(--panel)}
.pill.live{color:var(--good);border-color:rgba(65,128,58,.35)}
.pill.live .dot{background:var(--good);animation:pulse 1.6s ease-in-out infinite}
.pill .dot{width:7px;height:7px;border-radius:50%;background:var(--ink3)}
@keyframes pulse{50%{opacity:.25}}

/* ---- stat tiles: one hero + compact companions ---- */
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.tile{background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:12px 18px 10px;min-width:128px;
  box-shadow:var(--shadow)}
.tile .v{font-size:22px;font-weight:700;font-family:var(--mono);
  letter-spacing:-.01em}
.tile .k{font-size:10px;color:var(--ink2);text-transform:uppercase;
  letter-spacing:.11em;margin-top:3px;font-weight:600}
.tile .sub{font-size:11px;color:var(--ink3);margin-top:5px}
.tile .sub b{color:var(--ink2);font-weight:600}
.tile.hot{border-color:rgba(176,72,15,.35);
  background:linear-gradient(160deg,#fff8f2,var(--panel) 55%)}
.tile.hot .v{color:var(--accent);font-size:26px}

.panel{background:var(--panel);border:1px solid var(--line);
  border-radius:14px;padding:16px 18px;margin:14px 0;
  box-shadow:var(--shadow)}
.panel h2{font-size:11px;color:var(--ink2);text-transform:uppercase;
  letter-spacing:.14em;margin-bottom:12px;font-weight:700}
.panel h2 .count{color:var(--ink3);font-weight:400;letter-spacing:.02em;
  text-transform:none;margin-left:6px}

table{border-collapse:collapse;width:100%;font-size:12.5px}
th{font-size:10px;color:var(--ink3);text-transform:uppercase;
  letter-spacing:.09em;text-align:left;padding:4px 12px 7px 0;
  border-bottom:1px solid var(--line);font-weight:600}
td{padding:7px 12px 7px 0;border-bottom:1px solid var(--line2);
  vertical-align:middle}
tr:last-child td{border-bottom:0}
tr:hover td{background:rgba(176,72,15,.03)}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}

.badge{display:inline-block;font-size:10px;font-weight:600;
  padding:2px 8px;border-radius:6px;border:1px solid var(--line);
  color:var(--ink2);background:var(--panel2);letter-spacing:.03em}
.badge.found{color:var(--good);border-color:rgba(65,128,58,.35)}
.badge.inject{color:var(--gold);border-color:rgba(171,122,12,.4)}
.badge.crossover{color:var(--plum);border-color:rgba(141,74,158,.35)}
.badge.good{color:var(--good);border-color:rgba(65,128,58,.35)}
.badge.warn{color:var(--warn);border-color:rgba(168,132,44,.4)}
.badge.bad{color:var(--bad);border-color:rgba(168,68,52,.4)}
.badge.champ{color:var(--accent);border-color:rgba(176,72,15,.45);
  background:#fff6ef}
.statusdot{display:inline-block;width:8px;height:8px;border-radius:50%;
  margin-right:6px;vertical-align:baseline;background:var(--ink3)}
.statusdot.running{background:var(--good);
  animation:pulse 1.6s ease-in-out infinite}
.scorebar{position:relative;display:inline-block;width:84px;height:5px;
  background:var(--line2);border-radius:3px;overflow:hidden;
  margin-right:8px;vertical-align:middle}
.scorebar i{position:absolute;left:0;top:0;bottom:0;
  background:linear-gradient(90deg,#7fa055,var(--leaf));border-radius:3px}

.events{max-height:280px;overflow-y:auto;font-size:12px}
.events .row{display:flex;gap:10px;padding:4px 0;align-items:baseline;
  border-bottom:1px solid var(--line2)}
.events .row:last-child{border-bottom:0}
.events .t{color:var(--ink3);font-family:var(--mono);font-size:10.5px;
  flex:0 0 auto}
.events .echip{flex:0 0 78px;text-align:center;font-size:9.5px;
  font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  border-radius:5px;padding:1px 0;background:var(--panel2);
  border:1px solid var(--line);color:var(--ink2)}
.events .echip.report{color:var(--teal);border-color:rgba(0,137,161,.3)}
.events .echip.kill,.events .echip.void{color:var(--bad);
  border-color:rgba(168,68,52,.35)}
.events .echip.audit{color:var(--gold);border-color:rgba(171,122,12,.4)}
.events .echip.share,.events .echip.incorporate,.events .echip.compact{color:var(--leaf);
  border-color:rgba(65,128,58,.35)}
.events .echip.found,.events .echip.inject,.events .echip.crossover{
  color:var(--plum);border-color:rgba(141,74,158,.3)}
.events .x{min-width:0;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}

.grid{display:grid;
  grid-template-columns:repeat(auto-fill,minmax(min(360px,100%),1fr));
  gap:14px}
.card{background:var(--panel);border:1px solid var(--line);
  border-radius:14px;padding:13px 15px;box-shadow:var(--shadow);
  transition:border-color .2s}
.card:hover{border-color:rgba(176,72,15,.4)}
.card .name{font-weight:700;font-size:13.5px}
.card .meta{font-size:11px;color:var(--ink2);margin:3px 0 8px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .thumbrow{display:flex;gap:6px;margin:2px 0 8px;flex-wrap:wrap}
.card .thumbrow img{width:56px;height:auto;border:1px solid var(--line);
  border-radius:6px}

.chart-note{font-size:12px;color:var(--ink2);margin:0 0 8px;max-width:52em}
.chart-note.warn{color:var(--bad);font-weight:600}
.chartwrap{position:relative}
.tip{position:absolute;pointer-events:none;background:var(--panel);
  border:1px solid var(--line);border-radius:8px;padding:7px 11px;
  font-size:11px;display:none;z-index:9;white-space:nowrap;
  box-shadow:var(--shadow)}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-thumb{background:#d9cfba;border-radius:4px}

.mediagrid{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start}
.mediacard{background:var(--panel2);border:1px solid var(--line);
  border-radius:10px;padding:10px;margin:0}
.mediacard figcaption{font-size:11px;color:var(--ink2);margin-top:6px}
img.pix{image-rendering:pixelated}
.mediacard img.big{width:192px;height:auto;display:block;border-radius:4px}
.filmstrip{display:flex;gap:3px;margin-top:6px;flex-wrap:wrap}
.filmstrip img{width:34px;height:auto;opacity:.75;border-radius:2px}
.mediatext{font-size:11px;max-width:420px;max-height:220px;
  overflow:auto;white-space:pre-wrap}
.svgwrap{max-width:420px;overflow:auto}

/* ================= Tree of Life ================= */
.tol-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  margin-bottom:12px}
.tol-controls input[type=search],.tol-controls select{
  border:1px solid var(--line);background:var(--panel2);color:var(--ink);
  border-radius:8px;padding:6px 10px;font:12px/1.3 inherit;outline:none}
.tol-controls input[type=search]{min-width:230px}
.tol-controls input[type=search]:focus{border-color:rgba(176,72,15,.5)}
.tol-stats{font-size:11px;color:var(--ink3);margin-left:auto}
.tol-stats .bad{color:var(--bad);font-weight:600}
.chips{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:10.5px;
  font-weight:600;color:var(--ink2);border:1px solid var(--line);
  background:var(--panel2);border-radius:999px;padding:3px 10px;
  cursor:pointer;user-select:none;transition:all .15s}
.chip .swatch{width:14px;height:0;border-top:2.5px solid;border-radius:2px}
.chip .swatch.dashed{border-top-style:dashed}
.chip.off{opacity:.4;background:transparent}
.chip:hover{border-color:var(--ink3)}
.tol-time{display:flex;align-items:center;gap:8px;font-size:11px;
  color:var(--ink2)}
.tol-time input[type=range]{width:150px;accent-color:var(--accent)}
.tol-zoom{display:inline-flex;border:1px solid var(--line);
  border-radius:8px;overflow:hidden}
.tol-zoom button{border:0;background:var(--panel2);color:var(--ink3);
  font:12px/1 inherit;padding:5px 9px;cursor:pointer}
.tol-zoom button.on{background:var(--panel);color:var(--ink);
  font-weight:700}

.tol-grid{display:grid;grid-template-columns:212px minmax(0,1fr) 304px;
  gap:0 12px;align-items:start}
.tol-rail{display:flex;flex-direction:column;padding-top:4px}
.tol-rail .trunkcard,.lanecard{border:1px solid var(--line);
  border-radius:10px;background:var(--panel2);padding:7px 10px;
  margin-bottom:0;cursor:pointer;transition:border-color .15s,opacity .3s}
.lanecard:hover{border-color:var(--ink3)}
.lanecard.selected{border-color:var(--accent);
  box-shadow:0 0 0 1px rgba(176,72,15,.25)}
.lanecard.faded{opacity:.35}
.lanecard .r1{display:flex;align-items:center;gap:6px}
.lanecard .r1 .id{font-family:var(--mono);font-weight:700;font-size:12.5px}
.lanecard .r1 .kindmark{font-size:10px;color:var(--ink3)}
.lanecard .r1 .best{margin-left:auto;font-family:var(--mono);
  font-weight:700;font-size:12.5px;font-variant-numeric:tabular-nums}
.lanecard .r1 .best.champ{color:var(--accent)}
.lanecard .r2{font-size:10px;color:var(--ink3);margin-top:1px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lanecard.killed .r1 .id{color:var(--ink3);text-decoration:line-through}
.trunkcard .r1{display:flex;align-items:center;gap:6px;font-weight:700;
  font-size:12px}
.trunkcard .r1 .v{margin-left:auto;font-family:var(--mono);
  color:var(--gold)}
.trunkcard .r2{font-size:10px;color:var(--ink3);margin-top:1px}
.tol-canvas{overflow-x:auto;overflow-y:hidden;border:1px solid var(--line);
  border-radius:12px;background:var(--panel2);position:relative}
.tol-canvas svg{display:block;max-width:none;font-family:inherit}
.tol-tip{position:fixed;pointer-events:none;background:var(--panel);
  border:1px solid var(--line);border-radius:9px;padding:8px 11px;
  font-size:11px;display:none;z-index:40;max-width:340px;
  box-shadow:0 6px 24px rgba(94,80,63,.18)}
.tol-tip .tt-id{font-family:var(--mono);font-weight:700;font-size:12px}
.tol-tip .tt-score{font-family:var(--mono);font-weight:700}
.tol-tip .tt-sum{color:var(--ink2);margin-top:3px;white-space:normal}
.tol-inspect{border:1px solid var(--line);border-radius:12px;
  padding:13px 15px;background:var(--panel2);min-height:200px;
  max-height:640px;overflow:auto;font-size:12px}
.tol-inspect h3{font-size:13px;margin-bottom:2px;font-family:var(--mono)}
.tol-inspect .sub{color:var(--ink3);font-size:11px;margin-bottom:8px}
.tol-inspect .badges{display:flex;gap:5px;flex-wrap:wrap;margin:7px 0}
.tol-inspect .bigscore{font-family:var(--mono);font-size:21px;
  font-weight:700;margin:4px 0 0}
.tol-inspect .bigscore .delta{font-size:12px;font-weight:600;
  margin-left:7px}
.tol-inspect .delta.up{color:var(--good)} .tol-inspect .delta.down{color:var(--bad)}
.tol-inspect .src{font-size:10.5px;color:var(--ink3);margin-bottom:8px}
.tol-inspect p.prose{font-family:var(--mono);font-size:11.5px;
  line-height:1.55;white-space:pre-wrap;overflow-wrap:anywhere;
  margin:8px 0;color:var(--ink)}
.tol-inspect .sec{font-size:9.5px;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink3);margin:12px 0 4px}
.tol-inspect button.jump{display:block;width:100%;text-align:left;
  border:1px solid var(--line);border-radius:7px;background:var(--panel);
  color:var(--ink2);font:11px/1.4 var(--mono);cursor:pointer;
  padding:4px 8px;margin:3px 0;overflow-wrap:anywhere}
.tol-inspect button.jump:hover{border-color:var(--accent);
  color:var(--accent)}
.tol-inspect code.token{display:block;margin-top:7px;padding:7px 9px;
  background:#f2ecdd;border-radius:7px;overflow-wrap:anywhere;
  font-size:10.5px;cursor:pointer}
.tol-inspect code.token:hover{background:#ece4d0}
.tol-inspect .warnrow{color:var(--bad);font-size:10.5px;margin-top:8px}
.tol-inspect pre.decoderpre{font-size:10.5px;line-height:1.5;
  background:#f2ecdd;border-radius:7px;padding:9px;overflow:auto;
  max-height:260px;white-space:pre-wrap;margin-top:8px}

.tol-node{cursor:pointer}
.tol-node text{pointer-events:none}
.tol-node.faded,.tol-edge.faded,.tol-lanetrack.faded{opacity:.13}
.tol-edge{transition:opacity .15s}
.tol-edge:hover{opacity:1 !important}
.decoderpanel pre{font-family:var(--mono);font-size:11.5px;
  line-height:1.55;white-space:pre-wrap;background:var(--panel2);
  border:1px solid var(--line2);border-radius:9px;padding:12px 14px;
  margin-top:10px;max-height:340px;overflow:auto}
.decoderpanel summary{cursor:pointer;font-size:11px;color:var(--ink2);
  text-transform:uppercase;letter-spacing:.14em;font-weight:700}
.decoderpanel summary .count{color:var(--ink3);font-weight:400;
  letter-spacing:.02em;text-transform:none;margin-left:6px}
@media(max-width:1100px){
  .tol-grid{grid-template-columns:180px minmax(0,1fr)}
  .tol-inspect{grid-column:1/-1;margin-top:12px;max-height:380px}}
"""

CHART_JS = """
const PALETTE=['#b0480f','#0089a1','#ab7a0c','#8d4a9e','#41803a','#3b6fb0'];
function esc(s){return String(s).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function fmt(v){if(v===null||v===undefined)return '—';
  const a=Math.abs(v);return a!==0&&(a<1e-3||a>=1e5)?
  v.toExponential(3):(+v.toFixed(5)).toString();}
const AUDIT_SUFFIX=' · audit-passed';
function seriesStyle(name){
  return name.endsWith(AUDIT_SUFFIX)||name.includes(' · holdout');}
function seriesColor(name,baseNames){
  const base=name.endsWith(AUDIT_SUFFIX)?
    name.slice(0,-AUDIT_SUFFIX.length):name;
  const i=Math.max(0,baseNames.indexOf(base));
  return PALETTE[i%PALETTE.length];}
function legendLabel(name){
  return name.replace(/^[^·]+ · /,'').replace(AUDIT_SUFFIX,' ✓audit')
    .replace('sealed','held-in');}
function drawChart(el,series,points,opts){
  opts=opts||{}; const W=opts.w||el.clientWidth||700,H=opts.h||240;
  const mini=!!opts.mini, ML=mini?6:74, MR=mini?6:14,
        legendRows=mini?0:Math.ceil(Object.keys(series).filter(
          k=>series[k].length>=2).length/3),
        MT=mini?6:(12+legendRows*16), MB=mini?6:26;
  const names=Object.keys(series).filter(k=>series[k].length>=2).sort();
  if(!names.length){el.innerHTML=
    '<div class="dim" style="padding:30px 10px">awaiting data…</div>';return;}
  const baseNames=[...new Set(names.map(n=>n.endsWith(AUDIT_SUFFIX)?
    n.slice(0,-AUDIT_SUFFIX.length):n))].sort();
  let xs=[],ys=[];
  names.forEach(n=>series[n].forEach(p=>{xs.push(p[0]);ys.push(p[1]);}));
  (points||[]).forEach(p=>ys.push(p[1]));
  const xmax=Math.max(...xs)*1.04+1e-9;
  let lo=Math.min(...ys),hi=Math.max(...ys);
  const pad=Math.max((hi-lo)*.14,1e-9); lo-=pad; hi+=pad;
  const X=x=>ML+(W-ML-MR)*x/xmax, Y=y=>MT+(H-MT-MB)*(hi-y)/(hi-lo);
  let s='<svg width="'+W+'" height="'+H+'">';
  if(!mini){for(let i=0;i<=3;i++){const gy=lo+pad+(hi-lo-2*pad)*i/3;
    s+='<line x1="'+ML+'" y1="'+Y(gy)+'" x2="'+(W-MR)+'" y2="'+Y(gy)+
       '" stroke="#f0ead9" stroke-width="1"/>'+
       '<text x="'+(ML-8)+'" y="'+(Y(gy)+3)+'" font-size="10" '+
       'fill="#a49a82" text-anchor="end" font-family="ui-monospace,Menlo,monospace">'+
       fmt(gy)+'</text>';}}
  names.forEach(n=>{
    const c=seriesColor(n,baseNames),cv=series[n],
          dashed=seriesStyle(n);
    let d='M '+X(cv[0][0])+' '+Y(cv[0][1]);
    for(let j=1;j<cv.length;j++)
      d+=' L '+X(cv[j][0])+' '+Y(cv[j-1][1])+' L '+X(cv[j][0])+' '+Y(cv[j][1]);
    if(baseNames.length===1&&!dashed&&!mini){
      s+='<path d="'+d+' L '+X(cv[cv.length-1][0])+' '+Y(lo)+' L '+
         X(cv[0][0])+' '+Y(lo)+' Z" fill="'+c+'" opacity="0.07"/>';}
    s+='<path d="'+d+'" fill="none" stroke="'+c+'" stroke-width="'+
       (mini?1.6:(dashed?1.7:2.2))+'"'+
       (dashed?' stroke-dasharray="5 4" opacity="0.85"':'')+'/>';
    if(!mini){const last=cv[cv.length-1];
      s+='<circle cx="'+X(last[0])+'" cy="'+Y(last[1])+'" r="3.2" fill="'+
         c+'"/>';}});
  if(!mini&&names.length>1){let lx=ML+6,ly=10;
    names.forEach((n,i)=>{const c=seriesColor(n,baseNames),
      dashed=seriesStyle(n), label=legendLabel(n);
      if(i&&i%3===0){lx=ML+6; ly+=16;}
      s+='<line x1="'+lx+'" y1="'+ly+'" x2="'+(lx+16)+'" y2="'+ly+
        '" stroke="'+c+'" stroke-width="2.5"'+
        (dashed?' stroke-dasharray="4 3"':'')+'/>'+
        '<text x="'+(lx+21)+'" y="'+(ly+4)+'" font-size="10.5" '+
        'fill="#6c6252">'+esc(label)+'</text>';
      lx+=36+label.length*6.4;});}
  else if(!mini){const n=names[0],cv=series[n],last=cv[cv.length-1];
    s+='<text x="'+(X(last[0])-8)+'" y="'+(Y(last[1])-9)+
       '" font-size="11" fill="'+seriesColor(n,baseNames)+
       '" text-anchor="end">'+esc(n)+' '+fmt(last[1])+'</text>';}
  if(!mini)(points||[]).forEach(p=>{
    s+='<circle cx="'+X(p[0])+'" cy="'+Y(p[1])+'" r="2.4" fill="#c9bfae" opacity="0.8"/>';});
  if(!mini)s+='<text x="'+ML+'" y="'+(H-6)+'" font-size="10" '+
    'fill="#a49a82">'+esc(opts.xlabel||'evaluations')+
    ' → &nbsp;·&nbsp; '+esc(opts.footer||'best so far ↑')+'</text>';
  s+='</svg>';
  el.innerHTML=s;
  if(mini)return;
  const tip=el.parentElement.querySelector('.tip'); if(!tip)return;
  el.onmousemove=e=>{
    const r=el.getBoundingClientRect(),mx=e.clientX-r.left;
    const tx=(mx-ML)/(W-ML-MR)*xmax; let best=null;
    names.forEach(n=>{const cv=series[n];
      let p=cv[0]; for(const q of cv){if(q[0]<=tx)p=q; else break;}
      if(!best||Math.abs(p[0]-tx)<Math.abs(best.p[0]-tx))
        best={n:n,p:p,c:seriesColor(n,baseNames)};});
    if(!best){tip.style.display='none';return;}
    tip.style.display='block';
    tip.style.left=Math.min(mx+14,W-170)+'px';
    tip.style.top='14px';
    tip.innerHTML='<span style="color:'+best.c+'">●</span> '+esc(best.n)+
      '<br>x '+best.p[0]+' · <b class="num">'+fmt(best.p[1])+'</b>';};
  el.onmouseleave=()=>{tip.style.display='none';};
}
"""


def page(title, body, boot_js):
    """Full HTML page: shared skin + a boot script that polls JSON and
    re-renders. body holds the static shells the JS fills."""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head><body>
{body}
<script>{CHART_JS}
{boot_js}</script></body></html>"""
