"""The shared design system for every Finch 4 page — the dashboard
hub and each run's page wear the same skin: warm paper background,
earthy accents, live JSON polling (no full-page refresh), labeled
multi-series fitness charts with hover readout. Self-contained: inline
CSS/JS, no external assets, served by stdlib HTTP."""

CSS = """
:root{
  --bg:#faf8f3; --panel:#ffffff; --panel2:#fdfcf9; --line:#e6dfd2;
  --ink:#2e2a23; --ink2:#6f675a; --ink3:#a59c8c;
  --accent:#b65c38; --leaf:#5f7a3d; --good:#4c7a4c; --warn:#a8842c;
  --bad:#a84434;
  --shadow:0 1px 3px rgba(94,80,63,.08), 0 4px 14px rgba(94,80,63,.06);
}
*{box-sizing:border-box;margin:0;padding:0}
svg{max-width:100%}
body{background:
  radial-gradient(1100px 480px at 75% -12%, rgba(182,92,56,.05), transparent),
  var(--bg);
  color:var(--ink);
  font:14px/1.45 ui-monospace,'SF Mono',Menlo,monospace;
  padding:22px 26px;min-height:100vh}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
h1{font-size:16px;letter-spacing:.05em;font-weight:700}
.small{font-size:11px;color:var(--ink2)}
.hdr{display:flex;align-items:baseline;gap:14px;margin-bottom:16px;
  border-bottom:2px solid var(--line);padding-bottom:12px}
.hdr .sub{color:var(--ink2);font-size:12px}
.pill{display:inline-flex;align-items:center;gap:6px;font-size:11px;
  padding:2px 10px;border-radius:999px;border:1px solid var(--line);
  color:var(--ink2);background:var(--panel)}
.pill.live{color:var(--good);border-color:rgba(76,122,76,.4)}
.pill.live .dot{width:7px;height:7px;border-radius:50%;
  background:var(--good);animation:pulse 1.6s ease-in-out infinite}
.pill .dot{width:7px;height:7px;border-radius:50%;background:var(--ink3)}
@keyframes pulse{50%{opacity:.3}}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.tile{background:var(--panel);border:1px solid var(--line);
  border-radius:10px;padding:10px 16px;min-width:130px;
  box-shadow:var(--shadow)}
.tile .v{font-size:20px;font-weight:700;letter-spacing:.02em}
.tile .k{font-size:10px;color:var(--ink2);text-transform:uppercase;
  letter-spacing:.12em;margin-top:2px}
.tile.hot .v{color:var(--accent)}
.panel{background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:14px 16px;margin:12px 0;
  box-shadow:var(--shadow)}
.panel h2{font-size:11px;color:var(--ink2);text-transform:uppercase;
  letter-spacing:.14em;margin-bottom:10px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{font-size:10px;color:var(--ink3);text-transform:uppercase;
  letter-spacing:.1em;text-align:left;padding:4px 10px 6px 0;
  border-bottom:1px solid var(--line)}
td{padding:5px 10px 5px 0;border-bottom:1px solid #f0ebe0;
  vertical-align:top}
tr:hover td{background:rgba(182,92,56,.035)}
.mono2{color:var(--ink2)} .dim{color:var(--ink3)}
.scorebar{position:relative;display:inline-block;width:90px;height:6px;
  background:#efe9dc;border-radius:3px;overflow:hidden;margin-right:8px;
  vertical-align:middle}
.scorebar i{position:absolute;left:0;top:0;bottom:0;
  background:linear-gradient(90deg,#8a9a55,var(--leaf));
  border-radius:3px}
.badge{font-size:10px;padding:1px 7px;border-radius:5px;
  border:1px solid var(--line);color:var(--ink2);background:var(--panel2)}
.badge.found{color:var(--good);border-color:rgba(76,122,76,.35)}
.badge.refound{color:var(--warn);border-color:rgba(168,132,44,.4)}
.badge.bad{color:var(--bad);border-color:rgba(168,68,52,.35)}
.events{max-height:260px;overflow-y:auto;font-size:12px}
.events div{padding:2.5px 0;border-bottom:1px solid #f0ebe0}
.events .t{color:var(--ink3);margin-right:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(360px,100%),1fr));
  gap:14px}
.card{background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:12px 14px;box-shadow:var(--shadow);
  transition:border-color .2s}
.card:hover{border-color:rgba(182,92,56,.45)}
.card .name{font-weight:700;font-size:13.5px}
.card .meta{font-size:11px;color:var(--ink2);margin:3px 0 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chartwrap{position:relative}
.tip{position:absolute;pointer-events:none;background:#fffdf8;
  border:1px solid var(--line);border-radius:8px;padding:6px 10px;
  font-size:11px;display:none;z-index:9;white-space:nowrap;
  box-shadow:var(--shadow)}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-thumb{background:#ddd4c2;border-radius:4px}
"""

CHART_JS = """
const PALETTE=['#5f7a3d','#b65c38','#3f7a74','#a8842c','#7d5a8a','#5b7fa6'];
function esc(s){return String(s).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function fmt(v){if(v===null||v===undefined)return '—';
  const a=Math.abs(v);return a!==0&&(a<1e-3||a>=1e5)?
  v.toExponential(3):(+v.toFixed(5)).toString();}
function drawChart(el,series,points,opts){
  opts=opts||{}; const W=opts.w||el.clientWidth||700,H=opts.h||240;
  const mini=!!opts.mini, ML=mini?6:74, MR=mini?6:14,
        MT=mini?6:10, MB=mini?6:24;
  const names=Object.keys(series).filter(k=>series[k].length>=2).sort();
  if(!names.length){el.innerHTML=
    '<div class="dim" style="padding:30px 10px">awaiting data…</div>';return;}
  let xs=[],ys=[];
  names.forEach(n=>series[n].forEach(p=>{xs.push(p[0]);ys.push(p[1]);}));
  (points||[]).forEach(p=>ys.push(p[1]));
  const xmax=Math.max(...xs)*1.04+1e-9;
  let lo=Math.min(...ys),hi=Math.max(...ys);
  const pad=Math.max((hi-lo)*.14,1e-9); lo-=pad; hi+=pad;
  const X=x=>ML+(W-ML-MR)*x/xmax, Y=y=>MT+(H-MT-MB)*(hi-y)/(hi-lo);
  let s='<svg width="'+W+'" height="'+H+'">';
  s+='<defs><filter id="gl"><feDropShadow dx="0" dy="0" '+
     'stdDeviation="1.6" flood-color="#8a7a5e" flood-opacity="0.25"/>'+
     '</filter></defs>';
  if(!mini){for(let i=0;i<=3;i++){const gy=lo+pad+(hi-lo-2*pad)*i/3;
    s+='<line x1="'+ML+'" y1="'+Y(gy)+'" x2="'+(W-MR)+'" y2="'+Y(gy)+
       '" stroke="#efe9dc" stroke-width="1"/>'+
       '<text x="'+(ML-8)+'" y="'+(Y(gy)+3)+'" font-size="10" '+
       'fill="#a59c8c" text-anchor="end">'+fmt(gy)+'</text>';}}
  names.forEach((n,i)=>{
    const c=PALETTE[i%PALETTE.length],cv=series[n];
    let d='M '+X(cv[0][0])+' '+Y(cv[0][1]);
    for(let j=1;j<cv.length;j++)
      d+=' L '+X(cv[j][0])+' '+Y(cv[j-1][1])+' L '+X(cv[j][0])+' '+Y(cv[j][1]);
    if(names.length===1&&!mini){
      s+='<path d="'+d+' L '+X(cv[cv.length-1][0])+' '+Y(lo)+' L '+
         X(cv[0][0])+' '+Y(lo)+' Z" fill="'+c+'" opacity="0.09"/>';}
    s+='<path d="'+d+'" fill="none" stroke="'+c+'" stroke-width="'+
       (mini?1.6:2.2)+'"'+(mini?'':' filter="url(#gl)"')+'/>';
    if(!mini){const last=cv[cv.length-1];
      s+='<circle cx="'+X(last[0])+'" cy="'+Y(last[1])+'" r="3.5" fill="'+
         c+'"/>'+'<text x="'+(X(last[0])-8)+'" y="'+(Y(last[1])-9)+
         '" font-size="11" fill="'+c+'" text-anchor="end">'+esc(n)+' '+
         fmt(last[1])+'</text>';}});
  if(!mini)(points||[]).forEach(p=>{
    s+='<circle cx="'+X(p[0])+'" cy="'+Y(p[1])+'" r="2.6" fill="#c9bfae"/>';});
  if(!mini)s+='<text x="'+ML+'" y="'+(H-6)+'" font-size="10" '+
    'fill="#a59c8c">'+esc(opts.xlabel||'evaluations')+
    ' → &nbsp;·&nbsp; best so far ↑</text>';
  s+='</svg>';
  el.innerHTML=s;
  if(mini)return;
  const tip=el.parentElement.querySelector('.tip'); if(!tip)return;
  el.onmousemove=e=>{
    const r=el.getBoundingClientRect(),mx=e.clientX-r.left;
    const tx=(mx-ML)/(W-ML-MR)*xmax; let best=null;
    names.forEach((n,i)=>{const cv=series[n];
      let p=cv[0]; for(const q of cv){if(q[0]<=tx)p=q; else break;}
      if(!best||Math.abs(p[0]-tx)<Math.abs(best.p[0]-tx))
        best={n:n,p:p,c:PALETTE[i%PALETTE.length]};});
    if(!best){tip.style.display='none';return;}
    tip.style.display='block';
    tip.style.left=Math.min(mx+14,W-160)+'px';
    tip.style.top='14px';
    tip.innerHTML='<span style="color:'+best.c+'">●</span> '+esc(best.n)+
      '<br>x '+best.p[0]+' · '+fmt(best.p[1]);};
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
