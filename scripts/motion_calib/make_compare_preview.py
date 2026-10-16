#!/usr/bin/env python
"""make_compare_preview.py - interactive old-vs-new reference skeleton preview.

Renders two 30-column G1 reference CSVs (e.g. raw GMR vs MC-calibrated) as
overlaid animated skeletons in an in-session HTML fragment: front (X-Z) and
side (Y-Z) views, red ghost = old, cyan solid = new, scrub bar, per-frame
joint-step chart, and clickable time-window markers. Body positions come from
local MuJoCo FK (measure.py's Rigger), so no NPZ or 224 access is needed.

Adapted from take2_r5_work/make_preview.py (same player, 15 Hz ticks for
30 fps CSVs instead of 25 Hz for 50 Hz NPZs).

Usage:
  python make_compare_preview.py --old RAW.csv --new MC.csv --label take6 \
      --windows '[{"t":9.5,"label":"深蹲单支撑(F18 死点)"}]' --out out.html
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from measure import Rigger, measure_fk, default_g1_xml  # noqa: E402

B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def enc2(v):
    q = int(round(v)) + 200
    q = 0 if q < 0 else (1295 if q > 1295 else q)
    return B36[q // 36] + B36[q % 36]


def prep(rig, path):
    m = np.loadtxt(path, delimiter=",")
    bp, _, _, _, _ = measure_fk(rig, m)
    step = np.abs(np.diff(m[:, 7:], axis=0)).max(axis=1)
    n = len(step) // 2
    s2 = np.maximum(step[0:2 * n:2], step[1:2 * n:2])  # per-tick max, @30Hz rad/frame
    frames = bp[0:2 * n:2] * 100.0  # cm, 15 Hz ticks
    return frames, np.concatenate([s2, [0.0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--label", default="ref")
    ap.add_argument("--xml", default=default_g1_xml())
    ap.add_argument("--windows", default="[]")
    ap.add_argument("--old-name", default="原始 GMR")
    ap.add_argument("--new-name", default="MC 校准")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rig = Rigger(args.xml)
    assert rig.nb == 24, "expected 24 bodies, got %d" % rig.nb
    old_f, old_s = prep(rig, args.old)
    new_f, new_s = prep(rig, args.new)
    npt = min(len(old_f), len(new_f))
    old_f, new_f, old_s, new_s = old_f[:npt], new_f[:npt], old_s[:npt], new_s[:npt]

    # center both onto a common origin so +-2 m encoding range suffices
    off = (np.concatenate([old_f, new_f]).reshape(-1, 3).max(axis=0)
           + np.concatenate([old_f, new_f]).reshape(-1, 3).min(axis=0)) / 2
    old_f -= off
    new_f -= off

    d_old = "".join(enc2(v) for v in old_f.ravel())
    d_new = "".join(enc2(v) for v in new_f.ravel())
    s_old = "".join(enc2(v * 100) for v in old_s)
    s_new = "".join(enc2(v * 100) for v in new_s)

    ext = {}
    for ax, name in zip(range(3), "xyz"):
        lo = min(old_f[:, :, ax].min(), new_f[:, :, ax].min()) - 10
        hi = max(old_f[:, :, ax].max(), new_f[:, :, ax].max()) + 10
        ext[name] = (float(lo), float(hi))
    smax = float(max(old_s.max(), new_s.max()))

    wins = json.loads(args.windows)
    win_js = ",\n  ".join('{t:%.2f,label:"%s"}' % (w["t"], w["label"]) for w in wins)

    html = """<div id="mccmpviz">
<h3>__LABEL__ 参考:__OLDNAME__(红, 悬浮/出界带病) vs __NEWNAME__(青, 校准后)</h3>
<div class="viz-controls">
  <button class="btn btn-primary" id="mcplay" type="button">播放</button>
  <label class="form-label" for="mcspeed" style="margin:0 0 0 8px">速度</label>
  <select class="form-select" id="mcspeed" style="width:auto">
    <option value="0.25">0.25×</option>
    <option value="0.5" selected>0.5×</option>
    <option value="1">1×</option>
  </select>
  <input class="form-range" id="mcscrub" type="range" min="0" max="__NPT__" value="0" step="1" aria-label="时间轴">
</div>
<div class="viz-controls" id="mcwins"></div>
<div class="viz-row" style="gap:12px">
  <span class="text-small"><span style="display:inline-block;width:14px;height:3px;background:var(--red);opacity:.65;vertical-align:middle"></span> __OLDNAME__</span>
  <span class="text-small"><span style="display:inline-block;width:14px;height:3px;background:var(--viz-series-1);vertical-align:middle"></span> __NEWNAME__</span>
</div>
<svg id="mcchart" role="img" aria-label="单帧最大关节步长:旧与新参考对比" style="width:100%;height:180px"></svg>
<div class="viz-row" style="align-items:stretch;gap:8px;flex-wrap:wrap">
  <svg id="mcfront" viewBox="0 0 330 400" role="img" aria-label="正视图骨架对比" style="flex:1 1 300px;min-width:260px;width:100%"></svg>
  <svg id="mcside" viewBox="0 0 330 400" role="img" aria-label="侧视图骨架对比" style="flex:1 1 300px;min-width:260px;width:100%"></svg>
</div>
<p class="text-small" aria-live="polite" id="mcreadout"></p>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(function(){
const NP=__NPT__, NPT2=NP*72;
const DO="__DOLD__", DN="__DNEW__", SO="__SOLD__", SN="__SNEW__";
function dec2(s,i){let a=s.charCodeAt(i),b=s.charCodeAt(i+1);a=a<=57?a-48:a-87;b=b<=57?b-48:b-87;return a*36+b;}
const po=new Float32Array(NPT2),pn=new Float32Array(NPT2),so=new Float32Array(NP),sn=new Float32Array(NP);
for(let k=0;k<NPT2;k++){po[k]=(dec2(DO,2*k)-200)/100;pn[k]=(dec2(DN,2*k)-200)/100;}
for(let k=0;k<NP;k++){so[k]=(dec2(SO,2*k)-200)/100;sn[k]=(dec2(SN,2*k)-200)/100;}
const BONES=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[16,17],[17,18],[13,19],[19,20],[20,21],[21,22],[22,23]];
const EXT=__EXT__, SMAX=__SMAX__;
let cur=0,playing=false,speed=0.5,acc=0,last=null;
const chart=d3.select("#mcchart");
const cw=()=>document.getElementById("mcchart").clientWidth||700;
function drawChart(){
  const W=cw(),H=180,M={l:36,r:8,t:18,b:24};
  chart.selectAll("*").remove();
  chart.attr("viewBox",`0 0 ${W} ${H}`);
  const x=d3.scaleLinear().domain([0,NP/15]).range([M.l,W-M.r]);
  const y=d3.scaleLinear().domain([0,Math.ceil(SMAX*20)/20]).range([H-M.b,M.t]);
  const g=chart.append("g");
  g.append("rect").attr("data-chart-frame").attr("x",M.l).attr("y",M.t).attr("width",W-M.l-M.r).attr("height",H-M.t-M.b).attr("fill","none").attr("stroke","var(--border)");
  y.ticks(4).forEach(t=>{g.append("line").attr("x1",M.l).attr("x2",W-M.r).attr("y1",y(t)).attr("y2",y(t)).attr("stroke","var(--border)").attr("stroke-width",.5).attr("opacity",.5);g.append("text").attr("x",M.l-4).attr("y",y(t)+4).attr("text-anchor","end").attr("font-size",11).attr("fill","var(--muted-foreground)").text(t.toFixed(1));});
  x.ticks(6).forEach(t=>{g.append("text").attr("x",x(t)).attr("y",H-6).attr("text-anchor","middle").attr("font-size",11).attr("fill","var(--muted-foreground)").text(t+"s");});
  g.append("text").attr("class","text-small").attr("x",M.l).attr("y",12).attr("font-size",12).attr("fill","var(--foreground)").text("单帧最大关节步长 (rad,@30Hz)");
  const line=k=>d3.line().x((d,i)=>x(i/15)).y(d=>y(k?sn[d]:so[d]));
  const idx=d3.range(NP);
  g.append("path").datum(idx).attr("d",line(0)).attr("fill","none").attr("stroke","var(--red)").attr("stroke-width",1).attr("opacity",.7);
  g.append("path").datum(idx).attr("d",line(1)).attr("fill","none").attr("stroke","var(--viz-series-1)").attr("stroke-width",1.2);
  const cg=g.append("line").attr("id","mccursor").attr("y1",M.t).attr("y2",H-M.b).attr("stroke","var(--foreground)").attr("stroke-width",1).attr("opacity",.8);
  chart.on("click",ev=>{const [mx]=d3.pointer(ev);seek(Math.round(x.invert(mx)*15));});
  window.mcdrawCursor=t=>cg.attr("x1",x(t/15)).attr("x2",x(t/15));
}
function panel(id,fx,fy,ex,ey){
  const svg=document.getElementById(id);
  const W=330,H=400,P=14;
  const sx=v=>P+(v-ex[0])/(ex[1]-ex[0])*(W-2*P);
  const sy=v=>H-P-(v-ey[0])/(ey[1]-ey[0])*(H-2*P);
  let inner='';
  inner+=`<line x1="${P}" y1="${sy(0)}" x2="${W-P}" y2="${sy(0)}" stroke="var(--border)" stroke-width="1"/>`;
  for(const [a,b] of BONES){
    inner+=`<line class="bo" data-a="${a}" data-b="${b}" stroke="var(--red)" stroke-width="1.4" opacity="0.45"/>`;
  }
  for(const [a,b] of BONES){
    inner+=`<line class="bn" data-a="${a}" data-b="${b}" stroke="var(--viz-series-1)" stroke-width="2.4" stroke-linecap="round"/>`;
  }
  inner+=`<text x="${W/2}" y="${H-2}" text-anchor="middle" font-size="12" fill="var(--muted-foreground)">${id==="#mcfront"?"正视图 (X-Z)":"侧视图 (Y-Z)"}</text>`;
  svg.innerHTML=inner;
  const bo=[...svg.querySelectorAll(".bo")],bn=[...svg.querySelectorAll(".bn")];
  function setLines(cls,P,k){
    for(const ln of cls){
      const a=+ln.dataset.a,b=+ln.dataset.b;
      ln.setAttribute("x1",sx(P[k*72+a*3+fx]));ln.setAttribute("y1",sy(P[k*72+a*3+fy]));
      ln.setAttribute("x2",sx(P[k*72+b*3+fx]));ln.setAttribute("y2",sy(P[k*72+b*3+fy]));
    }
  }
  return k=>{setLines(bo,po,k);setLines(bn,pn,k);};
}
const drawF=panel("mcfront",0,2,EXT.x,EXT.z), drawS=panel("mcside",1,2,EXT.y,EXT.z);
const wins=__WINS__;
const wdiv=document.getElementById("mcwins");
wins.forEach(w=>{
  const b=document.createElement("button");
  b.className="btn btn-ghost";b.type="button";b.textContent=w.label;
  b.addEventListener("click",()=>{seek(Math.round(w.t*15));setPlay(true);});
  wdiv.appendChild(b);
});
const playBtn=document.getElementById("mcplay"),scrub=document.getElementById("mcscrub"),
      spd=document.getElementById("mcspeed"),ro=document.getElementById("mcreadout");
function setPlay(p){playing=p;playBtn.textContent=p?"暂停":"播放";}
playBtn.addEventListener("click",()=>setPlay(!playing));
spd.addEventListener("change",()=>speed=+spd.value);
scrub.addEventListener("input",()=>seek(+scrub.value));
function seek(k){cur=Math.max(0,Math.min(NP-1,k));render();}
function render(){
  scrub.value=cur;
  drawF(cur);drawS(cur);
  if(window.mcdrawCursor)window.mcdrawCursor(cur);
  ro.textContent=`t = ${(cur/15).toFixed(2)} s · 该时刻单帧最大关节步长 — 原始 ${so[cur].toFixed(3)} rad / 校准 ${sn[cur].toFixed(3)} rad`;
}
function tick(ts){
  if(last==null)last=ts;
  acc+=(ts-last)*speed;last=ts;
  while(acc>=66){acc-=66;cur++;if(cur>=NP){cur=NP-1;setPlay(false);break;}render();}
  requestAnimationFrame(tick);
}
new ResizeObserver(drawChart).observe(document.getElementById("mcchart"));
drawChart();render();requestAnimationFrame(tick);
})();
</script>
</div>
"""

    html = (html.replace("__LABEL__", args.label)
                .replace("__OLDNAME__", args.old_name)
                .replace("__NEWNAME__", args.new_name)
                .replace("__NPT__", str(npt))
                .replace("__DOLD__", d_old)
                .replace("__DNEW__", d_new)
                .replace("__SOLD__", s_old)
                .replace("__SNEW__", s_new)
                .replace("__EXT__", json.dumps({k: list(v) for k, v in ext.items()}))
                .replace("__SMAX__", "%.3f" % smax)
                .replace("__WINS__", "[" + win_js + "]"))
    with open(args.out, "w") as f:
        f.write(html)
    print("wrote %s: %.0f KB, %d ticks @15Hz, extents %s, smax %.3f"
          % (args.out, len(html) / 1024, npt, ext, smax))


if __name__ == "__main__":
    main()
