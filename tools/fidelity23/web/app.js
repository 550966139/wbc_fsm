/* fidelity23 workbench: version comparison from evaluator metrics. */
const COLORS = ["#4fc3f7", "#ffb74d", "#66bb6a", "#ef5350", "#ba68c8", "#ffd54f"];
const state = { data: null, sel: new Set(), runs: {}, charts: {} };

const fmt = (v, d = 3) => (v == null || Number.isNaN(v)) ? "—" : (v * (Math.abs(v) < 1 ? 100 : 1)).toFixed(d).replace(/\.?0+$/, "") + (Math.abs(v) < 1 ? "%" : "");
const pct = (v, d = 1) => (v == null || Number.isNaN(v)) ? "—" : (v * 100).toFixed(d) + "%";
const m3 = (v, d = 3) => (v == null || Number.isNaN(v)) ? "—" : v.toFixed(d);

const METRICS = [
  { key: "mpjpe_world_p50", name: "世界系误差 p50", unit: "m", better: "low", dec: 3 },
  { key: "fsr_ref_L", name: "足滑移率 左(参考支撑相)", unit: "%", better: "low", pctv: true },
  { key: "fsr_ref_R", name: "足滑移率 右(参考支撑相)", unit: "%", better: "low", pctv: true },
  { key: "slide_ref_p50_L", name: "滑移速度 左 p50", unit: "m/s", better: "low", dec: 3 },
  { key: "float_L", name: "悬浮时长占比 左", unit: "%", better: "low", pctv: true },
  { key: "float_R", name: "悬浮时长占比 右", unit: "%", better: "low", pctv: true },
  { key: "penetration", name: "穿地时长占比", unit: "%", better: "low", pctv: true },
  { key: "kick_apex_ratio_L", name: "左踢顶点比", unit: "%", better: "high", pctv: true },
  { key: "kick_apex_cm_L", name: "左踢顶点", unit: "cm", better: "high", dec: 1 },
  { key: "drift_net", name: "结尾净漂移", unit: "m", better: "low", dec: 2 },
  { key: "drift_path", name: "结尾路径长", unit: "m", better: "low", dec: 2 },
  { key: "ankle_margin_rad", name: "支撑踝余量", unit: "rad", better: "high", dec: 3 },
];

function baseOpt(title) {
  return {
    title: { text: title, textStyle: { color: "#8a97a3", fontSize: 13 }, left: 8, top: 4 },
    backgroundColor: "transparent",
    tooltip: { trigger: "axis", backgroundColor: "#171d24", borderColor: "#26303a", textStyle: { color: "#d7dee6" } },
    legend: { textStyle: { color: "#8a97a3" }, top: 4, right: 8 },
    grid: { left: 56, right: 20, top: 40, bottom: 28 },
    xAxis: { type: "value", axisLabel: { color: "#8a97a3" }, splitLine: { lineStyle: { color: "#202a33" } } },
    yAxis: { type: "value", axisLabel: { color: "#8a97a3" }, splitLine: { lineStyle: { color: "#202a33" } } },
  };
}

function init() {
  fetch("/api/overview").then(r => r.json()).then(d => {
    state.data = d;
    const vers = Object.keys(d);
    vers.forEach((v, i) => {
      state.sel.add(v);
      state.runs[v] = d[v].representative;
    });
    renderVersions();
    renderRuns();
    renderOverview();
    renderTimeline();
    renderFeet();
    renderJoints();
  });
  document.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    document.getElementById("tab-" + b.dataset.tab).classList.add("active");
    Object.values(state.charts).forEach(c => c && c.resize());
  }));
  window.addEventListener("resize", () => Object.values(state.charts).forEach(c => c && c.resize()));
}

function renderVersions() {
  const box = document.getElementById("versionList");
  box.innerHTML = "";
  Object.entries(state.data).forEach(([v, info], i) => {
    const lb = document.createElement("label");
    lb.className = "vrow";
    lb.innerHTML = `<span style="width:10px;height:10px;background:${COLORS[i % COLORS.length]};border-radius:2px"></span>
      <input type="checkbox" checked> ${v}
      ${info.meta.in_service ? '<span class="svc">在役</span>' : ""}
      <span style="color:var(--dim);font-size:11px"> n=${info.n}</span>`;
    lb.querySelector("input").addEventListener("change", e => {
      e.target.checked ? state.sel.add(v) : state.sel.delete(v);
      renderOverview(); renderTimeline(); renderFeet(); renderJoints();
    });
    box.appendChild(lb);
  });
}

function renderRuns() {
  const box = document.getElementById("runList");
  box.innerHTML = "";
  Object.entries(state.data).forEach(([v, info]) => {
    const h = document.createElement("div");
    h.style.color = COLORS[Object.keys(state.data).indexOf(v) % COLORS.length];
    h.style.marginTop = "6px";
    h.textContent = v;
    box.appendChild(h);
    info.runs.forEach(t => {
      const lb = document.createElement("label");
      lb.className = "rrow";
      lb.innerHTML = `<input type="radio" name="run-${v}" ${t === info.representative ? "checked" : ""}> ${t}
        ${info.t_entry_source[t] === "inferred" ? '<span style="color:var(--warn)" title="入场时刻由轨迹对齐推断">Ⓘ</span>' : ""}`;
      lb.querySelector("input").addEventListener("change", () => {
        state.runs[v] = t;
        renderTimeline(); renderFeet();
      });
      box.appendChild(lb);
    });
  });
}

function selVers() { return Object.keys(state.data).filter(v => state.sel.has(v)); }

function renderOverview() {
  const vers = selVers();
  const t = document.getElementById("metricTable");
  let html = "<tr><th>指标</th>" + vers.map(v => `<th>${v}${state.data[v].meta.in_service ? " (在役)" : ""}</th>`).join("") + "</tr>";
  for (const mm of METRICS) {
    const vals = vers.map(v => state.data[v].median[mm.key]);
    let best = null;
    if (vals.length > 1 && vals.every(x => x != null && !Number.isNaN(x))) {
      best = mm.better === "low" ? Math.min(...vals) : Math.max(...vals);
    }
    html += `<tr><td>${mm.name}${mm.unit ? ` (${mm.unit})` : ""}</td>` + vals.map(v => {
      const txt = mm.pctv ? pct(v) : m3(v, mm.dec);
      const cls = (best != null && v === best) ? "best" : "";
      return `<td class="${cls}">${txt}</td>`;
    }).join("") + "</tr>";
  }
  t.innerHTML = html;

  const cards = document.getElementById("routingCards");
  cards.innerHTML = "";
  for (const v of vers) {
    const rt = state.data[v].routing || [];
    if (!rt.length) continue;
    const h = document.createElement("div");
    h.className = "ver";
    h.textContent = `${v} 路由：`;
    cards.appendChild(h);
    rt.forEach(r => {
      const c = document.createElement("div");
      c.className = "card " + (r.severity || "");
      c.innerHTML = `<b>[${r.severity}] ${r.area}</b><div class="ev">证据：${r.evidence}</div><div>建议：${r.suggestion}</div>`;
      cards.appendChild(c);
    });
  }
  if (!cards.children.length) cards.innerHTML = '<p class="hint">当前选择无超阈值项。</p>';
}

async function runSeries(v) {
  const tag = state.runs[v];
  const r = await fetch(`/api/run/${v}/${tag}`).then(x => x.json());
  return r;
}

async function renderTimeline() {
  const vers = selVers();
  const vidx = v => Object.keys(state.data).indexOf(v) % COLORS.length;
  const mkOpt = (title, yname) => {
    const o = baseOpt(title);
    o.yAxis.name = yname || "";
    return o;
  };
  const build = async (id, title, yname, seriesOf) => {
    if (!state.charts[id]) state.charts[id] = echarts.init(document.getElementById(id));
    const series = [];
    for (const v of vers) {
      const r = await runSeries(v);
      const s = r.series;
      for (const sr of seriesOf(v, s, vidx(v))) series.push(sr);
    }
    state.charts[id].setOption(Object.assign(mkOpt(title, yname), { series }), true);
  };
  const line = (name, color, data, dash) => ({
    name, type: "line", showSymbol: false, animation: false,
    lineStyle: { width: 1.6, color, type: dash ? "dashed" : "solid" },
    itemStyle: { color }, data,
  });
  const tdxy = (s, arr) => s.td.map((t, i) => [+t.toFixed(2), arr[i]]).filter(d => d[1] != null);
  await build("chart-mpjpe", "世界系 24 体位置误差 p（m）", "m",
    (v, s, ci) => [line(`${v}`, COLORS[ci], tdxy(s, s.mpjpe_world))]);
  await build("chart-sole", "鞋底高度（实线=左 虚线=右，m）", "m",
    (v, s, ci) => [line(`${v} 左`, COLORS[ci], tdxy(s, s.soleL)),
                   line(`${v} 右`, COLORS[ci], tdxy(s, s.soleR), true)]);
  await build("chart-speed", "足底水平速度（实线=左 虚线=右，m/s）", "m/s",
    (v, s, ci) => [line(`${v} 左`, COLORS[ci], tdxy(s, s.speedL)),
                   line(`${v} 右`, COLORS[ci], tdxy(s, s.speedR), true)]);
  await build("chart-drift", "骨盆累计漂移（m，td≥28.5 结尾段）", "m",
    (v, s, ci) => [line(`${v}`, COLORS[ci], tdxy(s, s.drift))]);
}

async function renderFeet() {
  const id = "chart-feet";
  if (!state.charts[id]) state.charts[id] = echarts.init(document.getElementById(id));
  const opt = baseOpt("足底世界轨迹（左/右 · 实线=机器人 虚线=参考）");
  opt.xAxis.name = "x (m)"; opt.yAxis.name = "y (m)";
  opt.legend.data = [];
  const series = [];
  for (const v of selVers()) {
    const r = await runSeries(v);
    const s = r.series;
    const ci = Object.keys(state.data).indexOf(v) % COLORS.length;
    for (const [nm, act, ref] of [["左", s.xyL, s.refxyL], ["右", s.xyR, s.refxyR]]) {
      series.push({ name: `${v} ${nm}`, type: "line", showSymbol: false, animation: false,
        lineStyle: { width: 1.8, color: COLORS[ci], type: nm === "左" ? "solid" : "dashed" },
        itemStyle: { color: COLORS[ci] }, data: act.map(p => [+p[0].toFixed(3), +p[1].toFixed(3)]) });
    }
  }
  opt.series = series;
  state.charts[id].setOption(opt, true);
}

async function renderJoints() {
  const id1 = "chart-jerr", id2 = "chart-std";
  if (!state.charts[id1]) state.charts[id1] = echarts.init(document.getElementById(id1));
  if (!state.charts[id2]) state.charts[id2] = echarts.init(document.getElementById(id2));
  const vers = selVers();
  const names = state.data[vers[0]].joints.names;
  const order = names.map((_, j) => j).sort((a, b) =>
    Math.max(...vers.map(v => state.data[v].joints.err_p90[a] || 0)) -
    Math.max(...vers.map(v => state.data[v].joints.err_p90[b] || 0))).slice(0, 23);
  const o1 = baseOpt("每关节跟踪误差 p90（rad）");
  o1.xAxis = { type: "category", data: order.map(j => names[j]), axisLabel: { color: "#8a97a3", rotate: 60, fontSize: 10 } };
  o1.series = vers.map((v, i) => ({ name: v, type: "bar", data: order.map(j => +(state.data[v].joints.err_p90[j] || 0).toFixed(3)),
    itemStyle: { color: COLORS[i % COLORS.length] } }));
  state.charts[id1].setOption(o1, true);
  const o2 = baseOpt("幅度比 std(actual)/std(ref)（<0.8 幅度亏损）");
  o2.xAxis = { type: "category", data: names, axisLabel: { color: "#8a97a3", rotate: 60, fontSize: 10 } };
  o2.series = vers.map((v, i) => ({ name: v, type: "bar", data: state.data[v].joints.std_ratio.map(x => +x.toFixed(2)),
    itemStyle: { color: COLORS[i % COLORS.length] } }));
  o2.series.push({ name: "0.8 线", type: "line", data: names.map(() => 0.8), lineStyle: { color: "#ef5350", type: "dashed" }, showSymbol: false });
  state.charts[id2].setOption(o2, true);
}

document.addEventListener("DOMContentLoaded", init);
