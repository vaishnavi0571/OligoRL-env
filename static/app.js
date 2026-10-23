/* OpenEnv Market -- dashboard client.
 * Thin client: every rule lives in the Python simulator; this file only
 * sends actions to the API and draws what comes back.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  meta: null,
  difficulty: "easy",
  history: [],      // [{obs, reward, details}] -- index 0 is the reset state
  done: false,
  busy: false,
  charts: {},
};

// ---------------------------------------------------------------- utils
const safeStorage = {
  get(store, key) { try { return window[store].getItem(key); } catch (e) { return null; } },
  set(store, key, val) { try { window[store].setItem(key, val); } catch (e) { /* ignore */ } },
};

function makeId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return "s-" + Math.random().toString(36).slice(2) + Date.now().toString(36);
}
const sessionId = safeStorage.get("sessionStorage", "openenv-session") || makeId();
safeStorage.set("sessionStorage", "openenv-session", sessionId);

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const firmColor = (i) => (i < 0 ? css("--pool") : css(`--firm-${i}`));
const money = (v) => (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
const money2 = (v) => "$" + v.toFixed(2);
const cost = (v) => (Math.abs(v) < 0.5 ? "$0" : "-" + money(v));
const compact = (v) => {
  const a = Math.abs(v), s = v < 0 ? "-" : "";
  if (a >= 1e6) return `${s}$${(a / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${s}$${(a / 1e3).toFixed(a >= 1e4 ? 0 : 1)}K`;
  return `${s}$${a.toFixed(0)}`;
};
const pct = (v, d = 0) => (v * 100).toFixed(d) + "%";
const firmName = (i) => (i === -1 ? "Unemployed" : state.meta.firm_names[i]);

async function api(path, opts = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: { "Content-Type": "application/json", "X-Session-Id": sessionId, ...(opts.headers || {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { const body = await res.json(); detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail); } catch (e) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, 3500);
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

// ---------------------------------------------------------------- theme
function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  Object.values(state.charts).forEach(restyleChart);
  renderLegends();
  if (state.history.length) { renderLabor(); rebuildCharts(); }
}
$("btn-theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : window.matchMedia("(prefers-color-scheme: dark)").matches;
  const next = dark ? "light" : "dark";
  safeStorage.set("localStorage", "openenv-theme", next);
  applyTheme(next);
});

// ---------------------------------------------------------------- charts
const crosshair = {
  id: "crosshair",
  afterDatasetsDraw(chart) {
    const active = chart.tooltip && chart.tooltip.getActiveElements();
    if (!active || !active.length) return;
    const x = active[0].element.x;
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = css("--axis");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.restore();
  },
};

function baseOptions(yFormat) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: "index", intersect: false },
    layout: { padding: { top: 4, right: 8 } },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: css("--surface"),
        titleColor: css("--ink-2"),
        bodyColor: css("--ink"),
        borderColor: css("--border"),
        borderWidth: 1,
        padding: 10,
        boxWidth: 12,
        boxHeight: 2,
        usePointStyle: false,
        titleFont: { weight: "500" },
        bodyFont: { weight: "600" },
        callbacks: {
          title: (items) => `Step ${items[0].label}`,
          label: (item) => ` ${yFormat(item.parsed.y)}  ${item.dataset.label}`,
        },
      },
    },
    scales: {
      x: {
        grid: { display: false },
        border: { color: css("--axis") },
        ticks: { color: css("--muted"), maxTicksLimit: 8, maxRotation: 0 },
      },
      y: {
        grid: { color: css("--grid"), lineWidth: 1 },
        border: { display: false },
        ticks: { color: css("--muted"), maxTicksLimit: 5, callback: (v) => yFormat(v) },
      },
    },
    elements: {
      line: { borderWidth: 2, tension: 0.25, borderCapStyle: "round", borderJoinStyle: "round" },
      point: { radius: 0, hoverRadius: 4, hoverBorderWidth: 2 },
    },
  };
}

function makeChart(id, datasets, yFormat) {
  const chart = new Chart($(id), {
    type: "line",
    data: { labels: [], datasets: datasets.map((d) => ({ ...d, data: [] })) },
    options: baseOptions(yFormat),
    plugins: [crosshair],
  });
  chart.$colorKeys = datasets.map((d) => d.colorKey);
  chart.$yFormat = yFormat;
  restyleChart(chart);
  return chart;
}

function restyleChart(chart) {
  chart.data.datasets.forEach((ds, i) => {
    const c = css(chart.$colorKeys[i]);
    ds.borderColor = c;
    ds.backgroundColor = c;
    ds.pointHoverBackgroundColor = c;
    ds.pointHoverBorderColor = css("--surface");
  });
  const o = baseOptions(chart.$yFormat);
  chart.options.plugins.tooltip = o.plugins.tooltip;
  chart.options.scales = o.scales;
  chart.update("none");
}

function initCharts() {
  const names = state.meta.firm_names;
  state.charts.cash = makeChart("c-cash",
    names.map((n, i) => ({ label: n, colorKey: `--firm-${i}`, order: i === 0 ? 0 : 1 })), compact);
  state.charts.reward = makeChart("c-reward",
    [{ label: "Reward", colorKey: "--firm-0" }], (v) => v.toFixed(2));
  state.charts.price = makeChart("c-price", [
    { label: "You", colorKey: "--firm-0" },
    { label: "Rivals' average", colorKey: "--muted" },
  ], money2);
  state.charts.wage = makeChart("c-wage", [
    { label: "You", colorKey: "--firm-0" },
    { label: "Market (median)", colorKey: "--muted" },
  ], (v) => money(v));
  renderLegends();
}

function renderLegends() {
  const fill = (id, chart) => {
    if (!chart) return;
    const box = $(id);
    box.replaceChildren();
    chart.data.datasets.forEach((ds, i) => {
      const s = el("span");
      const key = el("i");
      key.style.background = css(chart.$colorKeys[i]);
      s.append(key, document.createTextNode(ds.label));
      box.append(s);
    });
  };
  fill("legend-cash", state.charts.cash);
  fill("legend-price", state.charts.price);
  fill("legend-wage", state.charts.wage);
}

function rivalsAvgPrice(obs) {
  const p = obs.firm_prices.slice(1).filter((_, i) => obs.firm_active[i + 1]);
  return p.length ? p.reduce((a, b) => a + b, 0) / p.length : state.meta.defaults.price;
}

function rebuildCharts() {
  const h = state.history;
  const labels = h.map((x) => x.obs.step);
  const c = state.charts;
  c.cash.data.labels = labels;
  c.cash.data.datasets.forEach((ds, i) => { ds.data = h.map((x) => x.obs.agent_wealth[i]); });
  c.reward.data.labels = labels;
  c.reward.data.datasets[0].data = h.map((x) => (x.reward ? x.reward.composite : null));
  c.price.data.labels = labels;
  c.price.data.datasets[0].data = h.map((x) => x.obs.current_price);
  c.price.data.datasets[1].data = h.map((x) => (x.details ? x.details.competitor_avg_price : rivalsAvgPrice(x.obs)));
  c.wage.data.labels = labels;
  c.wage.data.datasets[0].data = h.map((x) => x.obs.wage);
  c.wage.data.datasets[1].data = h.map((x) => x.obs.market_wage);
  // Keep small noise from looking like big swings, and keep lines off the axis
  padRange(c.price, 0.15 * state.meta.defaults.price);
  padRange(c.wage, 0.10 * state.meta.defaults.wage);
  padRange(c.reward, 0.2);
  Object.values(c).forEach((ch) => ch.update("none"));
}

function padRange(chart, minSpan) {
  const vals = chart.data.datasets.flatMap((ds) => ds.data).filter((v) => v !== null);
  if (!vals.length) return;
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const mid = (lo + hi) / 2;
  const half = Math.max((hi - lo) / 2 * 1.1, minSpan / 2);
  const y = chart.options.scales.y;
  y.suggestedMin = lo >= 0 ? Math.max(0, mid - half) : mid - half;
  y.suggestedMax = mid + half;
}

// ---------------------------------------------------------------- controls
const inputs = {
  price: $("in-price"), wage: $("in-wage"), hire: $("in-hire"),
  capex: $("in-capex"), lobby: $("in-lobby"),
};

function syncOutputs() {
  $("out-price").textContent = money2(+inputs.price.value);
  $("out-wage").textContent = money(+inputs.wage.value);
  const hire = +inputs.hire.value;
  $("out-hire").textContent = hire > 0 ? `+${hire}` : `${hire}`;
  $("out-capex").textContent = money(+inputs.capex.value);
  const lobby = +inputs.lobby.value;
  $("out-lobby").textContent = `${lobby > 0 ? "+" : ""}${(lobby * 100).toFixed(0)} pts`;
}
Object.values(inputs).forEach((inp) => inp.addEventListener("input", syncOutputs));

function applyBounds() {
  const b = state.meta.bounds;
  inputs.price.min = Math.max(b.input_cost_per_unit + 1, 1);
  inputs.wage.min = b.min_wage;
  inputs.wage.max = b.max_wage;
  inputs.hire.max = b.max_hire_per_step;
  inputs.lobby.min = b.lobby_min;
  inputs.lobby.max = b.lobby_max;
}

function setControlsFromObs(obs) {
  inputs.price.value = obs.current_price;
  inputs.wage.value = Math.round(obs.wage / 10) * 10;
  resetOneTimeControls();
}

function resetOneTimeControls() {
  inputs.hire.value = 0;
  inputs.capex.value = 0;
  inputs.lobby.value = 0;
  syncOutputs();
}

function currentAction(includeOneTime) {
  return {
    set_price: +inputs.price.value,
    set_wage: +inputs.wage.value,
    hire_workers: includeOneTime ? +inputs.hire.value : 0,
    capex: includeOneTime ? +inputs.capex.value : 0,
    lobby_tax_change: includeOneTime ? +inputs.lobby.value : 0,
  };
}

function setBusy(busy) {
  state.busy = busy;
  ["btn-step", "btn-auto5", "btn-end"].forEach((id) => { $(id).disabled = busy || state.done; });
  $("btn-reset").disabled = busy;
}

// ---------------------------------------------------------------- render
function last() { return state.history[state.history.length - 1]; }
function prev() { return state.history[state.history.length - 2]; }

function setDelta(id, cur, before, fmt, eps = 1e-9) {
  const node = $(id);
  if (before === undefined || before === null) { node.textContent = ""; node.className = "tile-delta"; return; }
  const d = cur - before;
  if (Math.abs(d) < eps) { node.textContent = "no change"; node.className = "tile-delta"; return; }
  node.textContent = `${d > 0 ? "▲" : "▼"} ${fmt(Math.abs(d))} vs last step`;
  node.className = "tile-delta " + (d > 0 ? "up" : "down");
}

function renderTiles() {
  const cur = last(), before = prev();
  const o = cur.obs;
  $("t-cash").textContent = money(o.firm_cash);
  setDelta("d-cash", o.firm_cash, before && before.obs.firm_cash, money);
  $("t-workers").textContent = o.worker_count;
  setDelta("d-workers", o.worker_count, before && before.obs.worker_count, (v) => `${v}`);
  const share = cur.details ? cur.details.market_share : 1 / state.meta.num_firms;
  $("t-share").textContent = pct(share, 1);
  const prevShare = before ? (before.details ? before.details.market_share : 1 / state.meta.num_firms) : null;
  setDelta("d-share", share, prevShare, (v) => pct(v, 1), 0.0005);
  $("t-reward").textContent = cur.reward ? cur.reward.composite.toFixed(3) : "–";
  setDelta("d-reward", cur.reward ? cur.reward.composite : 0,
    before && before.reward ? before.reward.composite : null, (v) => v.toFixed(3), 0.0005);
}

function renderHeader() {
  const o = last().obs;
  $("step-label").textContent = `Step ${o.step} / ${o.episode_length}`;
  $("progress-bar").style.width = pct(o.step / o.episode_length, 1);
}

function renderHints() {
  const o = last().obs;
  $("hint-price").textContent = `Rivals charge about ${money2(rivalsAvgPrice(o))} · inputs cost $${state.meta.bounds.input_cost_per_unit}/unit`;
  $("hint-wage").textContent = `Current ${money(o.wage)} · market ${money(o.market_wage)} · moves ≤${pct(state.meta.bounds.wage_max_change)}/step`;
  $("hint-hire").textContent = `One-time this step · ${o.unemployed} unemployed`;
  $("hint-lobby").textContent = `Tax rate ${pct(o.tax_rate)} · one-time this step`;
}

function renderSales() {
  const d = last().details;
  const kv = $("sales-kv");
  kv.replaceChildren();
  if (!d) {
    $("meter-fill").style.width = "0";
    kv.append(el("dt", "", "Take a step to see sales and costs."), el("dd", "", ""));
    return;
  }
  $("meter-fill").style.width = pct(d.max_output ? Math.min(d.units_sold / d.max_output, 1) : 0, 1);
  const rows = [
    ["Own demand", `${Math.round(d.demand).toLocaleString()} units`],
    ["Sold / capacity", `${Math.round(d.units_sold).toLocaleString()} / ${Math.round(d.max_output).toLocaleString()}`],
    ["From rivals' overflow", `${Math.round(d.spill_units).toLocaleString()} units`],
    ["Revenue", money(d.revenue)],
    ["Wages", cost(d.labor_cost)],
    ["Inputs", cost(d.input_cost)],
    ["Hiring / firing", cost(d.hiring_cost)],
    ["Tax", cost(d.tax_paid)],
    ["Capex", cost(d.capex_spent)],
  ];
  rows.forEach(([k, v]) => kv.append(el("dt", "", k), el("dd", "", v)));
  kv.append(el("dt", "total", "Net cash flow"), el("dd", "total", money(d.net_cash_flow)));
}

function renderBreakdown() {
  const box = $("breakdown");
  box.replaceChildren();
  const r = last().reward;
  if (!r) { box.append(el("p", "caption", "No reward yet.")); return; }
  const w = state.meta.presets[state.difficulty].weights;
  const parts = [
    ["Profit", w.profit * r.profit],
    ["Solvency", w.solvency * r.solvency],
    ["GDP growth", w.gdp * r.gdp],
    ["Inequality", -Math.abs(w.gini) * r.gini_pen],
  ];
  const scale = Math.max(...parts.map((p) => Math.abs(p[1])), 0.05);
  parts.forEach(([name, val]) => {
    const row = el("div", "bd-row");
    row.append(el("span", "bd-name", name), el("span", "bd-val", (val >= 0 ? "+" : "") + val.toFixed(3)));
    const track = el("div", "bd-track");
    const bar = el("div", "bd-bar" + (val < 0 ? " neg" : ""));
    bar.style.left = "0";
    bar.style.width = pct(Math.abs(val) / scale, 1);
    track.append(bar);
    row.append(track);
    box.append(row);
  });
  const total = el("div", "bd-row");
  total.append(el("span", "bd-name", "Total"), el("span", "bd-val", r.composite.toFixed(3)));
  box.append(total);
  if (r.activity < 1) {
    box.append(el("p", "caption", `Solvency scaled by activity ${pct(r.activity)}`));
  }
}

function renderLabor() {
  const cur = last();
  const o = cur.obs, d = cur.details;
  const box = $("firms");
  box.replaceChildren();
  const n = state.meta.num_firms;
  for (let i = 0; i <= n; i++) {
    const isPool = i === n;
    const card = el("div", "firm" + (i === 0 ? " you" : "") + (!isPool && !o.firm_active[i] ? " exited" : ""));
    const name = el("div", "firm-name");
    const sw = el("i");
    sw.style.background = isPool ? "transparent" : firmColor(i);
    if (isPool) sw.style.boxShadow = `inset 0 0 0 1.5px ${css("--pool")}`;
    name.append(sw, document.createTextNode(isPool ? "Unemployed" : firmName(i)));
    card.append(name);

    const count = isPool ? o.unemployed : o.firm_workers[i];
    let meta;
    if (isPool) meta = `${count} looking for work`;
    else if (!o.firm_active[i]) meta = "Exited (bankrupt)";
    else meta = `${count} workers · ${money(o.firm_wages[i])}`;
    card.append(el("div", "firm-meta", meta));
    if (!isPool) card.append(el("div", "firm-meta", `Cash ${compact(o.agent_wealth[i])} · ${money2(o.firm_prices[i])}`));

    const dots = el("div", "dots");
    const gained = d ? (isPool ? d.labor_flows.filter((f) => f[1] === -1).reduce((a, f) => a + f[2], 0)
      : d.firm_hires[i]) : 0;
    for (let k = 0; k < count; k++) {
      const dot = el("span", "dot" + (isPool ? " hollow" : "") + (k >= count - gained ? " new" : ""));
      if (!isPool) dot.style.background = firmColor(i);
      dots.append(dot);
    }
    card.append(dots);

    const badges = el("div", "badges");
    if (d && !isPool) {
      if (d.firm_hires[i]) badges.append(el("span", "badge up", `+${d.firm_hires[i]} hired`));
      if (d.firm_quits[i]) badges.append(el("span", "badge down", `−${d.firm_quits[i]} quit`));
      if (d.firm_poached_out[i]) badges.append(el("span", "badge down", `−${d.firm_poached_out[i]} poached`));
      if (i === 0 && d.fired) badges.append(el("span", "badge down", `−${d.fired} fired`));
    }
    card.append(badges);
    box.append(card);
  }

  $("labor-caption").textContent =
    `${o.labor_force} workers · ${o.unemployed} unemployed · market wage ${money(o.market_wage)}`;

  const flows = $("flows");
  flows.replaceChildren();
  if (!d || !d.labor_flows.length) {
    flows.append(el("li", "", d ? "No workers moved this step." : "Worker moves will appear here."));
    return;
  }
  d.labor_flows.forEach(([src, dst, k]) => {
    const li = el("li");
    const dot = el("i");
    dot.style.background = firmColor(src);
    if (src === -1) dot.style.boxShadow = `inset 0 0 0 1.5px ${css("--pool")}`, dot.style.background = "transparent";
    li.append(dot, document.createTextNode(`${firmName(src)} → ${firmName(dst)} ×${k}`));
    flows.append(li);
  });
}

function renderFeedItem(entry) {
  const r = entry.reward;
  if (!r) return;
  const parts = r.feedback.split(" | ");
  const score = parts.find((p) => p.includes("(R=")) || "";
  const because = parts.find((p) => p.startsWith("Because: "));
  const status = parts.filter((p) => p.startsWith(">>>") || p.startsWith("WARNING")).map((p) => p.replace(">>> ", ""));
  const isFinancial = (p) => /^(Rev|Costs|Net|Cash) /.test(p) || p.startsWith("[Step");
  const assess = parts.filter((p) => !isFinancial(p) && p !== score && p !== because && !status.includes(p.replace(">>> ", "")));
  const li = el("li", /BANKRUPT|CRITICAL|WARNING/.test(r.feedback) ? "alert" : "");
  li.append(el("span", "f-step", `Step ${entry.obs.step}`));
  const body = el("div");
  body.append(el("div", "f-head", [score, ...status].filter(Boolean).join(" · ")));
  body.append(el("div", "f-why", assess.join(" · ")));
  if (because) body.append(el("div", "f-why", because));
  li.append(body);
  $("feed").prepend(li);
}

function renderTableRow(entry) {
  const o = entry.obs;
  const tr = el("tr");
  [
    o.step, money(o.firm_cash), o.worker_count, money2(o.current_price), money(o.wage),
    entry.details ? Math.round(entry.details.units_sold) : "–",
    entry.details ? pct(entry.details.market_share, 1) : "–",
    entry.reward ? entry.reward.composite.toFixed(3) : "–",
  ].forEach((v) => tr.append(el("td", "", String(v))));
  $("history-table").tBodies[0].prepend(tr);
}

function renderAll() {
  renderHeader();
  renderTiles();
  renderHints();
  renderSales();
  renderBreakdown();
  renderLabor();
  rebuildCharts();
}

// ---------------------------------------------------------------- game flow
async function resetGame() {
  setBusy(true);
  try {
    state.difficulty = $("difficulty").value;
    const seed = parseInt($("seed").value, 10);
    const obs = await api("/reset", {
      method: "POST",
      body: JSON.stringify({ difficulty: state.difficulty, seed: Number.isFinite(seed) ? seed : 42 }),
    });
    state.history = [{ obs, reward: null, details: null }];
    state.done = false;
    $("feed").replaceChildren();
    $("history-table").tBodies[0].replaceChildren();
    renderTableRow(state.history[0]);
    $("modal").hidden = true;
    setControlsFromObs(obs);
    renderAll();
  } catch (e) {
    toast(`Could not start a game: ${e.message}`);
  } finally {
    setBusy(false);
  }
}

async function stepOnce(includeOneTime) {
  const res = await api("/step", { method: "POST", body: JSON.stringify(currentAction(includeOneTime)) });
  const entry = { obs: res.observation, reward: res.reward, details: res.details };
  state.history.push(entry);
  state.done = res.terminated || res.truncated;
  state.bankrupt = res.terminated;
  renderFeedItem(entry);
  renderTableRow(entry);
  return res;
}

async function runSteps(count) {
  if (state.busy || state.done) return;
  setBusy(true);
  try {
    for (let k = 0; k < count && !state.done; k++) {
      await stepOnce(k === 0);
      renderAll();
    }
    resetOneTimeControls();
    if (state.done) showSummary();
  } catch (e) {
    toast(`Step failed: ${e.message}`);
  } finally {
    setBusy(false);
  }
}

function showSummary() {
  const o = last().obs;
  const bankrupt = state.bankrupt;
  const worth = o.firm_cash + o.capital_stock - (bankrupt ? state.meta.bankruptcy_penalty : 0);
  const cash = o.agent_wealth.slice(0, state.meta.num_firms);
  const rank = 1 + cash.slice(1).filter((c) => c > cash[0]).length;
  const totalReward = state.history.reduce((a, h) => a + (h.reward ? h.reward.composite : 0), 0);
  $("modal-title").textContent = bankrupt ? `Bankrupt at step ${o.step}` : "Episode complete";
  $("modal-sub").textContent = bankrupt
    ? `Net worth includes the ${money(state.meta.bankruptcy_penalty)} bankruptcy penalty.`
    : `You ran your firm for all ${o.episode_length} steps.`;
  const stats = $("modal-stats");
  stats.replaceChildren();
  [
    ["Net worth", money(worth)],
    ["Rank by cash", `${rank} of ${state.meta.num_firms}`],
    ["Total reward", totalReward.toFixed(2)],
    ["Workers at end", `${o.worker_count}`],
  ].forEach(([k, v]) => {
    const box = el("div");
    box.append(el("span", "", k), el("strong", "", v));
    stats.append(box);
  });
  $("modal").hidden = false;
  $("btn-again").focus();
}

$("btn-step").addEventListener("click", () => runSteps(1));
$("btn-auto5").addEventListener("click", () => runSteps(5));
$("btn-end").addEventListener("click", () => runSteps(1000));
$("btn-reset").addEventListener("click", resetGame);
$("btn-again").addEventListener("click", resetGame);
$("btn-close").addEventListener("click", () => { $("modal").hidden = true; });
$("difficulty").addEventListener("change", resetGame);

document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat) return;
  const t = e.target;
  if (t.tagName === "BUTTON" || t.tagName === "SELECT" || (t.tagName === "INPUT" && t.type !== "range")) return;
  if (!$("modal").hidden) return;
  e.preventDefault();
  runSteps(1);
});

// ---------------------------------------------------------------- boot
async function boot() {
  // URL options for demos: ?theme=dark|light  ?steps=N (auto-play N steps)
  const params = new URLSearchParams(location.search);
  const saved = params.get("theme") || safeStorage.get("localStorage", "openenv-theme");
  if (saved === "dark" || saved === "light") document.documentElement.dataset.theme = saved;
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(document.documentElement.dataset.theme));
  try {
    state.meta = await api("/meta");
  } catch (e) {
    toast(`Could not reach the server: ${e.message}`);
    return;
  }
  const sel = $("difficulty");
  Object.entries(state.meta.presets).forEach(([name, p]) => {
    const opt = el("option", "", `${name[0].toUpperCase()}${name.slice(1)} · ${p.episode_length} steps`);
    opt.value = name;
    sel.append(opt);
  });
  sel.value = state.meta.default_difficulty;
  applyBounds();
  initCharts();
  await resetGame();
  const autoSteps = parseInt(params.get("steps"), 10);
  if (autoSteps > 0) {
    inputs.wage.value = params.get("wage") || inputs.wage.value;
    inputs.hire.value = params.get("hire") || 0;
    syncOutputs();
    await runSteps(autoSteps);
  }
}
boot();
