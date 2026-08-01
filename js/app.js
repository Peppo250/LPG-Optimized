const API = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.hostname === ''
  ? 'http://localhost:8000'
  : (window.location.origin.includes('github.io') || window.location.origin.includes('vercel.app') || window.location.protocol === 'file:'
     ? 'https://lpg-optimized.onrender.com' // Replace with your actual Render app URL (e.g. lpg-catering-intelligence.onrender.com)
     : window.location.origin);

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const COLORS = ['#f78166', '#58a6ff', '#3fb950', '#d29922', '#8b949e', '#39d353', '#f85149', '#7f77dd'];
let METRICS = null, FI = null;

// ── API ──────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  // Cache-bust GET requests so browser never shows stale data
  const url = API + path + (opts.method ? '' : (path.includes('?') ? '&' : '?') + '_t=' + Date.now());
  const r = await fetch(url, {
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' },
    ...opts
  });
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

async function checkAPI() {
  try {
    await api('/healthz');
    document.getElementById('apibadge').className = 'badge badge-green';
    document.getElementById('apibadge').textContent = '\u25CF Live';
    document.getElementById('sstatus').textContent = '\u25CF API connected';
    document.getElementById('sstatus').style.color = 'var(--green)';
    return true;
  } catch {
    document.getElementById('apibadge').className = 'badge badge-red';
    document.getElementById('apibadge').textContent = '\u25CF API offline';
    document.getElementById('sstatus').textContent = '\u25CF API offline';
    document.getElementById('sstatus').style.color = 'var(--red)';
    toast('API offline \u2014 run: uvicorn api:app --port 8000 --reload');
    return false;
  }
}

// ── Navigation ───────────────────────────────────────────────────
const TITLES = {
  dashboard: 'Dashboard',
  predict: 'Predict Event',
  optimize: 'Refill Optimizer',
  regional: 'Regional Demand',
  simulation: 'Simulation',
  models: 'Model Metrics',
  features: 'Feature Importance'
};

function nav(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n => {
    if (n.textContent.trim().toLowerCase().replace(/\s+/g, '').includes(id.replace(/-/g, ''))) {
      n.classList.add('active');
    }
  });
  document.getElementById('ptitle').textContent = TITLES[id] || id;
  if (id === 'features' && FI) renderFI(FI);
  if (id === 'models' && METRICS) renderModelCards(METRICS);
}

// ── Dashboard ────────────────────────────────────────────────────
async function loadDash() {
  try {
    METRICS = await api('/api/metrics');
    const mp = METRICS.model_performance, ds = METRICS.dataset;
    // KPIs
    document.getElementById('k-r2').textContent = (mp.consumption_r2 * 100).toFixed(1) + '%';
    document.getElementById('k-mae').textContent = mp.consumption_mae_kg + ' kg';
    document.getElementById('k-auc').textContent = mp.stockout_auc.toFixed(3);
    document.getElementById('k-price').textContent = '\u20B9' + Number(METRICS.cylinder_price_inr).toLocaleString('en-IN');
    // Dataset
    document.getElementById('d-ev').textContent = Number(ds.total_events).toLocaleString('en-IN');
    document.getElementById('d-ft').textContent = ds.feature_columns;
    document.getElementById('d-et').textContent = ds.event_types;
    document.getElementById('d-so').textContent = Number(ds.stockout_events).toLocaleString('en-IN');
    // Sidebar
    document.getElementById('sfooter').innerHTML = 'v3 &middot; GBM+MLP<br>' + Number(ds.total_events).toLocaleString() + ' events &middot; ' + ds.feature_columns + ' features<br><span id="sstatus" style="font-size:10px;font-family:var(--mono);color:var(--green)">\u25CF API connected</span>';
    // Perf bars
    renderPerfBars(mp);
    // Charts
    if (METRICS.monthly_avg_consumption) renderMonthly(METRICS.monthly_avg_consumption);
    if (METRICS.event_distribution) renderDonut(METRICS.event_distribution);
    renderModelCards(METRICS);
  } catch (e) {
    document.getElementById('k-r2').textContent = 'offline';
    toast('Could not load metrics \u2014 is uvicorn running?');
  }
}

function renderPerfBars(mp) {
  const rows = [
    ['Consumption R\u00B2', mp.consumption_r2, 'pg'],
    ['Cylinders R\u00B2', mp.cylinders_r2, 'pg'],
    ['Stockout F1', mp.stockout_f1, 'pa'],
    ['Stockout AUC', mp.stockout_auc, 'pbl'],
  ];
  document.getElementById('perfbars').innerHTML = rows.map(([l, v, c]) => `
    <div style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px"><span>${l}</span><span style="font-family:var(--mono)">${v.toFixed(4)}</span></div>
      <div class="pb"><div class="pf ${c}" style="width:${(v * 100).toFixed(1)}%"></div></div>
    </div>`).join('');
}

function renderMonthly(data) {
  const vals = MONTHS.map((_, i) => data[i + 1] || 0);
  const ctx = document.getElementById('c-monthly').getContext('2d');
  if (window._mc) window._mc.destroy();
  window._mc = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: MONTHS,
      datasets: [{
        data: vals,
        backgroundColor: vals.map(v => v > 80 ? 'rgba(247,129,102,0.7)' : 'rgba(88,166,255,0.5)'),
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e', font: { size: 10 } } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e', font: { size: 10 } } }
      }
    }
  });
}

function renderDonut(dist) {
  const labels = Object.keys(dist).map(k => k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()));
  const vals = Object.values(dist);
  const ctx = document.getElementById('c-events').getContext('2d');
  if (window._ec) window._ec.destroy();
  window._ec = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{ data: vals, backgroundColor: COLORS, borderWidth: 0, hoverOffset: 4 }]
    },
    options: { cutout: '70%', plugins: { legend: { display: false } }, responsive: false }
  });
  document.getElementById('c-legend').innerHTML = labels.map((l, i) => `<div class="rl"><div class="rdot" style="background:${COLORS[i]}"></div><span style="flex:1">${l}</span><span style="font-family:var(--mono);color:var(--muted)">${vals[i].toFixed(1)}%</span></div>`).join('');
}

function renderModelCards(m) {
  const mp = m.model_performance, ds = m.dataset;
  function pr(l, v, c) {
    const pct = Math.min(100, parseFloat(v) * 100);
    return `<div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px"><span>${l}</span><span style="font-family:var(--mono);color:var(--${c})">${v}</span></div>
      <div class="pb"><div class="pf p${c}" style="width:${isNaN(pct) ? 0 : pct}%"></div></div>
    </div>`;
  }
  document.getElementById('mcards').innerHTML = `
    <div class="card"><div class="card-title">Consumption (kg) &middot; Regression</div>
      ${pr('GBM R\u00B2', mp.consumption_r2.toFixed(4), 'g')}
      ${pr('GBM MAE', mp.consumption_mae_kg + ' kg', 'bl')}
      ${pr('MLP R\u00B2', mp.mlp_consumption_r2.toFixed(4), 'a')}
      <div style="margin-top:12px"><span class="badge badge-green" style="font-size:11px">GBM wins &uarr;${((mp.consumption_r2 - mp.mlp_consumption_r2) * 100).toFixed(2)}% R\u00B2</span></div>
    </div>
    <div class="card"><div class="card-title">Cylinders Needed &middot; Regression</div>
      ${pr('GBM R\u00B2', mp.cylinders_r2.toFixed(4), 'g')}
      ${pr('GBM MAE', mp.cylinders_mae + ' cyl', 'bl')}
      ${pr('MLP R\u00B2', mp.mlp_cylinders_r2.toFixed(4), 'a')}
      <div style="margin-top:12px"><span class="badge badge-green" style="font-size:11px">GBM wins &uarr;${((mp.cylinders_r2 - mp.mlp_cylinders_r2) * 100).toFixed(2)}% R\u00B2</span></div>
    </div>
    <div class="card"><div class="card-title">Stockout &middot; Classifier</div>
      ${pr('F1 Score', mp.stockout_f1.toFixed(4), 'g')}
      ${pr('ROC-AUC', mp.stockout_auc.toFixed(4), 'bl')}
      ${pr('Accuracy', (mp.stockout_acc * 100).toFixed(1) + '%', 'a')}
      <div style="margin-top:12px"><span class="badge badge-blue" style="font-size:11px">GBM + SMOTE oversampling</span></div>
    </div>`;
  if (ds.real_sources && ds.real_sources.length) {
    document.getElementById('msources').style.display = 'block';
    document.getElementById('msrclist').innerHTML = ds.real_sources.map(s => `<div style="padding:6px 0;border-bottom:1px solid var(--border);font-size:13px">\u2192 ${s}</div>`).join('');
  }
}

// ── Feature importance ───────────────────────────────────────────
async function loadFI() {
  try {
    FI = await api('/api/feature-importance');
    // Top 5 on dashboard
    const top5 = FI.stockout_top10.slice(0, 5);
    const clrs = ['var(--accent)', 'var(--blue)', 'var(--green)', 'var(--amber)', 'var(--muted)'];
    document.getElementById('topfeats').innerHTML = top5.map((f, i) => `<div class="rl"><div class="rdot" style="background:${clrs[i]}"></div><span style="flex:1;font-size:12px">${f.feature}</span><span style="font-family:var(--mono);color:var(--muted);font-size:11px">${(f.stockout * 100).toFixed(1)}%</span></div>`).join('');
    renderFI(FI);
  } catch (e) {
    document.getElementById('topfeats').innerHTML = '<span style="color:var(--muted);font-size:12px">Feature data unavailable</span>';
  }
}

function renderFI(fi) {
  ['stockout', 'consumption'].forEach(key => {
    const id = 'c-fi' + (key === 'stockout' ? 's' : 'c');
    const el = document.getElementById(id); if (!el) return;
    if (el._ch) el._ch.destroy();
    const data = key === 'stockout' ? fi.stockout_top10 : fi.consumption_top10;
    if (!data) return;
    const labels = data.map(d => d.feature).reverse();
    const vals = data.map(d => d[key]).reverse();
    el._ch = new Chart(el.getContext('2d'), {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data: vals,
          backgroundColor: key === 'stockout' ? 'rgba(247,129,102,0.7)' : 'rgba(88,166,255,0.6)',
          borderRadius: 3
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e', font: { size: 10 } } },
          y: { grid: { display: false }, ticks: { color: '#8b949e', font: { size: 10 } } }
        }
      }
    });
  });
}

// ── Predict ──────────────────────────────────────────────────────
async function runPredict() {
  const btn = document.getElementById('pred-btn'); btn.disabled = true;
  document.getElementById('pred-spin').innerHTML = '<div class="spin" style="margin-right:6px"></div>';
  const body = {
    caterer_id: 'CAT001',
    caterer_name: document.getElementById('pn').value,
    experience_yrs: +document.getElementById('pe').value,
    num_burners: +document.getElementById('pb').value,
    business_size: document.getElementById('ps').value,
    event_date: document.getElementById('pd').value || new Date(Date.now() + 7 * 86400000).toISOString().slice(0, 10),
    event_type: document.getElementById('pt').value,
    headcount: +document.getElementById('phc').value,
    num_dishes: +document.getElementById('pnd').value,
    duration_hrs: +document.getElementById('pdr').value,
    menu_profile: document.getElementById('pm').value,
    is_festival_season: document.getElementById('pf').checked
  };
  try {
    const resp = await api('/api/predict', { method: 'POST', body: JSON.stringify(body) });
    const r = resp.data;
    document.getElementById('rp').classList.add('show');
    document.getElementById('r-cons').textContent = r.predicted_consumption_kg + ' kg';
    document.getElementById('r-cyl').textContent = r.cylinders_to_order;
    document.getElementById('r-cost').textContent = '\u20B9' + Number(r.estimated_cost_inr).toLocaleString('en-IN');
    document.getElementById('r-date').textContent = r.recommended_order_date;
    document.getElementById('r-risk').textContent = r.stockout_risk_pct + '%';
    document.getElementById('r-risk').style.color = r.stockout_risk_pct > 50 ? 'var(--red)' : r.stockout_risk_pct > 25 ? 'var(--amber)' : 'var(--green)';
    document.getElementById('r-eff').textContent = r.efficiency_score + '/100';
    const t = document.getElementById('r-tier'); t.textContent = r.recommendation_tier; t.className = 'tier tier-' + r.recommendation_tier;
    document.getElementById('r-actions').innerHTML = r.action_items.map(a => `<li>${a}</li>`).join('');
    document.getElementById('r-notes').textContent = r.optimization_notes || '';
    document.getElementById('r-src').textContent = r.ml_consumption_kg ? `\u00B7 ML: ${r.ml_consumption_kg} kg (blended)` : `\u00B7 Rule-based (run train_final.py to enable ML)`;
    toast('Prediction from API');
  } catch (e) {
    toast('Predict failed \u2014 API offline?');
  }
  btn.disabled = false; document.getElementById('pred-spin').innerHTML = '';
}

// ── Optimize ─────────────────────────────────────────────────────
const DEVENTS = [
  { n: 'Murugan Grand Catering', e: 15, b: 12, s: 'large', et: 'wedding', hc: 800, nd: 9, d: 8.0, m: 'nonveg_elaborate', f: true },
  { n: 'Meenakshi Events', e: 7, b: 6, s: 'medium', et: 'corporate_lunch', hc: 250, nd: 5, d: 3.5, m: 'veg_simple', f: false },
  { n: 'Balaji Caterers', e: 2, b: 3, s: 'small', et: 'birthday_party', hc: 80, nd: 4, d: 3.0, m: 'mixed_standard', f: false },
  { n: 'Annapoorna Services', e: 10, b: 8, s: 'medium', et: 'festival_event', hc: 400, nd: 7, d: 6.0, m: 'veg_elaborate', f: true },
  { n: 'Royal Weddings', e: 20, b: 18, s: 'large', et: 'wedding', hc: 1500, nd: 10, d: 9.0, m: 'nonveg_elaborate', f: true },
  { n: 'College Canteen', e: 5, b: 4, s: 'small', et: 'college_canteen', hc: 300, nd: 4, d: 2.0, m: 'veg_simple', f: false },
];

function dfn(n) {
  const d = new Date();
  d.setDate(d.getDate() + n);
  return d.toISOString().slice(0, 10);
}

async function loadOptTable() {
  document.getElementById('opt-hint').textContent = 'Loading predictions from API...';
  const ps = DEVENTS.map((e, i) => api('/api/predict', {
    method: 'POST',
    body: JSON.stringify({
      caterer_id: `CAT00${i + 1}`,
      caterer_name: e.n,
      experience_yrs: e.e,
      num_burners: e.b,
      business_size: e.s,
      event_date: dfn(3 + i * 2),
      event_type: e.et,
      headcount: e.hc,
      num_dishes: e.nd,
      duration_hrs: e.d,
      menu_profile: e.m,
      is_festival_season: e.f
    })
  }).then(r => ({ ...r.data, name: e.n })).catch(() => null));

  const res = (await Promise.all(ps)).filter(Boolean);
  if (!res.length) {
    document.getElementById('etbody').innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:20px">API offline</td></tr>';
    document.getElementById('opt-hint').textContent = 'Start uvicorn to load live predictions.';
    return;
  }
  document.getElementById('etbody').innerHTML = res.map(r => {
    const tc = r.recommendation_tier === 'GREEN' ? 'badge-green' : r.recommendation_tier === 'AMBER' ? 'badge-amber' : 'badge-red';
    return `<tr>
      <td>${r.name || r.caterer_id}</td>
      <td style="font-family:var(--mono)">${r.event_date}</td>
      <td>${(r.event_type || '').replace(/_/g, ' ')}</td>
      <td style="font-family:var(--mono)">${r.headcount || '\u2014'}</td>
      <td style="font-family:var(--mono)">${r.predicted_consumption_kg} kg</td>
      <td style="font-family:var(--mono)">${r.cylinders_to_order}</td>
      <td><span class="badge ${tc}">${r.recommendation_tier}</span></td>
    </tr>`;
  }).join('');
  document.getElementById('opt-hint').textContent = `LP optimizer minimizes total cost+wastage across ${res.length} events. All values from trained GBM model.`;
  toast('Events loaded from API');
}

async function runBatch() {
  document.getElementById('bspin').innerHTML = '<div class="spin" style="margin-right:6px"></div>';
  try {
    const evs = DEVENTS.map((e, i) => ({
      caterer_id: `CAT00${i + 1}`,
      caterer_name: e.n,
      experience_yrs: e.e,
      num_burners: e.b,
      business_size: e.s,
      event_date: dfn(3 + i * 2),
      event_type: e.et,
      headcount: e.hc,
      num_dishes: e.nd,
      duration_hrs: e.d,
      menu_profile: e.m,
      is_festival_season: e.f
    }));
    const r = await api('/api/batch-optimize', { method: 'POST', body: JSON.stringify({ events: evs, use_lp: true }) });
    const s = r.summary;
    document.getElementById('lpkpis').style.display = 'grid';
    document.getElementById('lp-cyl').textContent = s.total_cylinders;
    document.getElementById('lp-cost').textContent = '\u20B9' + Number(s.total_cost_inr).toLocaleString('en-IN');
    document.getElementById('lp-risk').textContent = s.high_risk_events;
    toast('LP optimization from API');
  } catch (e) {
    toast('Batch optimize failed');
  }
  document.getElementById('bspin').innerHTML = '';
}

// ── Regional ─────────────────────────────────────────────────────
let RC = null;
async function runRegional() {
  document.getElementById('regspin').innerHTML = '<div class="spin" style="margin-right:6px"></div>';
  try {
    const mo = document.getElementById('sm') ? document.getElementById('sm').value : 11;
    const r = await api(`/api/regional-forecast?month=${mo}&n_caterers=40`);
    const cur = r.demand_curve || [], s = r.summary || {}, im = r.improvement || {};
    document.getElementById('regkpis').style.display = 'grid';
    document.getElementById('regchart').style.display = 'block';
    document.getElementById('reg-avg').textContent = s.avg_daily_demand || '\u2014';
    document.getElementById('reg-peak').textContent = s.peak_demand_cylinders || '\u2014';
    document.getElementById('reg-spk').textContent = s.spike_days_count || '\u2014';
    document.getElementById('reg-wr').textContent = (im.wastage_reduction_pct || '\u2014') + '%';
    if (RC) RC.destroy();
    RC = new Chart(document.getElementById('c-reg').getContext('2d'), {
      type: 'line',
      data: {
        labels: cur.map(r => r.date ? r.date.slice(5) : ''),
        datasets: [
          {
            label: 'Raw',
            data: cur.map(r => Math.round(r.raw_demand || 0)),
            borderColor: '#f85149',
            backgroundColor: 'rgba(248,81,73,0.08)',
            fill: true,
            tension: 0.3,
            pointRadius: 0
          },
          {
            label: 'Smoothed',
            data: cur.map(r => Math.round(r.smoothed_demand || r.raw_demand || 0)),
            borderColor: '#3fb950',
            backgroundColor: 'rgba(63,185,80,0.08)',
            fill: true,
            tension: 0.4,
            pointRadius: 0
          },
          {
            label: 'Capacity',
            data: Array(cur.length).fill(s.dealer_capacity || 200),
            borderColor: 'rgba(210,153,34,0.6)',
            borderDash: [6, 3],
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: true, labels: { color: '#8b949e', font: { size: 11 } } } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e', font: { size: 10 } } },
          y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e', font: { size: 10 } } }
        }
      }
    });
    toast('Forecast from API');
  } catch (e) {
    toast('Regional forecast failed');
  }
  document.getElementById('regspin').innerHTML = '';
}

// ── Simulation ───────────────────────────────────────────────────
let SC = null;
async function runSim() {
  document.getElementById('sspin').innerHTML = '<div class="spin" style="margin-right:6px"></div>';
  try {
    const n = document.getElementById('sn').value, mo = document.getElementById('sm').value;
    const r = await api(`/api/simulation?n_caterers=${n}&month=${mo}`);
    const d = r.data, b = d.before_optimization, a = d.after_optimization, im = d.improvement;
    document.getElementById('simres').style.display = 'block';
    document.getElementById('simchart').style.display = 'block';
    document.getElementById('simmet').innerHTML = `<table style="font-size:13px;width:100%">
      <tr><td style="color:var(--muted);padding:6px 0">Caterers</td><td style="text-align:right;font-family:var(--mono)">${d.simulation.caterers}</td></tr>
      <tr><td style="color:var(--muted);padding:6px 0">Total events</td><td style="text-align:right;font-family:var(--mono)">${d.simulation.total_events}</td></tr>
      <tr><td style="color:var(--green);padding:6px 0">Wastage reduction</td><td style="text-align:right;font-family:var(--mono);color:var(--green)">${im.wastage_reduction_pct}%</td></tr>
      <tr><td style="color:var(--green);padding:6px 0">Peak demand reduction</td><td style="text-align:right;font-family:var(--mono);color:var(--green)">${im.peak_demand_reduction_pct}%</td></tr>
      <tr><td style="color:var(--blue);padding:6px 0">Cost saving</td><td style="text-align:right;font-family:var(--mono);color:var(--blue)">\u20B9${Number(im.cost_saving_inr).toLocaleString('en-IN')}</td></tr>
    </table>`;
    if (SC) SC.destroy();
    SC = new Chart(document.getElementById('c-sim').getContext('2d'), {
      type: 'bar',
      data: {
        labels: ['Wastage (kg)', 'Peak Demand', 'Cost (\u00F71000)'],
        datasets: [
          {
            label: 'Before',
            data: [b.total_wastage_kg, b.peak_daily_demand, Math.round(b.total_cost_inr / 1000)],
            backgroundColor: 'rgba(248,81,73,0.6)'
          },
          {
            label: 'After',
            data: [a.total_wastage_kg, a.peak_daily_demand, Math.round(a.total_cost_inr / 1000)],
            backgroundColor: 'rgba(63,185,80,0.6)'
          }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: true, labels: { color: '#8b949e', font: { size: 11 } } } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e', font: { size: 10 } } },
          y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e', font: { size: 10 } } }
        }
      }
    });
    toast('Simulation from API \u2014 ' + im.wastage_reduction_pct + '% wastage reduction');
  } catch (e) {
    toast('Simulation failed');
  }
  document.getElementById('sspin').innerHTML = '';
}

// ── Toast ────────────────────────────────────────────────────────
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = '\u2713 ' + msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2800);
}

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  document.getElementById('tdate').textContent = new Date().toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
  const d = new Date();
  d.setDate(d.getDate() + 7);
  document.getElementById('pd').value = d.toISOString().slice(0, 10);
  await checkAPI();
  await loadDash();
  await loadOptTable();
  await loadFI();
  try {
    const m = await api('/api/metrics');
    console.log('[LPG Dashboard] Live API metrics:', {
      r2: m.model_performance.consumption_r2,
      mae: m.model_performance.consumption_mae_kg,
      auc: m.model_performance.stockout_auc,
      rows: m.dataset.total_events,
      stockout: m.dataset.stockout_events,
    });
  } catch (e) { console.warn('Could not verify API metrics:', e); }
  setInterval(checkAPI, 30000);
});
