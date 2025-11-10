const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));
const sleep = ms => new Promise(r => setTimeout(r, ms));

function toast(msg, kind = "ok") {
  const wrap = $("#toasts");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  wrap.appendChild(el);
  setTimeout(() => { el.remove(); }, 4000);
}

function setBtnLoading(btn, on) {
  if (!btn) return;
  btn.disabled = !!on;
  if (on) btn.classList.add("loading"); else btn.classList.remove("loading");
}

async function api(path, opts = {}) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), opts.timeout || 10000);
  try {
    const res = await fetch(path, { ...opts, signal: ctrl.signal, headers: { 'Content-Type': 'application/json', ...(opts.headers||{}) } });
    const ct = res.headers.get('content-type') || '';
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    if (ct.includes('application/json')) return await res.json();
    return await res.text();
  } finally {
    clearTimeout(t);
  }
}

async function refreshStatus() {
  try {
    const s = await api('/api/status');
    $('#mode-chip').textContent = `mode: ${s.mode}`;
    $('#active-chip').textContent = s.active ? 'active' : 'stopped';
    $('#active-chip').style.background = s.active ? '#14532d' : '#1f2937';
    $('#errors-chip').textContent = `errors: ${s.error_count || 0}`;
  } catch (e) {
    $('#errors-chip').textContent = 'errors: ?';
  }
  try {
    const h = await api('/health');
    $('#health-chip').textContent = h.ok ? 'ok' : 'degraded';
  } catch {
    $('#health-chip').textContent = 'down';
  }
}

async function hookControls() {
  $('#btn-start').addEventListener('click', async (e) => {
    setBtnLoading(e.target, true);
    try { await api('/start', { method: 'POST' }); toast('Engine started'); } catch (err) { toast('Start failed', 'err'); } finally { setBtnLoading(e.target, false); refreshStatus(); }
  });
  $('#btn-stop').addEventListener('click', async (e) => {
    setBtnLoading(e.target, true);
    try { await api('/agent/stop', { method: 'POST' }); toast('Engine stopped'); } catch (err) { toast('Stop failed', 'err'); } finally { setBtnLoading(e.target, false); refreshStatus(); }
  });
  $('#btn-reset').addEventListener('click', async (e) => {
    setBtnLoading(e.target, true);
    try { await api('/api/state/reset', { method: 'POST', body: JSON.stringify({ confirm: true }) }); toast('State reset'); await refreshAll(); } catch (err) { toast('Reset failed', 'err'); } finally { setBtnLoading(e.target, false); refreshStatus(); }
  });
}

async function refreshFusion() {
  try {
    const data = await api('/fusion/ai');
    $('#fusion-outlook').innerHTML = `<pre class="pre">${JSON.stringify(data, null, 2)}</pre>`;
  } catch (e) {
    $('#fusion-outlook').innerHTML = `<div class="placeholder">Failed to load</div>`;
  }
}

async function hookFusion() {
  $('#btn-fusion-refresh').addEventListener('click', async (e) => {
    setBtnLoading(e.target, true);
    try { await api('/fusion/refresh', { method: 'POST' }); await refreshFusion(); toast('Fusion refreshed'); } catch { toast('Fusion refresh failed', 'err'); } finally { setBtnLoading(e.target, false); }
  });
}

function renderMoversList(el, arr, keySym = 'sym') {
  el.innerHTML = '';
  for (const it of arr || []) {
    const sym = (it.symbol || it[keySym] || '').toUpperCase();
    const chg = typeof it.change_pct === 'number' ? it.change_pct : (typeof it['change_24h'] === 'number' ? it['change_24h'] : 0);
    const gps = typeof it.gps === 'number' ? it.gps : (typeof it.gps_score === 'number' ? it.gps_score : null);
    const li = document.createElement('li');
    const sign = chg >= 0 ? 'up' : 'down';
    li.className = 'movers-item';
    li.innerHTML = `<span class="sym">${sym}</span><span class="chg ${sign}">${chg.toFixed(2)}%</span>${gps!=null?`<span class="gps">GPS ${gps}</span>`:''}`;
    el.appendChild(li);
  }
}

async function refreshMovers() {
  try {
    const data = await api('/api/top_movers');
    renderMoversList($('#movers-stocks'), data.stocks || [], 'sym');
    renderMoversList($('#movers-crypto'), data.crypto || [], 'id');
  } catch (e) { /* ignore */ }
}

function heatColor(score) {
  const s = Math.max(0, Math.min(10, Number(score || 0)));
  const g = Math.round((s/10)*255);
  const r = Math.round((1 - s/10)*255);
  return `rgb(${r}, ${g}, 64)`;
}

function renderHeatmap(items) {
  const box = $('#heatmap');
  if (!items || !items.length) {
    box.innerHTML = `<div class="placeholder">No data</div>`;
    return;
  }
  const grid = document.createElement('div');
  grid.style.display = 'grid';
  grid.style.gridTemplateColumns = 'repeat(4, minmax(0, 1fr))';
  grid.style.gap = '8px';
  for (const it of items) {
    const sym = (it.symbol || it.id || it.name || '?').toUpperCase();
    const gps = Number(it.gps || it.gps_score || it.score || 5);
    const cell = document.createElement('div');
    cell.style.padding = '10px';
    cell.style.border = '1px solid #1f2937';
    cell.style.borderRadius = '8px';
    cell.style.background = '#0b1220';
    cell.innerHTML = `<div style="font-weight:700">${sym}</div><div style="margin-top:4px"><span class="gps">GPS ${gps.toFixed(1)}</span></div>`;
    cell.querySelector('.gps').style.background = '#334155';
    cell.querySelector('.gps').style.padding = '2px 6px';
    cell.querySelector('.gps').style.borderRadius = '999px';
    cell.style.boxShadow = `inset 0 0 0 2px ${heatColor(gps)}22`;
    grid.appendChild(cell);
  }
  box.innerHTML = '';
  box.appendChild(grid);
}

async function refreshHeatmap() {
  try {
    let data = await api('/heatmap');
    if (!Array.isArray(data) || data.length === 0) {
      // Fallback: synthesize from /api/signals
      const sig = await api('/api/signals');
      const items = [];
      for (const [sym, v] of Object.entries(sig || {})) {
        items.push({ symbol: sym, gps: (v && typeof v.confidence === 'number') ? (v.confidence*10) : 5 });
      }
      data = items.slice(0, 8);
    }
    renderHeatmap(data);
  } catch (e) {
    $('#heatmap').innerHTML = `<div class="placeholder">Failed to load</div>`;
  }
}

async function refreshDiagnostics(run = false) {
  const btn = $('#btn-diag-refresh');
  setBtnLoading(btn, run);
  try {
    const data = run ? await api('/diagnostics/run') : await api('/diagnostics/summary');
    $('#diag-json').textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    $('#diag-json').textContent = 'Failed to fetch diagnostics';
  } finally { setBtnLoading(btn, false); }
}

function hookDiagnostics() {
  $('#btn-diag-refresh').addEventListener('click', async () => { await refreshDiagnostics(true); toast('Diagnostics refreshed'); });
}

async function refreshBank() {
  try {
    const b = await api('/api/bank');
    $('#bank-summary').textContent = `Cash $${(b.cash_balance||0).toFixed(2)} | Stock $${(b.stock_cash||0).toFixed(2)} | Crypto $${(b.crypto_cash||0).toFixed(2)}`;
  } catch { $('#bank-summary').textContent = 'Bank unavailable'; }
}

function hookBank() {
  $('#btn-set-cash').addEventListener('click', async (e) => {
    const amt = Number($('#set-cash').value||0);
    if (!amt) return;
    setBtnLoading(e.target, true);
    try { await api('/api/set_cash', { method:'POST', body: JSON.stringify({ amount: amt })}); toast('Cash set'); await refreshBank(); } catch { toast('Set cash failed', 'err'); } finally { setBtnLoading(e.target, false); }
  });
  $('#btn-reset-bank').addEventListener('click', async (e) => {
    const amt = Number($('#reset-cash').value||1000);
    setBtnLoading(e.target, true);
    try { await api('/api/bank/reset', { method:'POST', body: JSON.stringify({ amount: amt })}); toast('Bank reset'); await refreshBank(); } catch { toast('Reset bank failed', 'err'); } finally { setBtnLoading(e.target, false); }
  });
}

async function refreshPortfolio() {
  try {
    const p = await api('/portfolio');
    const tbody = $('#portfolio-table tbody');
    tbody.innerHTML = '';
    for (const row of (p.positions||[])) {
      const tr = document.createElement('tr');
      const pnl = (Number(row.current_price||0) - Number(row.entry_price||0)) * Number(row.quantity||0);
      tr.innerHTML = `<td>${row.symbol}</td><td>${row.type}</td><td>${row.quantity}</td><td>${row.entry_price?.toFixed?row.entry_price.toFixed(2):row.entry_price}</td><td>${row.current_price?.toFixed?row.current_price.toFixed(2):row.current_price}</td><td>${pnl.toFixed(2)}</td><td>${(row.gps||'')}</td><td><div class="row-actions"><button class="btn btn-sm" data-sell="${row.symbol}">Close 1</button></div></td>`;
      tbody.appendChild(tr);
    }
    $$('#portfolio-table [data-sell]').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const sym = e.target.getAttribute('data-sell');
        setBtnLoading(e.target, true);
        try { await api('/api/positions/close', { method:'POST', body: JSON.stringify({ symbol: sym, qty: 1 })}); toast(`Closed 1 ${sym}`); await refreshPortfolio(); await refreshBank(); } catch { toast('Close failed', 'err'); } finally { setBtnLoading(e.target, false); }
      });
    });
  } catch { /* ignore */ }
}

function hookPortfolioModal() {
  const modal = $('#modal');
  $('#btn-add-pos').addEventListener('click', () => { modal.classList.remove('hidden'); });
  $('#btn-modal-close').addEventListener('click', () => { modal.classList.add('hidden'); });
  $('#btn-modal-save').addEventListener('click', async () => {
    const symbol = String($('#pos-symbol').value||'').trim().toUpperCase();
    const type = $('#pos-type').value;
    const qty = Number($('#pos-qty').value||0);
    const price = Number($('#pos-price').value||0);
    if (!symbol || !qty || !price) { toast('Fill symbol, qty, price', 'err'); return; }
    setBtnLoading($('#btn-modal-save'), true);
    try { await api('/api/bank/add_position', { method:'POST', body: JSON.stringify({ symbol, quantity: qty, price, type }) }); toast('Position added'); modal.classList.add('hidden'); await refreshPortfolio(); await refreshBank(); } catch { toast('Add position failed', 'err'); } finally { setBtnLoading($('#btn-modal-save'), false); }
  });
}

function hookWatchlist() {
  $('#btn-import-watchlist').addEventListener('click', async (e) => {
    const stocks = $('#wl-stocks').value;
    const crypto = $('#wl-crypto').value;
    setBtnLoading(e.target, true);
    try { await api('/watchlist/import', { method:'POST', body: JSON.stringify({ stocks, crypto })}); toast('Watchlist imported'); } catch { toast('Import failed', 'err'); } finally { setBtnLoading(e.target, false); }
  });
  $('#btn-load-watchlist').addEventListener('click', async (e) => {
    setBtnLoading(e.target, true);
    try { const wl = await api('/watchlist'); $('#wl-stocks').value = (wl.assets||[]).filter(a=>a.type==='stock').map(a=>a.symbol).join(', '); $('#wl-crypto').value = (wl.assets||[]).filter(a=>a.type==='crypto').map(a=>a.symbol).join(', '); } catch { toast('Load failed', 'err'); } finally { setBtnLoading(e.target, false); }
  });
  $('#btn-clear-watchlist').addEventListener('click', async (e) => {
    setBtnLoading(e.target, true);
    try { await api('/watchlist/clear', { method:'POST' }); toast('Watchlist cleared'); } catch { toast('Clear failed', 'err'); } finally { setBtnLoading(e.target, false); }
  });
  $('#btn-remove-watchlist').addEventListener('click', async (e) => {
    const symbols = $('#remove-input').value;
    if (!symbols) return;
    setBtnLoading(e.target, true);
    try { await api('/watchlist/remove', { method:'POST', body: JSON.stringify({ symbols })}); toast('Symbols removed'); } catch { toast('Remove failed', 'err'); } finally { setBtnLoading(e.target, false); }
  });
}

function connectSSE() {
  const chip = $('#sse-chip');
  try {
    const es = new EventSource('/events');
    es.onopen = () => { chip.classList.add('connected'); };
    es.onerror = () => { chip.classList.remove('connected'); };
    es.onmessage = (e) => { /* could parse snapshot if desired */ };
  } catch { /* ignore */ }
}

async function refreshAll() {
  await Promise.all([
    refreshStatus(),
    refreshFusion(),
    refreshHeatmap(),
    refreshMovers(),
    refreshDiagnostics(false),
    refreshBank(),
    refreshPortfolio(),
  ]);
}

async function main() {
  hookControls();
  hookFusion();
  hookDiagnostics();
  hookBank();
  hookPortfolioModal();
  hookWatchlist();
  connectSSE();
  await refreshAll();
  // periodic refresh
  setInterval(refreshStatus, 5000);
  setInterval(refreshMovers, 15000);
}

main().catch(err => console.error(err));
