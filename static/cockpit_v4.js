/* Ghost Protocol v4 — Verification-First Dashboard
 *
 * Design rules:
 * 1. Every number on screen must match what Telegram sends
 * 2. Picks come from ghost_tracked_picks (same DB the Telegram reads)
 * 3. Confidence < 50% = HOLD (no contradictory arrows)
 * 4. One card = one truth. No duplicates.
 */

// ─── STATE ───
let _picks = [];
let _trades = [];
let _watchlist = [];
let _news = [];
let _historyFilter = 'all';

// ─── BOOT ───
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initHeader();
    startClock();
    loadAll();
    setInterval(loadAll, 30000);
});

// ─── TAB NAVIGATION ───
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.tab-page').forEach(p => p.classList.remove('active'));
            const page = document.getElementById('tab-' + btn.dataset.tab);
            if (page) page.classList.add('active');
        });
    });
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            _historyFilter = btn.dataset.filter;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderHistory();
        });
    });
}

// ─── HEADER ───
function initHeader() {
    const startBtn = document.getElementById('btn-start');
    const stopBtn = document.getElementById('btn-stop');
    if (startBtn) startBtn.addEventListener('click', () => postJSON('/api/cockpit', { action: 'start' }));
    if (stopBtn) stopBtn.addEventListener('click', () => postJSON('/api/cockpit', { action: 'stop' }));
}

function startClock() {
    const el = document.getElementById('system-time');
    if (!el) return;
    const tick = () => { el.textContent = new Date().toLocaleTimeString('en-US', { hour12: true }); };
    tick();
    setInterval(tick, 1000);
}

// ─── MASTER LOADER ───
async function loadAll() {
    const results = await Promise.allSettled([
        fetchJSON('/api/v4/picks'),            // 0 – picks from same source as Telegram
        fetchJSON('/api/v3/paper/trades?limit=200'), // 1 – paper trades
        fetchJSON('/api/v3/watchlist'),         // 2 – watchlist
        fetchJSON('/api/v3/news/feed'),         // 3 – news
        fetchJSON('/api/v3/accuracy/summary'),  // 4 – accuracy
        fetchJSON('/api/v3/heartbeat/status'),  // 5 – heartbeat
        fetchJSON('/integrity/audit/readonly'), // 6 – integrity audit
    ]);

    const val = i => results[i].status === 'fulfilled' ? results[i].value : null;

    // Store data
    const picksData = val(0);
    const tradesData = val(1);
    const watchData = val(2);
    const newsData = val(3);
    const accData = val(4);
    const hbData = val(5);
    const auditData = val(6);

    if (picksData?.ok) _picks = picksData.picks || [];
    if (tradesData?.ok) _trades = tradesData.trades || [];
    if (watchData?.ok) _watchlist = watchData.items || watchData.watchlist || [];
    if (newsData?.ok) _news = newsData.articles || newsData.feed || [];

    // ── Header health pill ──
    const pill = document.getElementById('health-pill');
    if (pill && auditData) {
        const s = auditData.health_score ?? '--';
        pill.textContent = s + '/100';
        pill.style.borderColor = s >= 70 ? 'var(--green)' : s >= 40 ? 'var(--yellow)' : 'var(--red)';
    }

    // ── Status dot ──
    setStatus(!!picksData || !!tradesData);

    // ── Greeting bar ──
    const dateEl = document.getElementById('greeting-date');
    const subEl = document.getElementById('greeting-sub');
    if (dateEl) {
        dateEl.textContent = new Date().toLocaleDateString('en-US', {
            weekday: 'long', month: 'long', day: 'numeric', year: 'numeric'
        });
    }
    if (subEl && accData) {
        const pct = accData.accuracy_pct ?? 0;
        const correct = accData.correct_predictions ?? 0;
        const total = accData.total_predictions ?? 0;
        const status = accData.accuracy_status || '';
        const pickCount = _picks.length;
        subEl.textContent = `${pickCount} picks today · ${pct}% accuracy · ${correct}/${total} correct · ${status}`;
    }

    // ── Render all sections ──
    renderPicks();
    renderActiveTrades('stock', 'stock-active-trades');
    renderActiveTrades('crypto', 'crypto-active-trades');
    renderWatchlist('stock', 'stock-watchlist');
    renderWatchlist('crypto', 'crypto-watchlist');
    renderNews('stock-news');
    renderNews('crypto-news');
    renderHistory();
    renderHealth(accData, hbData, auditData);
}

function setStatus(alive) {
    const dot = document.getElementById('status-indicator');
    const txt = document.getElementById('status-text');
    if (dot) dot.style.background = alive ? 'var(--green)' : 'var(--red)';
    if (txt) {
        txt.textContent = alive ? 'LIVE' : 'OFFLINE';
        txt.style.color = alive ? 'var(--green)' : 'var(--red)';
    }
}

// ═══════════════════════════════════════════
// PICKS TAB — Telegram-format cards
// ═══════════════════════════════════════════
function renderPicks() {
    const el = document.getElementById('all-picks');
    if (!el) return;

    if (!_picks.length) {
        el.innerHTML = '<div class="no-picks">No picks right now — Ghost is watching</div>';
        return;
    }

    el.innerHTML = _picks.map((p, i) => {
        const isUp = (p.direction || '').toUpperCase() === 'UP';
        const sideClass = isUp ? 'bullish' : 'bearish';
        const emoji = isUp ? '🟢' : '🔴';
        const dirWord = isUp ? 'UP' : 'DOWN';
        const star = p.whitelisted ? ' <span class="pick-star">⭐</span>' : '';

        const entry = fmtPrice(p.entry_price);
        const target = fmtPrice(p.target_price);
        const stop = fmtPrice(p.stop_loss);
        const gainPct = p.gain_pct != null ? Math.abs(p.gain_pct).toFixed(1) : '--';
        const returnVal = p.gain_pct != null ? (100 + Math.abs(p.gain_pct)).toFixed(2) : '--';
        const deadline = p.done_by || '--';

        // Status badge
        const status = (p.status || 'pending').toLowerCase();
        let statusClass = 'pending';
        let statusLabel = 'PENDING';
        if (status === 'won' || status === 'win' || status === 'correct' || status === 'target_hit') {
            statusClass = 'won'; statusLabel = 'WON';
        } else if (status === 'lost' || status === 'loss' || status === 'incorrect' || status === 'stop_hit') {
            statusClass = 'lost'; statusLabel = 'LOST';
        } else if (status === 'expired') {
            statusClass = 'expired'; statusLabel = 'EXPIRED';
        }

        const typeLabel = (p.type || p.market || '').toUpperCase();

        return `
        <div class="pick-card ${sideClass}">
            <div class="pick-headline">
                ${emoji} <strong>${p.symbol}</strong> is going <strong>${dirWord}</strong>${star}
            </div>
            <div class="pick-body">
                <div class="pick-row">
                    <span class="pick-label">Get in at</span>
                    <span class="pick-val">${entry}</span>
                </div>
                <div class="pick-row">
                    <span class="pick-label">Get out at</span>
                    <span class="pick-val green">${target}  (you make ${gainPct}%)</span>
                </div>
                <div class="pick-row">
                    <span class="pick-label">Run away at</span>
                    <span class="pick-val red">${stop}</span>
                </div>
                <div class="pick-row">
                    <span class="pick-label">Done by</span>
                    <span class="pick-val">${deadline}</span>
                </div>
            </div>
            <div class="pick-footer">
                <span class="pick-return green">$100 in → $${returnVal} back</span>
                <span class="pick-status ${statusClass}">${statusLabel}</span>
            </div>
        </div>`;
    }).join('');
}

// ═══════════════════════════════════════════
// ACTIVE TRADES (Stocks / Crypto tabs)
// ═══════════════════════════════════════════
function renderActiveTrades(assetType, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const pending = _trades.filter(t => {
        const outcome = (t.outcome || t.status || '').toLowerCase();
        if (outcome && outcome !== 'pending' && outcome !== 'open') return false;
        const tType = (t.market || t.type || '').toLowerCase();
        return assetType === 'stock'
            ? (tType === 'stock' || tType === 'stocks')
            : tType === 'crypto';
    });

    if (!pending.length) {
        el.innerHTML = '<div class="loading-msg">No active trades</div>';
        return;
    }

    el.innerHTML = pending.slice(0, 8).map(t => {
        const pnl = t.unrealized_pnl || t.pnl || 0;
        const pnlClass = pnl >= 0 ? 'green' : 'red';
        const pnlStr = (pnl >= 0 ? '+' : '') + '$' + Math.abs(pnl).toFixed(2);
        const dir = (t.direction || t.signal_direction || '--').toUpperCase();
        return `
        <div class="trade-card">
            <div class="trade-left">
                <span class="trade-sym">${t.symbol || '--'}</span>
                <span class="trade-meta">${dir} · ${fmtPrice(t.entry_price || t.signal_price)}</span>
            </div>
            <div class="trade-right">
                <span class="trade-pnl ${pnlClass}">${pnlStr}</span>
                <span class="trade-status">Open</span>
            </div>
        </div>`;
    }).join('');
}

// ═══════════════════════════════════════════
// WATCHLIST (Stocks / Crypto tabs)
// ═══════════════════════════════════════════
function renderWatchlist(assetType, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const items = _watchlist.filter(w => {
        const wType = (w.type || '').toLowerCase();
        return assetType === 'stock' ? wType === 'stock' : wType === 'crypto';
    });

    if (!items.length) {
        el.innerHTML = '<div class="loading-msg">No items</div>';
        return;
    }

    el.innerHTML = items.map(w => {
        const changePct = w.change_pct || 0;
        const changeClass = changePct >= 0 ? 'green' : 'red';
        const changeStr = (changePct >= 0 ? '+' : '') + changePct.toFixed(2) + '%';

        // Direction: only show if confidence > 50%
        const conf = w.ghost_confidence || 0;
        let dirLabel, dirClass;
        if (conf < 50) {
            dirLabel = 'HOLD'; dirClass = 'hold';
        } else {
            const dir = (w.ghost_direction || '').toUpperCase();
            dirLabel = dir === 'UP' ? '↑ UP' : dir === 'DOWN' ? '↓ DOWN' : 'HOLD';
            dirClass = dir === 'UP' ? 'up' : dir === 'DOWN' ? 'down' : 'hold';
        }

        return `
        <div class="wl-card">
            <div>
                <div class="wl-sym">${w.symbol}</div>
                <div class="wl-price">${fmtPrice(w.price)}</div>
            </div>
            <div class="wl-right">
                <div class="wl-change ${changeClass}">${changeStr}</div>
                <div class="wl-dir ${dirClass}">${dirLabel}</div>
            </div>
        </div>`;
    }).join('');
}

// ═══════════════════════════════════════════
// NEWS (shown on both Stocks/Crypto tabs)
// ═══════════════════════════════════════════
function renderNews(containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const articles = _news.slice(0, 8);
    if (!articles.length) {
        el.innerHTML = '<div class="loading-msg">No news</div>';
        return;
    }

    el.innerHTML = articles.map(a => {
        const title = a.title || a.headline || 'Untitled';
        const url = a.url || a.link || '#';
        const time = fmtTimeAgo(a.published_at || a.timestamp || a.published);
        const sent = (a.sentiment || 'neutral').toLowerCase();
        const sentClass = sent === 'bullish' ? 'bullish' : sent === 'bearish' ? 'bearish' : 'neutral';
        return `
        <a class="news-item" href="${url}" target="_blank" rel="noopener">
            <span class="news-title">${esc(title)}</span>
            <div class="news-meta">
                <span class="news-sent ${sentClass}">${sent}</span>
                <span class="news-time">${time}</span>
            </div>
        </a>`;
    }).join('');
}

// ═══════════════════════════════════════════
// HISTORY TAB — Resolved trades
// ═══════════════════════════════════════════
function renderHistory() {
    // Build resolved list from paper trades
    let resolved = _trades.filter(t => {
        const o = (t.outcome || '').toLowerCase();
        return o && o !== 'pending' && o !== 'open' && o !== '';
    });

    // Apply filter
    if (_historyFilter === 'stock') resolved = resolved.filter(t => isStock(t));
    else if (_historyFilter === 'crypto') resolved = resolved.filter(t => !isStock(t));
    else if (_historyFilter === 'win') resolved = resolved.filter(t => isWin(t));
    else if (_historyFilter === 'loss') resolved = resolved.filter(t => !isWin(t));

    // Sort newest first
    resolved.sort((a, b) => (ts(b) || 0) - (ts(a) || 0));

    // Stats (from ALL resolved, not filtered)
    const allResolved = _trades.filter(t => {
        const o = (t.outcome || '').toLowerCase();
        return o && o !== 'pending' && o !== 'open' && o !== '';
    });
    const wins = allResolved.filter(t => isWin(t)).length;
    const losses = allResolved.length - wins;
    const totalPnl = allResolved.reduce((s, t) => s + (t.pnl || t.realized_pnl || 0), 0);
    const winRate = allResolved.length > 0 ? (wins / allResolved.length * 100).toFixed(1) : '--';

    setText('hist-total', allResolved.length);
    setTextColor('hist-wins', wins, 'green');
    setTextColor('hist-losses', losses, 'red');
    setText('hist-winrate', winRate === '--' ? '--' : winRate + '%');
    const pnlEl = document.getElementById('hist-pnl');
    if (pnlEl) {
        pnlEl.textContent = (totalPnl >= 0 ? '+' : '') + '$' + Math.abs(totalPnl).toFixed(2);
        pnlEl.className = 'hstat-val ' + (totalPnl >= 0 ? 'green' : 'red');
    }

    // Table
    const tbody = document.getElementById('history-tbody');
    if (!tbody) return;

    if (!resolved.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="loading-msg">No resolved trades</td></tr>';
        return;
    }

    tbody.innerHTML = resolved.slice(0, 200).map(t => {
        const won = isWin(t);
        const pnl = t.pnl || t.realized_pnl || 0;
        const pnlStr = (pnl >= 0 ? '+' : '') + '$' + Math.abs(pnl).toFixed(2);
        const dir = (t.direction || t.signal_direction || '--').toUpperCase();
        const date = ts(t) ? fmtDate(ts(t)) : '--';
        return `<tr>
            <td><strong>${t.symbol || '--'}</strong></td>
            <td>${dir}</td>
            <td>${fmtPrice(t.entry_price || t.signal_price)}</td>
            <td>${fmtPrice(t.exit_price || t.close_price)}</td>
            <td class="${pnl >= 0 ? 'result-win' : 'result-loss'}">${pnlStr}</td>
            <td class="${won ? 'result-win' : 'result-loss'}">${won ? 'WIN' : 'LOSS'}</td>
            <td>${date}</td>
        </tr>`;
    }).join('');
}

// ═══════════════════════════════════════════
// HEALTH TAB — Honest numbers only
// ═══════════════════════════════════════════
function renderHealth(accuracy, heartbeat, audit) {
    // ── Top line: one honest sentence ──
    const topEl = document.getElementById('health-topline');
    if (topEl) {
        if (accuracy && audit) {
            const pct = accuracy.accuracy_pct ?? 0;
            const correct = accuracy.correct_predictions ?? 0;
            const total = accuracy.total_predictions ?? 0;
            const status = accuracy.accuracy_status || 'UNKNOWN';
            const score = audit.health_score ?? '--';
            const issues = audit.issues_remaining ?? 0;
            topEl.innerHTML = `
                <span class="hl-big">${pct}%</span> accuracy ·
                ${correct}/${total} correct ·
                Status: <strong>${status}</strong> ·
                System: ${score}/100 ·
                ${issues} issue${issues !== 1 ? 's' : ''}
            `;
        } else {
            topEl.textContent = 'Unable to load health data';
        }
    }

    // ── Accuracy cards ──
    if (accuracy) {
        setText('acc-24h', (accuracy.daily_accuracy_pct ?? 0) + '%');
        setText('acc-7d', (accuracy.weekly_accuracy_pct ?? 0) + '%');
        setText('acc-30d', (accuracy.monthly_accuracy_pct ?? 0) + '%');
        setText('acc-record', `${accuracy.correct_predictions || 0}W / ${((accuracy.total_predictions || 0) - (accuracy.correct_predictions || 0))}L`);
    }

    // ── Heartbeat grid ──
    const hbEl = document.getElementById('heartbeat-grid');
    if (hbEl && heartbeat?.tasks) {
        const entries = Object.entries(heartbeat.tasks);
        if (!entries.length) {
            hbEl.innerHTML = '<div class="loading-msg">No tasks registered</div>';
        } else {
            hbEl.innerHTML = entries.map(([name, info]) => {
                const status = info.status || (info.alive ? 'alive' : 'dead');
                const dotClass = status === 'alive' ? 'alive' : status === 'stale' ? 'stale' : status === 'never' ? 'never' : 'dead';
                const ago = info.last_pulse ? fmtTimeAgo(info.last_pulse) : 'never';
                return `
                <div class="hb-card">
                    <span class="hb-dot ${dotClass}"></span>
                    <span class="hb-name">${esc(name.replace(/-/g, ' '))}</span>
                    <span class="hb-ago">${ago}</span>
                </div>`;
            }).join('');
        }
    }

    // ── Issues list ──
    const issEl = document.getElementById('issues-list');
    if (issEl && audit?.issues) {
        const issues = audit.issues || [];
        if (!issues.length) {
            issEl.innerHTML = '<div class="loading-msg" style="color:var(--green)">✓ No issues</div>';
        } else {
            issEl.innerHTML = issues.map(iss => {
                const sev = (iss.severity || 'info').toLowerCase();
                return `
                <div class="issue-item">
                    <span class="issue-sev ${sev}">${sev}</span>
                    <span class="issue-detail">${esc(iss.detail || iss.message || iss.type || '')}</span>
                </div>`;
            }).join('');
        }
    }
}

// ─── UTILITIES ───
async function fetchJSON(url) {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status);
    return r.json();
}

async function postJSON(url, body) {
    try {
        await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
    } catch (e) { console.warn('POST failed:', e); }
}

function fmtPrice(v) {
    if (v == null || v === 0) return '--';
    return v >= 1
        ? '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
        : '$' + Number(v).toFixed(6);
}

function fmtDate(ts) {
    if (!ts) return '--';
    const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
    return isNaN(d) ? '--' : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function fmtTimeAgo(ts) {
    if (!ts) return '';
    const d = typeof ts === 'number' ? new Date(ts > 1e12 ? ts : ts * 1000) : new Date(ts);
    if (isNaN(d)) return '';
    const s = Math.floor((Date.now() - d.getTime()) / 1000);
    if (s < 60) return 'just now';
    if (s < 3600) return Math.floor(s / 60) + 'm ago';
    if (s < 86400) return Math.floor(s / 3600) + 'h ago';
    return Math.floor(s / 86400) + 'd ago';
}

function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }
function setText(id, v) { const el = document.getElementById(id); if (el) el.textContent = v; }
function setTextColor(id, v, color) {
    const el = document.getElementById(id);
    if (el) { el.textContent = v; el.className = 'hstat-val ' + color; }
}
function isWin(t) { const o = (t.outcome || '').toLowerCase(); return o === 'win' || o === 'correct'; }
function isStock(t) { const m = (t.market || t.type || '').toLowerCase(); return m === 'stock' || m === 'stocks'; }
function ts(t) { return t.resolved_at || t.closed_at || t.updated_at || null; }
