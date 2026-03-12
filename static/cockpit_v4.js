/* Ghost Protocol v4 — Clean Cockpit JS
 * Tab-based dashboard: Stocks · Crypto · History · Health
 * Every number matches what Telegram sends. No vanity scores.
 */

// ─── STATE ───
let _activeTab = 'stocks';
let _allPicks = [];
let _allTrades = [];
let _allWatchlist = [];
let _allNews = [];
let _historyData = [];
let _historyFilter = 'all';

// ─── BOOT ───
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initHeader();
    startClock();
    loadAll();
    // Poll every 30s
    setInterval(loadAll, 30000);
});

// ─── TAB NAVIGATION ───
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            _activeTab = tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.tab-page').forEach(p => p.classList.remove('active'));
            const page = document.getElementById('tab-' + tab);
            if (page) page.classList.add('active');
        });
    });

    // History filter buttons
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
    if (startBtn) startBtn.addEventListener('click', () => sendControl('start'));
    if (stopBtn) stopBtn.addEventListener('click', () => sendControl('stop'));
}

function startClock() {
    const el = document.getElementById('system-time');
    if (!el) return;
    setInterval(() => {
        const now = new Date();
        el.textContent = now.toLocaleTimeString('en-US', { hour12: false });
    }, 1000);
}

async function sendControl(action) {
    try {
        await fetch('/api/cockpit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action })
        });
    } catch (e) { console.warn('Control failed:', e); }
}

// ─── LOAD ALL DATA ───
async function loadAll() {
    const [picks, trades, watchlist, news, health, accuracy, heartbeat, audit] = await Promise.allSettled([
        fetchJSON('/api/v4/picks'),
        fetchJSON('/api/v3/paper/trades?limit=100'),
        fetchJSON('/api/v3/watchlist'),
        fetchJSON('/api/v3/news/feed'),
        fetchJSON('/api/v3/cockpit/status'),
        fetchJSON('/api/v3/accuracy/summary'),
        fetchJSON('/api/v3/heartbeat/status'),
        fetchJSON('/integrity/audit/readonly'),
    ]);

    // Picks
    if (picks.status === 'fulfilled' && picks.value?.ok) {
        _allPicks = picks.value.picks || [];
    }

    // Trades
    if (trades.status === 'fulfilled' && trades.value?.ok) {
        _allTrades = trades.value.trades || [];
    }

    // Watchlist
    if (watchlist.status === 'fulfilled' && watchlist.value?.ok) {
        _allWatchlist = watchlist.value.items || watchlist.value.watchlist || [];
    }

    // News
    if (news.status === 'fulfilled' && news.value?.ok) {
        _allNews = news.value.articles || news.value.feed || [];
    }

    // Health pill in header
    if (audit.status === 'fulfilled') {
        const score = audit.value?.health_score ?? '--';
        const pill = document.getElementById('health-pill');
        if (pill) pill.textContent = score + ' / 100';
    }

    // Status indicator
    if (health.status === 'fulfilled' && health.value?.status === 'ok') {
        setStatus(true);
    }

    // Render everything
    renderPicks('stock', 'stock-picks');
    renderPicks('crypto', 'crypto-picks');
    renderActiveTrades('stock', 'stock-active-trades');
    renderActiveTrades('crypto', 'crypto-active-trades');
    renderWatchlist('stock', 'stock-watchlist');
    renderWatchlist('crypto', 'crypto-watchlist');
    renderNews('stock', 'stock-news');
    renderNews('crypto', 'crypto-news');

    // History (from resolved trades)
    if (trades.status === 'fulfilled' && trades.value?.ok) {
        _historyData = (_allTrades || []).filter(t => t.outcome && t.outcome !== 'pending');
        renderHistory();
    }

    // Health tab
    renderHealthTab(
        accuracy.status === 'fulfilled' ? accuracy.value : null,
        heartbeat.status === 'fulfilled' ? heartbeat.value : null,
        audit.status === 'fulfilled' ? audit.value : null
    );
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

// ─── PICK CARDS (Telegram-style) ───
function renderPicks(assetType, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const picks = _allPicks.filter(p => {
        const pType = (p.type || p.market || '').toLowerCase();
        if (assetType === 'stock') return pType === 'stock';
        return pType === 'crypto';
    });

    if (!picks.length) {
        el.innerHTML = '<div class="no-picks">No active picks right now</div>';
        return;
    }

    el.innerHTML = picks.map(p => {
        const isUp = (p.direction || '').toUpperCase() === 'UP';
        const dirClass = isUp ? 'bullish' : 'bearish';
        const dirLabel = isUp ? 'UP' : 'DOWN';
        const dirBtnClass = isUp ? 'up' : 'down';
        const emoji = isUp ? '🟢' : '🔴';
        const entry = fmtPrice(p.entry_price);
        const target = fmtPrice(p.target_price);
        const stop = fmtPrice(p.stop_loss);
        const gainPct = p.gain_pct != null ? Math.abs(p.gain_pct).toFixed(1) : '--';
        const returnVal = p.gain_pct != null ? (100 + Math.abs(p.gain_pct)).toFixed(2) : '--';
        const deadline = p.done_by || '--';

        return `
        <div class="pick-card ${dirClass}">
            <div class="pick-header">
                <span class="pick-symbol">${emoji} ${p.symbol}</span>
                <span class="pick-dir ${dirBtnClass}">${dirLabel}</span>
            </div>
            <div class="pick-body">
                <span class="pick-label">Get in at</span>
                <span class="pick-val">${entry}</span>
                <span class="pick-label">Get out at</span>
                <span class="pick-val green">${target} (${gainPct}%)</span>
                <span class="pick-label">Run away at</span>
                <span class="pick-val red">${stop}</span>
                <span class="pick-label">Done by</span>
                <span class="pick-val">${deadline}</span>
            </div>
            <div class="pick-footer">
                <span>Confidence: ${(p.confidence || 0).toFixed(0)}%</span>
                <span class="pick-return green">$100 → $${returnVal}</span>
            </div>
        </div>`;
    }).join('');
}

// ─── ACTIVE TRADES ───
function renderActiveTrades(assetType, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const pending = _allTrades.filter(t => {
        const status = (t.outcome || t.status || '').toLowerCase();
        if (status !== 'pending' && status !== 'open' && status !== '') return false;
        const tType = (t.market || t.type || '').toLowerCase();
        if (assetType === 'stock') return tType === 'stock' || tType === 'stocks';
        return tType === 'crypto';
    });

    if (!pending.length) {
        el.innerHTML = '<div class="loading-msg">No active trades</div>';
        return;
    }

    // Show max 6 per tab
    el.innerHTML = pending.slice(0, 6).map(t => {
        const pnl = t.unrealized_pnl || t.pnl || 0;
        const pnlClass = pnl >= 0 ? 'green' : 'red';
        const pnlStr = (pnl >= 0 ? '+' : '') + '$' + Math.abs(pnl).toFixed(2);
        const dir = (t.direction || t.signal_direction || '--').toUpperCase();

        return `
        <div class="trade-card">
            <div class="trade-left">
                <span class="trade-sym">${t.symbol || '--'}</span>
                <span class="trade-meta">${dir} · Entry: ${fmtPrice(t.entry_price || t.signal_price)}</span>
            </div>
            <div class="trade-right">
                <span class="trade-pnl ${pnlClass}">${pnlStr}</span>
                <span class="trade-status">Open</span>
            </div>
        </div>`;
    }).join('');
}

// ─── WATCHLIST ───
function renderWatchlist(assetType, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const items = _allWatchlist.filter(w => {
        const wType = (w.type || '').toLowerCase();
        if (assetType === 'stock') return wType === 'stock';
        return wType === 'crypto';
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
            dirLabel = 'HOLD';
            dirClass = 'hold';
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

// ─── NEWS ───
function renderNews(assetType, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;

    // News doesn't have a clean type field — show all news on both tabs
    // but limit to 5 per tab
    const articles = _allNews.slice(0, 8);

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
        const sentLabel = sent.charAt(0).toUpperCase() + sent.slice(1);

        return `
        <a class="news-item" href="${url}" target="_blank" rel="noopener">
            <span class="news-title">${escHtml(title)}</span>
            <div class="news-meta">
                <span class="news-sent ${sentClass}">${sentLabel}</span>
                <span class="news-time">${time}</span>
            </div>
        </a>`;
    }).join('');
}

// ─── HISTORY TAB ───
function renderHistory() {
    let data = [..._historyData];

    // Apply filters
    if (_historyFilter === 'stock') data = data.filter(t => (t.market || t.type || '').toLowerCase() === 'stock');
    else if (_historyFilter === 'crypto') data = data.filter(t => (t.market || t.type || '').toLowerCase() === 'crypto');
    else if (_historyFilter === 'win') data = data.filter(t => (t.outcome || '').toLowerCase() === 'win' || (t.outcome || '').toLowerCase() === 'correct');
    else if (_historyFilter === 'loss') data = data.filter(t => (t.outcome || '').toLowerCase() === 'loss' || (t.outcome || '').toLowerCase() === 'incorrect');

    // Sort by most recent
    data.sort((a, b) => (b.resolved_at || b.closed_at || 0) - (a.resolved_at || a.closed_at || 0));

    // Stats
    const wins = data.filter(t => ['win', 'correct'].includes((t.outcome || '').toLowerCase())).length;
    const losses = data.filter(t => ['loss', 'incorrect'].includes((t.outcome || '').toLowerCase())).length;
    const totalPnl = data.reduce((s, t) => s + (t.pnl || t.realized_pnl || 0), 0);
    const winRate = (wins + losses) > 0 ? (wins / (wins + losses) * 100).toFixed(1) : '--';

    const winsEl = document.getElementById('hist-wins');
    const lossEl = document.getElementById('hist-losses');
    const wrEl = document.getElementById('hist-winrate');
    const pnlEl = document.getElementById('hist-pnl');

    if (winsEl) { winsEl.textContent = wins; winsEl.className = 'hstat-val green'; }
    if (lossEl) { lossEl.textContent = losses; lossEl.className = 'hstat-val red'; }
    if (wrEl) wrEl.textContent = winRate === '--' ? '--' : winRate + '%';
    if (pnlEl) {
        pnlEl.textContent = (totalPnl >= 0 ? '+' : '') + '$' + Math.abs(totalPnl).toFixed(2);
        pnlEl.className = 'hstat-val ' + (totalPnl >= 0 ? 'green' : 'red');
    }

    // Table
    const tbody = document.getElementById('history-tbody');
    if (!tbody) return;

    if (!data.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="loading-msg">No resolved trades yet</td></tr>';
        return;
    }

    tbody.innerHTML = data.slice(0, 200).map(t => {
        const outcome = (t.outcome || '').toLowerCase();
        const isWin = outcome === 'win' || outcome === 'correct';
        const resultClass = isWin ? 'result-win' : 'result-loss';
        const resultLabel = isWin ? 'WIN' : 'LOSS';
        const pnl = t.pnl || t.realized_pnl || 0;
        const pnlStr = (pnl >= 0 ? '+' : '') + '$' + Math.abs(pnl).toFixed(2);
        const dir = (t.direction || t.signal_direction || '--').toUpperCase();
        const date = t.resolved_at || t.closed_at ? fmtDate(t.resolved_at || t.closed_at) : '--';

        return `<tr>
            <td><strong>${t.symbol || '--'}</strong></td>
            <td>${dir}</td>
            <td>${fmtPrice(t.entry_price || t.signal_price)}</td>
            <td>${fmtPrice(t.exit_price || t.close_price)}</td>
            <td class="${pnl >= 0 ? 'result-win' : 'result-loss'}">${pnlStr}</td>
            <td class="${resultClass}">${resultLabel}</td>
            <td>${date}</td>
        </tr>`;
    }).join('');
}

// ─── HEALTH TAB ───
function renderHealthTab(accuracy, heartbeat, audit) {
    // Top line — honest one-liner
    const topEl = document.getElementById('health-topline');
    if (topEl && accuracy) {
        const accPct = accuracy.accuracy_pct ?? 0;
        const correct = accuracy.correct_predictions ?? 0;
        const total = accuracy.total_predictions ?? 0;
        const status = accuracy.accuracy_status || 'UNKNOWN';
        const auditScore = audit?.health_score ?? '--';

        topEl.innerHTML = `
            <span class="hl-score">${accPct}%</span> accuracy · 
            ${correct}/${total} correct · 
            Status: <strong>${status}</strong> · 
            System: ${auditScore}/100
        `;
    }

    // Accuracy cards
    if (accuracy) {
        setText('acc-24h', (accuracy.daily_accuracy_pct ?? 0) + '%');
        setText('acc-7d', (accuracy.weekly_accuracy_pct ?? 0) + '%');
        setText('acc-30d', (accuracy.monthly_accuracy_pct ?? 0) + '%');
        setText('acc-total', `${accuracy.correct_predictions || 0}W / ${(accuracy.total_predictions || 0) - (accuracy.correct_predictions || 0)}L`);
    }

    // Heartbeat
    const hbEl = document.getElementById('heartbeat-grid');
    if (hbEl && heartbeat && heartbeat.tasks) {
        const tasks = heartbeat.tasks || {};
        const entries = Object.entries(tasks);
        if (!entries.length) {
            hbEl.innerHTML = '<div class="loading-msg">No heartbeat data</div>';
        } else {
            hbEl.innerHTML = entries.map(([name, info]) => {
                const alive = info.alive || info.status === 'alive';
                const warming = info.status === 'warming';
                const dotClass = alive ? 'alive' : warming ? 'warming' : 'dead';
                const ago = info.last_pulse ? fmtTimeAgo(info.last_pulse) : 'never';
                return `
                <div class="hb-card">
                    <span class="hb-dot ${dotClass}"></span>
                    <span class="hb-name">${name.replace(/-/g, ' ')}</span>
                    <span class="hb-ago">${ago}</span>
                </div>`;
            }).join('');
        }
    }

    // Issues
    const issEl = document.getElementById('issues-list');
    if (issEl && audit && audit.issues) {
        const issues = audit.issues || [];
        if (!issues.length) {
            issEl.innerHTML = '<div class="loading-msg" style="color:var(--green)">✓ No issues detected</div>';
        } else {
            issEl.innerHTML = issues.map(iss => {
                const sev = (iss.severity || 'info').toLowerCase();
                return `
                <div class="issue-item">
                    <span class="issue-sev ${sev}">${sev}</span>
                    <span class="issue-detail">${escHtml(iss.detail || iss.message || iss.type || 'Unknown issue')}</span>
                </div>`;
            }).join('');
        }
    }
}

// ─── HELPERS ───
async function fetchJSON(url) {
    const resp = await fetch(url, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`${resp.status}`);
    return resp.json();
}

function fmtPrice(v) {
    if (v == null || v === 0) return '--';
    if (v >= 1) return '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    return '$' + Number(v).toFixed(6);
}

function fmtDate(ts) {
    if (!ts) return '--';
    const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
    if (isNaN(d.getTime())) return '--';
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function fmtTimeAgo(ts) {
    if (!ts) return '';
    const d = typeof ts === 'number' ? new Date(ts > 1e12 ? ts : ts * 1000) : new Date(ts);
    if (isNaN(d.getTime())) return '';
    const secs = Math.floor((Date.now() - d.getTime()) / 1000);
    if (secs < 60) return 'just now';
    if (secs < 3600) return Math.floor(secs / 60) + 'm ago';
    if (secs < 86400) return Math.floor(secs / 3600) + 'h ago';
    return Math.floor(secs / 86400) + 'd ago';
}

function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}
