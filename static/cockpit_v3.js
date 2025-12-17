// Ghost Protocol v3 - Minimal UI JavaScript

// State
let currentTab = 'stocks';
let currentForecastSymbol = 'BTC';  // Default to BTC (has active predictions)
let updateInterval = null;
let watchlistMode = 'personal';  // 'personal' or 'market'
let watchlistFilter = 'all';     // 'all', 'stocks', 'crypto'
let sharedWatchlistData = [];    // Shared cache for cross-panel data (Major Caps, XRP VIP)

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    updateSystemTime();
    
    // Load status indicator immediately
    loadCockpitStatus();
    
    // Sync forecast input with default symbol
    document.getElementById('forecast-symbol').value = currentForecastSymbol;
    
    // Load all panels IMMEDIATELY on startup (don't wait for intervals)
    loadAllPanels();
    
    // Pre-load goals for modal (silent load, no UI update needed)
    // PERFORMANCE FIX: Defer health score load by 500ms to prevent blocking page load
    setTimeout(() => loadHealthScore(), 500);
    
    // OPTIMIZED: Set smart update intervals (reduced from 5s to prevent hammering)
    // Goals/Stats: 30s (slow-changing data)
    // Predictions/Forecast: 15s (medium-priority)
    // Top Movers/Hunter: 10s (fast-moving opportunities)
    // Time display: 1s (real-time clock)
    
    // Store intervals in window so handleModeChange can clear them for FIXED mode
    window.updateInterval = setInterval(() => updateSystemTime(), 1000);  // Clock: every 1s
    window.statusInterval = setInterval(() => loadCockpitStatus(), 30000);  // Status: every 30s
    window.healthInterval = setInterval(() => loadHealthScore(), 30000);  // Goals/Health: every 30s
    window.accuracyInterval = setInterval(() => loadAccuracyChart(), 30000);  // Accuracy Chart: every 30s
    window.forecastInterval = setInterval(() => loadForecast(), 15000);  // Forecast: every 15s
    window.topMoversInterval = setInterval(() => loadTopMovers(), 10000);  // Top Movers: every 10s (includes hunter feed)
    window.watchlistInterval = setInterval(() => loadWatchlistByMode(), 15000);  // Watchlist: every 15s (mode-aware)
    window.vipInterval = setInterval(() => loadVIPCoins(), 15000);  // VIP Coins: every 15s
    
    console.log('✅ Ghost Protocol Cockpit v3 initialized');
}

// Event Listeners
function setupEventListeners() {
    // Header controls
    document.getElementById('btn-start').addEventListener('click', () => controlAction('start'));
    document.getElementById('btn-stop').addEventListener('click', () => controlAction('stop'));
    document.getElementById('btn-reset').addEventListener('click', () => controlAction('reset'));
    document.getElementById('mode-selector').addEventListener('change', handleModeChange);
    
    // Goals Settings Modal
    document.getElementById('btn-settings').addEventListener('click', openGoalsModal);
    document.getElementById('modal-close').addEventListener('click', closeGoalsModal);
    document.getElementById('cancel-goals').addEventListener('click', closeGoalsModal);
    document.getElementById('save-goals').addEventListener('click', saveGoals);
    
    // Close modal on outside click
    document.getElementById('goals-modal').addEventListener('click', (e) => {
        if (e.target.id === 'goals-modal') {
            closeGoalsModal();
        }
    });
    
    // Tabs - handle both mode tabs (data-mode) and filter tabs (data-tab)
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const mode = e.target.dataset.mode;
            const tabType = e.target.dataset.tab;
            
            if (mode) {
                // Mode tab (Personal/Market)
                switchTab(e.target.closest('.tabs'), mode);
            } else if (tabType) {
                // Filter tab (Stocks/Crypto/All)
                switchTab(e.target.closest('.tabs'), tabType);
            }
        });
    });
    
    // Forecast symbol search
    const forecastInput = document.getElementById('forecast-symbol');
    forecastInput.addEventListener('change', (e) => {
        currentForecastSymbol = e.target.value.toUpperCase();
        loadForecast();
    });
    
    // Refresh buttons
    document.querySelectorAll('.refresh-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const panel = e.target.dataset.panel;
            refreshPanel(panel);
        });
    });
}

// System Time Update
function updateSystemTime() {
    const timeEl = document.getElementById('system-time');
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    timeEl.textContent = `${hours}:${minutes}:${seconds}`;
}

// Control Actions
async function controlAction(action) {
    try {
        const response = await fetch(`/api/cockpit/${action}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
            const data = await response.json();
            updateStatusIndicator(data.active);
            console.log(`${action} successful:`, data);
        }
    } catch (error) {
        console.error(`Error in ${action}:`, error);
    }
}

function handleModeChange(e) {
    const mode = e.target.value;
    console.log('[MODE] Changed to:', mode);
    
    if (mode === 'fixed') {
        // FIXED MODE: Freeze all auto-refresh intervals
        if (window.updateInterval) clearInterval(window.updateInterval);
        if (window.statusInterval) clearInterval(window.statusInterval);
        if (window.topMoversInterval) clearInterval(window.topMoversInterval);
        if (window.vipInterval) clearInterval(window.vipInterval);
        if (window.watchlistInterval) clearInterval(window.watchlistInterval);
        if (window.forecastInterval) clearInterval(window.forecastInterval);
        if (window.healthInterval) clearInterval(window.healthInterval);
        if (window.accuracyInterval) clearInterval(window.accuracyInterval);
        
        document.getElementById('status-text').textContent = 'FIXED MODE';
        document.getElementById('status-text').style.color = 'var(--accent-yellow)';
        console.log('[MODE] All intervals frozen');
    } else {
        // LIVE MODE: Resume auto-refresh intervals
        document.getElementById('status-text').textContent = 'LIVE MODE';
        document.getElementById('status-text').style.color = 'var(--accent-green)';
        
        // Restart all intervals
        initializeApp();
        console.log('[MODE] All intervals resumed');
    }
}

function updateStatusIndicator(isActive) {
    const dot = document.getElementById('status-indicator');
    const text = document.getElementById('status-text');
    
    if (isActive) {
        dot.style.background = 'var(--accent-green)';
        dot.style.display = 'inline-block';
        text.textContent = 'RUNNING';
        text.style.color = 'var(--accent-green)';
    } else {
        dot.style.background = 'var(--accent-red)';
        dot.style.display = 'inline-block';
        text.textContent = 'STOPPED';
        text.style.color = 'var(--accent-red)';
    }
}

// Load Cockpit Status
async function loadCockpitStatus() {
    try {
        const response = await fetch('/api/v3/cockpit/status');
        if (response.ok) {
            const data = await response.json();
            updateStatusIndicator(data.active);
            
            // Update system stats in health panel
            if (data.uptime_seconds !== undefined) {
                const uptimeHours = (data.uptime_seconds / 3600).toFixed(1);
                const uptimeEl = document.getElementById('system-uptime');
                if (uptimeEl) {
                    uptimeEl.textContent = `${uptimeHours}h`;
                }
            }
            
            if (data.predictions_today !== undefined) {
                const countEl = document.getElementById('predictions-count');
                if (countEl) {
                    countEl.textContent = data.predictions_today || 0;
                }
            }
        }
    } catch (error) {
        console.error('Error loading cockpit status:', error);
        // Show stopped state on error
        updateStatusIndicator(false);
    }
}

// Tab Switching
function switchTab(tabsContainer, tabType) {
    const tabs = tabsContainer.querySelectorAll('.tab');
    tabs.forEach(t => t.classList.remove('active'));
    
    // Handle mode tabs (Personal/Market) vs filter tabs (Stocks/Crypto/All)
    if (tabsContainer.id === 'watchlist-mode-tabs') {
        // Switching between Personal and Market watchlist
        const modeButton = tabsContainer.querySelector(`[data-mode="${tabType}"]`);
        if (modeButton) {
            modeButton.classList.add('active');
            watchlistMode = tabType;
            loadWatchlistByMode();
        }
    } else if (tabsContainer.id === 'watchlist-filter-tabs') {
        // Switching between Stocks/Crypto/All filters
        const filterButton = tabsContainer.querySelector(`[data-tab="${tabType}"]`);
        if (filterButton) {
            filterButton.classList.add('active');
            watchlistFilter = tabType;
            // Update filter in personal watchlist OR reload market watchlist
            if (watchlistMode === 'personal' && typeof updateWatchlistTab === 'function') {
                updateWatchlistTab(tabType);
            } else {
                loadWatchlistByMode();
            }
        }
    } else {
        // Other panels (top movers, etc.)
        const button = tabsContainer.querySelector(`[data-tab="${tabType}"]`);
        if (button) {
            button.classList.add('active');
            currentTab = tabType;
        }
        
        // Reload relevant panel
        if (tabsContainer.closest('#panel-movers')) {
            loadTopMovers();
        }
    }
}

// Load All Panels
async function loadAllPanels() {
    // PERFORMANCE FIX: Load panels independently without blocking
    // Don't wait for slow endpoints (hunter feed) to show fast data
    
    // Fast panels - load first (no await)
    loadCockpitSnapshot().catch(e => console.error('Snapshot error:', e));
    loadLatestBTCPrediction().catch(e => console.error('BTC prediction error:', e));
    loadWatchlistByMode().catch(e => console.error('Watchlist error:', e));
    loadForecast().catch(e => console.error('Forecast error:', e));
    loadAccuracyChart().catch(e => console.error('Accuracy error:', e));
    
    // Slow panels - load in background (may take 10-30s on first load)
    setTimeout(() => {
        loadTopMovers().catch(e => console.error('Top movers error:', e));
        loadVIPCoins().catch(e => console.error('VIP coins error:', e));
        loadNews().catch(e => console.error('News error:', e));
        loadHealthScore().catch(e => console.error('Health score error:', e));
    }, 100);
    
    console.log('✅ Fast panels loaded, slow panels loading in background');
}

// Panel 0: Latest BTC Prediction with CASCADE VIEW
async function loadLatestBTCPrediction() {
    try {
        // Fetch both cascade data and momentum data
        const [cascadeResp, predResp, momentumResp] = await Promise.all([
            fetch('/api/v3/cascade/list?symbol=BTC&active_only=true'),
            fetch('/api/v3/predictions/latest?symbol=BTC'),
            fetch('/api/v3/momentum/BTC')
        ]);
        
        const cascadeData = cascadeResp.ok ? await cascadeResp.json() : null;
        const predData = predResp.ok ? await predResp.json() : null;
        const momentumData = momentumResp.ok ? await momentumResp.json() : null;
        
        const activeCascade = cascadeData?.cascades?.[0] || null;
        const latestPred = predData?.predictions?.[0] || null;
        const momentum = momentumData?.momentum || null;
        
        console.log('[BTC] Cascade:', activeCascade);
        console.log('[BTC] Momentum:', momentum);
        console.log('[BTC] Prediction:', latestPred);
        
        if (activeCascade) {
            // Render CASCADE VIEW if active cascade exists
            renderBTCCascade(activeCascade, momentum);
        } else if (latestPred) {
            // Fallback to single prediction with momentum
            renderBTCPrediction(latestPred, momentum);
        } else {
            renderBTCPrediction(null, null);
        }
    } catch (error) {
        console.error('[BTC] Load error:', error);
        renderBTCPrediction(null, null);
    }
}

function renderBTCPrediction(prediction, momentum) {
    const container = document.getElementById('btc-prediction');
    
    if (!prediction) {
        container.innerHTML = '<div class="pred-loading">⏳ No BTC prediction available yet</div>';
        return;
    }
    
    const direction = prediction.direction || 'FLAT';
    const confidence = Math.round((prediction.confidence || 0) * 100);
    const expectedMove = (prediction.expected_move || 0).toFixed(2);
    const horizonH = prediction.horizon_h || 6;
    
    // Calculate time remaining
    const runAt = prediction.run_at || Date.now() / 1000;
    const now = Date.now() / 1000;
    const ageSeconds = now - runAt;
    const ageMinutes = Math.floor(ageSeconds / 60);
    const timeRemaining = (horizonH * 3600) - ageSeconds;
    const hoursRemaining = Math.floor(timeRemaining / 3600);
    const minutesRemaining = Math.floor((timeRemaining % 3600) / 60);
    
    // Direction styling
    const directionIcon = direction === 'UP' ? '📈' : direction === 'DOWN' ? '📉' : '➡️';
    const directionClass = direction === 'UP' ? 'bullish' : direction === 'DOWN' ? 'bearish' : 'neutral';
    const directionLabel = direction === 'UP' ? 'BULLISH' : direction === 'DOWN' ? 'BEARISH' : 'NEUTRAL';
    
    // Confidence styling
    const confidenceClass = confidence >= 55 ? 'high' : confidence >= 45 ? 'medium' : 'low';
    
    // Momentum rendering
    let momentumHTML = '';
    if (momentum && momentum.status) {
        const momentumClass = momentum.status.toLowerCase().replace(' ', '-');
        const momentumDelta = momentum.change_pct || 0;
        const momentumSign = momentumDelta > 0 ? '+' : '';
        momentumHTML = `
            <div class="btc-momentum ${momentumClass}">
                <span class="momentum-emoji">${momentum.emoji || '➡️'}</span>
                <span class="momentum-label">${momentum.status || 'STABLE'}</span>
                <span class="momentum-delta">${momentumSign}${momentumDelta.toFixed(1)}%</span>
            </div>
        `;
    }
    
    container.innerHTML = `
        <div class="btc-pred-main">
            <div class="btc-pred-direction ${directionClass}">
                <span class="direction-icon">${directionIcon}</span>
                <span class="direction-label">${directionLabel}</span>
            </div>
            <div class="btc-pred-confidence ${confidenceClass}">
                <span class="confidence-value">${confidence}%</span>
                <span class="confidence-label">Confidence</span>
            </div>
            <div class="btc-pred-move">
                <span class="move-value">${expectedMove > 0 ? '+' : ''}${expectedMove}%</span>
                <span class="move-label">Expected Move</span>
            </div>
        </div>
        ${momentumHTML}
        <div class="btc-pred-meta">
            <div class="meta-item">
                <span class="meta-icon">⏱️</span>
                <span class="meta-text">${hoursRemaining}h ${minutesRemaining}m remaining</span>
            </div>
            <div class="meta-item">
                <span class="meta-icon">🕐</span>
                <span class="meta-text">Generated ${ageMinutes}m ago</span>
            </div>
            <div class="meta-item">
                <span class="meta-icon">🎯</span>
                <span class="meta-text">6-Hour Horizon (GHOST MAX v2.0)</span>
            </div>
        </div>
    `;
}

// NEW: Render CASCADE VIEW (48h → 24h → 6h → Outcome)
function renderBTCCascade(cascade, momentum) {
    const container = document.getElementById('btc-prediction');
    const now = Date.now() / 1000;
    const createdAt = cascade.created_at;
    const elapsed = now - createdAt;
    
    // Calculate time until next updates
    const h24_time = createdAt + (24 * 3600);
    const h42_time = createdAt + (42 * 3600);
    const h48_time = createdAt + (48 * 3600);
    
    const secondsUntil24h = h24_time - now;
    const secondsUntil42h = h42_time - now;
    const secondsUntil48h = h48_time - now;
    
    // Format time remaining
    function formatTimeRemaining(seconds) {
        if (seconds < 0) return 'Complete';
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${mins}m`;
    }
    
    // Elapsed time
    const elapsedHours = Math.floor(elapsed / 3600);
    const elapsedMins = Math.floor((elapsed % 3600) / 60);
    
    // Stage data
    const h48 = cascade.h48;
    const h24 = cascade.h24;
    const h6 = cascade.h6;
    const outcome = cascade.outcome;
    
    // Momentum rendering
    let momentumHTML = '';
    if (momentum && momentum.status) {
        const momentumClass = momentum.status.toLowerCase().replace(' ', '-');
        const momentumDelta = momentum.change_pct || 0;
        const momentumSign = momentumDelta > 0 ? '+' : '';
        momentumHTML = `
            <div class="cascade-momentum ${momentumClass}">
                <span class="momentum-emoji">${momentum.emoji || '➡️'}</span>
                <span class="momentum-label">${momentum.status || 'STABLE'}</span>
                <span class="momentum-delta">${momentumSign}${momentumDelta.toFixed(1)}%</span>
            </div>
        `;
    }
    
    // Build stage HTML
    const stageHTML = `
        <div class="cascade-stage ${h48 ? 'completed' : 'pending'}">
            <div class="stage-header">
                <span class="stage-icon">🔔</span>
                <span class="stage-title">48H ALERT</span>
                <span class="stage-status">${h48 ? '✅ Sent' : '⏳ Pending'}</span>
            </div>
            ${h48 ? `
                <div class="stage-body">
                    <div class="stage-prediction">
                        <span class="pred-direction ${h48.direction.toLowerCase()}">${h48.direction === 'UP' ? '📈' : '📉'} ${h48.direction}</span>
                        <span class="pred-confidence">${Math.round(h48.confidence * 100)}%</span>
                    </div>
                    <div class="stage-meta">Entry: $${h48.price.toLocaleString()}</div>
                </div>
            ` : '<div class="stage-pending-text">Not started</div>'}
        </div>
        
        <div class="cascade-stage ${h24 ? 'completed' : secondsUntil24h > 0 ? 'pending' : 'due'}">
            <div class="stage-header">
                <span class="stage-icon">📈</span>
                <span class="stage-title">24H UPDATE</span>
                <span class="stage-status">${h24 ? '✅ Updated' : '⏰ ' + formatTimeRemaining(secondsUntil24h)}</span>
            </div>
            ${h24 ? `
                <div class="stage-body">
                    <div class="stage-prediction">
                        <span class="pred-direction ${h24.direction.toLowerCase()}">${h24.direction === 'UP' ? '📈' : '📉'} ${h24.direction}</span>
                        <span class="pred-confidence">${Math.round(h24.confidence * 100)}%</span>
                    </div>
                    ${h48 ? `<div class="stage-delta">Change: ${(h24.confidence - h48.confidence) > 0 ? '+' : ''}${((h24.confidence - h48.confidence) * 100).toFixed(1)}%</div>` : ''}
                </div>
            ` : '<div class="stage-pending-text">Will re-evaluate with fresh data</div>'}
        </div>
        
        <div class="cascade-stage ${h6 ? 'completed' : secondsUntil42h > 0 ? 'pending' : 'due'}">
            <div class="stage-header">
                <span class="stage-icon">✅</span>
                <span class="stage-title">6H FINAL</span>
                <span class="stage-status">${h6 ? '✅ Final' : '⏰ ' + formatTimeRemaining(secondsUntil42h)}</span>
            </div>
            ${h6 ? `
                <div class="stage-body">
                    <div class="stage-prediction">
                        <span class="pred-direction ${h6.direction.toLowerCase()}">${h6.direction === 'UP' ? '📈' : '📉'} ${h6.direction}</span>
                        <span class="pred-confidence">${Math.round(h6.confidence * 100)}%</span>
                    </div>
                    ${h24 ? `<div class="stage-delta">Change: ${(h6.confidence - h24.confidence) > 0 ? '+' : ''}${((h6.confidence - h24.confidence) * 100).toFixed(1)}%</div>` : ''}
                </div>
            ` : '<div class="stage-pending-text">High-confidence final call</div>'}
        </div>
        
        <div class="cascade-stage ${outcome ? 'completed' : secondsUntil48h > 0 ? 'pending' : 'due'}">
            <div class="stage-header">
                <span class="stage-icon">🎯</span>
                <span class="stage-title">OUTCOME</span>
                <span class="stage-status">${outcome ? '✅ Scored' : '⏰ ' + formatTimeRemaining(secondsUntil48h)}</span>
            </div>
            ${outcome ? `
                <div class="stage-body">
                    <div class="stage-outcome">
                        <span class="outcome-result">${outcome.actual_direction || 'TBD'}</span>
                        <span class="outcome-score">Score: ${outcome.stages_correct || 0}/3</span>
                    </div>
                </div>
            ` : '<div class="stage-pending-text">Final validation & scoring</div>'}
        </div>
    `;
    
    container.innerHTML = `
        <div class="cascade-container">
            <div class="cascade-header">
                <div class="cascade-title">
                    <span class="cascade-icon">🔮</span>
                    <span>BTC PREDICTION CASCADE</span>
                </div>
                <div class="cascade-subtitle">ADAPTIVE 48H → 24H → 6H</div>
                <div class="cascade-time-running">Running ${elapsedHours}h ${elapsedMins}m</div>
            </div>
            ${momentumHTML}
            <div class="cascade-timeline">
                ${stageHTML}
            </div>
        </div>
    `;
}

// Panel 1: Top Movers
async function loadTopMovers() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);  // 3s timeout (hunter can be slow on first load)
        
        const response = await fetch('/api/v3/hunter/feed', { signal: controller.signal });
        clearTimeout(timeoutId);
        
        if (!response.ok) throw new Error('Failed to load movers');
        
        const data = await response.json();
        const container = document.getElementById('movers-list');
        
        // V3 format: {movers: [...], timestamp: N}
        const movers = data.movers || [];
        
        if (!movers || movers.length === 0) {
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; color: var(--text-secondary);">
                    <div style="font-size: 48px; margin-bottom: 20px;">👁️</div>
                    <div style="font-size: 18px; font-weight: 600; margin-bottom: 10px;">No High-Quality Opportunities</div>
                    <div style="font-size: 14px; opacity: 0.7;">Ghost filters out noise. Only 20%+ gains with 70%+ confidence appear here.</div>
                    <div style="font-size: 14px; opacity: 0.7; margin-top: 10px;">Market is quiet. Ghost is watching.</div>
                </div>
            `;
            return;
        }
        
        // Filter by current tab
        let filtered = movers;
        if (currentTab === 'stocks') {
            filtered = movers.filter(item => item.type === 'stock');
        } else if (currentTab === 'crypto') {
            filtered = movers.filter(item => item.type === 'crypto');
        }
        
        container.innerHTML = filtered.slice(0, 10).map(item => {
            // Show confidence only if it's meaningful (not default 50%)
            const hasRealPrediction = item.confidence && item.confidence !== 50;
            const confidenceDisplay = hasRealPrediction ? 
                `Ghost: ${item.confidence}%` : 
                `Ghost: --`;
            
            return `
                <div class="mover-card">
                    <div class="mover-left">
                        <div class="mover-icon">${getSymbolIcon(item.symbol)}</div>
                        <div class="mover-info">
                            <div class="mover-name">${item.symbol}</div>
                            <div class="mover-symbol">${item.name || item.symbol}</div>
                        </div>
                    </div>
                    <div class="mover-right">
                        <div class="mover-change ${item.change >= 0 ? 'positive' : 'negative'}">
                            ${item.change >= 0 ? '+' : ''}${item.change?.toFixed(2)}%
                        </div>
                        <div class="mover-confidence">${confidenceDisplay}</div>
                    </div>
                </div>
            `;
        }).join('');
    } catch (error) {
        console.error('[MOVERS] Error:', error);
        const container = document.getElementById('movers-list');
        if (error.name === 'AbortError') {
            container.innerHTML = '<p style="color: var(--accent-orange); text-align: center; padding: 20px;">⏱️ Connection timeout - retrying...</p>';
        } else {
            container.innerHTML = '<p style="color: var(--accent-red); text-align: center; padding: 20px;">❌ Failed to load movers</p>';
        }
    }
}

// Panel VIP: XRP Tracker + VIP Sniper Coins + Major Caps
async function loadVIPCoins() {
    // Load all three data sources in parallel
    try {
        const [xrpResponse, presaleResponse] = await Promise.all([
            fetch('/api/xrp/tracker').catch(e => ({ ok: false, error: e })),
            fetch('/api/presale/watch').catch(e => ({ ok: false, error: e }))
        ]);
        
        // XRP Tracker (Priority) - Enhanced with Watchlist 24h data
        if (xrpResponse.ok) {
            const xrpData = await xrpResponse.json();
            
            // CRITICAL FIX: Use Watchlist 24h change instead of XRP tracker's change_24h_pct
            // This ensures consistency across the dashboard
            const xrpWatchlistData = sharedWatchlistData.find(item => item.symbol === 'XRP');
            if (xrpWatchlistData && xrpWatchlistData.change_pct !== undefined) {
                console.log('[VIP] XRP sync - Before:', xrpData.change_24h_pct, '% (Tracker native)');
                xrpData.change_24h_pct = xrpWatchlistData.change_pct;
                console.log('[VIP] XRP sync - After:', xrpData.change_24h_pct, '% (Watchlist synced)');
            } else {
                console.warn('[VIP] XRP NOT found in Watchlist - using Tracker native 24h:', xrpData.change_24h_pct, '%');
            }
            
            renderXRPTracker(xrpData);
        } else {
            document.getElementById('xrp-tracker').innerHTML = '<p style="color: var(--text-secondary); font-size: 13px;">XRP tracker offline</p>';
        }
        
        // VIP Sniper Coins (WEPE, LILPEPE, DORKL, SLOTH, APC)
        if (presaleResponse.ok) {
            const presaleData = await presaleResponse.json();
            renderVIPSniperCoins(presaleData.presales || []);
        } else {
            document.getElementById('vip-sniper-list').innerHTML = '<p style="color: var(--text-secondary); font-size: 13px;">Sniper coins loading...</p>';
        }
        
        // CRITICAL FIX: Major Caps now uses Watchlist data (VIP snapshot is broken)
        // Pull BTC and ETH from the shared watchlist cache
        const majorsFromWatchlist = sharedWatchlistData.filter(item => ['BTC', 'ETH'].includes(item.symbol));
        
        if (majorsFromWatchlist.length > 0) {
            console.log('[VIP] Major Caps pulled from Watchlist:', majorsFromWatchlist);
            renderMajorCaps(majorsFromWatchlist);
        } else {
            console.warn('[VIP] No BTC/ETH found in Watchlist cache yet');
            document.getElementById('vip-majors-list').innerHTML = '<p style="color: var(--text-secondary); font-size: 13px;">Loading...</p>';
        }
        
    } catch (error) {
        console.error('[VIP] Error loading panel:', error);
        document.getElementById('xrp-tracker').innerHTML = '<p style="color: var(--accent-red);">❌ VIP panel error</p>';
    }
}

// Render XRP Tracker Widget
function renderXRPTracker(data) {
    const container = document.getElementById('xrp-tracker');
    
    // Eye indicator color based on bullish_eye value
    let eyeEmoji = '🟢';
    let eyeLabel = 'BULLISH';
    if (data.bullish_eye < 40) {
        eyeEmoji = '🔴';
        eyeLabel = 'BEARISH';
    } else if (data.bullish_eye < 60) {
        eyeEmoji = '🟡';
        eyeLabel = 'NEUTRAL';
    }
    
    const signalColor = data.signal === 'BUY' ? 'var(--accent-green)' : 
                        data.signal === 'SELL' ? 'var(--accent-red)' : 
                        'var(--accent-orange)';
    
    container.innerHTML = `
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border); border-radius: 8px; padding: 15px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 24px;">${eyeEmoji}</span>
                    <div>
                        <div style="font-weight: 600; font-size: 16px;">XRP ${eyeLabel}</div>
                        <div style="font-size: 12px; color: var(--text-secondary);">Price: $${data.price?.toFixed(4) || '--'}</div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 18px; font-weight: 600; color: ${signalColor};">${data.signal || 'HOLD'}</div>
                    <div style="font-size: 12px; color: var(--text-secondary);">Confidence: ${((data.confidence || 0) * 100).toFixed(0)}%</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--text-secondary);">
                <span>Eye Score: ${data.bullish_eye || 0}/100</span>
                <span>24h: ${data.change_24h_pct ? (data.change_24h_pct >= 0 ? '+' : '') + data.change_24h_pct.toFixed(2) + '%' : '--'}</span>
            </div>
        </div>
    `;
}

// Render VIP Sniper Coins (WEPE, LILPEPE, DORKL, SLOTH, APC)
function renderVIPSniperCoins(coins) {
    const container = document.getElementById('vip-sniper-list');
    
    if (!coins || coins.length === 0) {
        container.innerHTML = '<p style="color: var(--text-secondary); font-size: 13px;">No sniper coins in watch</p>';
        return;
    }
    
    container.innerHTML = coins.map(coin => {
        const statusColor = coin.status === 'Active' ? 'var(--accent-green)' : 
                           coin.status === 'Monitoring' ? 'var(--accent-orange)' : 
                           'var(--text-secondary)';
        
        return `
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border); border-radius: 6px; padding: 10px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 16px;">${getSymbolIcon(coin.symbol || coin.name)}</span>
                        <div>
                            <div style="font-weight: 600; font-size: 14px;">${coin.name || coin.symbol}</div>
                            <div style="font-size: 11px; color: var(--text-secondary);">${coin.category || 'Presale'}</div>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 12px; font-weight: 600; color: ${statusColor};">${coin.status || 'Watching'}</div>
                        ${coin.price ? `<div style="font-size: 11px; color: var(--text-secondary);">$${coin.price.toFixed(6)}</div>` : ''}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// Render Major Caps Reference (BTC, ETH)
// UPDATED: Now accepts Watchlist data format (change_pct, no status field)
function renderMajorCaps(coins) {
    const container = document.getElementById('vip-majors-list');
    
    if (!coins || coins.length === 0) {
        console.warn('[VIP] renderMajorCaps: No coins provided');
        container.innerHTML = '<p style="color: var(--text-secondary); font-size: 13px; grid-column: 1 / -1;">Loading majors...</p>';
        return;
    }
    
    console.log('[VIP] Rendering', coins.length, 'major caps:', coins);
    
    container.innerHTML = coins.map(coin => {
        // Watchlist format: {symbol, price, change_pct, ghost_confidence, ghost_direction, type}
        const isOffline = !coin.price || coin.price === 0;
        const priceDisplay = isOffline ? '--' : `$${coin.price.toLocaleString()}`;
        const changePct = coin.change_pct ?? 0;
        const changeClass = changePct >= 0 ? 'positive' : 'negative';
        const changeDisplay = isOffline ? '--' : 
            `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`;
        
        console.log(`[VIP] ${coin.symbol}: price=${priceDisplay}, change=${changeDisplay}, isOffline=${isOffline}`);
        
        return `
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border); border-radius: 6px; padding: 10px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <span style="font-size: 18px;">${getSymbolIcon(coin.symbol)}</span>
                    <span style="font-weight: 600; font-size: 14px;">${coin.symbol}</span>
                </div>
                <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 4px;">${priceDisplay}</div>
                <div class="${changeClass}" style="font-size: 12px; font-weight: 600;">${changeDisplay}</div>
            </div>
        `;
    }).join('');
}

// Panel 2: Forecast
// currentForecastSymbol already declared at top of file (line 5)

async function loadForecast() {
    const labelEl = document.getElementById('forecast-symbol-label');
    
    try {
        // Show loading state
        if (labelEl) labelEl.textContent = `Loading ${currentForecastSymbol}...`;
        
        const response = await fetch(`/api/v3/predictions/latest?symbol=${currentForecastSymbol}`);
        if (!response.ok) throw new Error('Failed to load forecast');
        
        const data = await response.json();
        
        // V3 format: {predictions: [{direction, confidence, horizon_h}]}
        const predictions = data.predictions || [];
        const pred = predictions[0] || {};
        
        // Update label to show current symbol
        if (labelEl) labelEl.textContent = `Forecast for ${currentForecastSymbol}`;
        
        console.log(`[FORECAST] Loaded for ${currentForecastSymbol}:`, pred);
        console.log(`[FORECAST] Direction: ${pred.direction}, Confidence: ${pred.confidence}, Move: ${pred.expected_move}, Horizon: ${pred.horizon_h}h`);
        
        // NEW: Display 6-hour prediction in card 0, price range in card 1, target in card 2
        updateForecastCard6h(pred);
    } catch (error) {
        console.error('[GHOST V3] Error loading forecast:', error);
        
        // Show error in label
        if (labelEl) labelEl.textContent = `❌ ${currentForecastSymbol} unavailable`;
        
        // Graceful degradation: show "no data" state
        const cards = document.querySelectorAll('.forecast-card');
        cards.forEach(card => {
            const probEl = card.querySelector('.prob-value');
            const moveEl = card.querySelector('.move-value');
            const lowEl = card.querySelector('.low-value');
            const highEl = card.querySelector('.high-value');
            const currentEl = card.querySelector('.current-value');
            const targetEl = card.querySelector('.target-value');
            
            if (probEl) probEl.textContent = '--';
            if (moveEl) moveEl.textContent = '--';
            if (lowEl) lowEl.textContent = '--';
            if (highEl) highEl.textContent = '--';
            if (currentEl) currentEl.textContent = '--';
            if (targetEl) targetEl.textContent = '--';
        });
    }
}

function updateForecastCard6h(prediction) {
    const cards = document.querySelectorAll('.forecast-card');
    if (cards.length < 3) {
        console.error('[FORECAST] Not enough cards in DOM');
        return;
    }
    
    const direction = prediction.direction || 'FLAT';
    let confidence = (prediction.confidence || 0) * 100;
    const expectedMove = prediction.expected_move || 0;
    
    // Card 0: 6-Hour Forecast (Main prediction)
    const card0 = cards[0];
    card0.querySelector('.forecast-direction').textContent = 
        direction === 'UP' ? '↑ BUY' : direction === 'DOWN' ? '↓ SELL' : '→ FLAT';
    card0.querySelector('.forecast-direction').className = 
        `forecast-direction ${direction === 'UP' ? 'bullish' : direction === 'DOWN' ? 'bearish' : 'neutral'}`;
    card0.querySelector('.prob-value').textContent = confidence.toFixed(0);
    card0.querySelector('.move-value').textContent = expectedMove.toFixed(2);
    
    // Card 1: Expected Range (calculate from expected move)
    const card1 = cards[1];
    const lowMove = expectedMove * 0.5;  // Conservative estimate
    const highMove = expectedMove * 1.5;  // Aggressive estimate
    card1.querySelector('.low-value').textContent = `${lowMove > 0 ? '+' : ''}${lowMove.toFixed(2)}%`;
    card1.querySelector('.high-value').textContent = `${highMove > 0 ? '+' : ''}${highMove.toFixed(2)}%`;
    
    // Card 2: Target Price (fetch from shared watchlist data)
    const card2 = cards[2];
    const watchlistSymbol = sharedWatchlistData.find(s => s.symbol === currentForecastSymbol);
    if (watchlistSymbol && watchlistSymbol.price) {
        const currentPrice = watchlistSymbol.price;
        const targetPrice = currentPrice * (1 + (expectedMove / 100));
        card2.querySelector('.current-value').textContent = `$${currentPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        card2.querySelector('.target-value').textContent = `$${targetPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    } else {
        card2.querySelector('.current-value').textContent = '--';
        card2.querySelector('.target-value').textContent = '--';
    }
    
    console.log(`[FORECAST 6H] Direction=${direction}, Confidence=${confidence.toFixed(0)}%, Move=${expectedMove.toFixed(2)}%`);
}

function updateForecastCard(index, prediction, icon, timeframe, confidenceMultiplier = 1.0) {
    const cards = document.querySelectorAll('.forecast-card');
    if (!cards[index]) {
        console.error(`[FORECAST] Card ${index} not found in DOM`);
        return;
    }
    
    const card = cards[index];
    const direction = prediction.direction || 'FLAT';
    let confidence = prediction.confidence || 0;
    
    // Convert confidence from 0-1 scale to percentage (0-100)
    if (confidence > 0 && confidence <= 1) {
        confidence = confidence * 100;
    }
    
    // Apply time decay to confidence
    const originalConfidence = confidence;
    confidence = confidence * confidenceMultiplier;
    
    console.log(`[FORECAST] ${timeframe}: orig_conf=${originalConfidence.toFixed(1)}, multiplier=${confidenceMultiplier}, final=${confidence.toFixed(1)}`);
    
    // Use backend expected_move if available, otherwise calculate from confidence
    let expectedMove = prediction.expected_move !== undefined 
        ? prediction.expected_move 
        : (confidence > 0 ? (confidence * 0.15) : 0);
    
    // Scale expected move by timeframe (longer = larger potential move)
    const timeframeMultipliers = {
        '24h': 1.0,
        '2-5d': 1.8,
        '7-14d': 2.5
    };
    const originalMove = expectedMove;
    expectedMove = expectedMove * (timeframeMultipliers[timeframe] || 1.0);
    
    console.log(`[FORECAST] ${timeframe}: orig_move=${originalMove.toFixed(2)}, multiplier=${timeframeMultipliers[timeframe]}, final=${expectedMove.toFixed(2)}`);
    
    card.querySelector('.forecast-icon').textContent = icon;
    
    // Graceful degradation: show "--" if no data
    const directionText = direction === 'UP' ? '↑ BUY' : 
                         direction === 'DOWN' ? '↓ SELL' : 
                         direction === 'FLAT' ? '→ FLAT' : '--';
    
    card.querySelector('.forecast-direction').textContent = directionText;
    card.querySelector('.prob-value').textContent = confidence > 0 ? confidence.toFixed(0) : '--';
    card.querySelector('.move-value').textContent = expectedMove !== 0 ? Math.abs(expectedMove).toFixed(2) : '--';
}

// Panel 3: News Feed
async function loadNews() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000);  // 10s timeout
        
        const response = await fetch('/api/v3/news/feed?limit=10', { signal: controller.signal });
        clearTimeout(timeoutId);
        
        if (!response.ok) throw new Error('Failed to load news');
        
        const data = await response.json();
        const container = document.getElementById('news-list');
        
        console.log('[NEWS] Loaded items:', data?.items?.length || 0);
        
        if (!data || !data.items || data.items.length === 0) {
            console.error('[NEWS] No items in response:', data);
            container.innerHTML = '<p style="color: var(--text-secondary); text-align: center;">No news available yet</p>';
            return;
        }
        
        // Debug: Log first item's sentiment
        if (data.items[0]) {
            console.log('[GHOST V3] News sentiment debug:', {
                headline: data.items[0].headline,
                sentiment: data.items[0].sentiment,
                type: typeof data.items[0].sentiment,
                formatted: formatSentiment(data.items[0].sentiment)
            });
        }
        
        container.innerHTML = data.items.slice(0, 10).map(article => `
            <div class="news-item">
                <div class="news-headline">${article.headline || article.title || 'No headline'}</div>
                <div class="news-meta">
                    <span class="news-sentiment ${getSentimentClass(article.sentiment)}">
                        ${formatSentiment(article.sentiment)}
                    </span>
                    <span class="news-time">${formatTime(article.timestamp)}</span>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('[NEWS] Error:', error);
        const container = document.getElementById('news-list');
        if (error.name === 'AbortError') {
            container.innerHTML = '<p style="color: var(--accent-orange); text-align: center; padding: 20px;">⏱️ Loading news...</p>';
        } else {
            container.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 20px;">📰 News feed temporarily unavailable</p>';
        }
    }
}

// Panel 4: Accuracy Chart
async function loadAccuracyChart() {
    try {
        const response = await fetch('/api/v3/accuracy/summary');
        if (!response.ok) throw new Error('Failed to load accuracy data');
        
        const data = await response.json();
        
        // Handle API's {ok: false, error: "..."} format
        if (!data.ok) {
            console.log('[ACCURACY] API returned no data:', data.error);
            renderAccuracyChart(null);
            return;
        }
        
        renderAccuracyChart(data);
    } catch (error) {
        console.error('[GHOST V3] Error loading accuracy chart:', error);
        renderAccuracyChart(null);
    }
}

function renderAccuracyChart(accuracyData) {
    const canvas = document.getElementById('accuracy-chart');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!accuracyData) {
        // Show friendly "waiting for data" message
        ctx.fillStyle = 'var(--text-secondary)';
        ctx.font = '13px var(--font-mono)';
        ctx.textAlign = 'center';
        ctx.fillText('⏳ Waiting for predictions to mature...', rect.width / 2, rect.height / 2 - 10);
        ctx.font = '11px var(--font-mono)';
        ctx.fillText('(Predictions need 48 hours to reconcile)', rect.width / 2, rect.height / 2 + 10);
        return;
    }
    
    // Extract metrics
    const dailyAcc = accuracyData.daily_accuracy_pct || 0;
    const weeklyAcc = accuracyData.weekly_accuracy_pct || 0;
    const monthlyAcc = accuracyData.monthly_accuracy_pct || 0;
    const status = accuracyData.accuracy_status || 'NO_DATA';
    const meetsThreshold = accuracyData.meets_70pct_threshold || false;
    
    // Draw 70% threshold line
    const thresholdY = rect.height * 0.3;  // 70% from top = 30% from top (inverted)
    ctx.strokeStyle = 'rgba(255, 193, 7, 0.3)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(40, thresholdY);
    ctx.lineTo(rect.width - 20, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw threshold label
    ctx.fillStyle = 'rgba(255, 193, 7, 0.6)';
    ctx.font = '11px var(--font-mono)';
    ctx.textAlign = 'left';
    ctx.fillText('70% TARGET', 45, thresholdY - 5);
    
    // Draw bars for Daily / Weekly / Monthly
    const barWidth = 50;
    const spacing = 90;
    const startX = rect.width / 2 - (spacing * 1.5);
    
    const bars = [
        { label: '24h', value: dailyAcc, x: startX + spacing * 0 },
        { label: '7d', value: weeklyAcc, x: startX + spacing * 1 },
        { label: '30d', value: monthlyAcc, x: startX + spacing * 2 }
    ];
    
    bars.forEach(bar => {
        const barHeight = (bar.value / 100) * (rect.height - 80);
        const barY = rect.height - 40 - barHeight;
        
        // Choose color based on value
        let barColor = 'var(--accent-red)';  // <50%
        if (bar.value >= 70) barColor = 'var(--accent-green)';  // >=70%
        else if (bar.value >= 50) barColor = 'var(--accent-yellow)';  // 50-70%
        
        // Draw bar
        ctx.fillStyle = barColor;
        ctx.fillRect(bar.x, barY, barWidth, barHeight);
        
        // Draw value on top
        ctx.fillStyle = 'var(--text-primary)';
        ctx.font = 'bold 16px var(--font-mono)';
        ctx.textAlign = 'center';
        ctx.fillText(`${bar.value.toFixed(1)}%`, bar.x + barWidth / 2, barY - 10);
        
        // Draw label at bottom
        ctx.fillStyle = 'var(--text-secondary)';
        ctx.font = '12px var(--font-mono)';
        ctx.fillText(bar.label, bar.x + barWidth / 2, rect.height - 20);
    });
    
    // Draw status badge
    ctx.font = 'bold 14px var(--font-mono)';
    ctx.textAlign = 'center';
    
    let statusText = status;
    let statusColor = 'var(--accent-red)';
    if (status === 'ACCURATE') {
        statusText = '✅ ACCURATE';
        statusColor = 'var(--accent-green)';
    } else if (status === 'BELOW_TARGET') {
        statusText = '⚠️ BELOW TARGET';
        statusColor = 'var(--accent-yellow)';
    } else {
        statusText = '❌ NO DATA';
        statusColor = 'var(--text-secondary)';
    }
    
    ctx.fillStyle = statusColor;
    ctx.fillText(statusText, rect.width / 2, 25);
    
    // Draw prediction count
    const totalPreds = accuracyData.total_predictions || 0;
    const correct = accuracyData.correct || 0;
    const wrong = accuracyData.wrong || 0;
    
    ctx.font = '11px var(--font-mono)';
    ctx.fillStyle = 'var(--text-secondary)';
    ctx.fillText(`${correct}W / ${wrong}L / ${totalPreds} Total`, rect.width / 2, 45);
}

// Panel 5: Watchlist - Master loader
async function loadWatchlistByMode() {
    if (watchlistMode === 'personal') {
        // Use personal watchlist from personal_watchlist_ui.js
        if (typeof loadPersonalWatchlist === 'function') {
            await loadPersonalWatchlist();
        } else {
            console.error('[WATCHLIST] personal_watchlist_ui.js not loaded');
            renderWatchlist([]);
        }
    } else {
        // Use market watchlist (existing behavior)
        await loadMarketWatchlist();
    }
}

// Panel 5: Market Watchlist (existing default watchlist)
async function loadMarketWatchlist() {
    try {
        // Use enriched watchlist endpoint that includes live prices AND predictions
        const response = await fetch('/api/v3/watchlist/enriched');
        if (!response.ok) throw new Error('Failed to load watchlist');
        
        const data = await response.json();
        const watchlistItems = data.items || [];
        
        console.log('[WATCHLIST] Loaded items:', watchlistItems.length);
        if (watchlistItems.length > 0) {
            console.log('[WATCHLIST] Sample item:', watchlistItems[0]);
        }
        
        // API already includes ghost_confidence and ghost_direction - use them directly!
        const watchlistData = watchlistItems.map(item => ({
            symbol: item.symbol,
            change_pct: item.change_pct || 0,  // Keep original field name
            price: item.price || 0,
            ghost_confidence: item.ghost_confidence || 0,  // Keep original field name
            ghost_direction: item.ghost_direction || 'FLAT', // Keep original field name
            type: item.type || 'stock',  // Default to stock if not specified
            asset_type: item.type || 'stock'  // Add asset_type alias
        }));
        
        // Apply filter (stocks/crypto/all)
        let filteredData = watchlistData;
        if (watchlistFilter === 'stocks') {
            filteredData = watchlistData.filter(item => item.type === 'stock');
        } else if (watchlistFilter === 'crypto') {
            filteredData = watchlistData.filter(item => item.type === 'crypto');
        }
        
        // CRITICAL: Populate shared cache for Major Caps and XRP VIP panels
        sharedWatchlistData = filteredData;
        
        renderWatchlist(filteredData);
    } catch (error) {
        console.error('[GHOST V3] Error loading market watchlist:', error);
        renderWatchlist([]);
    }
}

// Keep old loadWatchlist() as alias for backward compatibility
async function loadWatchlist() {
    await loadWatchlistByMode();
}

function renderWatchlist(data) {
    const container = document.getElementById('watchlist-table');
    
    if (!data || data.length === 0) {
        container.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 20px;">Watchlist empty - add symbols to track</p>';
        return;
    }
    
    container.innerHTML = data.slice(0, 15).map(item => {
        // API returns: price, ghost_confidence, ghost_direction, change_pct, type
        const priceDisplay = item.price ? `$${item.price.toFixed(2)}` : '--';
        
        // Use change_pct (from API) instead of change_24h
        const changePct = item.change_pct ?? item.change ?? 0;
        const changeDisplay = changePct !== 0 ? 
            `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%` : 
            '--';
        
        const scoreDisplay = item.ghost_confidence && item.ghost_confidence > 0 ? 
            `${item.ghost_confidence.toFixed(0)}%` : 
            '--';
        
        // Asset type display
        const assetType = (item.type || item.asset_type || 'stock').toUpperCase();
        
        // Direction from ghost_direction field
        const direction = item.ghost_direction || 'FLAT';
        const directionEmoji = direction === 'UP' ? '↑' : direction === 'DOWN' ? '↓' : '→';
        const changeClass = changePct >= 0 ? 'positive' : 'negative';
        
        // Momentum indicator (if available)
        let momentumHTML = '';
        if (item.momentum || item.momentum_status) {
            const momentum = item.momentum || {};
            const status = momentum.status || item.momentum_status || 'STABLE';
            const emoji = momentum.emoji || '➡️';
            const delta = momentum.change_pct || 0;
            const momentumClass = status.toLowerCase().replace(' ', '-');
            
            momentumHTML = `
                <div class="watchlist-momentum ${momentumClass}" title="Momentum: ${status}">
                    <span class="momentum-icon">${emoji}</span>
                    ${delta !== 0 ? `<span class="momentum-change">${delta > 0 ? '+' : ''}${delta.toFixed(1)}%</span>` : ''}
                </div>
            `;
        }
        
        return `
            <div class="watchlist-row">
                <div class="watchlist-left">
                    <div class="watchlist-icon">${getSymbolIcon(item.symbol)}</div>
                    <div class="watchlist-ticker">
                        ${item.symbol}
                        <span style="font-size: 10px; color: var(--text-secondary); margin-left: 4px;">${assetType}</span>
                    </div>
                    <div class="watchlist-price">${priceDisplay}</div>
                </div>
                <div class="watchlist-right">
                    <div class="watchlist-move ${changeClass}">
                        ${changeDisplay}
                    </div>
                    <div class="watchlist-score">${directionEmoji} Ghost: ${scoreDisplay}</div>
                    ${momentumHTML}
                </div>
            </div>
        `;
    }).join('');
}

// Panel 6: Health Score
async function loadHealthScore() {
    try {
        // Fetch both goals and health metrics in parallel
        const [goalsResponse, healthResponse] = await Promise.all([
            fetch('/api/v3/goals/snapshot'),
            fetch('/api/v3/health/metrics')
        ]);
        
        if (!goalsResponse.ok) throw new Error('Failed to load goals');
        
        const goalsData = await goalsResponse.json();
        
        // Use ghost_score from goals API (85 = 85%)
        const score = goalsData.ghost_score || 0;
        const grade = calculateGrade(score);
        
        console.log('[HEALTH] Ghost score:', score, 'Grade:', grade);
        
        document.getElementById('health-score-value').textContent = score > 0 ? score.toFixed(0) : '--';
        document.getElementById('health-grade').textContent = grade;
        
        // Get real health metrics or use goals as fallback
        let healthMetrics = {
            daily: goalsData.daily_goal_pct || 0,
            weekly: goalsData.weekly_goal_pct || 0,
            monthly: goalsData.monthly_goal_pct || 0,
            data_health: 85,  // Fallback
            ai_activity: 75,  // Fallback
            accuracy: 70      // Fallback
        };
        
        if (healthResponse.ok) {
            const healthData = await healthResponse.json();
            healthMetrics.data_health = healthData.data_health || 85;
            healthMetrics.ai_activity = healthData.ai_activity || 75;
            healthMetrics.accuracy = healthData.accuracy || 70;
        }
        
        // Update health metrics display
        renderHealthMetrics(healthMetrics);
    } catch (error) {
        console.error('[HEALTH] Error loading health score:', error);
        document.getElementById('health-score-value').textContent = '--';
        document.getElementById('health-grade').textContent = 'F';
    }
}

function calculateGrade(score) {
    if (score >= 90) return 'A';
    if (score >= 80) return 'B';
    if (score >= 70) return 'C';
    if (score >= 60) return 'D';
    return 'F';
}

function renderHealthMetrics(metrics) {
    const container = document.getElementById('health-metrics');
    
    // Handle both old format (provider_health) and new format (daily/weekly/monthly)
    const metricsList = [];
    
    if (metrics.daily !== undefined) {
        // V3 format with goal progress + real health metrics
        metricsList.push(
            { name: 'Daily Goal', value: metrics.daily },
            { name: 'Weekly Goal', value: metrics.weekly },
            { name: 'Monthly Goal', value: metrics.monthly },
            { name: 'Data Health', value: metrics.data_health || 50 },  // Real value from API
            { name: 'AI Activity', value: metrics.ai_activity || 50 },  // Real value from API
            { name: 'Accuracy', value: metrics.accuracy || 50 }  // Real value from API
        );
    } else {
        // V2 format (backward compatibility)
        metricsList.push(
            { name: 'Providers', value: metrics.provider_health || 0 },
            { name: 'Predictions', value: metrics.prediction_coverage || 0 },
            { name: 'News Pipeline', value: metrics.news_health || 0 },
            { name: 'AI Engine', value: metrics.ai_health || 0 },
            { name: 'Latency', value: metrics.latency_score || 0 },
            { name: 'Error Rate', value: 100 - (metrics.error_rate || 0) }
        );
    }
    
    container.innerHTML = metricsList.map(metric => `
        <div class="health-metric">
            <span class="metric-name">${metric.name}</span>
            <div class="metric-bar">
                <div class="metric-fill ${getHealthClass(metric.value)}" style="width: ${metric.value}%"></div>
            </div>
            <span class="metric-value">${metric.value > 0 ? metric.value.toFixed(0) : '--'}%</span>
        </div>
    `).join('');
}

// Cockpit Snapshot (for system state)
async function loadCockpitSnapshot() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);  // 5s timeout
        
        const response = await fetch('/api/v3/cockpit/status', { signal: controller.signal });
        clearTimeout(timeoutId);
        
        if (!response.ok) throw new Error('Failed to load cockpit snapshot');
        
        const data = await response.json();
        
        // Update system status (use 'active' field from API)
        updateStatusIndicator(data.active !== undefined ? data.active : true);
        
        // Update header with last update time if available
        if (data.last_update_ts) {
            const lastUpdateEl = document.getElementById('last-update-time');
            if (lastUpdateEl) {
                const date = new Date(data.last_update_ts * 1000);
                lastUpdateEl.textContent = `Last updated: ${date.toLocaleTimeString()}`;
            }
        }
    } catch (error) {
        console.error('[GHOST V3] Error loading cockpit snapshot:', error);
        updateStatusIndicator(false);
    }
}

// Refresh specific panel
function refreshPanel(panel) {
    console.log(`[GHOST V3] Refreshing panel: ${panel}`);
    switch(panel) {
        case 'movers': loadTopMovers(); break;
        case 'forecast': loadForecast(); break;
        case 'news': loadNews(); break;
        case 'accuracy': loadAccuracyChart(); break;
        case 'watchlist': loadPersonalWatchlist(); break;
        case 'health': loadHealthScore(); break;
    }
}

// Helper Functions
function getSymbolIcon(symbol) {
    const icons = {
        'BTC': '₿',
        'ETH': 'Ξ',
        'WOLF': '🐺',
        'AAPL': '🍎',
        'TSLA': '🚗',
        'NVDA': '🎮',
        'MSFT': '🪟'
    };
    return icons[symbol] || '📈';
}

function getSentimentClass(sentiment) {
    if (sentiment > 0.3) return 'positive';
    if (sentiment < -0.3) return 'negative';
    return 'neutral';
}

function formatSentiment(sentiment) {
    if (!sentiment) return 'Neutral';
    if (sentiment > 0.3) return 'Bullish';
    if (sentiment < -0.3) return 'Bearish';
    return 'Neutral';
}

function formatTime(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000 / 60); // minutes ago
    
    if (diff < 60) return `${diff}m ago`;
    if (diff < 1440) return `${Math.floor(diff / 60)}h ago`;
    return `${Math.floor(diff / 1440)}d ago`;
}

function getHealthClass(value) {
    if (value >= 80) return 'healthy';
    if (value >= 50) return 'warning';
    return 'critical';
}

// Goals Settings Modal Functions
async function openGoalsModal() {
    try {
        console.log('[GOALS] Opening modal, fetching current goals...');
        
        // Fetch current goals
        const response = await fetch('/api/v3/goals/snapshot');
        if (!response.ok) {
            throw new Error(`API returned ${response.status}`);
        }
        
        const data = await response.json();
        console.log('[GOALS] API response:', JSON.stringify(data, null, 2));
        
        // Populate input fields with current goals
        if (data.ok && data.goals) {
            // API returns {daily: 500, weekly: 2500, ...}
            const daily = data.goals.daily || 500;
            const weekly = data.goals.weekly || 2500;
            const monthly = data.goals.monthly || 10000;
            const yearly = data.goals.yearly || 120000;
            
            console.log('[GOALS] Setting input values:', { daily, weekly, monthly, yearly });
            
            // Set values with verification
            const dailyInput = document.getElementById('goal-daily');
            const weeklyInput = document.getElementById('goal-weekly');
            const monthlyInput = document.getElementById('goal-monthly');
            const yearlyInput = document.getElementById('goal-yearly');
            
            if (dailyInput) {
                dailyInput.value = daily;
                console.log(`[GOALS] Set dailyInput.value = ${daily}, actual value = ${dailyInput.value}`);
            }
            if (weeklyInput) {
                weeklyInput.value = weekly;
                console.log(`[GOALS] Set weeklyInput.value = ${weekly}, actual value = ${weeklyInput.value}`);
            }
            if (monthlyInput) {
                monthlyInput.value = monthly;
                console.log(`[GOALS] Set monthlyInput.value = ${monthly}, actual value = ${monthlyInput.value}`);
            }
            if (yearlyInput) {
                yearlyInput.value = yearly;
                console.log(`[GOALS] Set yearlyInput.value = ${yearly}, actual value = ${yearlyInput.value}`);
            }
            
            console.log('[GOALS] All input values after setting:', {
                daily: dailyInput?.value,
                weekly: weeklyInput?.value,
                monthly: monthlyInput?.value,
                yearly: yearlyInput?.value
            });
        } else {
            console.warn('[GOALS] No goals data in response, inputs will be empty');
            console.warn('[GOALS] Response structure:', data);
        }
        
        // Show modal AFTER values are set
        const modal = document.getElementById('goals-modal');
        modal.classList.add('active');
        console.log('[GOALS] Modal displayed with active class');
        
        // Verify values are still set after modal is shown
        setTimeout(() => {
            const dailyInput = document.getElementById('goal-daily');
            const weeklyInput = document.getElementById('goal-weekly');
            const monthlyInput = document.getElementById('goal-monthly');
            const yearlyInput = document.getElementById('goal-yearly');
            
            console.log('[GOALS] Post-display verification (50ms later):', {
                daily: dailyInput?.value,
                weekly: weeklyInput?.value,
                monthly: monthlyInput?.value,
                yearly: yearlyInput?.value
            });
        }, 50);
    } catch (error) {
        console.error('[GOALS] Error loading goals:', error);
        // Show modal anyway with empty inputs
        document.getElementById('goals-modal').classList.add('active');
    }
}

function closeGoalsModal() {
    document.getElementById('goals-modal').classList.remove('active');
}

async function saveGoals() {
    try {
        const daily = parseFloat(document.getElementById('goal-daily').value) || 0;
        const weekly = parseFloat(document.getElementById('goal-weekly').value) || 0;
        const monthly = parseFloat(document.getElementById('goal-monthly').value) || 0;
        const yearly = parseFloat(document.getElementById('goal-yearly').value) || 0;
        
        // Save each goal using v3 API endpoint
        const periods = [
            { period: 'daily', amount: daily },
            { period: 'weekly', amount: weekly },
            { period: 'monthly', amount: monthly },
            { period: 'yearly', amount: yearly }
        ];
        
        for (const goal of periods) {
            if (goal.amount > 0) {
                console.log(`Saving ${goal.period} goal: $${goal.amount}`);
                const response = await fetch(`/api/v3/goals/set?period=${goal.period}&target_amount=${goal.amount}`, {
                    method: 'POST'
                });
                
                const data = await response.json();
                console.log(`Response for ${goal.period}:`, data);
                
                if (!response.ok || !data.ok) {
                    const errorMsg = data.error || `HTTP ${response.status}`;
                    throw new Error(`Failed to set ${goal.period} goal: ${errorMsg}`);
                }
            }
        }
        
        // Close modal
        closeGoalsModal();
        
        // Refresh goals panel
        await loadHealthScore();
        
        // Show success message with visual confirmation
        alert('✅ Goals saved successfully!\n\nYour new targets:\n' +
              `Daily: $${daily}\n` +
              `Weekly: $${weekly}\n` +
              `Monthly: $${monthly}\n` +
              `Yearly: $${yearly}`);
        console.log('✅ Goals saved successfully!');
    } catch (error) {
        console.error('Error saving goals:', error);
        alert(`Failed to save goals: ${error.message}\n\nCheck browser console for details.`);
    }
}

// CASCADE SYSTEM FUNCTIONS
async function updateCascadeView() {
    try {
        const response = await fetch('/api/v3/cascade/list?active_only=true');
        const data = await response.json();
        
        if (data.ok && data.cascades && data.cascades.length > 0) {
            const cascade = data.cascades[0]; // Most recent cascade
            
            const now = Date.now() / 1000;
            const elapsed = now - cascade.created_at;
            const hoursElapsed = Math.floor(elapsed / 3600);
            const minsElapsed = Math.floor((elapsed % 3600) / 60);
            
            const timing = {
                hoursElapsed,
                minsElapsed,
                h24_remaining_hours: Math.max(0, 24 - hoursElapsed),
                h24_remaining_mins: minsElapsed > 0 ? 60 - minsElapsed : 0,
                h42_remaining_hours: Math.max(0, 42 - hoursElapsed),
                h6_remaining_mins: minsElapsed > 0 ? 60 - minsElapsed : 0,
                h48_remaining_hours: Math.max(0, 48 - hoursElapsed),
                h48_remaining_mins: minsElapsed > 0 ? 60 - minsElapsed : 0
            };
            
            renderCascadeCard(cascade, timing);
        } else {
            const cascadeCard = document.getElementById('cascade-card');
            if (cascadeCard) cascadeCard.style.display = 'none';
        }
    } catch (error) {
        console.error('Failed to fetch cascade data:', error);
    }
}

function renderCascadeCard(cascade, timing) {
    const cascadeCard = document.getElementById('cascade-card');
    if (!cascadeCard) return;
    
    const { hoursElapsed, minsElapsed, h24_remaining_hours, h24_remaining_mins,
            h42_remaining_hours, h6_remaining_mins, h48_remaining_hours, h48_remaining_mins } = timing;
    
    const directionEmoji = cascade.h48_direction === 'UP' ? '📈' : '📉';
    const directionClass = cascade.h48_direction === 'UP' ? 'bullish' : 'bearish';
    
    let html = `
        <div class="cascade-header">
            <h3>🔮 ${cascade.symbol} PREDICTION CASCADE</h3>
            <span class="cascade-running">Running ${hoursElapsed}h ${minsElapsed}m</span>
        </div>
        
        <div class="cascade-stages">
            <!-- 48h Alert Stage -->
            <div class="cascade-stage completed">
                <div class="stage-marker"></div>
                <div class="stage-content">
                    <h4>🔔 48H ALERT</h4>
                    <div class="stage-prediction ${directionClass}">
                        ${directionEmoji} ${cascade.h48_direction} @ ${(cascade.h48_confidence * 100).toFixed(1)}%
                    </div>
                    <div class="stage-details">
                        Entry: $${cascade.h48_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                    </div>
                    <div class="stage-status">✅ Sent</div>
                </div>
            </div>
    `;
    
    // 24h Update Stage
    if (cascade.h24_sent_at) {
        const h24DirectionEmoji = cascade.h24_direction === 'UP' ? '📈' : '📉';
        const h24DirectionClass = cascade.h24_direction === 'UP' ? 'bullish' : 'bearish';
        const directionChanged = cascade.h24_direction_changed ? ' 🔄' : '';
        const confidenceDelta = cascade.h24_confidence_delta ? 
            ` (${cascade.h24_confidence_delta > 0 ? '+' : ''}${(cascade.h24_confidence_delta * 100).toFixed(1)}%)` : '';
        
        html += `
            <div class="cascade-stage completed">
                <div class="stage-marker"></div>
                <div class="stage-content">
                    <h4>📈 24H UPDATE${directionChanged}</h4>
                    <div class="stage-prediction ${h24DirectionClass}">
                        ${h24DirectionEmoji} ${cascade.h24_direction} @ ${(cascade.h24_confidence * 100).toFixed(1)}%${confidenceDelta}
                    </div>
                    <div class="stage-details">
                        Price: $${cascade.h24_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                    </div>
                    <div class="stage-status">✅ Updated</div>
                </div>
            </div>
        `;
    } else {
        html += `
            <div class="cascade-stage pending">
                <div class="stage-marker"></div>
                <div class="stage-content">
                    <h4>📈 24H UPDATE</h4>
                    <div class="stage-countdown">
                        ⏰ In ${h24_remaining_hours}h ${h24_remaining_mins}m
                    </div>
                    <div class="stage-details">
                        Will re-evaluate with fresh data
                    </div>
                </div>
            </div>
        `;
    }
    
    // 6h Final Stage
    if (cascade.h6_sent_at) {
        const h6DirectionEmoji = cascade.h6_direction === 'UP' ? '📈' : '📉';
        const h6DirectionClass = cascade.h6_direction === 'UP' ? 'bullish' : 'bearish';
        const directionChanged = cascade.h6_direction_changed ? ' 🔄' : '';
        const confidenceDelta = cascade.h6_confidence_delta ? 
            ` (${cascade.h6_confidence_delta > 0 ? '+' : ''}${(cascade.h6_confidence_delta * 100).toFixed(1)}%)` : '';
        
        html += `
            <div class="cascade-stage completed">
                <div class="stage-marker"></div>
                <div class="stage-content">
                    <h4>✅ 6H FINAL CALL${directionChanged}</h4>
                    <div class="stage-prediction ${h6DirectionClass}">
                        ${h6DirectionEmoji} ${cascade.h6_direction} @ ${(cascade.h6_confidence * 100).toFixed(1)}%${confidenceDelta}
                    </div>
                    <div class="stage-details">
                        Price: $${cascade.h6_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                    </div>
                    <div class="stage-status">✅ Final Call Made</div>
                </div>
            </div>
        `;
    } else {
        html += `
            <div class="cascade-stage pending">
                <div class="stage-marker"></div>
                <div class="stage-content">
                    <h4>✅ 6H FINAL CALL</h4>
                    <div class="stage-countdown">
                        ⏰ In ${h42_remaining_hours}h ${h6_remaining_mins}m
                    </div>
                    <div class="stage-details">
                        High-confidence prediction
                    </div>
                </div>
            </div>
        `;
    }
    
    // Outcome Stage
    if (cascade.evaluated_at) {
        const h48Status = cascade.h48_correct ? '✅' : '❌';
        const h24Status = cascade.h24_correct !== null ? (cascade.h24_correct ? '✅' : '❌') : '⏳';
        const h6Status = cascade.h6_correct !== null ? (cascade.h6_correct ? '✅' : '❌') : '⏳';
        const correctCount = [cascade.h48_correct, cascade.h24_correct, cascade.h6_correct].filter(x => x === true).length;
        
        html += `
            <div class="cascade-stage evaluated">
                <div class="stage-marker"></div>
                <div class="stage-content">
                    <h4>🎯 OUTCOME</h4>
                    <div class="stage-prediction">
                        Score: ${correctCount}/3 stages correct
                    </div>
                    <div class="stage-details">
                        48h ${h48Status} | 24h ${h24Status} | 6h ${h6Status}
                    </div>
                    <div class="stage-status">✅ Evaluated</div>
                </div>
            </div>
        `;
    } else {
        html += `
            <div class="cascade-stage pending">
                <div class="stage-marker"></div>
                <div class="stage-content">
                    <h4>🎯 OUTCOME</h4>
                    <div class="stage-countdown">
                        ⏰ In ${h48_remaining_hours}h ${h48_remaining_mins}m
                    </div>
                    <div class="stage-details">
                        Validation & scoring
                    </div>
                </div>
            </div>
        `;
    }
    
    html += `</div>`; // Close cascade-stages
    
    cascadeCard.innerHTML = html;
    cascadeCard.style.display = 'block';
}

// Add cascade update to intervals
function initializeCascadeUpdates() {
    updateCascadeView(); // Initial load
    window.cascadeInterval = setInterval(() => updateCascadeView(), 60000); // Update every minute
}

// Call on app init
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeCascadeUpdates);
} else {
    initializeCascadeUpdates();
}

// Cleanup on unload
window.addEventListener('beforeunload', () => {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    if (window.cascadeInterval) {
        clearInterval(window.cascadeInterval);
    }
});
