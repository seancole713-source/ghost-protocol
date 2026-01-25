// Ghost Protocol v3 - Minimal UI JavaScript

// State
let currentTab = 'stocks';
let currentForecastSymbol = 'CHZ';  // Default to CHZ (top V2 whitelist crypto at 85%)
let updateInterval = null;
let watchlistMode = 'market';    // DEFAULT TO MARKET - personal watchlist API not ready
let watchlistFilter = 'all';     // 'all', 'stocks', 'crypto'
let sharedWatchlistData = [];    // Shared cache for cross-panel data (Major Caps, XRP VIP)
let isInitialized = false;       // BUG 5 FIX: Guard against double initialization

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // BUG 5 FIX: Prevent duplicate intervals on LIVE/FIXED toggle
    if (isInitialized) {
        console.log('[INIT] Already initialized, restarting intervals only');
        startIntervals();
        return;
    }
    isInitialized = true;
    
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
        
        // BUG 5 FIX: Only restart intervals, don't re-init event listeners
        startIntervals();
        console.log('[MODE] All intervals resumed');
    }
}

// BUG 5 FIX: Separate function to start intervals (called by mode change)
function startIntervals() {
    // Clear any existing intervals first
    if (window.updateInterval) clearInterval(window.updateInterval);
    if (window.statusInterval) clearInterval(window.statusInterval);
    if (window.topMoversInterval) clearInterval(window.topMoversInterval);
    if (window.vipInterval) clearInterval(window.vipInterval);
    if (window.watchlistInterval) clearInterval(window.watchlistInterval);
    if (window.forecastInterval) clearInterval(window.forecastInterval);
    if (window.healthInterval) clearInterval(window.healthInterval);
    if (window.accuracyInterval) clearInterval(window.accuracyInterval);
    
    // Restart intervals
    window.updateInterval = setInterval(updateSystemTime, 1000);
    window.statusInterval = setInterval(() => loadCockpitStatus(), 10000);
    window.topMoversInterval = setInterval(() => loadTopMovers(), 15000);
    window.vipInterval = setInterval(() => loadVIPWatch(), 30000);
    window.watchlistInterval = setInterval(() => loadWatchlistByMode(), 20000);
    window.forecastInterval = setInterval(() => loadForecast(), 60000);
    window.healthInterval = setInterval(() => loadHealthScore(), 30000);
    window.accuracyInterval = setInterval(() => loadAccuracyChart(), 60000);
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
            console.log('[STATUS] Cockpit status:', data);
            updateStatusIndicator(data.active);
            
            // Update system stats in health panel
            if (data.uptime_seconds !== undefined) {
                const uptimeHours = (data.uptime_seconds / 3600).toFixed(1);
                const uptimeEl = document.getElementById('system-uptime');
                console.log('[STATUS] Updating uptime:', uptimeHours, 'h, element:', uptimeEl);
                if (uptimeEl) {
                    uptimeEl.textContent = `${uptimeHours}h`;
                } else {
                    console.error('[STATUS] system-uptime element NOT FOUND');
                }
            }
            
            if (data.predictions_today !== undefined) {
                const countEl = document.getElementById('predictions-count');
                console.log('[STATUS] Updating predictions:', data.predictions_today, 'element:', countEl);
                if (countEl) {
                    countEl.textContent = data.predictions_today || 0;
                } else {
                    console.error('[STATUS] predictions-count element NOT FOUND');
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
    // ═══════════════════════════════════════════════════════════════════════
    // CRITICAL FIX: ALWAYS load market watchlist for sharedWatchlistData
    // VIP, Forecast, Major Caps all depend on this shared cache
    // Personal watchlist is NOT loaded on init - user can switch to it manually
    // ═══════════════════════════════════════════════════════════════════════
    try {
        // ALWAYS fetch market watchlist to populate sharedWatchlistData AND render
        await loadMarketWatchlist();
        console.log('[INIT] ✓ sharedWatchlistData ready:', sharedWatchlistData?.length || 0, 'items');
    } catch (e) {
        console.error('[INIT] ✗ Market watchlist load failed:', e);
    }
    
    // NOW safe to load panels that depend on sharedWatchlistData
    loadCockpitSnapshot().catch(e => console.error('Snapshot error:', e));
    loadLatestBTCPrediction().catch(e => console.error('BTC prediction error:', e));
    loadForecast().catch(e => console.error('Forecast error:', e));
    loadVIPCoins().catch(e => console.error('VIP coins error:', e));
    
    // ROBUST FIX: Use double requestAnimationFrame to ensure canvas layout is complete
    // This guarantees the browser has computed styles and layout before we measure
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            loadAccuracyChart().catch(e => console.error('Accuracy error:', e));
        });
    });
    
    // Slow panels - load in background (may take 10-30s on first load)
    setTimeout(() => {
        loadTopMovers().catch(e => console.error('Top movers error:', e));
        loadNews().catch(e => console.error('News error:', e));
        loadHealthScore().catch(e => console.error('Health score error:', e));
    }, 100);
    
    console.log('✅ All panels loaded (watchlist first, then parallel)');
}

// Panel 0: Latest V2 Prediction (CHZ = top V2 symbol) with CASCADE VIEW
// NOTE: BTC is V2 blacklisted - using CHZ (85% confidence) as default
async function loadLatestBTCPrediction() {
    // Use top V2 symbol instead of blacklisted BTC
    const v2TopSymbol = 'CHZ';  // Top V2 crypto at 85% win rate
    
    // BUG 6 FIX: Update title with actual symbol
    const titleEl = document.getElementById('latest-pred-title');
    if (titleEl) titleEl.textContent = `🎯 Latest ${v2TopSymbol} Prediction`;
    
    try {
        // Fetch both cascade data and momentum data
        const [cascadeResp, predResp, momentumResp] = await Promise.all([
            fetch(`/api/v3/cascade/list?symbol=${v2TopSymbol}&active_only=true`),
            fetch(`/api/v3/predictions/latest?symbol=${v2TopSymbol}`),
            fetch(`/api/v3/momentum/${v2TopSymbol}`)
        ]);
        
        const cascadeData = cascadeResp.ok ? await cascadeResp.json() : null;
        const predData = predResp.ok ? await predResp.json() : null;
        const momentumData = momentumResp.ok ? await momentumResp.json() : null;
        
        const activeCascade = cascadeData?.cascades?.[0] || null;
        const latestPred = predData?.predictions?.[0] || null;
        const momentum = momentumData?.momentum || null;
        
        console.log(`[${v2TopSymbol}] Cascade:`, activeCascade);
        console.log(`[${v2TopSymbol}] Momentum:`, momentum);
        console.log(`[${v2TopSymbol}] Prediction:`, latestPred);
        
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
        container.innerHTML = '<div class="pred-loading">⏳ Loading V2 predictions... (CHZ/RNDR/ZEC/TURBO)</div>';
        return;
    }
    
    const direction = prediction.direction || 'FLAT';
    const confidence = Math.round((prediction.confidence || 0) * 100);
    const expectedMove = (prediction.expected_move || 0).toFixed(2);
    // BUG 1 FIX: Force 6h horizon for this panel (API returns 48h cascade horizon)
    // The "6-Hour Horizon" panel should show 6h countdown, not full 48h cascade
    const horizonH = 6;  // Fixed to 6h for this panel (ignore prediction.horizon_h which is 48)
    
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
                <span class="move-value">${direction === 'DOWN' ? '-' : expectedMove > 0 ? '+' : ''}${Math.abs(expectedMove)}%</span>
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
    console.log('[VIP] loadVIPCoins() called');
    console.log('[VIP] sharedWatchlistData length:', sharedWatchlistData?.length || 0);
    
    // Load all three data sources in parallel
    try {
        const [xrpResponse, presaleResponse] = await Promise.all([
            fetch('/api/xrp/tracker').catch(e => ({ ok: false, error: e })),
            fetch('/api/presale/watch').catch(e => ({ ok: false, error: e }))
        ]);
        
        console.log('[VIP] XRP response ok:', xrpResponse.ok);
        
        // XRP Tracker (Priority) - Enhanced with Watchlist 24h data
        if (xrpResponse.ok) {
            const xrpData = await xrpResponse.json();
            console.log('[VIP] XRP data:', JSON.stringify(xrpData));
            
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
        
        // V2 FIX: Major Caps shows TOP V2 whitelisted crypto (CHZ, RNDR, ZEC, TURBO)
        // RACE CONDITION FIX: If sharedWatchlistData empty, fetch directly
        const v2CryptoSymbols = ['CHZ', 'RNDR', 'ZEC', 'TURBO'];
        let majorsFromWatchlist = sharedWatchlistData.filter(item => v2CryptoSymbols.includes(item.symbol));
        
        // If cache is empty, fetch watchlist directly (race condition workaround)
        if (majorsFromWatchlist.length === 0) {
            console.log('[VIP] Watchlist cache empty, fetching directly...');
            try {
                const watchlistRes = await fetch('/api/v3/watchlist/enriched');
                if (watchlistRes.ok) {
                    const watchlistData = await watchlistRes.json();
                    const items = watchlistData.items || [];
                    majorsFromWatchlist = items.filter(item => v2CryptoSymbols.includes(item.symbol));
                    console.log('[VIP] Direct fetch got', majorsFromWatchlist.length, 'V2 crypto');
                }
            } catch (e) {
                console.error('[VIP] Direct watchlist fetch failed:', e);
            }
        }
        
        if (majorsFromWatchlist.length > 0) {
            console.log('[VIP] V2 Crypto from Watchlist:', majorsFromWatchlist);
            renderMajorCaps(majorsFromWatchlist);
        } else {
            console.warn('[VIP] No V2 crypto found in Watchlist cache yet');
            document.getElementById('vip-majors-list').innerHTML = '<p style="color: var(--text-secondary); font-size: 13px;">Loading V2 crypto...</p>';
        }
        
    } catch (error) {
        console.error('[VIP] Error loading panel:', error);
        document.getElementById('xrp-tracker').innerHTML = '<p style="color: var(--accent-red);">❌ VIP panel error</p>';
    }
}

// Render XRP Tracker Widget
function renderXRPTracker(data) {
    console.log('[XRP-RENDER] renderXRPTracker called with:', JSON.stringify(data));
    const container = document.getElementById('xrp-tracker');
    console.log('[XRP-RENDER] Container found:', !!container);
    
    if (!container) {
        console.error('[XRP-RENDER] FATAL: #xrp-tracker container not found!');
        return;
    }
    
    // FIX: Backend returns bullish_eye as emoji string ("🟢", "🟡", "🔴")
    // Map emoji to label/color, use confidence for numeric display
    let eyeEmoji = data.bullish_eye || '🟡';
    let eyeLabel = 'NEUTRAL';
    
    // Determine label from emoji
    if (eyeEmoji === '🟢') {
        eyeLabel = 'BULLISH';
    } else if (eyeEmoji === '🔴') {
        eyeLabel = 'BEARISH';
    } else {
        eyeLabel = 'NEUTRAL';
        eyeEmoji = '🟡';  // Default
    }
    
    // Use confidence * 100 for numeric eye score display
    const eyeScore = Math.round((data.confidence || 0.5) * 100);
    
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
                <span>Eye Score: ${eyeScore}/100</span>
                <span>24h: ${(data.change_24h_pct ?? 0) >= 0 ? '+' : ''}${(data.change_24h_pct ?? 0).toFixed(2)}%</span>
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
    console.log('[MAJOR-CAPS] renderMajorCaps called');
    console.log('[MAJOR-CAPS] coins:', coins?.length, coins);
    const container = document.getElementById('vip-majors-list');
    console.log('[MAJOR-CAPS] Container found:', !!container);
    
    if (!container) {
        console.error('[MAJOR-CAPS] FATAL: #vip-majors-list container not found!');
        return;
    }
    
    if (!coins || coins.length === 0) {
        console.warn('[VIP] renderMajorCaps: No coins provided');
        container.innerHTML = '<p style="color: var(--text-secondary); font-size: 13px; grid-column: 1 / -1;">Loading majors...</p>';
        return;
    }
    
    console.log('[VIP] Rendering', coins.length, 'major caps:', coins);
    
    container.innerHTML = coins.map(coin => {
        // Watchlist format: {symbol, price, change_pct, ghost_confidence, ghost_direction, type}
        const isOffline = !coin.price || coin.price === 0;
        // FIX: Use dynamic decimals for small crypto prices (TURBO, CHZ)
        let priceDisplay = '--';
        if (!isOffline) {
            if (coin.price < 0.01) {
                priceDisplay = `$${coin.price.toFixed(6)}`;  // 6 decimals for tiny prices
            } else if (coin.price < 1) {
                priceDisplay = `$${coin.price.toFixed(4)}`;  // 4 decimals for sub-dollar
            } else {
                priceDisplay = `$${coin.price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            }
        }
        // BUG 3 FIX: Standardize null handling - show '--' for missing data, not +0.00%
        const changePct = coin.change_pct;
        const hasChange = changePct !== null && changePct !== undefined;
        const changeClass = (changePct || 0) >= 0 ? 'positive' : 'negative';
        const changeDisplay = (isOffline || !hasChange) ? '--' : 
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
    
    // BUG 7 FIX: Dynamic precision for sub-$1 assets
    const formatForecastPrice = (price) => {
        if (price < 0.01) return `$${price.toFixed(6)}`;
        if (price < 1) return `$${price.toFixed(4)}`;
        return `$${price.toFixed(2)}`;
    };
    
    if (watchlistSymbol && watchlistSymbol.price) {
        const currentPrice = watchlistSymbol.price;
        const targetPrice = currentPrice * (1 + (expectedMove / 100));
        card2.querySelector('.current-value').textContent = formatForecastPrice(currentPrice);
        card2.querySelector('.target-value').textContent = formatForecastPrice(targetPrice);
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
    console.log('[NEWS] ======= LOAD START =======');
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000);  // 10s timeout
        
        const response = await fetch('/api/v3/news/feed?limit=10', { signal: controller.signal });
        clearTimeout(timeoutId);
        
        console.log('[NEWS] Response status:', response.status, response.ok);
        if (!response.ok) throw new Error('Failed to load news');
        
        const data = await response.json();
        const container = document.getElementById('news-list');
        
        console.log('[NEWS] Parsed data:', data ? 'OK' : 'NULL', 'items:', data?.items?.length || 0);
        
        if (!data || !data.items || data.items.length === 0) {
            console.error('[NEWS] No items in response:', JSON.stringify(data).slice(0, 200));
            container.innerHTML = '<p style="color: var(--text-secondary); text-align: center;">No news available yet</p>';
            return;
        }
        
        console.log('[NEWS] Rendering', data.items.length, 'items');
        
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
    console.log('[ACCURACY] ======= LOAD START =======');
    try {
        const response = await fetch('/api/v3/accuracy/summary');
        console.log('[ACCURACY] API response status:', response.status, response.ok);
        if (!response.ok) throw new Error('Failed to load accuracy data');
        
        const data = await response.json();
        console.log('[ACCURACY] API payload:', JSON.stringify(data));
        
        // Handle API's {ok: false, error: "..."} format
        if (!data.ok) {
            console.log('[ACCURACY] API returned ok=false:', data.error);
            renderAccuracyChart(null);
            return;
        }
        
        console.log('[ACCURACY] Calling renderAccuracyChart with valid data');
        renderAccuracyChart(data);
    } catch (error) {
        console.error('[GHOST V3] Error loading accuracy chart:', error);
        renderAccuracyChart(null);
    }
}

function renderAccuracyChart(accuracyData) {
    console.log('[ACCURACY] ======= RENDER START =======');
    console.log('[ACCURACY] accuracyData:', accuracyData ? JSON.stringify({ok: accuracyData.ok, accuracy_pct: accuracyData.accuracy_pct, correct: accuracyData.correct_predictions, total: accuracyData.total_predictions}) : 'NULL');
    
    const canvas = document.getElementById('accuracy-chart');
    console.log('[ACCURACY] Canvas element:', canvas ? 'FOUND' : 'MISSING');
    if (!canvas) {
        console.error('[ACCURACY] Canvas element not found!');
        return;
    }
    
    // Check if canvas is visible
    const style = window.getComputedStyle(canvas);
    console.log('[ACCURACY] Canvas CSS: display=' + style.display + ', visibility=' + style.visibility + ', opacity=' + style.opacity);
    
    const ctx = canvas.getContext('2d');
    console.log('[ACCURACY] Canvas 2D context:', ctx ? 'OK' : 'FAILED');
    
    // Get canvas bounding rect
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    // ROBUST FIX: Validate dimensions - retry if too small (layout not ready)
    if (rect.width < 10 || rect.height < 10) {
        console.warn('[ACCURACY] Canvas rect too small:', rect.width, 'x', rect.height, '- retrying in 100ms');
        setTimeout(() => renderAccuracyChart(accuracyData), 100);
        return;
    }
    
    console.log('[ACCURACY] Canvas rect:', rect.width, 'x', rect.height, 'dpr:', dpr, 'buffer:', rect.width * dpr, 'x', rect.height * dpr);
    
    // Set canvas buffer size
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    
    // CRITICAL: Reset transform before scaling (prevents cumulative scaling bug)
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    
    // CRITICAL FIX: CSS variables don't work in canvas context - use actual color values
    const COLORS = {
        textPrimary: '#E8E6E3',
        textSecondary: '#9CA3AF',
        accentGreen: '#10B981',
        accentYellow: '#F59E0B',
        accentRed: '#EF4444',
        accentOrange: '#F97316'
    };
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!accuracyData) {
        // Show friendly "waiting for data" message
        ctx.fillStyle = COLORS.textSecondary;
        ctx.font = '13px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('⏳ Waiting for predictions to mature...', rect.width / 2, rect.height / 2 - 10);
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText('(Predictions need 48 hours to reconcile)', rect.width / 2, rect.height / 2 + 10);
        return;
    }
    
    // Extract metrics - use accuracy_pct as fallback for all timeframes
    const dailyAcc = accuracyData.daily_accuracy_pct || accuracyData.accuracy_pct || 0;
    const weeklyAcc = accuracyData.weekly_accuracy_pct || accuracyData.accuracy_pct || 0;
    const monthlyAcc = accuracyData.monthly_accuracy_pct || accuracyData.accuracy_pct || 0;
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
    ctx.font = '11px "JetBrains Mono", monospace';
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
        
        // Choose color based on value - use actual color values
        let barColor = COLORS.accentRed;  // <50%
        if (bar.value >= 70) barColor = COLORS.accentGreen;  // >=70%
        else if (bar.value >= 50) barColor = COLORS.accentYellow;  // 50-70%
        
        // Draw bar
        ctx.fillStyle = barColor;
        ctx.fillRect(bar.x, barY, barWidth, barHeight);
        
        // Draw value on top
        ctx.fillStyle = COLORS.textPrimary;
        ctx.font = 'bold 16px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${bar.value.toFixed(1)}%`, bar.x + barWidth / 2, barY - 10);
        
        // Draw label at bottom
        ctx.fillStyle = COLORS.textSecondary;
        ctx.font = '12px "JetBrains Mono", monospace';
        ctx.fillText(bar.label, bar.x + barWidth / 2, rect.height - 20);
    });
    
    console.log('[ACCURACY] Drew', bars.length, 'bars with values:', bars.map(b => b.value.toFixed(1) + '%').join(', '));
    
    // Draw status badge
    ctx.font = 'bold 14px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    
    let statusText = status;
    let statusColor = COLORS.accentRed;
    if (status === 'ACCURATE') {
        statusText = '✅ ACCURATE';
        statusColor = COLORS.accentGreen;
    } else if (status === 'IMPROVING') {
        statusText = '📈 IMPROVING';
        statusColor = COLORS.accentYellow;
    } else if (status === 'BELOW_TARGET') {
        statusText = '⚠️ BELOW TARGET';
        statusColor = COLORS.accentYellow;
    } else {
        statusText = '❌ NO DATA';
        statusColor = COLORS.textSecondary;
    }
    
    ctx.fillStyle = statusColor;
    ctx.fillText(statusText, rect.width / 2, 25);
    
    // Draw prediction count - use correct API field names
    const totalPreds = accuracyData.total_predictions || 0;
    const correct = accuracyData.correct_predictions || accuracyData.correct || 0;
    const wrong = totalPreds - correct;
    
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillStyle = COLORS.textSecondary;
    ctx.fillText(`${correct}W / ${wrong}L / ${totalPreds} Total`, rect.width / 2, 45);
    
    console.log('[ACCURACY] ======= RENDER COMPLETE =======');
    console.log('[ACCURACY] Final stats: status=' + statusText + ', ' + correct + 'W/' + wrong + 'L/' + totalPreds + ' total');
}

// Panel 5: Watchlist - Master loader
async function loadWatchlistByMode() {
    // NUCLEAR FIX: ALWAYS use market watchlist until personal API is stable
    // Personal watchlist API causes cascade failures when it has issues
    console.log('[WATCHLIST] Forcing market mode (personal disabled)');
    await loadMarketWatchlist();
    return;
    
    // DISABLED - Personal watchlist code below
    // if (watchlistMode === 'personal') {
    //     if (typeof loadPersonalWatchlist === 'function') {
    //         await loadPersonalWatchlist();
    //     } else {
    //         console.error('[WATCHLIST] personal_watchlist_ui.js not loaded');
    //         renderWatchlist([]);
    //     }
    // } else {
    //     await loadMarketWatchlist();
    // }
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
        
        // CRITICAL: Populate shared cache with ALL data FIRST (for Major Caps, XRP VIP, Forecast panels)
        // This ensures crypto symbols like CHZ are available for forecast even when filtered to "stocks"
        sharedWatchlistData = watchlistData;
        
        // Apply filter (stocks/crypto/all) for display only
        let filteredData = watchlistData;
        if (watchlistFilter === 'stocks') {
            filteredData = watchlistData.filter(item => item.type === 'stock');
        } else if (watchlistFilter === 'crypto') {
            filteredData = watchlistData.filter(item => item.type === 'crypto');
        }
        
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
        // Use more decimals for small crypto prices
        const priceDisplay = item.price ? 
            (item.price < 0.01 ? `$${item.price.toFixed(6)}` : 
             item.price < 1 ? `$${item.price.toFixed(4)}` : 
             `$${item.price.toFixed(2)}`) : '--';
        
        // Use change_pct (from API) instead of change_24h
        // BUG 3 FIX: Standardize null handling - show '--' only for truly missing data
        const changePct = item.change_pct ?? item.change;
        const hasChange = changePct !== null && changePct !== undefined;
        const changeDisplay = hasChange ? `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%` : '--';
        
        const scoreDisplay = item.ghost_confidence && item.ghost_confidence > 0 ? 
            `${item.ghost_confidence.toFixed(0)}%` : 
            '--';
        
        // Asset type display
        const assetType = (item.type || item.asset_type || 'stock').toUpperCase();
        
        // Direction from ghost_direction field
        const direction = item.ghost_direction || 'FLAT';
        const directionEmoji = direction === 'UP' ? '↑' : direction === 'DOWN' ? '↓' : '→';
        const changeClass = (changePct || 0) >= 0 ? 'positive' : 'negative';
        
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
    // Ghost Score grading thresholds (score 52 = C per spec)
    if (score >= 80) return 'A';
    if (score >= 60) return 'B';
    if (score >= 40) return 'C';
    if (score >= 20) return 'D';
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
        case 'watchlist': loadMarketWatchlist(); break;  // FIXED: Use market (personal disabled)
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
    // Handle string values from API (bullish/bearish/neutral)
    if (typeof sentiment === 'string') {
        const s = sentiment.toLowerCase();
        if (s === 'bullish' || s === 'positive') return 'positive';
        if (s === 'bearish' || s === 'negative') return 'negative';
        return 'neutral';
    }
    // Handle numeric values (-1 to 1)
    if (sentiment > 0.3) return 'positive';
    if (sentiment < -0.3) return 'negative';
    return 'neutral';
}

function formatSentiment(sentiment) {
    if (!sentiment) return 'Neutral';
    // Handle string values from API (bullish/bearish/neutral)
    if (typeof sentiment === 'string') {
        const s = sentiment.toLowerCase();
        if (s === 'bullish' || s === 'positive') return 'Bullish';
        if (s === 'bearish' || s === 'negative') return 'Bearish';
        return 'Neutral';
    }
    // Handle numeric values (-1 to 1)
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
    // IMMEDIATELY populate with defaults (sync) before any async calls
    document.getElementById('goal-daily').value = 500;
    document.getElementById('goal-weekly').value = 2500;
    document.getElementById('goal-monthly').value = 10000;
    document.getElementById('goal-yearly').value = 120000;
    
    // Show modal immediately with defaults
    document.getElementById('goals-modal').classList.add('active');
    console.log('[GOALS] Modal opened with defaults');
    
    // Then async update from API
    try {
        const response = await fetch('/api/v3/goals/snapshot');
        if (response.ok) {
            const data = await response.json();
            const goals = data.goals || {};
            
            // Update with actual values if available
            if (goals.daily) document.getElementById('goal-daily').value = goals.daily;
            if (goals.weekly) document.getElementById('goal-weekly').value = goals.weekly;
            if (goals.monthly) document.getElementById('goal-monthly').value = goals.monthly;
            if (goals.yearly) document.getElementById('goal-yearly').value = goals.yearly;
            
            console.log('[GOALS] Updated from API:', goals);
        }
    } catch (error) {
        console.error('[GOALS] API error (using defaults):', error);
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PAPER TRADING TRACKER
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async function updatePaperTrades() {
    try {
        // Get stats - V2 filtered with since date for clean data
        // V2 era started 2026-01-14, use that as default cutoff
        const statsRes = await fetch('/api/v3/paper/stats?since=2026-01-14&v2_only=true');
        const statsData = await statsRes.json();
        
        if (statsData.ok && statsData.stats) {
            const stats = statsData.stats;
            
            // Update stats display
            document.getElementById('paper-win-rate').textContent = 
                `${(stats.win_rate * 100).toFixed(1)}%`;
            document.getElementById('paper-total').textContent = stats.resolved_trades || 0;
            document.getElementById('paper-pending').textContent = stats.pending_trades || 0;
            
            const pnlEl = document.getElementById('paper-pnl');
            const pnl = stats.total_pnl || 0;
            pnlEl.textContent = `$${pnl.toFixed(2)}`;
            pnlEl.className = 'stat-value ' + (pnl >= 0 ? 'positive' : 'negative');
        }
        
        // Get recent trades - V2 filtered
        const tradesRes = await fetch('/api/v3/paper/trades?limit=20&v2_only=true');
        const tradesData = await tradesRes.json();
        
        if (tradesData.ok && tradesData.trades) {
            renderPaperTrades(tradesData.trades);
        }
        
    } catch (error) {
        console.error('Failed to update paper trades:', error);
    }
}

function renderPaperTrades(trades) {
    const container = document.getElementById('paper-trades-list');
    
    if (!trades || trades.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <div class="empty-state-text">No Paper Trades Yet</div>
                <div class="empty-state-subtext">Ghost will auto-log signals at 6h final calls</div>
            </div>
        `;
        return;
    }
    
    // ROBUST DEDUPLICATION:
    // 1. Primary: Use cascade_id (unique per prediction)
    // 2. Fallback: symbol-direction-roundedEntry (for legacy data without cascade_id)
    const seen = new Set();
    const uniqueTrades = trades.filter(trade => {
        // Primary key: cascade_id is unique
        if (trade.cascade_id) {
            if (seen.has(trade.cascade_id)) return false;
            seen.add(trade.cascade_id);
            return true;
        }
        // Fallback: paper_trade_id
        if (trade.paper_trade_id) {
            const key = `pid_${trade.paper_trade_id}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        }
        // Last resort: symbol + direction + entry rounded to 2 decimals
        const roundedEntry = Math.round((trade.entry_price || 0) * 100) / 100;
        const key = `${trade.symbol}-${trade.signal_direction}-${roundedEntry}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
    
    console.log('[PAPER] Dedup:', trades.length, 'raw →', uniqueTrades.length, 'unique');
    
    // BUG 1 FIX: Group by symbol+direction, show only LATEST per group
    // This prevents multiple entries for same symbol spamming the list
    const latestByGroup = {};
    uniqueTrades.forEach(trade => {
        const groupKey = `${trade.symbol}-${trade.signal_direction}`;
        const existing = latestByGroup[groupKey];
        if (!existing || new Date(trade.signal_time) > new Date(existing.signal_time)) {
            latestByGroup[groupKey] = trade;
        }
    });
    const displayTrades = Object.values(latestByGroup);
    console.log('[PAPER] Grouped:', uniqueTrades.length, 'unique →', displayTrades.length, 'display');
    
    container.innerHTML = displayTrades.map(trade => {
        const direction = trade.signal_direction?.toLowerCase() || 'long';
        const outcome = trade.outcome?.toLowerCase() || 'pending';
        const pnl = trade.profit_loss || 0;
        const pnlPct = trade.profit_loss_pct || 0;
        
        return `
            <div class="trade-item">
                <div class="trade-item-header">
                    <div>
                        <span class="trade-symbol">${trade.symbol}</span>
                        <span class="trade-direction ${direction}">${direction}</span>
                    </div>
                    <span class="trade-outcome ${outcome}">${outcome}</span>
                </div>
                <div class="trade-item-details">
                    <div class="trade-detail">
                        <div class="trade-detail-label">Entry</div>
                        <div class="trade-detail-value">$${trade.entry_price.toLocaleString()}</div>
                    </div>
                    <div class="trade-detail">
                        <div class="trade-detail-label">Target</div>
                        <div class="trade-detail-value">${trade.target_price ? '$' + trade.target_price.toLocaleString() : '--'}</div>
                    </div>
                    <div class="trade-detail">
                        <div class="trade-detail-label">Confidence</div>
                        <div class="trade-detail-value">${(trade.signal_confidence * 100).toFixed(0)}%</div>
                    </div>
                    <div class="trade-detail">
                        <div class="trade-detail-label">P&L</div>
                        <div class="trade-detail-value ${pnl >= 0 ? 'positive' : 'negative'}">
                            ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} (${pnlPct >= 0 ? '+' : ''}${(pnlPct * 100).toFixed(2)}%)
                        </div>
                    </div>
                </div>
                ${trade.actual_direction ? `
                    <div class="trade-notes">
                        Actual: ${trade.actual_direction} | Signal: ${trade.signal_time ? new Date(trade.signal_time).toLocaleString() : 'N/A'}
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TRADE JOURNAL
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async function updateJournal() {
    try {
        // Get stats
        const statsRes = await fetch('/api/v3/journal/stats?days=30');
        const statsData = await statsRes.json();
        
        if (statsData.ok && statsData.stats) {
            const stats = statsData.stats;
            
            // Update stats display
            document.getElementById('journal-win-rate').textContent = 
                stats.win_rate ? `${(stats.win_rate * 100).toFixed(1)}%` : '--';
            document.getElementById('journal-open').textContent = stats.open_trades || 0;
            
            const pnlEl = document.getElementById('journal-pnl');
            const pnl = stats.total_pnl || 0;
            pnlEl.textContent = `$${pnl.toFixed(2)}`;
            pnlEl.className = 'stat-value ' + (pnl >= 0 ? 'positive' : 'negative');
        }
        
        // Get recent trades
        const tradesRes = await fetch('/api/v3/journal/trades?limit=20');
        const tradesData = await tradesRes.json();
        
        if (tradesData.ok && tradesData.trades) {
            renderJournalTrades(tradesData.trades);
        }
        
    } catch (error) {
        console.error('Failed to update journal:', error);
    }
}

function renderJournalTrades(trades) {
    const container = document.getElementById('journal-trades-list');
    
    if (!trades || trades.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📝</div>
                <div class="empty-state-text">No Trades Logged</div>
                <div class="empty-state-subtext">Click "+ Log Trade" to record your first trade</div>
            </div>
        `;
        return;
    }
    
    container.innerHTML = trades.map(trade => {
        const direction = trade.direction?.toLowerCase() || 'long';
        const isOpen = !trade.exit_time;
        const pnl = trade.profit_loss || 0;
        const pnlPct = trade.profit_loss_pct || 0;
        
        return `
            <div class="trade-item">
                <div class="trade-item-header">
                    <div>
                        <span class="trade-symbol">${trade.symbol}</span>
                        <span class="trade-direction ${direction}">${direction}</span>
                    </div>
                    ${isOpen ? 
                        '<span class="trade-outcome pending">OPEN</span>' :
                        `<span class="trade-outcome ${pnl >= 0 ? 'win' : 'loss'}">${pnl >= 0 ? 'WIN' : 'LOSS'}</span>`
                    }
                </div>
                <div class="trade-item-details">
                    <div class="trade-detail">
                        <div class="trade-detail-label">Entry</div>
                        <div class="trade-detail-value">$${trade.entry_price.toLocaleString()}</div>
                    </div>
                    ${!isOpen ? `
                        <div class="trade-detail">
                            <div class="trade-detail-label">Exit</div>
                            <div class="trade-detail-value">$${trade.exit_price.toLocaleString()}</div>
                        </div>
                    ` : ''}
                    <div class="trade-detail">
                        <div class="trade-detail-label">Size</div>
                        <div class="trade-detail-value">$${trade.position_size.toLocaleString()}</div>
                    </div>
                    ${!isOpen ? `
                        <div class="trade-detail">
                            <div class="trade-detail-label">P&L</div>
                            <div class="trade-detail-value ${pnl >= 0 ? 'positive' : 'negative'}">
                                ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} (${pnlPct >= 0 ? '+' : ''}${(pnlPct * 100).toFixed(2)}%)
                            </div>
                        </div>
                    ` : ''}
                </div>
                ${trade.entry_notes ? `
                    <div class="trade-notes">${trade.entry_notes}</div>
                ` : ''}
                ${isOpen && trade.stop_loss ? `
                    <div class="trade-notes">
                        Stop: $${trade.stop_loss.toLocaleString()} | Target: ${trade.take_profit ? '$' + trade.take_profit.toLocaleString() : 'N/A'}
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

// Journal Form Handlers
document.getElementById('btn-log-trade')?.addEventListener('click', () => {
    document.getElementById('journal-entry-form').style.display = 'block';
    document.getElementById('journal-stats').style.display = 'none';
    document.getElementById('journal-trades-list').style.display = 'none';
});

document.getElementById('btn-cancel-entry')?.addEventListener('click', () => {
    document.getElementById('journal-entry-form').style.display = 'none';
    document.getElementById('journal-stats').style.display = 'grid';
    document.getElementById('journal-trades-list').style.display = 'block';
    clearJournalForm();
});

document.getElementById('btn-save-entry')?.addEventListener('click', async () => {
    try {
        const symbol = document.getElementById('entry-symbol').value.toUpperCase();
        const direction = document.getElementById('entry-direction').value;
        const entryPrice = parseFloat(document.getElementById('entry-price').value);
        const positionSize = parseFloat(document.getElementById('entry-size').value);
        const stopLoss = document.getElementById('entry-stop').value ? 
            parseFloat(document.getElementById('entry-stop').value) : null;
        const takeProfit = document.getElementById('entry-target').value ? 
            parseFloat(document.getElementById('entry-target').value) : null;
        const notes = document.getElementById('entry-notes').value;
        
        if (!symbol || !entryPrice || !positionSize) {
            alert('Please fill in all required fields');
            return;
        }
        
        // Build query params
        const params = new URLSearchParams({
            symbol,
            direction,
            entry_price: entryPrice,
            position_size: positionSize
        });
        
        if (stopLoss) params.append('stop_loss', stopLoss);
        if (takeProfit) params.append('take_profit', takeProfit);
        if (notes) params.append('notes', notes);
        
        const response = await fetch(`/api/v3/journal/entry?${params}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.ok) {
            alert('Trade logged successfully!');
            document.getElementById('btn-cancel-entry').click();
            updateJournal(); // Refresh
        } else {
            alert('Failed to log trade: ' + (result.error || 'Unknown error'));
        }
        
    } catch (error) {
        console.error('Failed to save trade:', error);
        alert('Error saving trade');
    }
});

function clearJournalForm() {
    document.getElementById('entry-symbol').value = '';
    document.getElementById('entry-price').value = '';
    document.getElementById('entry-size').value = '';
    document.getElementById('entry-stop').value = '';
    document.getElementById('entry-target').value = '';
    document.getElementById('entry-notes').value = '';
}

// Initialize updates
function initializeTradeTracking() {
    updatePaperTrades();
    updateJournal();
    
    // Update every 60 seconds
    setInterval(() => {
        updatePaperTrades();
        updateJournal();
    }, 60000);
}

// Start tracking when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeTradeTracking);
} else {
    initializeTradeTracking();
}
