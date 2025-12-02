// Ghost Protocol v3 - Minimal UI JavaScript

// State
let currentTab = 'stocks';
let currentForecastSymbol = 'BTC';  // Default to BTC (has active predictions)
let updateInterval = null;
let watchlistMode = 'personal';  // 'personal' or 'market'
let watchlistFilter = 'all';     // 'all', 'stocks', 'crypto'

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    updateSystemTime();
    loadAllPanels();
    
    // OPTIMIZED: Set smart update intervals (reduced from 5s to prevent hammering)
    // Goals/Stats: 30s (slow-changing data)
    // Predictions/Forecast: 15s (medium-priority)
    // Top Movers/Hunter: 10s (fast-moving opportunities)
    // Time display: 1s (real-time clock)
    
    setInterval(() => updateSystemTime(), 1000);  // Clock: every 1s
    setInterval(() => loadHealthScore(), 30000);  // Goals/Health: every 30s
    setInterval(() => loadForecast(), 15000);  // Forecast: every 15s
    setInterval(() => loadTopMovers(), 10000);  // Top Movers: every 10s (includes hunter feed)
    setInterval(() => loadWatchlistByMode(), 15000);  // Watchlist: every 15s (mode-aware)
    setInterval(() => loadVIPCoins(), 15000);  // VIP Coins: every 15s
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
    setInterval(() => {
        const now = new Date();
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        timeEl.textContent = `${hours}:${minutes}:${seconds}`;
    }, 1000);
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
    console.log('Mode changed to:', mode);
    // Could POST to /api/cockpit/mode if endpoint exists
}

function updateStatusIndicator(isActive) {
    const dot = document.getElementById('status-indicator');
    const text = document.getElementById('status-text');
    
    if (isActive) {
        dot.style.background = 'var(--accent-green)';
        text.textContent = 'LIVE';
        text.style.color = 'var(--accent-green)';
    } else {
        dot.style.background = 'var(--accent-red)';
        text.textContent = 'STOPPED';
        text.style.color = 'var(--accent-red)';
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
    try {
        await Promise.all([
            loadCockpitSnapshot(),
            loadTopMovers(),
            loadVIPCoins(),
            loadForecast(),
            loadNews(),
            loadWatchlistByMode(),  // Use mode-aware watchlist loader
            loadHealthScore()
        ]);
    } catch (error) {
        console.error('Error loading panels:', error);
    }
}

// Panel 1: Top Movers
async function loadTopMovers() {
    try {
        const response = await fetch('/api/v3/hunter/feed');
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
        console.error('Error loading movers:', error);
        document.getElementById('movers-list').innerHTML = '<p style="color: var(--accent-red);">Failed to load movers</p>';
    }
}

// Panel VIP: VIP Coins + XRP
async function loadVIPCoins() {
    const container = document.getElementById('vip-list');
    
    try {
        // Add 8-second timeout to prevent indefinite hanging
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000);
        
        const response = await fetch('/api/v3/vip/snapshot', { signal: controller.signal });
        clearTimeout(timeoutId);
        
        if (!response.ok) throw new Error('Failed to load VIP coins');
        
        const data = await response.json();
        
        console.log('[VIP] Loaded:', data.count || 0, 'coins');
        
        // Get VIP coins and XRP
        const vipCoins = data.vip_coins || [];
        const xrp = data.xrp || null;
        
        // Combine VIP + XRP
        const allCoins = [...vipCoins];
        if (xrp) {
            allCoins.push({
                symbol: xrp.symbol,
                price: xrp.price,
                change_pct: xrp.change_pct,
                status: xrp.provider !== 'offline' ? 'online' : 'offline'
            });
        }
        
        if (allCoins.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); text-align: center;">VIP data loading...</p>';
            return;
        }
        
        container.innerHTML = allCoins.map(coin => {
            const isOffline = coin.status === 'offline' || coin.price === 0;
            const priceDisplay = isOffline ? '--' : `$${coin.price.toFixed(6)}`;
            const changeDisplay = isOffline ? '--' : 
                `${coin.change_pct >= 0 ? '+' : ''}${coin.change_pct.toFixed(2)}%`;
            
            return `
                <div class="mover-card vip-card ${isOffline ? 'offline' : ''}">
                    <div class="mover-left">
                        <div class="mover-icon">${getSymbolIcon(coin.symbol)}</div>
                        <div class="mover-info">
                            <div class="mover-name">${coin.symbol}</div>
                            <div class="mover-symbol">${priceDisplay}</div>
                        </div>
                    </div>
                    <div class="mover-right">
                        <div class="mover-change ${coin.change_pct >= 0 ? 'positive' : 'negative'}">
                            ${changeDisplay}
                        </div>
                        <div class="mover-confidence">${isOffline ? 'Offline' : 'Live'}</div>
                    </div>
                </div>
            `;
        }).join('');
    } catch (error) {
        console.error('[VIP] Error loading coins:', error);
        const container = document.getElementById('vip-list');
        
        if (error.name === 'AbortError') {
            container.innerHTML = '<p style="color: var(--accent-orange);">⏱️ VIP data timeout (backend slow - fix in progress)</p>';
        } else {
            container.innerHTML = '<p style="color: var(--accent-red);">VIP data unavailable</p>';
        }
    }
}

// Panel 2: Forecast
let currentForecastSymbol = 'BTC';  // Default symbol

async function loadForecast() {
    try {
        const response = await fetch(`/api/v3/predictions/latest?symbol=${currentForecastSymbol}`);
        if (!response.ok) throw new Error('Failed to load forecast');
        
        const data = await response.json();
        
        // V3 format: {predictions: [{direction, confidence, horizon_h}]}
        const predictions = data.predictions || [];
        const pred = predictions[0] || {};
        
        console.log(`[FORECAST] Loaded for ${currentForecastSymbol}:`, pred);
        console.log(`[FORECAST] Direction: ${pred.direction}, Confidence: ${pred.confidence}, Move: ${pred.expected_move}`);
        
        // Generate differentiated forecasts for each timeframe
        // 24h: Full confidence
        updateForecastCard(0, pred, '☀️', '24h', 1.0);
        
        // 2-5d: Moderate confidence decay (70% of original)
        updateForecastCard(1, pred, '⛅', '2-5d', 0.7);
        
        // 7-14d: Lower confidence decay (50% of original)  
        updateForecastCard(2, pred, '🌤️', '7-14d', 0.5);
    } catch (error) {
        console.error('[GHOST V3] Error loading forecast:', error);
        // Graceful degradation: show "no data" state
        for (let i = 0; i < 3; i++) {
            updateForecastCard(i, {direction: 'FLAT', confidence: 0, expected_move: 0}, ['☀️', '⛅', '🌤️'][i], ['24h', '2-5d', '7-14d'][i], 1.0);
        }
    }
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
        const response = await fetch('/api/v3/news/feed?limit=10');
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
        console.error('[GHOST V3] Error loading news:', error);
        document.getElementById('news-list').innerHTML = '<p style="color: var(--text-secondary);">News feed temporarily unavailable</p>';
    }
}

// Panel 4: Accuracy Chart
async function loadAccuracyChart() {
    try {
        const response = await fetch('/api/v3/accuracy/summary');
        if (!response.ok) throw new Error('Failed to load accuracy data');
        
        const data = await response.json();
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
        // Show error message
        ctx.fillStyle = 'var(--text-secondary)';
        ctx.font = '14px var(--font-mono)';
        ctx.textAlign = 'center';
        ctx.fillText('No accuracy data available', rect.width / 2, rect.height / 2);
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
            change: item.change_pct || 0,
            price: item.price || 0,
            ghost_score: item.ghost_confidence || 0,  // Use API field directly
            direction: item.ghost_direction || 'FLAT', // Use API field directly
            type: item.type
        }));
        
        // Apply filter (stocks/crypto/all)
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
        const changeDisplay = item.change && item.change !== 0 ? 
            `${item.change >= 0 ? '+' : ''}${item.change.toFixed(2)}%` : 
            '--';
        
        const scoreDisplay = item.ghost_score && item.ghost_score > 0 ? 
            `${item.ghost_score.toFixed(0)}%` : 
            '--';
        
        // Direction emoji
        const directionEmoji = item.direction === 'UP' ? '↑' : 
                              item.direction === 'DOWN' ? '↓' : 
                              item.direction === 'FLAT' ? '→' : '';
        
        return `
            <div class="watchlist-row">
                <div class="watchlist-left">
                    <div class="watchlist-icon">${getSymbolIcon(item.symbol)}</div>
                    <div class="watchlist-ticker">${item.symbol}</div>
                </div>
                <div class="watchlist-right">
                    <div class="watchlist-move ${item.change >= 0 ? 'positive' : 'negative'}">
                        ${changeDisplay}
                    </div>
                    <div class="watchlist-score">${directionEmoji} Ghost: ${scoreDisplay}</div>
                </div>
            </div>
        `;
    }).join('');
}

// Panel 6: Health Score
async function loadHealthScore() {
    try {
        const response = await fetch('/api/v3/goals/snapshot');
        if (!response.ok) throw new Error('Failed to load health score');
        
        const data = await response.json();
        
        // Use ghost_health_score (real value from API) instead of ghost_score
        const score = data.ghost_health_score || data.ghost_health || 0;
        const grade = calculateGrade(score);
        
        document.getElementById('health-score-value').textContent = score > 0 ? score.toFixed(0) : '--';
        document.getElementById('health-grade').textContent = grade;
        
        // Update goal progress as health metrics
        renderHealthMetrics({
            daily: data.daily_goal_pct || 0,
            weekly: data.weekly_goal_pct || 0,
            monthly: data.monthly_goal_pct || 0
        });
    } catch (error) {
        console.error('[GHOST V3] Error loading health score:', error);
        document.getElementById('health-score-value').textContent = '--';
        document.getElementById('health-grade').textContent = 'N/A';
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
        // V3 format with goal progress
        metricsList.push(
            { name: 'Daily Goal', value: metrics.daily },
            { name: 'Weekly Goal', value: metrics.weekly },
            { name: 'Monthly Goal', value: metrics.monthly },
            { name: 'Data Health', value: 85 },  // Placeholder
            { name: 'AI Activity', value: 75 },  // Placeholder
            { name: 'Accuracy', value: 70 }  // Placeholder
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
        const response = await fetch('/api/v3/cockpit/status');
        if (!response.ok) throw new Error('Failed to load cockpit snapshot');
        
        const data = await response.json();
        
        // Update system status
        updateStatusIndicator(data.live || false);
        
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
        // Fetch current goals
        const response = await fetch('/api/v3/goals/snapshot');
        const data = await response.json();
        
        // Populate input fields with current goals
        if (data.goals) {
            document.getElementById('goal-daily').value = data.goals.daily?.target || 500;
            document.getElementById('goal-weekly').value = data.goals.weekly?.target || 2500;
            document.getElementById('goal-monthly').value = data.goals.monthly?.target || 10000;
            document.getElementById('goal-yearly').value = data.goals.yearly?.target || 120000;
        }
        
        // Show modal
        document.getElementById('goals-modal').classList.add('active');
    } catch (error) {
        console.error('Error loading goals:', error);
        // Show modal anyway with defaults
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

// Cleanup on unload
window.addEventListener('beforeunload', () => {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});
