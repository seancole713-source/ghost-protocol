// Ghost Protocol v3 - Minimal UI JavaScript

// State
let currentTab = 'stocks';
let currentForecastSymbol = 'WOLF';
let updateInterval = null;

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
    setInterval(() => loadGoals(), 30000);  // Goals: every 30s
    setInterval(() => loadStats(), 30000);  // Stats: every 30s
    setInterval(() => loadForecast(), 15000);  // Forecast: every 15s
    setInterval(() => loadTopMovers(), 10000);  // Top Movers: every 10s (includes hunter feed)
    setInterval(() => loadWatchlist(), 15000);  // Watchlist: every 15s
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
    
    // Tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const tabType = e.target.dataset.tab;
            switchTab(e.target.closest('.tabs'), tabType);
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
    tabsContainer.querySelector(`[data-tab="${tabType}"]`).classList.add('active');
    currentTab = tabType;
    
    // Reload relevant panel
    if (tabsContainer.closest('#panel-movers')) {
        loadTopMovers();
    } else if (tabsContainer.closest('#panel-watchlist')) {
        loadWatchlist();
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
            loadWatchlist(),
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
    try {
        const response = await fetch('/api/v3/vip/snapshot');
        if (!response.ok) throw new Error('Failed to load VIP coins');
        
        const data = await response.json();
        const container = document.getElementById('vip-list');
        
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
        console.error('[GHOST V3] Error loading VIP coins:', error);
        document.getElementById('vip-list').innerHTML = '<p style="color: var(--accent-red);">VIP data unavailable</p>';
    }
}

// Panel 2: Forecast
async function loadForecast() {
    try {
        const response = await fetch(`/api/v3/predictions/latest?symbol=${currentForecastSymbol}`);
        if (!response.ok) throw new Error('Failed to load forecast');
        
        const data = await response.json();
        
        // V3 format: {predictions: [{direction, confidence, horizon_h}]}
        const predictions = data.predictions || [];
        const pred = predictions[0] || {};
        
        // Map single prediction to all timeframes (simplified)
        updateForecastCard(0, pred, '☀️', '24h');
        updateForecastCard(1, pred, '⛅', '2-5d');
        updateForecastCard(2, pred, '🌤️', '7-14d');
    } catch (error) {
        console.error('[GHOST V3] Error loading forecast:', error);
        // Graceful degradation: show "no data" state
        for (let i = 0; i < 3; i++) {
            updateForecastCard(i, {direction: 'FLAT', confidence: 0, expected_move: 0}, ['☀️', '⛅', '🌤️'][i], ['24h', '2-5d', '7-14d'][i]);
        }
    }
}

function updateForecastCard(index, prediction, icon, timeframe) {
    const cards = document.querySelectorAll('.forecast-card');
    if (!cards[index]) return;
    
    const card = cards[index];
    const direction = prediction.direction || 'FLAT';
    let confidence = prediction.confidence || 0;
    
    // Convert confidence from 0-1 scale to percentage (0-100)
    if (confidence > 0 && confidence <= 1) {
        confidence = confidence * 100;
    }
    
    // Use backend expected_move if available, otherwise calculate from confidence
    // Backend provides: confidence (0-1) * base_volatility * direction
    const expectedMove = prediction.expected_move !== undefined 
        ? prediction.expected_move 
        : (confidence > 0 ? (confidence * 0.15) : 0);
    
    card.querySelector('.forecast-icon').textContent = icon;
    
    // Graceful degradation: show "--" if no data
    const directionText = direction === 'UP' ? '↑ BUY' : 
                         direction === 'DOWN' ? '↓ SELL' : 
                         direction === 'FLAT' ? '→ FLAT' : '--';
    
    card.querySelector('.forecast-direction').textContent = directionText;
    card.querySelector('.prob-value').textContent = confidence > 0 ? confidence.toFixed(0) : '--';
    card.querySelector('.move-value').textContent = expectedMove !== 0 ? Math.abs(expectedMove).toFixed(2) + '%' : '--';
}

// Panel 3: News Feed
async function loadNews() {
    try {
        const response = await fetch('/api/v3/news/feed?limit=10');
        if (!response.ok) throw new Error('Failed to load news');
        
        const data = await response.json();
        const container = document.getElementById('news-list');
        
        if (!data || !data.items || data.items.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); text-align: center;">No news available yet</p>';
            return;
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
        const response = await fetch('/api/v3/predictions/history?limit=100');
        if (!response.ok) throw new Error('Failed to load accuracy data');
        
        const data = await response.json();
        renderAccuracyChart(data);
    } catch (error) {
        console.error('[GHOST V3] Error loading accuracy chart:', error);
        renderAccuracyChart({predictions: []});
    }
}

function renderAccuracyChart(data) {
    const canvas = document.getElementById('accuracy-chart');
    const ctx = canvas.getContext('2d');
    
    // Simple line chart (can be replaced with Chart.js later)
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'var(--accent-green)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    // Draw placeholder line
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
}

// Panel 5: Watchlist
async function loadWatchlist() {
    try {
        // Use enriched watchlist endpoint that includes live prices
        const response = await fetch('/api/v3/watchlist/enriched');
        if (!response.ok) throw new Error('Failed to load watchlist');
        
        const data = await response.json();
        const watchlistItems = data.items || [];
        
        // Fetch predictions for all symbols
        const predResponse = await fetch('/api/v3/predictions/latest?limit=100');
        const predData = await predResponse.json();
        
        // Create lookup map for predictions
        const predMap = {};
        if (predData && predData.predictions) {
            predData.predictions.forEach(pred => {
                predMap[pred.symbol] = {
                    confidence: pred.confidence * 100,  // Convert 0.45 to 45
                    direction: pred.direction
                };
            });
        }
        
        // Enrich watchlist with prediction data
        const watchlistData = watchlistItems.map(item => ({
            symbol: item.symbol,
            change: item.change_pct || 0,  // Real price change from enriched endpoint
            price: item.price || 0,         // Real price
            ghost_score: predMap[item.symbol]?.confidence || 0,
            direction: predMap[item.symbol]?.direction || 'FLAT',
            type: item.type
        }));
        
        renderWatchlist(watchlistData);
    } catch (error) {
        console.error('[GHOST V3] Error loading watchlist:', error);
        renderWatchlist([]);
    }
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
        
        const score = data.ghost_score || 0;
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
        case 'watchlist': loadWatchlist(); break;
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
        
        // Save each goal
        const periods = [
            { period: 'daily', amount: daily },
            { period: 'weekly', amount: weekly },
            { period: 'monthly', amount: monthly },
            { period: 'yearly', amount: yearly }
        ];
        
        for (const goal of periods) {
            if (goal.amount > 0) {
                await fetch(`/api/goals/set?period=${goal.period}&target_amount=${goal.amount}`, {
                    method: 'POST'
                });
            }
        }
        
        // Close modal
        closeGoalsModal();
        
        // Refresh goals panel
        await loadGoals();
        
        // Show success message (simple alert for now)
        console.log('✅ Goals saved successfully!');
    } catch (error) {
        console.error('Error saving goals:', error);
        alert('Failed to save goals. Please try again.');
    }
}

// Cleanup on unload
window.addEventListener('beforeunload', () => {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});
