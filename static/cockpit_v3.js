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
    
    // Set update intervals
    updateInterval = setInterval(() => {
        loadAllPanels();
    }, 5000); // Refresh every 5 seconds
}

// Event Listeners
function setupEventListeners() {
    // Header controls
    document.getElementById('btn-start').addEventListener('click', () => controlAction('start'));
    document.getElementById('btn-stop').addEventListener('click', () => controlAction('stop'));
    document.getElementById('btn-reset').addEventListener('click', () => controlAction('reset'));
    document.getElementById('mode-selector').addEventListener('change', handleModeChange);
    
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
        
        if (!data || data.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); text-align: center;">No data available</p>';
            return;
        }
        
        // Filter by current tab
        let filtered = data;
        if (currentTab === 'stocks') {
            filtered = data.filter(item => item.type === 'stock');
        } else if (currentTab === 'crypto') {
            filtered = data.filter(item => item.type === 'crypto');
        }
        
        container.innerHTML = filtered.slice(0, 10).map(item => `
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
                    <div class="mover-confidence">Ghost: ${item.confidence || 0}%</div>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading movers:', error);
        document.getElementById('movers-list').innerHTML = '<p style="color: var(--accent-red);">Failed to load movers</p>';
    }
}

// Panel 2: Forecast
async function loadForecast() {
    try {
        const response = await fetch(`/api/predict/run?symbol=${currentForecastSymbol}`);
        if (!response.ok) throw new Error('Failed to load forecast');
        
        const data = await response.json();
        
        // Update forecast cards with real data
        updateForecastCard(0, data.short || {}, '☀️', '24h');
        updateForecastCard(1, data.medium || {}, '⛅', '2-5d');
        updateForecastCard(2, data.long || {}, '🌤️', '7-14d');
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
    const confidence = prediction.confidence || 0;
    const expectedMove = prediction.expected_move || 0;
    
    card.querySelector('.forecast-icon').textContent = icon;
    
    // Graceful degradation: show "--" if no data
    const directionText = direction === 'UP' ? '↑ BUY' : 
                         direction === 'DOWN' ? '↓ SELL' : 
                         direction === 'FLAT' ? '→ FLAT' : '--';
    
    card.querySelector('.forecast-direction').textContent = directionText;
    card.querySelector('.prob-value').textContent = confidence > 0 ? confidence.toFixed(0) : '--';
    card.querySelector('.move-value').textContent = expectedMove !== 0 ? expectedMove.toFixed(2) + '%' : '--';
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
        const response = await fetch('/api/v3/watchlist');
        if (!response.ok) throw new Error('Failed to load watchlist');
        
        const data = await response.json();
        
        // Combine all symbol groups
        const allSymbols = [
            ...(data.stocks || []).map(s => ({symbol: s, type: 'stock'})),
            ...(data.crypto || []).map(s => ({symbol: s, type: 'crypto'})),
            ...(data.vip || []).map(s => ({symbol: s, type: 'vip'}))
        ];
        
        // Fetch prices for each (simplified - in production would batch)
        const watchlistData = allSymbols.map(item => ({
            symbol: item.symbol,
            change: 0,  // TODO: Fetch real price changes
            ghost_score: 0,  // TODO: Fetch real Ghost scores
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
            `${item.ghost_score}%` : 
            '--';
        
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
                    <div class="watchlist-score">Ghost: ${scoreDisplay}</div>
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

// Cleanup on unload
window.addEventListener('beforeunload', () => {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});
