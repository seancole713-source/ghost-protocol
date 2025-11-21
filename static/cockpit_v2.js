/* ============================================================================
   GHOST HUNTER COCKPIT V2 - JAVASCRIPT MODULE
   Clean, modular, real-time data updates with graceful degradation
   ============================================================================ */

// === CONFIGURATION ===
const CONFIG = {
    UPDATE_INTERVAL: 5000,        // 5 seconds for most data
    FAST_UPDATE_INTERVAL: 2000,   // 2 seconds for critical data
    SLOW_UPDATE_INTERVAL: 30000,  // 30 seconds for slow-changing data
    MAX_RETRIES: 3,
    RETRY_DELAY: 1000,
};

// === STATE MANAGEMENT ===
const state = {
    environment: 'LOADING',
    ghostHealth: null,
    hunterFeed: [],
    vipCoins: {},
    portfolio: [],
    providers: {},
    lastUpdate: null,
    updateTimers: {},
};

// === UTILITY FUNCTIONS ===
const utils = {
    formatCurrency(value) {
        if (value === null || value === undefined) return '--';
        if (Math.abs(value) >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
        if (Math.abs(value) >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
        if (Math.abs(value) >= 1e3) return `$${(value / 1e3).toFixed(2)}K`;
        if (Math.abs(value) < 0.01) return `$${value.toFixed(6)}`;
        return `$${value.toFixed(2)}`;
    },

    formatPercent(value) {
        if (value === null || value === undefined) return '--%';
        const sign = value >= 0 ? '+' : '';
        return `${sign}${value.toFixed(2)}%`;
    },

    formatNumber(value, decimals = 2) {
        if (value === null || value === undefined) return '--';
        return value.toFixed(decimals);
    },

    formatTimestamp(timestamp) {
        if (!timestamp) return '--:--:--';
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', { hour12: false });
    },

    applyColorClass(element, value) {
        element.classList.remove('positive', 'negative', 'neutral');
        if (value > 0) element.classList.add('positive');
        else if (value < 0) element.classList.add('negative');
        else element.classList.add('neutral');
    },

    async fetchWithRetry(url, options = {}, retries = CONFIG.MAX_RETRIES) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            if (retries > 0) {
                console.warn(`Fetch failed, retrying... (${retries} left)`, error);
                await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY));
                return this.fetchWithRetry(url, options, retries - 1);
            }
            console.error('Fetch failed after retries:', error);
            throw error;
        }
    },

    showError(message) {
        console.error(message);
        // Could add toast notifications here
    },

    updateElement(id, value, formatter = null) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = formatter ? formatter(value) : value;
        }
    },
};

// === HEADER UPDATES ===
const headerModule = {
    init() {
        this.updateEnvironment();
        this.updateLastUpdateTime();
        setInterval(() => this.updateLastUpdateTime(), 1000);
    },

    async updateEnvironment() {
        try {
            const data = await utils.fetchWithRetry('/api/config/runtime');
            const simMode = data.SIM_MODE || false;
            const env = simMode ? 'SIM' : 'LIVE';
            
            state.environment = env;
            const badge = document.getElementById('envBadge');
            badge.textContent = env;
            badge.setAttribute('data-env', env);
        } catch (error) {
            utils.showError('Failed to fetch environment');
        }
    },

    updateLastUpdateTime() {
        const now = new Date();
        utils.updateElement('lastUpdate', `Last Update: ${utils.formatTimestamp(now)}`);
    },

    async updateGhostHealth() {
        try {
            const data = await utils.fetchWithRetry('/api/ghost/health');
            state.ghostHealth = data;

            utils.updateElement('healthScore', data.overall_health_score || '--');
            
            const gradeElement = document.getElementById('healthGrade');
            if (gradeElement && data.grade) {
                gradeElement.textContent = data.grade;
                gradeElement.className = `grade-badge grade-${data.grade}`;
            }

            utils.updateElement('healthStatus', data.status_description || 'Unknown');
            utils.updateElement('ghostScore', data.overall_health_score || '--');
            
            const scoreGradeElement = document.getElementById('ghostGrade');
            if (scoreGradeElement && data.grade) {
                scoreGradeElement.textContent = data.grade;
                scoreGradeElement.className = `score-grade grade-badge grade-${data.grade}`;
            }
        } catch (error) {
            utils.showError('Failed to fetch Ghost health');
        }
    },

    async updateStatusIndicators() {
        const indicators = [
            { id: 'dataStatus', endpoint: '/api/providers/health' },
            { id: 'aiStatus', endpoint: '/api/ghost/brain/status' },
            { id: 'riskStatus', endpoint: '/api/risk/status' },
        ];

        for (const indicator of indicators) {
            try {
                const data = await utils.fetchWithRetry(indicator.endpoint);
                const element = document.getElementById(indicator.id);
                if (!element) continue;

                const dot = element.querySelector('.status-dot');
                const text = element.querySelector('.status-text');

                if (data.healthy || data.status === 'healthy') {
                    dot.className = 'status-dot healthy';
                    text.textContent = 'OK';
                } else if (data.warning || data.status === 'warning') {
                    dot.className = 'status-dot warning';
                    text.textContent = 'WARN';
                } else {
                    dot.className = 'status-dot critical';
                    text.textContent = 'DOWN';
                }
            } catch (error) {
                // Graceful degradation
                const element = document.getElementById(indicator.id);
                if (element) {
                    const dot = element.querySelector('.status-dot');
                    const text = element.querySelector('.status-text');
                    dot.className = 'status-dot critical';
                    text.textContent = 'N/A';
                }
            }
        }
    },
};

// === GOALS & GHOST SCORE ===
const goalsModule = {
    async update() {
        try {
            const data = await utils.fetchWithRetry('/api/portfolio/goals');
            
            const goals = [
                { period: 'Daily', value: data.daily_progress || 0 },
                { period: 'Weekly', value: data.weekly_progress || 0 },
                { period: 'Monthly', value: data.monthly_progress || 0 },
                { period: 'Yearly', value: data.yearly_progress || 0 },
            ];

            goals.forEach(goal => {
                const valueId = `goal${goal.period}`;
                const barId = `goal${goal.period}Bar`;
                
                utils.updateElement(valueId, utils.formatPercent(goal.value));
                
                const bar = document.getElementById(barId);
                if (bar) {
                    const clampedValue = Math.max(0, Math.min(100, goal.value));
                    bar.style.width = `${clampedValue}%`;
                }
            });
        } catch (error) {
            utils.showError('Failed to fetch goals');
        }
    },
};

// === HUNTER FEED ===
const hunterFeedModule = {
    async update() {
        try {
            const data = await utils.fetchWithRetry('/api/hunter/feed');
            state.hunterFeed = data.opportunities || [];
            this.render();
        } catch (error) {
            utils.showError('Failed to fetch hunter feed');
            this.renderError();
        }
    },

    render() {
        const tbody = document.getElementById('hunterFeedBody');
        if (!tbody) return;

        if (state.hunterFeed.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="loading-row">No opportunities detected</td></tr>';
            return;
        }

        tbody.innerHTML = state.hunterFeed.map(opp => `
            <tr data-symbol="${opp.symbol}">
                <td class="font-bold">${opp.symbol}</td>
                <td>${opp.market || 'STOCK'}</td>
                <td class="font-mono">${utils.formatCurrency(opp.price)}</td>
                <td class="${opp.change_pct >= 0 ? 'text-bullish' : 'text-bearish'} font-bold">
                    ${utils.formatPercent(opp.change_pct)}
                </td>
                <td class="font-mono">${utils.formatNumber(opp.volume / 1e6, 2)}M</td>
                <td class="font-bold">${opp.momentum || '--'}</td>
                <td class="font-bold">${opp.gps_score || '--'}</td>
            </tr>
        `).join('');

        // Add click handlers
        tbody.querySelectorAll('tr').forEach(row => {
            row.addEventListener('click', () => {
                const symbol = row.getAttribute('data-symbol');
                if (symbol) this.handleSymbolClick(symbol);
            });
        });
    },

    renderError() {
        const tbody = document.getElementById('hunterFeedBody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="7" class="loading-row text-bearish">Failed to load opportunities</td></tr>';
        }
    },

    handleSymbolClick(symbol) {
        console.log(`Symbol clicked: ${symbol}`);
        // Could navigate to detailed view or open modal
    },
};

// === VIP COINS + XRP + PRESALES ===
const vipModule = {
    async update() {
        await Promise.all([
            this.updateVIPCoins(),
            this.updateXRP(),
            this.updatePresales(),
        ]);
    },

    async updateVIPCoins() {
        const vipSymbols = ['WEPE', 'LILPEPE', 'DORKL', 'SLOTH', 'APC'];
        
        for (const symbol of vipSymbols) {
            try {
                const data = await utils.fetchWithRetry(`/api/price/${symbol}`);
                
                utils.updateElement(`vip${symbol}`, utils.formatCurrency(data.price));
                
                const changeElement = document.getElementById(`vip${symbol}Change`);
                if (changeElement) {
                    changeElement.textContent = utils.formatPercent(data.change_pct || 0);
                    utils.applyColorClass(changeElement, data.change_pct || 0);
                }

                utils.updateElement(`vip${symbol}Status`, data.status || 'Tracking');
            } catch (error) {
                utils.updateElement(`vip${symbol}`, '$--');
                utils.updateElement(`vip${symbol}Change`, '--%');
                utils.updateElement(`vip${symbol}Status`, 'N/A');
            }
        }
    },

    async updateXRP() {
        try {
            const data = await utils.fetchWithRetry('/api/price/XRP');
            
            utils.updateElement('xrpPrice', utils.formatCurrency(data.price));
            
            const changeElement = document.getElementById('xrpChange');
            if (changeElement) {
                changeElement.textContent = utils.formatPercent(data.change_pct || 0);
                utils.applyColorClass(changeElement, data.change_pct || 0);
            }

            utils.updateElement('xrpTrend', data.trend || 'BULLISH');
        } catch (error) {
            utils.updateElement('xrpPrice', '$--');
            utils.updateElement('xrpChange', '--%');
            utils.updateElement('xrpTrend', '--');
        }
    },

    async updatePresales() {
        try {
            const data = await utils.fetchWithRetry('/api/presale/watch');
            const presaleList = document.getElementById('presaleList');
            if (!presaleList) return;

            if (!data.presales || data.presales.length === 0) {
                presaleList.innerHTML = '<div class="presale-item"><span class="presale-name">No presales tracked</span><span class="presale-status">--</span></div>';
                return;
            }

            presaleList.innerHTML = data.presales.map(presale => `
                <div class="presale-item">
                    <span class="presale-name">${presale.name}</span>
                    <span class="presale-status">${presale.status}</span>
                </div>
            `).join('');
        } catch (error) {
            // Silent fail - presales optional
        }
    },
};

// === MACRO & WORLD CONTEXT ===
const macroModule = {
    async update() {
        try {
            const data = await utils.fetchWithRetry('/api/world-context');
            
            const symbols = ['SPY', 'QQQ', 'VIX', 'BTC', 'DXY'];
            symbols.forEach(symbol => {
                const price = data[symbol]?.price;
                const change = data[symbol]?.change_pct;
                
                utils.updateElement(`macro${symbol}`, symbol === 'VIX' || symbol === 'DXY' ? 
                    utils.formatNumber(price, 2) : utils.formatCurrency(price));
                
                const changeElement = document.getElementById(`macro${symbol}Change`);
                if (changeElement && change !== undefined) {
                    changeElement.textContent = utils.formatPercent(change);
                    utils.applyColorClass(changeElement, change);
                }
            });

            utils.updateElement('marketRegime', data.regime || '--');
            utils.updateElement('regimeConfidence', utils.formatPercent(data.regime_confidence || 0));

            await this.updateNews();
        } catch (error) {
            utils.showError('Failed to fetch macro data');
        }
    },

    async updateNews() {
        try {
            const data = await utils.fetchWithRetry('/api/news/headlines?limit=3');
            const container = document.getElementById('newsHeadlines');
            if (!container) return;

            if (!data.headlines || data.headlines.length === 0) {
                container.innerHTML = '<div class="headline-item"><span class="headline-text">No recent news</span></div>';
                return;
            }

            container.innerHTML = data.headlines.map(headline => `
                <div class="headline-item">
                    <span class="headline-text">${headline.title}</span>
                    <span class="headline-sentiment ${headline.sentiment}">${headline.sentiment}</span>
                </div>
            `).join('');
        } catch (error) {
            // Silent fail - news optional
        }
    },
};

// === RISK ENGINE ===
const riskModule = {
    async update() {
        try {
            const data = await utils.fetchWithRetry('/api/risk/metrics');
            
            utils.updateElement('riskNAV', utils.formatCurrency(data.total_nav));
            utils.updateElement('riskOpen', utils.formatPercent(data.open_risk_pct));
            utils.updateElement('riskMaxPos', `${utils.formatCurrency(data.max_position)} / ${utils.formatPercent(data.max_position_pct)}`);
            utils.updateElement('riskVaR', utils.formatCurrency(data.var_95));
            utils.updateElement('riskDrawdown', utils.formatPercent(data.drawdown_pct));

            const statusIndicator = document.getElementById('riskStatusIndicator');
            if (statusIndicator) {
                const dot = statusIndicator.querySelector('.status-dot');
                const text = statusIndicator.querySelector('.status-text');
                
                if (data.risk_level === 'LOW') {
                    dot.className = 'status-dot healthy';
                    text.textContent = 'HEALTHY';
                } else if (data.risk_level === 'MODERATE') {
                    dot.className = 'status-dot warning';
                    text.textContent = 'MODERATE';
                } else {
                    dot.className = 'status-dot critical';
                    text.textContent = 'HIGH';
                }
            }
        } catch (error) {
            utils.showError('Failed to fetch risk metrics');
        }
    },
};

// === PORTFOLIO ===
const portfolioModule = {
    async update() {
        try {
            const data = await utils.fetchWithRetry('/api/portfolio/summary');
            
            utils.updateElement('portfolioValue', utils.formatCurrency(data.market_value));
            
            const pnlElement = document.getElementById('portfolioPnL');
            if (pnlElement) {
                pnlElement.textContent = `${utils.formatCurrency(data.total_pnl)} (${utils.formatPercent(data.total_pnl_pct)})`;
                utils.applyColorClass(pnlElement, data.total_pnl);
            }

            this.renderPositions(data.positions || []);
        } catch (error) {
            utils.showError('Failed to fetch portfolio');
        }
    },

    renderPositions(positions) {
        const tbody = document.getElementById('portfolioBody');
        if (!tbody) return;

        if (positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="loading-row">No open positions</td></tr>';
            return;
        }

        // Show top 5 positions
        const topPositions = positions.slice(0, 5);
        tbody.innerHTML = topPositions.map(pos => `
            <tr>
                <td class="font-bold">${pos.symbol}</td>
                <td>${pos.qty}</td>
                <td>${utils.formatCurrency(pos.avg_cost)}</td>
                <td>${utils.formatCurrency(pos.current_price)}</td>
                <td class="${pos.pnl_pct >= 0 ? 'text-bullish' : 'text-bearish'} font-bold">
                    ${utils.formatPercent(pos.pnl_pct)}
                </td>
            </tr>
        `).join('');
    },
};

// === PREDICTIONS ===
const predictionsModule = {
    async update() {
        try {
            const data = await utils.fetchWithRetry('/api/predictions/latest');
            
            if (!data.prediction) {
                utils.updateElement('predDirection', 'No predictions');
                return;
            }

            const pred = data.prediction;
            const dirElement = document.getElementById('predDirection');
            if (dirElement) {
                dirElement.textContent = pred.direction || '--';
                dirElement.className = `prediction-direction ${pred.direction?.toLowerCase()}`;
            }

            utils.updateElement('predConfidence', utils.formatPercent(pred.confidence || 0));
            utils.updateElement('predHorizon', pred.horizon || '--');
            utils.updateElement('predTimestamp', utils.formatTimestamp(pred.timestamp));

            await this.updateHistory();
        } catch (error) {
            utils.showError('Failed to fetch predictions');
        }
    },

    async updateHistory() {
        try {
            const data = await utils.fetchWithRetry('/api/predictions/history?limit=5');
            const tbody = document.getElementById('predictionHistoryBody');
            if (!tbody) return;

            if (!data.history || data.history.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4">No recent predictions</td></tr>';
                return;
            }

            tbody.innerHTML = data.history.map(pred => `
                <tr>
                    <td>${utils.formatTimestamp(pred.timestamp)}</td>
                    <td class="${pred.direction?.toLowerCase()}">${pred.direction}</td>
                    <td>${utils.formatPercent(pred.confidence)}</td>
                    <td>${pred.outcome || 'Pending'}</td>
                </tr>
            `).join('');
        } catch (error) {
            // Silent fail
        }
    },
};

// === AI BRAIN ===
const aiBrainModule = {
    async update() {
        try {
            const data = await utils.fetchWithRetry('/api/ghost/brain/stats');
            
            utils.updateElement('aiDecisions24h', data.decisions_24h || '--');
            utils.updateElement('aiToolCalls', data.tool_calls || '--');
            utils.updateElement('aiSuccessRate', utils.formatPercent(data.success_rate || 0));

            const statusIndicator = document.getElementById('aiStatusIndicator');
            if (statusIndicator) {
                const dot = statusIndicator.querySelector('.status-dot');
                const text = statusIndicator.querySelector('.status-text');
                
                if (data.status === 'active') {
                    dot.className = 'status-dot healthy';
                    text.textContent = 'ACTIVE';
                } else if (data.status === 'idle') {
                    dot.className = 'status-dot warning';
                    text.textContent = 'IDLE';
                } else {
                    dot.className = 'status-dot critical';
                    text.textContent = 'OFFLINE';
                }
            }

            this.renderActions(data.recent_actions || []);
        } catch (error) {
            utils.showError('Failed to fetch AI stats');
        }
    },

    renderActions(actions) {
        const container = document.getElementById('aiActionsList');
        if (!container) return;

        if (actions.length === 0) {
            container.innerHTML = '<div class="ai-action-item">No recent actions</div>';
            return;
        }

        container.innerHTML = actions.slice(0, 3).map(action => `
            <div class="ai-action-item">
                <span class="action-symbol">${action.symbol || '--'}</span>
                <span class="action-type">${action.type || '--'}</span>
                <span class="action-time">${utils.formatTimestamp(action.timestamp)}</span>
            </div>
        `).join('');
    },
};

// === ACCURACY ===
const accuracyModule = {
    async update() {
        try {
            const data = await utils.fetchWithRetry('/api/predictions/accuracy');
            
            utils.updateElement('accDaily', utils.formatPercent(data.daily_accuracy || 0));
            utils.updateElement('accWeekly', utils.formatPercent(data.weekly_accuracy || 0));
            utils.updateElement('accMonthly', utils.formatPercent(data.monthly_accuracy || 0));

            utils.updateElement('accCorrect', data.correct || '--');
            utils.updateElement('accWarning', data.warning || '--');
            utils.updateElement('accWrong', data.wrong || '--');
            utils.updateElement('accPending', data.pending || '--');

            utils.updateElement('lastTuneTime', utils.formatTimestamp(data.last_tune_timestamp));
            utils.updateElement('tuningConfig', data.tuning_config || '--');
        } catch (error) {
            utils.showError('Failed to fetch accuracy data');
        }
    },
};

// === PROVIDERS ===
const providersModule = {
    async update() {
        try {
            const data = await utils.fetchWithRetry('/api/providers/health');
            state.providers = data.providers || {};

            const providerNames = ['Polygon', 'Yahoo', 'AlphaVantage', 'Binance', 'CoinGecko', 'Reuters'];
            
            providerNames.forEach(name => {
                const provider = state.providers[name.toLowerCase()];
                if (!provider) return;

                const statusElement = document.getElementById(`provider${name}`);
                if (statusElement) {
                    const dot = statusElement.querySelector('.status-dot');
                    const text = statusElement.querySelector('.status-text');
                    
                    if (provider.healthy) {
                        dot.className = 'status-dot healthy';
                        text.textContent = 'OK';
                    } else {
                        dot.className = 'status-dot critical';
                        text.textContent = 'DOWN';
                    }
                }

                utils.updateElement(`provider${name}Latency`, `${provider.latency_ms || '--'} ms`);
            });
        } catch (error) {
            utils.showError('Failed to fetch provider health');
        }
    },
};

// === LOGS ===
const logsModule = {
    async update() {
        try {
            const data = await utils.fetchWithRetry('/api/logs/recent?limit=10');
            const container = document.getElementById('logsContainer');
            if (!container) return;

            if (!data.logs || data.logs.length === 0) {
                container.innerHTML = '<div class="log-item"><span>No recent logs</span></div>';
                return;
            }

            container.innerHTML = data.logs.map(log => `
                <div class="log-item">
                    <span class="log-time">${utils.formatTimestamp(log.timestamp)}</span>
                    <span class="log-level ${log.level}">${log.level}</span>
                    <span class="log-message">${log.message}</span>
                </div>
            `).join('');

            // Auto-scroll to bottom
            container.scrollTop = container.scrollHeight;
        } catch (error) {
            // Silent fail
        }
    },
};

// === CONFIG ===
const configModule = {
    async update() {
        try {
            const data = await utils.fetchWithRetry('/api/config/runtime');
            const container = document.getElementById('configGrid');
            if (!container) return;

            const configItems = Object.entries(data).map(([key, value]) => `
                <div class="config-item">
                    <span class="config-key">${key}</span>
                    <span class="config-value">${typeof value === 'boolean' ? (value ? 'TRUE' : 'FALSE') : value}</span>
                </div>
            `).join('');

            container.innerHTML = configItems;
        } catch (error) {
            utils.showError('Failed to fetch config');
        }
    },
};

// === EVENT HANDLERS ===
const eventHandlers = {
    init() {
        // Refresh buttons
        document.getElementById('btnHunterRefresh')?.addEventListener('click', () => hunterFeedModule.update());
        document.getElementById('btnMacroRefresh')?.addEventListener('click', () => macroModule.update());
        document.getElementById('btnPortfolioRefresh')?.addEventListener('click', () => portfolioModule.update());
        document.getElementById('btnProviderRefresh')?.addEventListener('click', () => providersModule.update());
        document.getElementById('btnLogsRefresh')?.addEventListener('click', () => logsModule.update());
        document.getElementById('btnConfigRefresh')?.addEventListener('click', () => configModule.update());

        // Prediction controls
        document.getElementById('btnRunPrediction')?.addEventListener('click', () => this.runPrediction());

        // Portfolio view
        document.getElementById('btnViewFullPortfolio')?.addEventListener('click', () => this.viewFullPortfolio());
    },

    async runPrediction() {
        const select = document.getElementById('predictionSymbol');
        if (!select) return;

        const symbol = select.value;
        try {
            await utils.fetchWithRetry('/api/predictions/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol }),
            });
            await predictionsModule.update();
        } catch (error) {
            utils.showError('Failed to run prediction');
        }
    },

    viewFullPortfolio() {
        window.location.href = '/portfolio';
    },
};

// === MAIN INITIALIZATION ===
const app = {
    async init() {
        console.log('🚀 Ghost Hunter Cockpit V2 initializing...');

        // Initialize modules
        headerModule.init();
        eventHandlers.init();

        // Initial data load
        await this.loadAllData();

        // Setup update intervals
        this.setupUpdateSchedule();

        console.log('✅ Ghost Hunter Cockpit V2 ready');
    },

    async loadAllData() {
        await Promise.all([
            headerModule.updateGhostHealth(),
            headerModule.updateStatusIndicators(),
            goalsModule.update(),
            hunterFeedModule.update(),
            vipModule.update(),
            macroModule.update(),
            riskModule.update(),
            portfolioModule.update(),
            predictionsModule.update(),
            aiBrainModule.update(),
            accuracyModule.update(),
            providersModule.update(),
            logsModule.update(),
            configModule.update(),
        ]);
    },

    setupUpdateSchedule() {
        // Fast updates (2s) - critical data
        setInterval(() => {
            headerModule.updateStatusIndicators();
            vipModule.updateVIPCoins();
            vipModule.updateXRP();
        }, CONFIG.FAST_UPDATE_INTERVAL);

        // Normal updates (5s) - most data
        setInterval(() => {
            hunterFeedModule.update();
            riskModule.update();
            portfolioModule.update();
            aiBrainModule.update();
        }, CONFIG.UPDATE_INTERVAL);

        // Slow updates (30s) - slow-changing data
        setInterval(() => {
            headerModule.updateGhostHealth();
            goalsModule.update();
            macroModule.update();
            predictionsModule.update();
            accuracyModule.update();
            providersModule.update();
            logsModule.update();
            configModule.update();
        }, CONFIG.SLOW_UPDATE_INTERVAL);
    },
};

// === START APPLICATION ===
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => app.init());
} else {
    app.init();
}
