/**
 * Ghost Prediction Panel
 * 
 * Live 48h stock price predictions with accuracy scoreboard.
 * Displays forecast vs actual overlay and historical performance metrics.
 */

// Global state
let predictionChart = null;
let currentSymbol = 'WOLF';
let refreshInterval = null;

// Shared auth helper for all prediction API calls
function ghostAuthHeaders(extra = {}) {
    const token = (window.GHOST_API_TOKEN || '').trim();
    const headers = { ...extra };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    initPredictionPanel();
});

function initPredictionPanel() {
    // Set up event listeners
    const symbolSelect = document.getElementById('predict-symbol');
    const runButton = document.getElementById('predict-run-btn');
    
    if (symbolSelect) {
        symbolSelect.addEventListener('change', (e) => {
            currentSymbol = e.target.value;
            refreshPredictionData();
        });
    }
    
    if (runButton) {
        runButton.addEventListener('click', () => runNewForecast());
    }
    
    // Initialize chart
    initPredictionChart();
    
    // Load initial data
    refreshPredictionData();
    
    // Auto-refresh every 15 seconds
    if (refreshInterval) clearInterval(refreshInterval);
    refreshInterval = setInterval(() => refreshPredictionData(), 15000);
}

function initPredictionChart() {
    const ctx = document.getElementById('prediction-chart');
    if (!ctx) return;
    
    if (predictionChart) {
        predictionChart.destroy();
    }
    
    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Actual Price',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    fill: false,
                    tension: 0.1,
                },
                {
                    label: 'Predicted Price',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.05)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 2,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.1,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'hour',
                        displayFormats: {
                            hour: 'MMM d, ha'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Time'
                    },
                    grid: {
                        display: true,
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price ($)'
                    },
                    grid: {
                        display: true,
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += '$' + context.parsed.y.toFixed(2);
                            }
                            return label;
                        }
                    }
                },
                annotation: {
                    annotations: {}
                }
            }
        }
    });
}

async function refreshPredictionData() {
    try {
        // Fetch series data (forecast + actual)
        const seriesResp = await fetch(
            `/api/predict/series?symbol=${encodeURIComponent(currentSymbol)}`,
            { headers: ghostAuthHeaders() }
        );
        const seriesData = await seriesResp.json();
        
        // Update chart
        updatePredictionChart(seriesData);
        
        // Fetch and update scoreboard
        await updateScoreboard();
        
        // Update status
        document.getElementById('predict-status').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
        
    } catch (error) {
        console.error('Failed to refresh prediction data:', error);
    document.getElementById('predict-status').textContent = '';
    }
}

function updatePredictionChart(data) {
    if (!predictionChart) return;
    
    const actualData = (data.actual || []).map(p => ({
        x: p.ts * 1000,
        y: p.price
    }));
    
    const forecastData = (data.forecast || []).map(p => ({
        x: p.ts * 1000,
        y: p.price
    }));
    
    predictionChart.data.datasets[0].data = actualData;
    predictionChart.data.datasets[1].data = forecastData;
    
    // Add forecast window shading annotation if we have a prediction
    if (data.last_prediction) {
        const runAt = data.last_prediction.run_at * 1000;
        const endAt = runAt + (data.last_prediction.horizon_h * 3600000);
        
        predictionChart.options.plugins.annotation.annotations = {
            forecastWindow: {
                type: 'box',
                xMin: runAt,
                xMax: endAt,
                backgroundColor: 'rgba(59, 130, 246, 0.05)',
                borderColor: 'rgba(59, 130, 246, 0.2)',
                borderWidth: 1,
                label: {
                    display: true,
                    content: '48h Forecast Window',
                    position: 'start'
                }
            }
        };
    }
    
    predictionChart.update();
}

async function updateScoreboard() {
    try {
        // Fetch history
        const historyResp = await fetch(
            `/api/predict/history?symbol=${encodeURIComponent(currentSymbol)}&limit=10`,
            { headers: ghostAuthHeaders() }
        );
        const history = await historyResp.json();
        
        // Fetch scoreboard stats
        const scoreResp = await fetch(
            `/api/predict/scoreboard?symbol=${encodeURIComponent(currentSymbol)}`,
            { headers: ghostAuthHeaders() }
        );
        const scores = await scoreResp.json();
        
        // Update table
        const tbody = document.getElementById('predict-scoreboard-body');
        if (!tbody) return;
        
        tbody.innerHTML = '';
        
        if (history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-gray-400">No predictions yet. Click "Run New Forecast" to start.</td></tr>';
            return;
        }
        
        history.forEach(pred => {
            const row = document.createElement('tr');
            row.className = 'border-b border-gray-700';
            
            const date = new Date(pred.run_at * 1000).toLocaleString();
            const dirIcon = pred.direction === 'UP' ? '↑' : pred.direction === 'DOWN' ? '↓' : '→';
            const conf = (pred.confidence * 100).toFixed(0);
            
            let statusCell = '';
            if (pred.closed) {
                const hitIcon = pred.hit_direction ? '✓' : '✗';
                const hitClass = pred.hit_direction ? 'text-green-400' : 'text-red-400';
                statusCell = `
                    <td class="${hitClass}">${hitIcon}</td>
                    <td>${pred.mae ? pred.mae.toFixed(4) : '-'}</td>
                    <td>${pred.map ? pred.map.toFixed(2) : '-'}%</td>
                    <td>${pred.rmse ? pred.rmse.toFixed(4) : '-'}</td>
                `;
            } else {
                statusCell = `
                    <td class="text-yellow-400">Pending</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                `;
            }
            
            row.innerHTML = `
                <td class="py-2 px-3">${date}</td>
                <td class="py-2 px-3">${dirIcon} ${pred.direction}</td>
                <td class="py-2 px-3">${conf}%</td>
                ${statusCell}
            `;
            
            tbody.appendChild(row);
        });
        
        // Update summary stats
        updateSummaryStats(scores);
        
    } catch (error) {
        console.error('Failed to update scoreboard:', error);
    }
}

function updateSummaryStats(scores) {
    const summaryEl = document.getElementById('predict-summary');
    if (!summaryEl) return;
    
    const overall = scores.overall || {};
    const w7d = scores.w7d || {};
    const w30d = scores.w30d || {};
    
    if (overall.count === 0) {
        summaryEl.innerHTML = '<p class="text-gray-400">No completed predictions yet.</p>';
        return;
    }
    
    summaryEl.innerHTML = `
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-gray-800 p-3 rounded">
                <div class="text-sm text-gray-400">7d Accuracy</div>
                <div class="text-xl font-bold ${w7d.hit_dir_pct >= 60 ? 'text-green-400' : 'text-yellow-400'}">
                    ${w7d.hit_dir_pct !== undefined ? w7d.hit_dir_pct.toFixed(1) : '0.0'}%
                </div>
                <div class="text-xs text-gray-500">${w7d.count || 0} predictions</div>
            </div>
            <div class="bg-gray-800 p-3 rounded">
                <div class="text-sm text-gray-400">30d Accuracy</div>
                <div class="text-xl font-bold ${w30d.hit_dir_pct >= 60 ? 'text-green-400' : 'text-yellow-400'}">
                    ${w30d.hit_dir_pct !== undefined ? w30d.hit_dir_pct.toFixed(1) : '0.0'}%
                </div>
                <div class="text-xs text-gray-500">${w30d.count || 0} predictions</div>
            </div>
            <div class="bg-gray-800 p-3 rounded">
                <div class="text-sm text-gray-400">Avg Confidence</div>
                <div class="text-xl font-bold text-blue-400">
                    ${overall.avg_conf ? (overall.avg_conf * 100).toFixed(1) : '0.0'}%
                </div>
            </div>
            <div class="bg-gray-800 p-3 rounded">
                <div class="text-sm text-gray-400">Calibration</div>
                <div class="text-xl font-bold ${overall.calibration_gap < 0.1 ? 'text-green-400' : 'text-orange-400'}">
                    ${overall.calibration_gap !== undefined ? (overall.calibration_gap * 100).toFixed(1) : '0.0'}%
                </div>
                <div class="text-xs text-gray-500">gap</div>
            </div>
        </div>
    `;
}

async function runNewForecast() {
    const button = document.getElementById('predict-run-btn');
    if (!button) return;
    
    // Disable button during request
    button.disabled = true;
    button.textContent = 'Running...';
    
    try {
        // Use GET to bypass POST model validation issues
        const resp = await fetch(
            `/api/predict/run?symbol=${encodeURIComponent(currentSymbol)}`,
            { headers: ghostAuthHeaders() }
        );
        
        if (!resp.ok) {
            const error = await resp.json();
            throw new Error(error.detail || 'Failed to run forecast');
        }
        
        const result = await resp.json();
        
        // Show success message
        showNotification('Forecast generated successfully!', 'success');
        
        // Refresh data immediately
        await refreshPredictionData();
        
    } catch (error) {
        console.error('Failed to run forecast:', error);
        showNotification('Failed to run forecast: ' + error.message, 'error');
    } finally {
        button.disabled = false;
        button.textContent = 'Run New Forecast';
    }
}

function showNotification(message, type = 'info') {
    // Simple notification (can be enhanced with a toast library)
    const color = type === 'success' ? 'green' : type === 'error' ? 'red' : 'blue';
    const notif = document.createElement('div');
    notif.className = `fixed top-4 right-4 bg-${color}-600 text-white px-6 py-3 rounded shadow-lg z-50`;
    notif.textContent = message;
    document.body.appendChild(notif);
    
    setTimeout(() => {
        notif.remove();
    }, 3000);
}
