#!/bin/bash
# GHOST Simulation Mode Launcher
# Starts server with mock data injection

set -e

echo "=================================="
echo "🚀 GHOST SIMULATION MODE LAUNCHER"
echo "=================================="

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Set environment variables
export GHOST_SIM_MODE=1
export SIM_MODE=1
export PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR:-/tmp/ghost_prom}
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

echo "✓ Environment: SIM_MODE=1"

# Import and patch before starting server
echo ""
echo "Starting server with simulation mode..."
echo "Press Ctrl+C to stop"
echo ""

# Launch server with simulation pre-loader
python3 -c "
import os
os.environ['GHOST_SIM_MODE'] = '1'
os.environ['SIM_MODE'] = '1'

# Import simulation functions first
from simulation_mode import (
    get_mock_portfolio, get_mock_watchlist, get_mock_forecast_48h,
    get_mock_trade_card, get_mock_market_mood, get_mock_news,
    get_mock_ai_preview, get_mock_risk_status, get_mock_top_movers,
    log_simulation
)

print('[SIMULATION] Mock data providers loaded ✓')

# Now import and patch wolf_app
import wolf_app
from unittest.mock import AsyncMock

# Patch endpoints
wolf_app.get_mock_portfolio = get_mock_portfolio
wolf_app.get_mock_watchlist = get_mock_watchlist
wolf_app.get_mock_forecast_48h = get_mock_forecast_48h
wolf_app.get_mock_trade_card = get_mock_trade_card
wolf_app.get_mock_market_mood = get_mock_market_mood
wolf_app.get_mock_news = get_mock_news
wolf_app.get_mock_ai_preview = get_mock_ai_preview
wolf_app.get_mock_risk_status = get_mock_risk_status
wolf_app.get_mock_top_movers = get_mock_top_movers

# Inject simulation flag check helper
wolf_app.is_sim_mode = lambda: os.getenv('SIM_MODE') == '1'

print('[SIMULATION] Wolf app patched ✓')
print('')
print('='*60)
print('✅ SIMULATION MODE ACTIVE')
print('='*60)
print('All UI panels will display mock data')
print('='*60)
print('')

# Start uvicorn
import uvicorn
uvicorn.run(wolf_app.app, host='0.0.0.0', port=5000, reload=False)
"
