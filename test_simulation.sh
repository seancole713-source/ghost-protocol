#!/bin/bash
# GHOST Simulation Mode - Quick Start
# Run this to test simulation data immediately

set -e

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  🎭 GHOST SIMULATION MODE - QUICK START"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if simulation data exists
if [ ! -f "public/simulation_data.json" ]; then
    echo "⚠️  Simulation data not found. Generating now..."
    python3 generate_simulation_data.py > /dev/null 2>&1
    echo "✅ Simulation data generated"
else
    echo "✅ Simulation data exists ($(stat -f%z public/simulation_data.json 2>/dev/null || stat -c%s public/simulation_data.json 2>/dev/null) bytes)"
fi

echo ""
echo "─────────────────────────────────────────────────────────────────"
echo "  📊 SIMULATION DATA AVAILABLE"
echo "─────────────────────────────────────────────────────────────────"
echo ""

# Show what's available
jq -r 'keys | .[] as $k | "  ✓ \($k)"' public/simulation_data.json

echo ""
echo "─────────────────────────────────────────────────────────────────"
echo "  🚀 QUICK START OPTIONS"
echo "─────────────────────────────────────────────────────────────────"
echo ""

# Check if server is running
if pgrep -f "uvicorn.*wolf_app" > /dev/null; then
    SERVER_PID=$(pgrep -f "uvicorn.*wolf_app" | head -1)
    echo "✅ GHOST server is running (PID: $SERVER_PID)"
    echo ""
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│  OPTION 1: Test Simulation Data Now (Browser Console)      │"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  1. Open: http://localhost:5000/cockpit.html"
    echo "  2. Press F12 to open Developer Tools"
    echo "  3. Go to Console tab"
    echo "  4. Paste and run:"
    echo ""
    echo "     fetch('/simulation_data.json')"
    echo "       .then(r => r.json())"
    echo "       .then(data => {"
    echo "         window.GHOST_SIM = data;"
    echo "         console.log('Portfolio:', data.portfolio);"
    echo "         console.log('Watchlist:', data.watchlist);"
    echo "         console.log('Forecast:', data.forecast);"
    echo "       });"
    echo ""
    echo "─────────────────────────────────────────────────────────────────"
    echo ""
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│  OPTION 2: View Simulation Data (Terminal)                 │"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  Run any of these commands:"
    echo ""
    echo "  # View portfolio"
    echo "  jq '.portfolio' public/simulation_data.json"
    echo ""
    echo "  # View watchlist"
    echo "  jq '.watchlist.tickers[]' public/simulation_data.json"
    echo ""
    echo "  # View forecast"
    echo "  jq '.forecast.points[0:3]' public/simulation_data.json"
    echo ""
    echo "  # View trade card"
    echo "  jq '.trade_card' public/simulation_data.json"
    echo ""
    echo "─────────────────────────────────────────────────────────────────"
    echo ""
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│  OPTION 3: Frontend Integration (RECOMMENDED)              │"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  Edit static/ghost.js and add simulation mode:"
    echo ""
    echo "  // At top of initCockpit() function"
    echo "  const urlParams = new URLSearchParams(window.location.search);"
    echo "  if (urlParams.get('sim') === '1') {"
    echo "    return loadSimulationData();"
    echo "  }"
    echo ""
    echo "  Then access: http://localhost:5000/cockpit.html?sim=1"
    echo ""
    echo "  📄 See: SIMULATION_MODE_COMPLETE.md for full instructions"
    echo ""
    
else
    echo "⚠️  GHOST server is NOT running"
    echo ""
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│  START SERVER FIRST                                         │"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  Option A: Normal mode"
    echo "    bash -c 'source .venv/bin/activate && uvicorn wolf_app:app --host 0.0.0.0 --port 5000'"
    echo ""
    echo "  Option B: Simulation mode (backend integration)"
    echo "    export SIM_MODE=1"
    echo "    bash start_simulation_mode.sh"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════"
echo ""

# Offer to view sample data
echo "Would you like to see a sample of the simulation data? (y/n)"
read -t 5 -n 1 SHOW_SAMPLE 2>/dev/null || SHOW_SAMPLE="n"
echo ""

if [ "$SHOW_SAMPLE" = "y" ] || [ "$SHOW_SAMPLE" = "Y" ]; then
    echo ""
    echo "─────────────────────────────────────────────────────────────────"
    echo "  📊 SAMPLE: Portfolio Mock Data"
    echo "─────────────────────────────────────────────────────────────────"
    jq '.portfolio' public/simulation_data.json
    echo ""
    
    echo "─────────────────────────────────────────────────────────────────"
    echo "  📊 SAMPLE: Watchlist Mock Data (first 2 tickers)"
    echo "─────────────────────────────────────────────────────────────────"
    jq '.watchlist.tickers[0:2]' public/simulation_data.json
    echo ""
fi

echo "✅ Simulation mode ready for testing"
echo ""
echo "📚 Full documentation: SIMULATION_MODE_COMPLETE.md"
echo "🚀 Quick reference: UI_PANEL_QUICK_REFERENCE.md"
echo ""
