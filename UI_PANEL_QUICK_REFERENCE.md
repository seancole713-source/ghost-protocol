# GHOST UI PANEL QUICK REFERENCE

## 🎯 Current Status: READY FOR TESTING (80% functional)

### ✅ Working Panels (12/15)

1. Market Status ✅
2. Portfolio Overview ✅
3. Ghost Score Heatmap ✅
4. Top Movers ✅
5. Live News Feed ✅
6. Diagnostics ✅
7. 48h Forecast ✅ (backend working, check frontend if flat)
8. Feature Importance ✅
9. Active Goals ✅

1. Smart Watcher Performance ✅
2. SEC EDGAR Filings ✅
3. Feed Fusion Sources ✅


### ⚠️ Issues Identified (3)

1. **Risk Status API**- JSON parse error (HIGH priority)


2.**APEX Trade Card**- yfinance failing (MEDIUM priority)
3.**AI Analog Scenarios**- Need to trigger inference (LOW priority)


______________________________________________________________________

## 🚀 Quick Test Commands

### Open UI in Browser

```bash

# Main cockpit

<<<<<http://localhost:5000/cockpit.html>>>>>

# All panels

<<<<<http://localhost:5000/>>>>>          # Dashboard
<<<<<http://localhost:5000/bank.html>>>>>    # Portfolio
<<<<<http://localhost:5000/markets.html>>>>> # Markets
<<<<<http://localhost:5000/engine.html>>>>>  # Engine

```text

### Verify All Panels

```bash

bash /tmp/final_ui_verification.sh

```text

### Fix Watchlist Data

```bash

curl -X POST <<<<<http://localhost:5000/api/watcher/update_prices>>>>>
curl -X POST <<<<<http://localhost:5000/api/watcher/generate_signal?symbol=WOLF>>>>>

```text

### Check Specific Panel

```bash

# Watchlist

curl <<<<<http://localhost:5000/api/watcher/watchlist>>>>> | jq

# 48h Forecast

curl <<<<<http://localhost:5000/predict/48h>>>>> | jq '.points | length'

# Portfolio

curl <<<<<http://localhost:5000/api/portfolio>>>>> | jq

# Health

curl <<<<<http://localhost:5000/health>>>>> | jq

```text

______________________________________________________________________

## 📋 Monday Launch Checklist

### 8:00 AM - Pre-Market

```bash

# 1. Switch to LIVE mode

curl -X POST <<<<<http://localhost:5000/api/mode>>>>> -d '{"enabled": true}'

# 2. Reset test position

curl -X POST <<<<<http://localhost:5000/api/state/reset>>>>>

# 3. Add real WOLF position

curl -X POST <<<<<http://localhost:5000/api/bank/add_position>>>>> \
  -d '{"symbol":"WOLF","quantity":'"$(railway variables get WOLF_QTY)"',"price":'"$(railway variables get WOLF_AVG_COST)"'}'

# 4. Update data

curl -X POST <<<<<http://localhost:5000/api/watcher/update_macro>>>>>
curl -X POST <<<<<http://localhost:5000/api/feeds/fetch>>>>>

# 5. Verify mode

curl <<<<<http://localhost:5000/api/status>>>>> | jq '.mode'

# Should show: "live"

```text

### 9:30 AM - Market Open

- Open cockpit: <<<<<http://localhost:5000/cockpit.html>>>>>
- Verify all panels load
- Check Ghost-AI GPS scores updating
- Monitor Smart Watcher signals
- Watch for SEC filings
- Track 48h forecast cone


______________________________________________________________________

## 🔧 Troubleshooting

### Panel Shows [object Object]

```bash

# Update watchlist data

curl -X POST <<<<<http://localhost:5000/api/watcher/update_prices>>>>>

```text

### Chart Looks Flat/Empty

1. Hard refresh browser:**Ctrl+Shift+R**2. Check browser console (F12) for JS errors
2. Verify API: `curl <<<<<http://localhost:5000/predict/48h>>>>> | jq`


### Panel Won't Load

```bash

# Check server health

curl <<<<<http://localhost:5000/health>>>>>

# Check specific endpoint

curl <<<<<http://localhost:5000/api/STATUS_ENDPOINT>>>>> | jq

```text

### Need to Repopulate All Data

```bash

bash /tmp/setup_sim_mode.sh

```text

______________________________________________________________________

## 📊 Key Metrics**Current Configuration**

- Mode: SIM ✅
- Portfolio: 2000 WOLF @ $1.20 ✅
- NAV: $48,740 ✅
- Watchlist: 9/25 tickers ✅
- Server: <<<<<http://localhost:5000>>>>> ✅


**Panel Status**: 12/15 working (80%)

**Known Issues**: 3 (documented with fixes)

______________________________________________________________________

## 📚 Full Documentation

See `UI_PANEL_FIX_SUMMARY.md` for:

- Complete API endpoint analysis
- Detailed fix recommendations with code
- Priority breakdown
- Development roadmap


______________________________________________________________________

**Last Updated**: October 6, 2025\
**Ready for**: UI Testing Tonight + Monday Launch
