# 🚀 GHOST - MONDAY MORNING LAUNCH PREP

**Status:**✅ SIMULATION MODE ACTIVE\**Date:**Sunday, October 6, 2025\**Ready For:**Monday Market Open (9:30 AM ET)

______________________________________________________________________

## 📊 CURRENT SIMULATION STATE

### Portfolio

-**Mode:**SIM (simulation mode - safe testing)
-**Position:**2,000 WOLF shares @ $1.20 avg cost
-**NAV:**$48,740
-**Cash:**$10,000 available
-**P&L:**Ready for real-time calculation

### Watchlist (5/25 tickers)

- WOLF (primary focus)
- AAPL (test ticker)
- MSFT (test ticker)
- TSLA (test ticker)
- NVDA (test ticker)

### Macro Tracking

-**Regime:**SIDEWAYS
-**Risk Level:**MEDIUM
-**SPY/QQQ/VIX:**Actively monitored

### Data Sources (All Active)

- ✅ Smart Watcher (Level 10)
- ✅ World Feed Fusion (8 RSS sources)
- ✅ SEC EDGAR (free filings)
- ✅ Polygon.io ($29/mo real-time)
- ✅ Online Calibration
- ✅ Ghost-AI v1 APEX Enhanced
- ✅ Algo Footprint Detection

______________________________________________________________________

## 🧪 UI TESTING (Tonight)

### Available UI Panels

1.**Cockpit**- [Cockpit panel](<<<<<http://localhost:5000/cockpit.htm>>>>>l)

- Real-time status
- Position summary
- AI signals

1.**Portfolio/Bank**- [Portfolio/Bank panel](<<<<<http://localhost:5000/bank.htm>>>>>l)

- Position management
- Cash tracking
- P&L history

1.**Markets**- [Markets panel](<<<<<http://localhost:5000/markets.htm>>>>>l)

- Watchlist view
- Top movers
- Market status

1.**Engine**- [Engine panel](<<<<<http://localhost:5000/engine.htm>>>>>l)

- Ghost-AI v1 preview
- Feature importance
- Signal confidence

1.**Main Dashboard**- [Main dashboard](<<<<<http://localhost:5000>>>>>/)

- Overview of all systems
- Quick actions

### Quick Test Commands

```bash

# Test portfolio endpoint

curl <<<<<http://localhost:5000/api/portfolio>>>>> | jq .

# Test watchlist

curl <<<<<http://localhost:5000/api/watcher/watchlist>>>>> | jq '.tickers | length'

# Test Ghost-AI v1

curl <<<<<http://localhost:5000/ai/preview>>>>> | jq '{gps, confidence}'

# Test macro state

curl -X POST <<<<<http://localhost:5000/api/watcher/update_macro>>>>> | jq '.macro'

# Test latest news

curl <<<<<http://localhost:5000/api/feeds/latest?limit=5>>>>> | jq '.count'

```text

______________________________________________________________________

## 🎯 MONDAY MORNING CHECKLIST

### Step 1: PRE-MARKET (8:00-9:30 AM ET)**A. Switch to LIVE Mode**```bash

curl -X POST <<<<<http://localhost:5000/api/mode>>>>> \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'

```text**B. Reset Position to Real Holdings**```bash

# Clear test position

curl -X POST <<<<<http://localhost:5000/api/state/reset>>>>>

# Add your actual WOLF position

WOLF_QTY="$(railway variables get WOLF_QTY)"
WOLF_AVG_COST="$(railway variables get WOLF_AVG_COST)"
curl -X POST <<<<<http://localhost:5000/api/bank/add_position>>>>> \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "WOLF",
    "quantity": '"$WOLF_QTY"',
    "price": '"$WOLF_AVG_COST"',
    "type": "stock"
  }'

```text

#### C. Update Watchlist

```bash

# Add real tickers you want to track

curl -X POST "<<<<<http://localhost:5000/api/watcher/add_ticker?symbol=TICKER">>>>>

```text

#### D. Fetch Pre-Market Data

```bash

# Update macro conditions

curl -X POST <<<<<http://localhost:5000/api/watcher/update_macro>>>>>

# Fetch latest news

curl -X POST <<<<<http://localhost:5000/api/feeds/fetch>>>>>

# Get SEC filings from weekend

curl "<<<<<http://localhost:5000/api/edgar/recent_filings?hours_back=72&limit=20">>>>>

```text

### Step 2: MARKET OPEN (9:30 AM ET)

#### A. Verify Real-Time Data

```bash

# Check WOLF price is live

curl "<<<<<http://localhost:5000/api/price/WOLF?force=1">>>>> | jq .

# Check market status

curl <<<<<http://localhost:5000/api/polygon/market_status>>>>> | jq '.market_status.market'

```text

#### B. Monitor Panels

- Open [Cockpit](<<<<<http://localhost:5000/cockpit.htm>>>>>l)
- Check [Portfolio panel](<<<<<http://localhost:5000/bank.htm>>>>>l) updates
- Verify Ghost-AI v1 signals in [Engine](<<<<<http://localhost:5000/engine.htm>>>>>l)


#### C. Enable Alert System (if ready)

```bash

# Check alert configuration

curl <<<<<http://localhost:5000/api/alerts/config>>>>> | jq .

# Test alert dispatch (dry run)

curl -X POST "<<<<<http://localhost:5000/api/alerts/dispatch?dry_run=1">>>>>

```text

### Step 3: THROUGHOUT DAY

#### Monitor

- [ ] Smart Watcher signals (proactive alerts)
- [ ] Ghost-AI v1 GPS scores
- [ ] Portfolio P&L updates
- [ ] SEC 8-K filings (breaking news)
- [ ] Macro risk changes (VIX spikes)
- [ ] Algo pattern detection (HFT activity)


#### Key URLs

- [Health](<<<<<http://localhost:5000/healt>>>>>h)
- [Diagnostics](<<<<<http://localhost:5000/diagnostics/summar>>>>>y)
- [Portfolio API](<<<<<http://localhost:5000/api/portfoli>>>>>o)


______________________________________________________________________

## ⚠️ IMPORTANT REMINDERS

### API Rate Limits

-**Polygon.io:**12 req/sec (Starter plan)
-**SEC EDGAR:**10 req/sec (free, must comply)
-**RSS Feeds:**Fetch every 15-30 minutes


### Data Freshness

-**Price TTL (market open):**15 seconds
-**Price TTL (market closed):**300 seconds
-**News TTL:** 900 seconds (15 min)


### Safety Checks

1. ✅ Simulation mode active NOW (safe testing)
2. ⚠️ Switch to LIVE mode Monday morning ONLY
3. ✅ All databases backed up
4. ✅ Zero critical errors in system
5. ✅ Server running stable (check `ps aux | grep uvicorn`)


______________________________________________________________________

## 🔧 TROUBLESHOOTING

### If UI Panel Not Loading

```bash

# Check server status

curl <<<<<http://localhost:5000/health>>>>>

# Restart if needed (from workspace root)

# Use task runner or uvicorn command

```text

### If No Price Data

```bash

# Check price providers

curl <<<<<http://localhost:5000/debug/price>>>>> | jq .

# Force refresh

curl "<<<<<http://localhost:5000/api/price/WOLF?force=1">>>>>

```text

### If Watchlist Empty

```bash

# Re-add tickers

curl -X POST "<<<<<http://localhost:5000/api/watcher/add_ticker?symbol=WOLF">>>>>
curl -X POST "<<<<<http://localhost:5000/api/watcher/add_ticker?symbol=AAPL">>>>>

```text

______________________________________________________________________

## 📞 QUICK COMMANDS CHEATSHEET

```bash

# Status check

curl <<<<<http://localhost:5000/health>>>>> | jq .

# Switch modes

curl -X POST <<<<<http://localhost:5000/api/mode>>>>> \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'  # LIVE

curl -X POST <<<<<http://localhost:5000/api/mode>>>>> \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}' # SIM

# Portfolio snapshot

curl <<<<<http://localhost:5000/api/portfolio>>>>> | jq '{nav, positions, cash}'

# Watchlist snapshot

curl <<<<<http://localhost:5000/api/watcher/watchlist>>>>> | jq '{count, tickers: [.tickers[].symbol]}'

# Ghost-AI v1 signal

curl <<<<<http://localhost:5000/ai/preview>>>>> | jq '{gps, confidence, reasons}'

# Latest news

curl "<<<<<http://localhost:5000/api/feeds/latest?limit=10">>>>> | jq '[.articles[] | {title, sentiment}]'

# Macro state

curl -X POST <<<<<http://localhost:5000/api/watcher/update_macro>>>>> | jq '.macro'

```text

______________________________________________________________________

## 🎉 CURRENT STATUS SUMMARY

```text

╔════════════════════════════════════════════════════════════╗
║              ✅ GHOST SIMULATION MODE ACTIVE               ║
╠════════════════════════════════════════════════════════════╣
║  Mode:           SIM (safe for testing)                    ║
║  Portfolio:      2000 WOLF @ $1.20                         ║
║  NAV:            $48,740                                   ║
║  Watchlist:      5/25 tickers                              ║
║  Data Sources:   7 active (Level 10 + APEX)               ║
║  Server:         localhost:5000 ✅ RUNNING                ║
║                                                            ║
║  🧪 Test all UI panels TODAY                               ║
║  🚀 Switch to LIVE mode MONDAY 8:00 AM                     ║
║  📊 Ready for launch day!                                  ║
╚════════════════════════════════════════════════════════════╝

```text

______________________________________________________________________

## 🎯 TODAY'S TESTING CHECKLIST

- [ ] Open [main dashboard](<<<<<http://localhost:500>>>>>0) and verify status
- [ ] Test [Cockpit panel](<<<<<http://localhost:5000/cockpit.htm>>>>>l)
- [ ] Test [Portfolio panel](<<<<<http://localhost:5000/bank.htm>>>>>l)
- [ ] Test [Markets panel](<<<<<http://localhost:5000/markets.htm>>>>>l)
- [ ] Test [Engine panel](<<<<<http://localhost:5000/engine.htm>>>>>l)
- [ ] Verify Ghost-AI v1 GPS scores showing
- [ ] Check Smart Watcher has 5 tickers
- [ ] Confirm portfolio shows 2000 WOLF shares
- [ ] Test news feed loading
- [ ] Verify macro state updates


______________________________________________________________________

## 🚀 MONDAY MORNING STEPS (TL;DR)

### 8:00 AM - Pre-Market

1. Switch to LIVE mode
2. Reset position to real holdings
3. Update watchlist with real tickers
4. Fetch pre-market data


### 9:30 AM - Market Open

1. Verify real-time prices flowing
2. Open cockpit panel
3. Monitor Ghost-AI v1 signals
4. Watch for SEC 8-K filings


### Throughout Day

- Monitor Smart Watcher proactive signals
- Check portfolio P&L updates
- Watch for macro risk changes (VIX)
- Track algo pattern detection


______________________________________________________________________

### Next Steps

1. ✅ Test all UI panels tonight
2. ✅ Verify data flows correctly
3. ✅ Get comfortable with cockpit/portfolio views
4. 🚀 Monday 8 AM: Run pre-market checklist
5. 🎯 Monday 9:30 AM: Switch to LIVE, monitor closely


Good luck on launch day! 🚀
