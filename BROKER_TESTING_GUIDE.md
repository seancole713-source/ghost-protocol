# 🧪 Broker Integration Testing Guide

**Created**: October 13, 2025\
**Deployment**: Railway automatically deploying commit `bf75f2c1`\
**Status**: 🚀 **READY TO TEST**______________________________________________________________________

## 🎯 WHAT WAS ADDED

### New Endpoints (12 total)

All broker endpoints require Bearer token authentication.

#### Broker Status & Info

```bash

# Check broker health

GET /api/broker/health

# Get all open positions

GET /api/broker/positions

# Get account info (cash, buying power, portfolio value)

GET /api/broker/account

# Check if market is open

GET /api/broker/clock

```text

#### Trading Operations

```bash

# Submit order (with risk checks)

POST /api/trade/submit
Body: {
  "symbol": "WOLF",
  "qty": 10,
  "side": "buy",  # or "sell"
  "type": "market",  # or "limit", "stop", etc.
  "time_in_force": "day",  # or "gtc", "ioc", "fok"
  "dry_run": false  # true = check risk only, don't submit
}

# Get orders (open, closed, or all)

GET /api/trade/orders?status=open&limit=50

# Get specific order

GET /api/trade/order/{order_id}

# Cancel order

DELETE /api/trade/order/{order_id}

# Cancel ALL orders

DELETE /api/trade/orders/cancel_all

# Close entire position (sell all shares)

POST /api/trade/position/close/{symbol}

```text

#### Risk Management

```bash

# Get risk status

GET /api/risk/status

# Scan positions for SL/TP triggers

GET /api/risk/scan_exits

```text

______________________________________________________________________

## ⚡ QUICK START TESTING

### 1. Wait for Deployment (1-2 minutes)

Railway is automatically deploying the new code.

Check status:

```bash

curl <<<<<https://web-production-8e9a0.up.railway.app/health>>>>>

```text

### 2. Test Broker Health

```bash

curl <<<<<https://web-production-8e9a0.up.railway.app/api/broker/health>>>>>

```text**Expected Response**:

```json

{
  "ok": true,
  "broker": "alpaca",
  "paper": true,
  "account_id": "...",
  "status": "ACTIVE",
  "buying_power": 100000.00,
  "cash": 100000.00,
  "portfolio_value": 100000.00,
  "positions_count": 0,
  "market_open": true/false
}

```text

### 3. Test Risk Status

```bash

curl <<<<<https://web-production-8e9a0.up.railway.app/api/risk/status>>>>>

```text

**Expected Response**:

```json

{
  "ok": true,
  "risk": {
    "enabled": true,
    "kill_switch": false,
    "limits": {
      "max_position_pct": 10.0,
      "stop_loss_pct": 3.0,
      "take_profit_pct": 6.0,
      "max_daily_drawdown_pct": 15.0
    },
    "current": {
      "portfolio_value": 100000.00,
      "peak_value": 100000.00,
      "drawdown_pct": 0.0
    },
    "status": {
      "level": "green",
      "message": "Risk levels normal"
    }
  }
}

```text

### 4. Test Dry Run Order (No Actual Trade)

```bash

curl -X POST <<<<<https://web-production-8e9a0.up.railway.app/api/trade/submit>>>>> \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "qty": 10,
    "side": "buy",
    "type": "market",
    "dry_run": true
  }'

```text

**Expected Response**:

```json

{
  "ok": true,
  "submitted": false,
  "dry_run": true,
  "risk_check": "PASSED",
  "reason": "✅ All risk checks passed",
  "order": {
    "symbol": "AAPL",
    "qty": 10,
    "side": "buy",
    "type": "market",
    "price": 225.50
  }
}

```text

### 5. Submit Real Paper Trade Order 🎯

**WARNING**: This will place a REAL paper trade order with Alpaca!

```bash

curl -X POST <<<<<https://web-production-8e9a0.up.railway.app/api/trade/submit>>>>> \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "qty": 1,
    "side": "buy",
    "type": "market",
    "time_in_force": "day",
    "dry_run": false
  }'

```text

**Expected Response**:

```json

{
  "ok": true,
  "submitted": true,
  "risk_check": "PASSED",
  "order": {
    "id": "...",
    "client_order_id": "...",
    "created_at": "2025-10-13T...",
    "updated_at": "2025-10-13T...",
    "submitted_at": "2025-10-13T...",
    "filled_at": null,
    "expired_at": null,
    "canceled_at": null,
    "failed_at": null,
    "replaced_at": null,
    "replaced_by": null,
    "replaces": null,
    "asset_id": "...",
    "symbol": "AAPL",
    "asset_class": "us_equity",
    "notional": null,
    "qty": "1",
    "filled_qty": "0",
    "filled_avg_price": null,
    "order_class": "",
    "order_type": "market",
    "type": "market",
    "side": "buy",
    "time_in_force": "day",
    "limit_price": null,
    "stop_price": null,
    "status": "pending_new",
    "extended_hours": false,
    "legs": null,
    "trail_percent": null,
    "trail_price": null,
    "hwm": null,
    "subtag": null,
    "source": null
  }
}

```text

### 6. Check Positions

```bash

curl <<<<<https://web-production-8e9a0.up.railway.app/api/broker/positions>>>>>

```text

**Expected Response**:

```json

{
  "ok": true,
  "count": 1,
  "positions": [
    {
      "asset_id": "...",
      "symbol": "AAPL",
      "exchange": "NASDAQ",
      "asset_class": "us_equity",
      "avg_entry_price": "225.50",
      "qty": "1",
      "side": "long",
      "market_value": "225.50",
      "cost_basis": "225.50",
      "unrealized_pl": "0.00",
      "unrealized_plpc": "0.0000",
      "unrealized_intraday_pl": "0.00",
      "unrealized_intraday_plpc": "0.0000",
      "current_price": "225.50",
      "lastday_price": "225.00",
      "change_today": "0.0022"
    }
  ]
}

```text

### 7. Test Risk Blocking (Oversized Order)

```bash

curl -X POST <<<<<https://web-production-8e9a0.up.railway.app/api/trade/submit>>>>> \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "qty": 10000,
    "side": "buy",
    "type": "market",
    "dry_run": false
  }'

```text

**Expected Response**(Should be BLOCKED):

```json

{
  "ok": false,
  "submitted": false,
  "blocked": true,
  "reason": "❌ Position size limit exceeded: 22.5% > 10.0% max ($2,255,000 of $10,000,000 portfolio)",
  "order": {
    "symbol": "AAPL",
    "qty": 10000,
    "side": "buy",
    "price": 225.50
  }
}

```text

______________________________________________________________________

## 🔬 ADVANCED TESTING

### Test Stop-Loss Trigger

1. Buy a stock via API
2. Wait for position to show up
3. Simulate price drop by calling `/api/risk/scan_exits`
4. If position is down 3%+, it will return exit signal


```bash

curl <<<<<https://web-production-8e9a0.up.railway.app/api/risk/scan_exits>>>>>

```text

### Test Take-Profit Trigger

Same as above, but if position is up 6%+

### Test Kill Switch

```bash

# Enable kill switch via Railway dashboard

# Set: RISK_KILL=1

# Then try to place order

curl -X POST <<<<<https://web-production-8e9a0.up.railway.app/api/trade/submit>>>>> \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "qty": 1,
    "side": "buy",
    "type": "market"
  }'

# Expected: {"ok": false, "blocked": true, "reason": "🛑 KILL SWITCH ACTIVE"}

```text

______________________________________________________________________

## 📊 VERIFICATION CHECKLIST

Before declaring broker integration complete, verify:

- [ ] `/api/broker/health` returns 200 OK with account info
- [ ] `/api/broker/positions` returns empty array (no positions yet)
- [ ] `/api/broker/account` shows $100,000 paper trading cash
- [ ] `/api/risk/status` shows all limits correctly
- [ ] Dry run order (`dry_run: true`) returns risk check without submitting
- [ ] Real paper order (`dry_run: false`) actually submits to Alpaca
- [ ] Order appears in `/api/trade/orders`
- [ ] Position appears in `/api/broker/positions` after fill
- [ ] Risk blocks oversized orders (>10% of portfolio)
- [ ] Can cancel order via `/api/trade/order/{id}` DELETE
- [ ] Can close position via `/api/trade/position/close/{symbol}`
- [ ] `/api/risk/scan_exits` identifies positions with SL/TP triggers


______________________________________________________________________

## 🚨 TROUBLESHOOTING

### "Broker not enabled"**Solution**: Check Railway environment variables

- `BROKER=alpaca`
- `ALPACA_KEY_ID=PKVUMLL1V91W9Y5QCG77`
- `ALPACA_SECRET_KEY=sw09z6TdIeXrs9G6fE5Lo9AayM44UmSWiEYcuXyk`
- `ALPACA_PAPER=1`


### "Module 'core.alpaca_broker' not found"

**Solution**: Wait for Railway deployment to complete (check logs)

### "Alpaca API error: 401 Unauthorized"

**Solution**: Check Alpaca API keys are valid. Login to Alpaca dashboard and regenerate
if needed.

### "Risk check blocks order"

**Solution**: This is WORKING AS INTENDED! Check risk reason:

- Position size too large? Reduce `qty`
- Daily drawdown exceeded? Reset NAV or wait for next trading day
- Kill switch active? Set `RISK_KILL=0` in Railway


### Order stuck in "pending_new"

**Solution**: Wait 1-2 seconds. Market orders fill immediately during market hours. If
outside market hours, order will remain pending until market opens.

______________________________________________________________________

## 🎯 SUCCESS CRITERIA

**GHOST is at 90%+ when**:

- ✅ All 12 broker endpoints respond without errors
- ✅ Can place paper trade order successfully
- ✅ Order shows in Alpaca dashboard (paper.alpaca.markets)
- ✅ Risk engine blocks bad orders
- ✅ Position tracking works
- ✅ Can cancel orders
- ✅ Can close positions


______________________________________________________________________

## 📈 WHAT'S NEXT (90% → 95%)

1. **Automated SL/TP Loop**(1 hour)

   - Background task scanning positions every 60 seconds
   - Auto-submit sell orders when SL/TP triggered
   - Log all auto-exits to database


1.**Telegram Trading Commands**(1 hour)

   - `/buy AAPL 10` - Buy 10 shares
   - `/sell AAPL` - Sell all AAPL
   - `/positions` - Show positions with P&L
   - `/orders` - Show open orders
   - `/cancel {id}` - Cancel order


1.**Trading Dashboard UI**(2 hours)

   - Add "Trading" tab to web UI
   - Show positions with SL/TP levels
   - "Place Order" button
   - Real-time order status updates


1.**Backtesting Engine**(4-6 hours)

   - Load historical OHLCV data
   - Simulate strategy on past data
   - Calculate metrics (Sharpe, max DD, win rate)
   - Store backtest results


______________________________________________________________________

## 🔥 DEPLOY STATUS**Git Commit**: `bf75f2c1`\

**Railway**: Deploying automatically\
**ETA**: 1-2 minutes from push\
**Check**: `curl <<<<<https://web-production-8e9a0.up.railway.app/health`>>>>>

______________________________________________________________________

**Ready to test? Start with Step 1 above!** 🚀

Once you verify the broker is working, we can add automated SL/TP monitoring and
Telegram trading commands!
