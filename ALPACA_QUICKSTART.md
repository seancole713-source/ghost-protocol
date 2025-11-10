# Alpaca Broker Quick Reference

## 🚀 Quick Start

```bash
# 1. Configure environment
export BROKER=alpaca
export ALPACA_KEY_ID=your_key
export ALPACA_SECRET_KEY=your_secret
export ALPACA_PAPER=1  # 1=paper, 0=LIVE

# 2. Test connection
python test_alpaca_broker.py

# 3. Check health via API
curl http://localhost:8444/api/broker/health

# 4. Submit test order
curl -X POST http://localhost:8444/api/trade/submit \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","qty":1,"side":"buy","type":"market","dry_run":true}'
```

## 📋 Essential Commands

| Action | Command | |--------|---------| | Test connection |
`python test_alpaca_broker.py` | | Check health | `curl /api/broker/health` | | Get
account | `curl /api/broker/account` | | List positions | `curl /api/broker/positions` |
| Market status | `curl /api/broker/clock` | | Submit order | `POST /api/trade/submit` |
| List orders | `GET /api/trade/orders` | | Cancel order |
`DELETE /api/trade/order/{id}` | | Close position |
`POST /api/trade/position/close/{symbol}` |

## 🔑 Environment Variables

| Variable | Paper | Live | Required | |----------|-------|------|----------| | BROKER |
alpaca | alpaca | ✅ Yes | | ALPACA_KEY_ID | paper_key | live_key | ✅ Yes | |
ALPACA_SECRET_KEY | paper_secret | live_secret | ✅ Yes | | ALPACA_PAPER | 1 | 0 | ✅ Yes
| | ALPACA_ORDER_RATE | 30 | 30 | ⚙️ Optional | | ALPACA_ORDER_WINDOW_S | 60 | 60 | ⚙️
Optional |

## 🛡️ Safety Checklist

**Before Live Trading:**

- [ ] Tested >50 orders in paper mode
- [ ] Risk limits configured
- [ ] Stop loss / take profit tested
- [ ] Live API keys (not paper keys!)
- [ ] `ALPACA_PAPER=0` verified
- [ ] Maximum loss per trade set
- [ ] Daily loss limits enabled
- [ ] Emergency stop tested

## 📊 Order Types

```python
# Market Order (executes immediately)
{"symbol":"AAPL","qty":10,"side":"buy","type":"market"}

# Limit Order (price or better)
{"symbol":"AAPL","qty":10,"side":"buy","type":"limit","limit_price":150.00}

# Stop Order (triggers at price)
{"symbol":"AAPL","qty":10,"side":"sell","type":"stop","stop_price":140.00}

# Stop-Limit Order
{"symbol":"AAPL","qty":10,"side":"sell","type":"stop_limit","stop_price":140.00,"limit_price":139.50}

# Trailing Stop (trails by %)
{"symbol":"AAPL","qty":10,"side":"sell","type":"trailing_stop","trail_percent":5.0}
```

## 🎯 Position Sizing

```python
from core.trading_automation import (
    build_order_from_prediction,
    PositionSizingMethod
)

# Method 1: Fixed Dollar
order = build_order_from_prediction(
    symbol="AAPL",
    prediction_pct=0.08,
    confidence=0.75,
    current_price=150.00,
    portfolio_value=100000,
    sizing_method=PositionSizingMethod.FIXED_DOLLAR,
    fixed_dollar=1000  # Always $1000 per trade
)

# Method 2: Percent Portfolio
order = build_order_from_prediction(
    symbol="AAPL",
    prediction_pct=0.08,
    confidence=0.75,
    current_price=150.00,
    portfolio_value=100000,
    sizing_method=PositionSizingMethod.PERCENT_PORTFOLIO,
    percent_portfolio=0.02  # 2% of portfolio per trade
)

# Method 3: Kelly Criterion
order = build_order_from_prediction(
    symbol="AAPL",
    prediction_pct=0.08,
    confidence=0.75,
    current_price=150.00,
    portfolio_value=100000,
    sizing_method=PositionSizingMethod.KELLY_CRITERION,
    win_rate=0.55,  # 55% win rate
    avg_win_loss_ratio=1.5  # Wins are 1.5x losses
)
```

## 📈 Risk Management

```python
from core.trading_automation import should_close_position, create_close_order

# Check if should close existing position
should_close, reason = should_close_position(
    symbol="AAPL",
    current_qty=100,
    current_price=150.00,
    entry_price=145.00,
    unrealized_pl_pct=0.034,  # +3.4%
    prediction_pct=-0.04,  # Now predicting -4%
    confidence=0.80,
    stop_loss_pct=-0.10,  # Stop loss at -10%
    take_profit_pct=0.20,  # Take profit at +20%
    reversal_threshold=-0.03  # Close if reverses >-3%
)

if should_close:
    close_order = create_close_order("AAPL", current_qty, reason)
    broker.submit_order(**close_order)
```

## 🔍 Monitoring

```bash
# Watch logs
tail -f logs/ghost.log | grep -E "trade|broker|order"

# Check positions
watch -n 5 'curl -s http://localhost:8444/api/broker/positions | jq'

# Check orders
curl http://localhost:8444/api/trade/orders?status=open | jq

# Check account
curl http://localhost:8444/api/broker/account | jq
```

## 🚨 Troubleshooting

| Error | Solution | |-------|----------| | "Broker not enabled" | Set `BROKER=alpaca` |
| "API keys not configured" | Set `ALPACA_KEY_ID` and `ALPACA_SECRET_KEY` | |
"Authentication failed" | Check keys are correct and for right mode | | "Order blocked
by risk" | Review risk limits in risk_engine.py | | "Rate limit exceeded" | Slow down
requests or increase `ALPACA_ORDER_RATE` | | "Insufficient buying power" | Check account
balance, reduce position size | | "Market closed" | Wait for market open or use
`extended_hours=true` |

## 📚 Documentation

- **Full Setup Guide**: `ALPACA_BROKER_SETUP.md`
- **Implementation Details**: `ALPACA_INTEGRATION_COMPLETE.md`
- **Test Suite**: `test_alpaca_broker.py`
- **Automation Library**: `core/trading_automation.py`
- **Broker Module**: `core/alpaca_broker.py`

## 🔗 Links

- **Alpaca Dashboard**: https://app.alpaca.markets/
- **API Docs**: https://alpaca.markets/docs/
- **Market Hours**: https://www.alpaca.markets/support/market-hours/
- **API Status**: https://status.alpaca.markets/

## ⚡ Example Workflow

```python
# 1. Get prediction
prediction_pct = 0.08  # +8% predicted
confidence = 0.75

# 2. Get account state
from core.alpaca_broker import get_broker
broker = get_broker()
account = broker.get_account()
portfolio_value = float(account["portfolio_value"])

# 3. Get current price
from wolf_app import fetch_price_live
price_data = fetch_price_live("AAPL")
current_price = price_data["price"]

# 4. Build order
from core.trading_automation import build_order_from_prediction, PositionSizingMethod
order = build_order_from_prediction(
    symbol="AAPL",
    prediction_pct=prediction_pct,
    confidence=confidence,
    current_price=current_price,
    portfolio_value=portfolio_value,
    sizing_method=PositionSizingMethod.PERCENT_PORTFOLIO,
    percent_portfolio=0.02,
    max_position_value=10000
)

# 5. Submit order (passes through risk engine automatically via API)
if order:
    result = broker.submit_order(**order)
    print(f"Order submitted: {result['id']}")

# 6. Monitor execution
import time
time.sleep(2)
order_status = broker.get_order(result['id'])
print(f"Order status: {order_status['status']}")
```

## 🎯 Production Checklist

**Local Testing:**

- [ ] `BROKER=alpaca` set
- [ ] Paper keys configured
- [ ] `ALPACA_PAPER=1` verified
- [ ] Test suite passes
- [ ] Dry run orders work
- [ ] Real paper orders execute

**Railway Deployment:**

- [ ] Environment variables set in dashboard
- [ ] Health endpoint returns OK
- [ ] Price providers working
- [ ] Orders submit successfully
- [ ] Risk engine active

**Live Trading Prep:**

- [ ] Paper testing complete (>50 orders)
- [ ] Live account funded
- [ ] Live API keys generated
- [ ] `ALPACA_PAPER=0` set
- [ ] URL validation passes
- [ ] Risk limits configured
- [ ] Monitoring enabled
- [ ] Start with small positions

______________________________________________________________________

**Status**: ✅ Ready for paper trading\
**Next**: Configure paper credentials and test
