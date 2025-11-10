# Alpaca Broker Setup Guide

## Overview

The Ghost trading system integrates with Alpaca for live and paper trading. This guide
covers configuration, safety measures, and operational procedures.

## Environment Configuration

### Required Environment Variables

```bash
# Enable Alpaca broker
BROKER=alpaca

# API Credentials
ALPACA_KEY_ID=your_api_key_id
ALPACA_SECRET_KEY=your_secret_key

# Trading Mode
ALPACA_PAPER=1              # 1 = Paper trading (default), 0 = LIVE trading
APCA_API_BASE_URL=          # Optional: Override base URL

# Rate Limiting (optional)
ALPACA_ORDER_RATE=30        # Max orders per window (default: 30)
ALPACA_ORDER_WINDOW_S=60    # Rate limit window in seconds (default: 60)
```

### Paper Trading vs Live Trading

**Paper Trading (ALPACA_PAPER=1):**

- Uses: `https://paper-api.alpaca.markets/v2`
- Risk: None (simulated money)
- Best for: Testing strategies, development, demos
- API Keys: Use paper trading keys from Alpaca dashboard

**Live Trading (ALPACA_PAPER=0):**

- Uses: `https://api.alpaca.markets/v2`
- Risk: Real money at risk
- Best for: Production trading with validated strategies
- API Keys: Use live trading keys (requires funded account)

⚠️ **WARNING**: Never use live API keys in paper mode or vice versa. Alpaca enforces
this at the API level.

## Getting API Keys

1. **Sign up for Alpaca**: https://alpaca.markets

2. **Navigate to Paper Trading Dashboard**:
   https://app.alpaca.markets/paper/dashboard/overview

3. **Generate API Keys**:

   - Go to "Your API Keys" section
   - Click "Generate New Key"
   - Save `API Key ID` and `Secret Key` securely
   - **Never commit keys to git or share publicly**

4. **For Live Trading** (after paper testing):

   - Fund your account
   - Generate separate live API keys
   - Update environment variables
   - Set `ALPACA_PAPER=0`

## Safety Features

### 1. Risk Engine Integration

All orders pass through the risk engine before submission:

- Position size limits
- Portfolio concentration checks
- Maximum loss per trade
- Daily loss limits

### 2. Rate Limiting

Built-in rate limiter prevents API throttling:

- Default: 30 orders per 60 seconds
- Configurable via env vars
- Uses AsyncRateLimiter for non-blocking operation

### 3. Dry Run Mode

Test orders without submission:

```bash
curl -X POST https://your-ghost-instance/api/trade/submit \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "WOLF",
    "qty": 10,
    "side": "buy",
    "type": "market",
    "dry_run": true
  }'
```

### 4. Pre-Flight Checks

Before each order:

- Verify market is open (unless extended_hours=true)
- Check account status (not blocked, not restricted)
- Validate sufficient buying power
- Confirm order parameters

### 5. Paper Mode Default

System defaults to paper trading even if ALPACA_PAPER is not set:

```python
self.paper = os.getenv("ALPACA_PAPER", "1") == "1"
```

## API Endpoints

### Broker Health Check

```bash
GET /api/broker/health
```

Returns:

- Connection status
- Account info
- Buying power
- Portfolio value
- Position count
- Market open status

### Get Account

```bash
GET /api/broker/account
```

Returns full account details including equity, cash, trading restrictions.

### Get Positions

```bash
GET /api/broker/positions
```

Returns all open positions with P&L, entry price, current value.

### Get Market Clock

```bash
GET /api/broker/clock
```

Returns market open/close times and current status.

### Submit Order

```bash
POST /api/trade/submit
{
  "symbol": "AAPL",
  "qty": 10,
  "side": "buy",
  "type": "market",
  "time_in_force": "day",
  "dry_run": false
}
```

### Get Orders

```bash
GET /api/trade/orders?status=open&limit=50
```

### Cancel Order

```bash
DELETE /api/trade/order/{order_id}
```

### Close Position

```bash
POST /api/trade/position/close/{symbol}
```

## Order Types

### Market Order

Executes immediately at current market price:

```json
{
  "symbol": "WOLF",
  "qty": 10,
  "side": "buy",
  "type": "market"
}
```

### Limit Order

Executes at specified price or better:

```json
{
  "symbol": "WOLF",
  "qty": 10,
  "side": "buy",
  "type": "limit",
  "limit_price": 15.50,
  "time_in_force": "gtc"
}
```

### Stop Order

Triggers market order when price reaches stop:

```json
{
  "symbol": "WOLF",
  "qty": 10,
  "side": "sell",
  "type": "stop",
  "stop_price": 14.00
}
```

### Stop-Limit Order

Triggers limit order at stop price:

```json
{
  "symbol": "WOLF",
  "qty": 10,
  "side": "sell",
  "type": "stop_limit",
  "stop_price": 14.00,
  "limit_price": 13.50
}
```

### Trailing Stop

Stop price trails market price by percentage or dollar amount:

```json
{
  "symbol": "WOLF",
  "qty": 10,
  "side": "sell",
  "type": "trailing_stop",
  "trail_percent": 5.0
}
```

## Time in Force Options

- **day**: Order active until end of trading day
- **gtc**: Good 'til canceled (active until filled or manually canceled)
- **ioc**: Immediate or cancel (fill immediately or cancel)
- **fok**: Fill or kill (fill entire order immediately or cancel)

## Testing Procedure

### 1. Initial Setup (Paper Trading)

```bash
# Set environment variables
export BROKER=alpaca
export ALPACA_KEY_ID=your_paper_key
export ALPACA_SECRET_KEY=your_paper_secret
export ALPACA_PAPER=1

# Run connection test
python test_alpaca_broker.py
```

### 2. Verify Health

```bash
curl https://your-ghost-instance/api/broker/health
```

Expected response:

```json
{
  "ok": true,
  "broker": "alpaca",
  "paper": true,
  "account_id": "...",
  "buying_power": 100000.00,
  "portfolio_value": 100000.00,
  "market_open": true
}
```

### 3. Test Dry Run Order

```bash
curl -X POST https://your-ghost-instance/api/trade/submit \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "qty": 1,
    "side": "buy",
    "type": "market",
    "dry_run": true
  }'
```

### 4. Submit Paper Trading Order

```bash
curl -X POST https://your-ghost-instance/api/trade/submit \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "qty": 1,
    "side": "buy",
    "type": "market",
    "dry_run": false
  }'
```

### 5. Verify Order Execution

```bash
curl https://your-ghost-instance/api/trade/orders?status=open
curl https://your-ghost-instance/api/broker/positions
```

### 6. Transition to Live (After Extensive Testing)

```bash
# DANGER: Real money at risk!
export ALPACA_PAPER=0
export ALPACA_KEY_ID=your_live_key
export ALPACA_SECRET_KEY=your_live_secret

# Restart Ghost
./scripts/restart_ghost.sh

# Verify live connection
python test_alpaca_broker.py
```

## Monitoring and Logging

All broker operations are logged:

- Order submissions
- Order status changes
- Position updates
- Risk check failures
- API errors

Check logs:

```bash
tail -f logs/ghost.log | grep -E "trade|broker|order"
```

## Troubleshooting

### "Broker not enabled"

- Verify `BROKER=alpaca` is set
- Check API keys are configured
- Restart Ghost after env changes

### "API keys not configured"

- Set `ALPACA_KEY_ID` and `ALPACA_SECRET_KEY`
- Ensure keys are for correct mode (paper vs live)

### "Order blocked by risk engine"

- Check risk limits in risk_engine.py
- Review portfolio concentration
- Verify position sizing

### "Insufficient buying power"

- Check account balance: `GET /api/broker/account`
- Review open positions
- Consider closing positions or reducing order size

### "Market closed"

- Orders queue until market opens (unless extended_hours=true)
- Check market hours: `GET /api/broker/clock`
- Use `time_in_force=gtc` for overnight orders

## Security Best Practices

1. **Never commit API keys to git**

   - Use environment variables
   - Add .env to .gitignore
   - Use secrets management in production

2. **Separate paper and live keys**

   - Use different keys for testing and production
   - Rotate keys periodically
   - Revoke unused keys

3. **Monitor account activity**

   - Set up email/SMS alerts in Alpaca dashboard
   - Review daily trading activity
   - Check for unexpected positions

4. **Start small in live trading**

   - Test with minimal capital first
   - Gradually increase position sizes
   - Monitor performance closely

5. **Use risk limits**

   - Configure max position size
   - Set daily loss limits
   - Enable circuit breakers

## Railway Deployment

For Railway production deployment:

1. **Set environment variables in Railway dashboard**:

   ```
   BROKER=alpaca
   ALPACA_KEY_ID=***
   ALPACA_SECRET_KEY=***
   ALPACA_PAPER=1 (or 0 for live)
   ```

2. **Deploy and verify**:

   ```bash
   curl https://ghost-sniper-bot-seancole713-production.up.railway.app/api/broker/health
   ```

3. **Monitor logs**:

   ```bash
   railway logs
   ```

## Support and Resources

- **Alpaca API Docs**: https://alpaca.markets/docs/api-documentation/
- **Ghost Issues**: https://github.com/your-repo/issues
- **Trading Hours**: https://www.alpaca.markets/support/market-hours/
- **API Status**: https://status.alpaca.markets/

## License and Disclaimer

This software is provided "as is" without warranty. Trading involves risk of loss.
Always test thoroughly in paper mode before live trading. The developers assume no
liability for trading losses.
