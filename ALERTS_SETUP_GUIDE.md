# 🔔 Alert System Setup Guide

## Quick Start

Ghost Protocol supports **3 alert channels**:
- **Slack** (webhook)
- **Discord** (webhook)
- **Email** (SendGrid)

Choose one or use all three! Takes ~5 minutes to set up.

---

## 🟢 Slack Setup

### 1. Create Slack Webhook

1. Go to https://api.slack.com/apps
2. Click **"Create New App"** → **"From scratch"**
3. Name: `Ghost Protocol Alerts`
4. Select your workspace
5. Click **"Incoming Webhooks"** → Toggle **ON**
6. Click **"Add New Webhook to Workspace"**
7. Select channel (e.g., `#trading-alerts`)
8. Copy the webhook URL (starts with `https://hooks.slack.com/services/...`)

### 2. Add to Railway

1. Go to Railway Dashboard → **ghost-protocol** → **Variables**
2. Click **"New Variable"**
3. Name: `SLACK_WEBHOOK_URL`
4. Value: Paste your webhook URL
5. Click **"Add"**

### 3. Test

```bash
# Test endpoint (sends test alert to Slack)
curl -X POST https://ghost-protocol-production.up.railway.app/api/v3/alerts/test \
  -H "Content-Type: application/json" \
  -d '{"channel": "slack", "message": "Test from Ghost Protocol"}' | jq
```

You should see a message in your Slack channel! 🎉

---

## 💜 Discord Setup

### 1. Create Discord Webhook

1. Open Discord → Go to your server
2. Right-click on channel → **"Edit Channel"**
3. Go to **"Integrations"** → **"Webhooks"**
4. Click **"New Webhook"**
5. Name: `Ghost Protocol`
6. Copy the **Webhook URL**

### 2. Add to Railway

1. Railway Dashboard → **Variables**
2. Add new variable:
   - Name: `DISCORD_WEBHOOK_URL`
   - Value: Your webhook URL

### 3. Test

```bash
curl -X POST https://ghost-protocol-production.up.railway.app/api/v3/alerts/test \
  -H "Content-Type: application/json" \
  -d '{"channel": "discord", "message": "Test from Ghost Protocol"}' | jq
```

---

## 📧 Email Setup (SendGrid)

### 1. Create SendGrid Account

1. Go to https://sendgrid.com/
2. Sign up (free tier: 100 emails/day)
3. Go to **Settings** → **API Keys**
4. Click **"Create API Key"**
5. Name: `Ghost Protocol`
6. Permissions: **Full Access**
7. Copy the API key (starts with `SG.`)

### 2. Add to Railway

Railway → Variables → Add:
- `SENDGRID_API_KEY`: Your API key
- `ALERT_EMAIL`: Your email address (where you want alerts)

### 3. Test

```bash
curl -X POST https://ghost-protocol-production.up.railway.app/api/v3/alerts/test \
  -H "Content-Type: application/json" \
  -d '{"channel": "email", "message": "Test from Ghost Protocol"}' | jq
```

---

## 🎯 Alert Types

Once configured, you'll automatically receive:

### 1. **Trade Execution Alerts**
```
📈 Trade Executed
Symbol: AAPL
Side: BUY
Quantity: 10
Price: $175.50
Time: 2025-12-11 14:30:00 UTC
```

### 2. **Circuit Breaker Alerts** (Critical)
```
🚨 CIRCUIT BREAKER ACTIVATED 🚨
Reason: Daily loss limit reached
Daily P&L: $-510.00
⚠️ Trading PAUSED
```

### 3. **Daily Summary** (EOD)
```
🎉 Daily Trading Summary
Date: 2025-12-11
Daily P&L: $425.75
Total Trades: 12
Win Rate: 66.7%
```

### 4. **Milestone Alerts**
```
🎯 Profit Target Reached!
Total P&L: $5,000.00
```

---

## ⚙️ Advanced Configuration

### Custom Alert Thresholds

Add to Railway Variables:

```bash
# Only alert if daily P&L exceeds this amount
ALERT_DAILY_PNL_THRESHOLD=1000

# Alert frequency (prevent spam)
ALERT_MIN_INTERVAL_SECONDS=300
```

### Disable Specific Alert Types

```bash
# Disable trade alerts (only get circuit breaker + daily summary)
ALERT_TRADES_ENABLED=false

# Disable daily summaries
ALERT_DAILY_SUMMARY_ENABLED=false
```

---

## 🧪 Testing Your Setup

### Test All Channels at Once

```bash
curl -X POST https://ghost-protocol-production.up.railway.app/api/v3/alerts/test-all | jq
```

This will send test messages to:
- Slack (if configured)
- Discord (if configured)
- Email (if configured)

### Manual Trade Alert Test

```bash
curl -X POST https://ghost-protocol-production.up.railway.app/api/v3/alerts/send \
  -H "Content-Type: application/json" \
  -d '{
    "type": "trade",
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 10,
    "price": 175.50,
    "pnl": 125.00
  }' | jq
```

---

## 📱 Mobile Notifications

### Slack Mobile
1. Install Slack app
2. Enable push notifications for your channel
3. Done! Get alerts on your phone

### Discord Mobile
1. Install Discord app
2. Enable push notifications
3. Mentions/channel settings for immediate alerts

### Email Mobile
- Use Gmail/Outlook app with push notifications
- Set up VIP/Priority for `alerts@ghost-protocol.com`

---

## 🔍 Troubleshooting

### "Alert send failed: 404"
- Check webhook URL is correct
- Webhook might have been deleted/regenerated
- Regenerate and update Railway variable

### "Alert send failed: 401"
- SendGrid API key invalid/expired
- Regenerate API key in SendGrid dashboard

### "Alert send failed: timeout"
- Check Railway logs for detailed error
- Webhook endpoint might be down
- Test webhook manually: `curl -X POST <your_webhook_url> -d '{"text":"test"}'`

### Not Receiving Alerts
1. Check Railway variables are set correctly
2. Restart Railway deployment to pick up new env vars
3. Test using `/api/v3/alerts/test` endpoint
4. Check spam folder for emails

---

## 🎨 Custom Alert Messages

Want to customize alert formatting? Edit `core/alert_system.py`:

```python
# Line ~55: Trade alert format
message = f"""📈 **Trade Executed**
Symbol: {symbol}
Side: {side}
Your custom fields here!
"""
```

Commit and push to deploy.

---

## 🚀 You're All Set!

Once configured, Ghost Protocol will automatically notify you:
- ✅ Every trade execution
- ✅ Daily performance summaries
- ✅ Critical safety alerts
- ✅ Performance milestones

**No more checking Railway logs!** Get real-time updates wherever you are.
