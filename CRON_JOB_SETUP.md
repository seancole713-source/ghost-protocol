# 🕐 External Cron Setup (cron-job.org)

Railway containers can restart, which disrupts internal schedulers. Using an external cron service like **cron-job.org** provides more reliable scheduling.

## Quick Setup (5 minutes)

### 1. Create Account
Go to [cron-job.org](https://cron-job.org) and sign up (free tier = 3 jobs)

### 2. Get Your Railway URL
Your app URL: `https://ghost-protocol-production.up.railway.app`

### 3. Create These Jobs

| Job Name | URL | Time (UTC) | Time (Central) | Method |
|----------|-----|------------|----------------|--------|
| Daily Scout | `/cron/daily-scout` | 12:00 | 6:00 AM | GET |
| Morning Alert | `/cron/morning-alert` | 14:00 | 8:00 AM | GET |
| Evening Resolve | `/cron/evening-resolve` | 00:00 | 6:00 PM | GET |

### 4. Job Details

#### 🌅 Daily Scout (6 AM Central)
- **URL:** `https://ghost-protocol-production.up.railway.app/cron/daily-scout`
- **Time:** `0 12 * * *` (12:00 UTC = 6 AM CT)
- **Purpose:** Scans all stocks + crypto, records trades

#### ☀️ Morning Alert (8 AM Central)
- **URL:** `https://ghost-protocol-production.up.railway.app/cron/morning-alert`
- **Time:** `0 14 * * *` (14:00 UTC = 8 AM CT)  
- **Purpose:** Sends TOP 5 picks to Telegram

#### 🌙 Evening Resolve (6 PM Central)
- **URL:** `https://ghost-protocol-production.up.railway.app/cron/evening-resolve`
- **Time:** `0 0 * * *` (00:00 UTC = 6 PM CT)
- **Purpose:** Resolves trades, updates win/loss rankings

---

## Security (Optional but Recommended)

### Add a Cron Secret

1. **Set in Railway:**
   ```bash
   railway variables set CRON_SECRET="your-secret-key-here"
   ```

2. **Add to cron-job.org:**
   - Go to each job → Advanced → Request Headers
   - Add: `X-Cron-Secret: your-secret-key-here`

   Or use query param:
   ```
   https://your-app.railway.app/cron/daily-scout?secret=your-secret-key-here
   ```

---

## Testing Endpoints

Test each endpoint manually:

```bash
# Test daily scout
curl https://ghost-protocol-production.up.railway.app/cron/daily-scout

# Test morning alert
curl https://ghost-protocol-production.up.railway.app/cron/morning-alert

# Test evening resolve
curl https://ghost-protocol-production.up.railway.app/cron/evening-resolve

# Health check
curl https://ghost-protocol-production.up.railway.app/cron/health
```

---

## Expected Responses

### Daily Scout
```json
{
  "ok": true,
  "job": "daily-scout",
  "timestamp": "2026-02-05T12:00:00Z",
  "stocks_scouted": 86,
  "crypto_scouted": 98,
  "total_scouted": 184
}
```

### Morning Alert
```json
{
  "ok": true,
  "job": "morning-alert",
  "telegram_sent": true,
  "stocks": ["NVDA", "META", "PLTR", "COIN", "MSTR"],
  "crypto": ["RNDR", "TURBO", "SOL", "BTC", "SUI"]
}
```

### Evening Resolve
```json
{
  "ok": true,
  "job": "evening-resolve",
  "result": {"resolved": 50, "winners": 32, "losers": 18}
}
```

---

## Alternative Services

If cron-job.org doesn't work:
- [EasyCron](https://www.easycron.com/)
- [cron-job.org Pro](https://cron-job.org/en/pricing/)
- [AWS EventBridge](https://aws.amazon.com/eventbridge/)
- GitHub Actions with scheduled workflows

---

## Troubleshooting

### Jobs not running?
1. Check cron-job.org dashboard for errors
2. Verify Railway app is up: `/cron/health`
3. Check Railway logs for errors

### Telegram not sending?
1. Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set
2. Test manually: `curl /cron/morning-alert`

### Stock prices still 0?
- Polygon API key issue - check `POLYGON_API_KEY` env var
- Free tier limit (5 calls/min) - the scout has delays built in
