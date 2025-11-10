# 🔐 RAILWAY SECRETS/ENVIRONMENT VARIABLES

## Required Secrets (8 total)

Add these to your Railway project via dashboard or CLI:

### 1. GHOST_API_TOKEN

**Purpose**: Authentication for Ghost API endpoints **Value**:
`e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0`

### 2. POLYGON_API_KEY

**Purpose**: Stock price data from Polygon.io **Value**:
`G1UkONuCx3Mpcngnvu239peiSyhNWRC3`

### 3. ALPHAVANTAGE_API_KEY

**Purpose**: Alternative stock price data provider **Value**: `3WNNLA81KS7BG4AK`

### 4. TELEGRAM_BOT_TOKEN

**Purpose**: Send alerts/notifications via Telegram **Value**:
`8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw`

### 5. TELEGRAM_CHAT_ID

**Purpose**: Your Telegram chat ID to receive alerts **Value**: `940596997`

### 6. GHOST_FOCUS_TICKER

**Purpose**: Primary ticker symbol to trade **Value**: `WOLF`

### 7. WOLF_PERSIST_MODE

**Purpose**: Database persistence mode **Value**: `sqlite`

### 8. SIM_MODE

**Purpose**: Simulation mode (0=live, 1=simulation) **Value**: `0`

______________________________________________________________________

## How to Add Secrets

### Option A: Railway Dashboard (Recommended)

1. Go to your Railway dashboard
2. Click on your `ghost-protocol` project
3. Click on the service/deployment
4. Go to "Variables" tab
5. Click "Add Variable" for each secret above
6. Copy the exact values (including quotes if shown)

### Option B: Railway CLI

```bash
# First, login to Railway
railway login

# Add all 8 variables at once
railway variables set GHOST_API_TOKEN="e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0"
railway variables set POLYGON_API_KEY="G1UkONuCx3Mpcngnvu239peiSyhNWRC3"
railway variables set ALPHAVANTAGE_API_KEY="3WNNLA81KS7BG4AK"
railway variables set TELEGRAM_BOT_TOKEN="8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw"
railway variables set TELEGRAM_CHAT_ID="940596997"
railway variables set GHOST_FOCUS_TICKER="WOLF"
railway variables set WOLF_PERSIST_MODE="sqlite"
railway variables set SIM_MODE="0"
```

### Verify Variables Are Set

```bash
railway variables
```

Should show all 8 variables listed.

______________________________________________________________________

## What Each Secret Does

| Variable | Used For | Required? | |----------|----------|-----------| |
**GHOST_API_TOKEN** | Authenticate API requests to Ghost | ✅ YES | | **POLYGON_API_KEY**
| Fetch stock prices from Polygon.io | ✅ YES | | **ALPHAVANTAGE_API_KEY** | Backup price
provider | ✅ YES | | **TELEGRAM_BOT_TOKEN** | Send trading alerts | ⚠️ Optional\* | |
**TELEGRAM_CHAT_ID** | Receive alerts on your phone | ⚠️ Optional\* | |
**GHOST_FOCUS_TICKER** | Which stock to trade (WOLF) | ✅ YES | | **WOLF_PERSIST_MODE** |
How to save data (sqlite) | ✅ YES | | **SIM_MODE** | Live trading (0) or simulation (1)
| ✅ YES |

\*Telegram is optional but recommended for alerts

______________________________________________________________________

## After Adding Secrets

Railway will automatically **redeploy** your app with the new environment variables.

**Wait 2-3 minutes** for the redeploy to complete, then test:

```bash
railway domain  # Get your URL
curl https://[your-url]/health
```

Expected: `{"ok": true, "ts": ...}`

______________________________________________________________________

## Security Notes

⚠️ **IMPORTANT**: These secrets are already exposed in this file. After deployment:

1. Consider rotating your API keys (get new ones)
2. Never commit `.env` files or secrets to GitHub
3. Use Railway's secret management (encrypted at rest)
4. Regenerate GHOST_API_TOKEN if you suspect compromise

To rotate GHOST_API_TOKEN:

```python
import secrets
new_token = secrets.token_urlsafe(32)
print(new_token)  # Use this as new GHOST_API_TOKEN
```

______________________________________________________________________

## Quick Copy-Paste (for CLI)

```bash
railway variables set GHOST_API_TOKEN="e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0" \
  && railway variables set POLYGON_API_KEY="G1UkONuCx3Mpcngnvu239peiSyhNWRC3" \
  && railway variables set ALPHAVANTAGE_API_KEY="3WNNLA81KS7BG4AK" \
  && railway variables set TELEGRAM_BOT_TOKEN="8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw" \
  && railway variables set TELEGRAM_CHAT_ID="940596997" \
  && railway variables set GHOST_FOCUS_TICKER="WOLF" \
  && railway variables set WOLF_PERSIST_MODE="sqlite" \
  && railway variables set SIM_MODE="0"
```

______________________________________________________________________

## Troubleshooting

**If Ghost crashes after deployment:**

- Check Railway logs: `railway logs --tail 50`
- Look for "environment variable not found" errors
- Verify all 8 variables are set: `railway variables`

**If prices don't work:**

- POLYGON_API_KEY and ALPHAVANTAGE_API_KEY must be valid
- Check provider status in logs
- Ghost will fall back to forecast data if providers fail

**If you don't receive alerts:**

- TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are optional
- Ghost will still trade without them
- Test Telegram: send `/start` to your bot

______________________________________________________________________

✅ **Add these 8 secrets, then Ghost will be fully operational on Railway!**
