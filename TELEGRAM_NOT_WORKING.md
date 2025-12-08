# Telegram Not Working - Root Cause & Fix

## Problem

Telegram feed shows `false` and `can_send: false`

## Root Cause

The env vars `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are loaded **once at startup**(line 806-807 in wolf_app.py):

```python
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

```text

If these weren't set in Railway**when the app was deployed**, they'll be empty strings.

## Solution

### Option 1: Redeploy After Setting Env Vars (Recommended)

1. Go to Railway → Project → Variables
2. Confirm these are set:
   - `TELEGRAM_BOT_TOKEN` = `8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw`
   - `TELEGRAM_CHAT_ID` = `940596997`
1. **Redeploy**the service (Railway → Deployments → Redeploy)
2. Wait 2-3 minutes for deployment
3. Test: `curl -X POST <<<<<https://web-production-8e9a0.up.railway.app/api/telegram/test`>>>>>


### Option 2: Use Telegram Reinit Endpoint

If the app has a reinit endpoint, you can reload env vars without redeploying:

```bash

curl -X POST <<<<<https://web-production-8e9a0.up.railway.app/api/telegram/reinit>>>>>

```text

## Verification

After fix, these should return `true`:

```bash

# Check feed status

curl -s <<<<<https://web-production-8e9a0.up.railway.app/api/cockpit>>>>> | jq '.status.feeds.telegram'

# Output: true

# Test sending

curl -X POST <<<<<https://web-production-8e9a0.up.railway.app/api/telegram/test>>>>> | jq '.can_send'

# Output: true

```text

## Why It Happened

The env vars were added to Railway**after**the initial deployment, so the running
instance never saw them. Environment variables need to be present**before**or**during** deployment to be loaded at
startup.
