# 📋 Ghost Daily Audit - Quick Reference

## ⚡ One-Line Setup

```bash
chmod +x ghost_daily_audit.sh && ./ghost_daily_audit.sh

```text

______________________________________________________________________

## 🕐 Cron Schedules (Copy & Paste)

```bash

# Daily at 9:00 AM

0 9 ***/workspaces/GHOST/ghost_daily_audit.sh

# Twice daily (9 AM & 9 PM)

0 9,21*** /workspaces/GHOST/ghost_daily_audit.sh

# Every 6 hours

0 */6 ***/workspaces/GHOST/ghost_daily_audit.sh

# With environment variables

0 9*** source /etc/ghost/audit.env && /workspaces/GHOST/ghost_daily_audit.sh >> /var/log/ghost/cron.log 2>&1

```text

______________________________________________________________________

## 📊 What Gets Checked (23 Tests)

| Category | Checks | |----------|--------| | **Core Health**| Basic & detailed system
status | |**AI Memory**| 58K+ decisions, activity, growth | |**Data Persistence**|
Portfolio, DB, cache | |**Price Providers**| Live data, API keys, fallbacks | |**Telegram Bot**| Bot status, webhook,
endpoint | |**Security**| All API keys/secrets
| |**Functionality** | Critical endpoints |

______________________________________________________________________

## 🎯 Health Score Interpretation

| Score | Status | Meaning | |-------|--------|---------| | 100% | 🎉 EXCELLENT | Perfect
\- no issues | | 90-99% | ✅ GOOD | Minor warnings only | | 70-89% | ⚠️ FAIR | Some issues
detected | | \<70% | ❌ POOR | Multiple failures |

______________________________________________________________________

## 📱 Telegram Notification Format

```text

🎉 Ghost Daily Audit Report

Status: EXCELLENT (100%)
Date: 2025-10-04 09:00:00

✅ Passed: 23
❌ Failed: 0
⚠️  Warnings: 0

AI Memory: 58226 decisions
Portfolio: 0 shares

```text

______________________________________________________________________

## 🔧 Environment Variables

```bash

GHOST_URL="<<<<<https://web-production-8e9a0.up.railway.app">>>>>
GHOST_API_TOKEN="your-token"
TELEGRAM_BOT_TOKEN="8229069551:..."
TELEGRAM_CHAT_ID="940596997"
LOG_FILE="/var/log/ghost/audit.log"

```text

______________________________________________________________________

## 📝 View Results

```bash

# Run audit manually

./ghost_daily_audit.sh

# View latest log

tail -f /tmp/ghost_audit_*.log

# View today's log

cat /tmp/ghost_audit_$(date +%Y%m%d).log

# Check cron execution

grep CRON /var/log/syslog

```text

______________________________________________________________________

## 🐛 Quick Troubleshooting

| Problem | Solution | |---------|----------| | Command not found |
`sudo apt-get install curl jq bc` | | Permission denied |
`chmod +x ghost_daily_audit.sh` | | Cron not running | `sudo systemctl status cron` | |
No Telegram alert | Check `TELEGRAM_BOT_TOKEN` set | | Env vars not found |
`source /etc/ghost/audit.env` |

______________________________________________________________________

## 🚀 Most Common Setup

```bash

# 1. Create config

cat > /etc/ghost/audit.env << 'EOF'
GHOST_URL=<<<<<https://web-production-8e9a0.up.railway.app>>>>>
GHOST_API_TOKEN=e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0
TELEGRAM_BOT_TOKEN=8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw
TELEGRAM_CHAT_ID=940596997
EOF

# 2. Add to crontab

(crontab -l 2>/dev/null; echo "0 9 ***source /etc/ghost/audit.env && /workspaces/GHOST/ghost_daily_audit.sh") |
crontab -

# 3. Test

source /etc/ghost/audit.env && /workspaces/GHOST/ghost_daily_audit.sh

```text

______________________________________________________________________

## ✅ First Run Checklist

- [ ] Script executable: `chmod +x ghost_daily_audit.sh`
- [ ] Dependencies installed: `jq`, `curl`, `bc`
- [ ] Environment variables set
- [ ] Manual test run successful
- [ ] Telegram notification received
- [ ] Cron job added
- [ ] Log directory created


______________________________________________________________________

## 📞 Sample Audit Results**Latest Test Run (Oct 4, 2025):**- ✅**Passed**: 20/23 tests

- ⚠️ **Warnings**: 3 (fallback price source, REDIS_URL missing)
- ❌ **Failed**: 0
- 📊 **Health Score**: 86% (GOOD)
- 🧠 **AI Memory**: 58,226 decisions
- 💰 **Portfolio**: 0 shares (empty)
- 📱 **Telegram**: Active, webhook configured


______________________________________________________________________

## 🎉 Done

Your Ghost system now:

- ✅ Self-checks daily at 9:00 AM
- ✅ Alerts you via Telegram
- ✅ Logs results to `/tmp/ghost_audit_*.log`
- ✅ Monitors 23 critical health points


**Full docs**: See `GHOST_DAILY_AUDIT_SETUP.md`
