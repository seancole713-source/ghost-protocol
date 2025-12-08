# Ghost 24-Hour Auto-Audit Setup Guide

## Daily Self-Check for Memory, Data, and Telegram Functions

______________________________________________________________________

## 📋 What It Does

The Ghost Daily Auto-Audit script performs comprehensive health checks:

### ✅ Checks Performed

1. **Core Health**- Basic & detailed system status


2.**AI Memory**- 58K+ decisions, recent activity, growth
3.**Data Persistence**- Portfolio, database, cache status
4.**Price Providers**- Data availability, API keys, fallbacks
5.**Telegram Bot**- Bot status, webhook config, endpoint
6.**Security**- API keys and secrets validation
7.**Functionality**- Critical endpoints spot checks


### 📊 Audit Results

-**Health Score**: Percentage of passed checks

- **Status**: EXCELLENT / GOOD / FAIR / POOR
- **Detailed Log**: Saved to `/tmp/ghost_audit_YYYYMMDD.log`
- **Telegram Alert**: Automatic notification with summary


______________________________________________________________________

## 🚀 Quick Setup

### 1. Install the Script

The script is already created at:

```bash
/workspaces/GHOST/ghost_daily_audit.sh

```text

Make it executable (already done):

```bash

chmod +x ghost_daily_audit.sh

```text

### 2. Set Environment Variables

The script uses these environment variables:

```bash

export GHOST_URL="<<<<<https://web-production-8e9a0.up.railway.app">>>>>
export GHOST_API_TOKEN="your-api-token-here"
export TELEGRAM_BOT_TOKEN="8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw"
export TELEGRAM_CHAT_ID="940596997"

```text

Or create a config file `/etc/ghost/audit.env`:

```bash

# Ghost Audit Configuration

GHOST_URL=<<<<<https://web-production-8e9a0.up.railway.app>>>>>
GHOST_API_TOKEN=e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0
TELEGRAM_BOT_TOKEN=8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw
TELEGRAM_CHAT_ID=940596997
LOG_FILE=/var/log/ghost/audit.log

```text

### 3. Manual Test Run

Run the audit manually to verify it works:

```bash

./ghost_daily_audit.sh

```text

Expected output:

```text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 Ghost Daily Auto-Audit - 2025-10-04 16:56:40
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 1. CORE HEALTH CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ PASS - Basic health endpoint responding
✅ PASS - Detailed health check: all systems operational

...

📊 AUDIT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ️  INFO - Tests Passed:  20
ℹ️  INFO - Tests Failed:  0
ℹ️  INFO - Warnings:      3
ℹ️  INFO - Health Score:  86%

✅ AUDIT RESULT: GOOD - Minor warnings detected

```text

______________________________________________________________________

## ⏰ Automated Daily Execution

### Option A: Cron (Recommended for Servers)

1. **Edit crontab**:


```bash

crontab -e

```text

1. **Add daily audit at 9:00 AM**:


```bash

# Ghost Daily Audit - runs every day at 9:00 AM

0 9 ***/workspaces/GHOST/ghost_daily_audit.sh >> /var/log/ghost/cron.log 2>&1

```text

1.**Alternative schedules**:


```bash

# Every day at 6:00 AM

0 6 ***/workspaces/GHOST/ghost_daily_audit.sh

# Twice daily: 9:00 AM and 9:00 PM

0 9,21*** /workspaces/GHOST/ghost_daily_audit.sh

# Every 12 hours

0 */12 *** /workspaces/GHOST/ghost_daily_audit.sh

# Every 6 hours (4 times daily)

0 */6 ***/workspaces/GHOST/ghost_daily_audit.sh

```text

1.**With environment variables**:


```bash

0 9 ***source /etc/ghost/audit.env && /workspaces/GHOST/ghost_daily_audit.sh

```text

### Option B: systemd Timer (Modern Linux)

1.**Create service file**`/etc/systemd/system/ghost-audit.service`:


```ini

[Unit]
Description=Ghost Daily Auto-Audit
After=network.target

[Service]
Type=oneshot
User=ghost
EnvironmentFile=/etc/ghost/audit.env
ExecStart=/workspaces/GHOST/ghost_daily_audit.sh
StandardOutput=journal
StandardError=journal

```text

1.**Create timer file** `/etc/systemd/system/ghost-audit.timer`:


```ini

[Unit]
Description=Run Ghost audit daily at 9 AM
Requires=ghost-audit.service

[Timer]
OnCalendar=daily
OnCalendar=*-*-* 09:00:00
Persistent=true

[Install]
WantedBy=timers.target

```text

1. **Enable and start**:


```bash

sudo systemctl daemon-reload
sudo systemctl enable ghost-audit.timer
sudo systemctl start ghost-audit.timer

```text

1. **Check status**:


```bash

sudo systemctl status ghost-audit.timer
sudo systemctl list-timers ghost-audit

```text

### Option C: Railway Cron Job

Railway doesn't natively support cron jobs, but you can:

1. **Use GitHub Actions**to trigger the audit:


Create `.github/workflows/ghost-audit.yml`:

```yaml

name: Ghost Daily Audit

on:
  schedule:

    - cron: '0 9***'  # 9:00 AM UTC daily


  workflow_dispatch:  # Manual trigger

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:

      - name: Checkout


        uses: actions/checkout@v3

      - name: Run Ghost Audit


        env:
          GHOST_URL: ${{ secrets.GHOST_URL }}
          GHOST_API_TOKEN: ${{ secrets.GHOST_API_TOKEN }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: |
          chmod +x ghost_daily_audit.sh
          ./ghost_daily_audit.sh

```text

1. **Use EasyCron**(external service):

   - Sign up at <<<<<https://www.easycron.com>>>>>
   - Create job: Run shell command or webhook
   - Point to: `curl <<<<<https://your-audit-endpoint.com/run-audit`>>>>>


1.**Use Cloud Scheduler**(GCP):

   - Create Cloud Scheduler job
   - Target: HTTP endpoint or Cloud Function
   - Schedule: `0 9***`


______________________________________________________________________

## 📱 Telegram Notifications

The audit automatically sends a Telegram message if `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID` are set.

### Sample Notification

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

### If Issues Detected

```text

⚠️  Ghost Daily Audit Report

Status: FAIR (73%)
Date: 2025-10-04 09:00:00

✅ Passed: 16
❌ Failed: 3
⚠️  Warnings: 3

AI Memory: 58226 decisions
Portfolio: 0 shares

🚨 Critical Issues:
→ Price provider timeout
→ Webhook endpoint 500 error
→ AI Memory database locked

```text

______________________________________________________________________

## 📊 Interpreting Results

### Health Scores

- **100%**: 🎉 EXCELLENT - Perfect health
- **90-99%**: ✅ GOOD - Minor warnings only
- **70-89%**: ⚠️ FAIR - Some issues detected
- **\<70%**: ❌ POOR - Multiple failures


### Common Warnings (Normal)

- ⚠️ Using fallback price source (markets closed)
- ⚠️ News cache aging (6+ hours old)
- ⚠️ Missing REDIS_URL (optional)
- ⚠️ AI Memory stale (weekend, no trading)


### Critical Issues (Action Required)

- ❌ Basic health endpoint failed
- ❌ AI Memory database error
- ❌ Portfolio persistence error
- ❌ Telegram bot not responding
- ❌ Price unavailable (all providers failed)


______________________________________________________________________

## 📝 Log Files

### Default Log Location

```bash

/tmp/ghost_audit_YYYYMMDD.log

```text

### Custom Log Location

Set `LOG_FILE` environment variable:

```bash

export LOG_FILE="/var/log/ghost/audit-$(date +%Y%m%d).log"

```text

### View Latest Log

```bash

tail -f /tmp/ghost_audit_*.log

```text

### Rotate Logs (Keep Last 7 Days)

Add to crontab:

```bash

0 0 *** find /tmp -name "ghost_audit_*.log" -mtime +7 -delete

```text

______________________________________________________________________

## 🔧 Customization

### Change Audit Schedule

Edit cron expression in crontab:

```bash

# Daily at 9 AM

0 9 ***# Twice daily (9 AM & 9 PM)

0 9,21***

# Every 6 hours

0 */6 ***

# Weekly on Monday at 9 AM

0 9 * * 1

# Monthly on 1st at 9 AM

0 9 1 * *

```text

### Add Custom Checks

Edit `ghost_daily_audit.sh` and add your checks:

```bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "🔍 8. CUSTOM CHECKS"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Example: Check disk space

disk_usage=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$disk_usage" -lt 80 ]; then
    log_pass "Disk usage OK: ${disk_usage}%"
else
    log_warn "Disk usage high: ${disk_usage}%"
fi

```text

### Disable Telegram Notifications

Unset the environment variables:

```bash

unset TELEGRAM_BOT_TOKEN
unset TELEGRAM_CHAT_ID

```text

Or comment out the Telegram section in the script.

______________________________________________________________________

## 🐛 Troubleshooting

### Issue: Cron job not running

**Check cron service**:

```bash

sudo systemctl status cron

```text

**Check cron logs**:

```bash

grep CRON /var/log/syslog

```text

**Test cron expression**:

```bash

# Run at next minute

*** * * /workspaces/GHOST/ghost_daily_audit.sh


```text

### Issue: Script fails with "command not found"

**Check dependencies**:

```bash

which curl
which jq

```text

**Install missing tools**:

```bash

# Debian/Ubuntu

sudo apt-get install curl jq bc

# RHEL/CentOS

sudo yum install curl jq bc

# macOS

brew install jq bc

```text

### Issue: Permission denied

**Fix permissions**:

```bash

chmod +x /workspaces/GHOST/ghost_daily_audit.sh

```text

**Check file ownership**:

```bash

ls -la /workspaces/GHOST/ghost_daily_audit.sh

```text

### Issue: Environment variables not found

**Source config file**:

```bash

source /etc/ghost/audit.env
./ghost_daily_audit.sh

```text

**Or inline**:

```bash

GHOST_URL=<<<<<https://...>>>>> GHOST_API_TOKEN=... ./ghost_daily_audit.sh

```text

______________________________________________________________________

## 📚 Example Cron Setup (Complete)

### 1. Create config file

```bash

sudo mkdir -p /etc/ghost
sudo tee /etc/ghost/audit.env << 'EOF'
GHOST_URL=<<<<<https://web-production-8e9a0.up.railway.app>>>>>
GHOST_API_TOKEN=e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0
TELEGRAM_BOT_TOKEN=8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw
TELEGRAM_CHAT_ID=940596997
LOG_FILE=/var/log/ghost/audit.log
EOF

```text

### 2. Create log directory

```bash

sudo mkdir -p /var/log/ghost
sudo chown $USER:$USER /var/log/ghost

```text

### 3. Add to crontab

```bash

crontab -e

```text

Add:

```bash

# Source env and run Ghost audit daily at 9 AM

0 9 ***source /etc/ghost/audit.env && /workspaces/GHOST/ghost_daily_audit.sh >> /var/log/ghost/cron.log 2>&1

# Rotate old logs (keep last 30 days)

0 0*** find /var/log/ghost -name "audit-*.log" -mtime +30 -delete

```text

### 4. Verify

```bash

# List cron jobs

crontab -l

# Test run

source /etc/ghost/audit.env && /workspaces/GHOST/ghost_daily_audit.sh

# Check Telegram for notification

```text

______________________________________________________________________

## 🎉 Success

You should now receive daily health reports via Telegram at 9:00 AM!

**Check Results:**- 📱 Telegram notification with health score

- 📝 Detailed log at `/var/log/ghost/audit.log`
- 🔍 Comprehensive 23-point system check**Next Steps:**- Monitor first few audits to ensure stability
- Adjust schedule if needed (twice daily for critical systems)
- Customize checks for your specific needs
- Set up log rotation for long-term storage


Ghost now**self-audits daily** and alerts you automatically! 🤖✅
