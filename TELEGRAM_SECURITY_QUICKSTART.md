# Telegram Bot Security - Quick Reference

## 🚀 Quick Start

### Step 1: Get Your Bot Token
```bash
# From Railway dashboard:
# 1. Go to ghost-protocol project
# 2. Click "Variables" tab
# 3. Copy TELEGRAM_BOT_TOKEN value

export TELEGRAM_BOT_TOKEN='paste_your_token_here'
export TELEGRAM_CHAT_ID='your_chat_id'  # Optional
```

### Step 2: Run Setup Script
```bash
./setup_telegram_security.sh
```

This will:
- ✅ Check current bot name
- ✅ Offer to reset if wrong
- ✅ Start monitoring if desired

---

## 📋 Manual Commands

### Check Bot Name Once
```bash
python monitor_telegram_bot.py
```

### Reset Bot Name
```bash
python reset_telegram_bot_name.py
```

### Start Continuous Monitoring
```bash
# Foreground (see output)
python monitor_telegram_bot.py --continuous

# Background (runs silently)
nohup python monitor_telegram_bot.py --continuous > /dev/null 2>&1 &
```

### Check Monitor Logs
```bash
tail -f logs/telegram_bot_monitor.log
```

### View Bot State
```bash
cat data/telegram_bot_state.json | python -m json.tool
```

---

## 🎯 What Each Tool Does

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `reset_telegram_bot_name.py` | Resets bot name to "Ghost Protocol Bot" | When name is wrong |
| `monitor_telegram_bot.py` | Checks bot name every 5 min, alerts on changes | Run 24/7 in background |
| `setup_telegram_security.sh` | Interactive setup wizard | First time setup |
| `TELEGRAM_BOT_SECURITY.md` | Full documentation | Read for details |

---

## ⚠️ If Name Is Wrong Right Now

**Option 1: Quick Fix (30 seconds)**
```bash
# Set token
export TELEGRAM_BOT_TOKEN='your_token'

# Reset name
python reset_telegram_bot_name.py

# Verify in Telegram app
```

**Option 2: Full Setup (2 minutes)**
```bash
# Set token
export TELEGRAM_BOT_TOKEN='your_token'

# Run interactive setup
./setup_telegram_security.sh
```

---

## 🔒 For Production (Railway)

### Add Monitor to Startup

**Option A: Update Procfile**
```procfile
web: python wolf_app.py
monitor: python monitor_telegram_bot.py --continuous
```

**Option B: Add to wolf_app.py startup**
```python
# In wolf_app.py startup
import subprocess
subprocess.Popen(['python', 'monitor_telegram_bot.py', '--continuous'])
```

### Set Environment Variables
Railway dashboard → Variables:
```bash
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here  # For alerts
```

---

## 📊 Expected Output

### ✅ Good (Name Correct)
```
============================================================
📊 BOT NAME STATUS REPORT
============================================================
Bot ID:       123456789
Username:     @GhostProtocolBot
Current Name: Ghost Protocol Bot
Expected:     Ghost Protocol Bot
Status:       ✅ OK
============================================================
```

### ⚠️ Bad (Name Wrong)
```
============================================================
📊 BOT NAME STATUS REPORT
============================================================
Bot ID:       123456789
Username:     @GhostProtocolBot
Current Name: hacked by @mishadox
Expected:     Ghost Protocol Bot
Status:       ⚠️  MISMATCH

🚨 SUSPICIOUS NAME DETECTED!
   Run: python reset_telegram_bot_name.py
============================================================
```

---

## 🆘 Troubleshooting

### "TELEGRAM_BOT_TOKEN not found"
```bash
# Check if set
echo $TELEGRAM_BOT_TOKEN

# If empty, set it
export TELEGRAM_BOT_TOKEN='your_token'
```

### "Could not retrieve bot info"
- Check token is correct
- Check internet connection
- Verify token hasn't been revoked in @BotFather

### Monitor not alerting
- Check `TELEGRAM_CHAT_ID` is set
- Verify chat ID is correct (get from @userinfobot)
- Check logs: `cat logs/telegram_bot_monitor.log`

---

## 🔗 Related Files

- **Full docs:** `TELEGRAM_BOT_SECURITY.md`
- **Monitor logs:** `logs/telegram_bot_monitor.log`
- **State file:** `data/telegram_bot_state.json`
- **Test messaging:** `test_telegram_send.py`

---

## ✅ Checklist

- [ ] Set `TELEGRAM_BOT_TOKEN` environment variable
- [ ] Run `python monitor_telegram_bot.py` (single check)
- [ ] If name wrong: Run `python reset_telegram_bot_name.py`
- [ ] Verify in Telegram app
- [ ] Start monitor: `python monitor_telegram_bot.py --continuous &`
- [ ] Test messaging: `python test_telegram_send.py`
- [ ] Add monitor to Railway startup (optional)

---

**Everything working?** Bot name should be "Ghost Protocol Bot" in Telegram. 🎉
