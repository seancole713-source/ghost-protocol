# 🚀 GHOST v10.2 - Complete Startup Guide

## ⚡ FASTEST METHOD (One Command)

```bash
cd /workspaces/GHOST
./start_ghost_live.sh
```

**This automated script does everything:**

- ✅ Installs dependencies
- ✅ Validates environment
- ✅ Starts server on port 5000
- ✅ Runs health checks
- ✅ Verifies all endpoints
- ✅ Shows you URLs to access

______________________________________________________________________

## 🔧 Setup (First Time Only)

### 1. Edit your API keys in .env file:

```bash
nano .env
```

Add your real keys:

```env
ALPHAVANTAGE_API_KEY=your_real_key_here
POLYGON_API_KEY=your_polygon_key_here
TELEGRAM_BOT_TOKEN=optional_for_notifications
TELEGRAM_CHAT_ID=optional_for_notifications
```

**Get free Alpha Vantage key:** https://www.alphavantage.co/support/#api-key

______________________________________________________________________

## 📖 Manual Method (Step by Step)

If you prefer manual control:

### Step 1: Install Dependencies

```bash
cd /workspaces/GHOST
source .venv/bin/activate  # or: python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Set Environment

```bash
export SIM_MODE=0
export ALPHAVANTAGE_API_KEY="your_key"
export POLYGON_API_KEY="your_key"
export GHOST_API_TOKEN="strong_random_token"
export ALLOWED_ORIGINS="*"
export PRICE_TTL_OPEN_S=60
```

### Step 3: Start Server

```bash
# Kill any existing instances
pkill -f "wolf_app" || true

# Start server
.venv/bin/python wolf_app.py
```

### Step 4: Verify (in another terminal)

```bash
# Wait 3 seconds, then test
sleep 3

# Health check
curl -s http://127.0.0.1:5000/health | jq

# Provider check
curl -s http://127.0.0.1:5000/api/price/diagnostics | jq

# Cockpit check
curl -s http://127.0.0.1:5000/api/cockpit | jq '.snapshot_id, .prices'
```

### Step 5: Force Data Refresh

```bash
# Refresh prices and generate forecast
curl -s -X POST http://127.0.0.1:5000/api/advisor_refresh | jq

# Check forecast
curl -s "http://127.0.0.1:5000/forecast/48h/recent?symbol=WOLF&limit=3" | jq

# Check forecast metrics
curl -s "http://127.0.0.1:5000/forecast/48h/metrics?symbol=WOLF" | jq
```

### Step 6: Check Watchlist/Top Movers

```bash
# Scan watchlist
curl -s -X POST "http://127.0.0.1:5000/api/watchlist/scan?threshold=7.0&limit=30" | jq

# Get top movers
curl -s "http://127.0.0.1:5000/api/top_movers?threshold=7.0&limit=10" | jq
```

______________________________________________________________________

## 🌐 Access URLs

### Local Development:

- **UI:** http://127.0.0.1:5000
- **Health:** http://127.0.0.1:5000/health
- **Cockpit:** http://127.0.0.1:5000/api/cockpit
- **API Docs:** http://127.0.0.1:5000/docs
- **Metrics:** http://127.0.0.1:5000/metrics

### Production (Railway):

- **UI:** https://web-production-8e9a0.up.railway.app
- **Health:** https://web-production-8e9a0.up.railway.app/health

______________________________________________________________________

## 📊 Key Endpoints Reference

```bash
# System Health
GET  /health
GET  /api/system/diagnostics

# Price Data
GET  /api/price/diagnostics
GET  /api/watchlist/price?symbol=WOLF

# Cockpit (Main Dashboard Data)
GET  /api/cockpit
GET  /api/cockpit/stream  # SSE stream

# Forecasting
POST /api/advisor_refresh
GET  /forecast/48h/recent?symbol=WOLF&limit=10
GET  /forecast/48h/metrics?symbol=WOLF
GET  /forecast/two_line  # Overlay chart data

# Watchlist & Top Movers
POST /api/watchlist/scan?threshold=7.0&limit=30
GET  /api/top_movers?threshold=7.0&limit=10

# Portfolio
GET  /api/portfolio
POST /api/orders

# News
GET  /api/news?symbol=WOLF&limit=10
```

______________________________________________________________________

## 🛑 Stop/Restart Server

```bash
# Stop
pkill -f wolf_app.py

# Or if you have PID file
kill $(cat ghost_server.pid)

# Restart
./start_ghost_live.sh
```

______________________________________________________________________

## 🐛 Troubleshooting

### Problem: Port 5000 already in use

```bash
# Find what's using port 5000
lsof -i :5000

# Kill it
kill $(lsof -t -i:5000)

# Or kill all wolf_app processes
pkill -f wolf_app.py
```

### Problem: No price data showing

```bash
# Check API keys are configured
curl http://127.0.0.1:5000/api/system/diagnostics | jq '.providers'

# Should show:
# "alphavantage": true,
# "polygon": true (if configured)

# Check price diagnostics
curl http://127.0.0.1:5000/api/price/diagnostics | jq
```

### Problem: UI not updating

1. **Hard refresh browser:** `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
2. **Clear browser cache**
3. **Try incognito/private window**
4. **Check SSE connection:** Open DevTools → Network tab → Look for
   `/api/cockpit/stream`

### Problem: Server won't start

```bash
# Check logs
tail -50 ghost_server.log

# Check if dependencies are installed
pip list | grep -i "fastapi\|uvicorn\|sqlalchemy"

# Reinstall dependencies
pip install -r requirements.txt
```

### Problem: Forecasts not generating

```bash
# Manually trigger refresh
curl -X POST http://127.0.0.1:5000/api/advisor_refresh

# Check if forecast data exists
curl "http://127.0.0.1:5000/forecast/48h/recent?symbol=WOLF&limit=1"

# Check database
ls -lh data/wolf.db
```

______________________________________________________________________

## 🐳 Docker Alternative (Optional)

If you prefer Docker:

```bash
# Build image
docker build -t ghost:v10.2 .

# Run container
docker run -d \
  --name ghost \
  -p 5000:5000 \
  -e ALPHAVANTAGE_API_KEY="your_key" \
  -e POLYGON_API_KEY="your_key" \
  -v $(pwd)/data:/app/data \
  ghost:v10.2

# Check logs
docker logs -f ghost

# Stop
docker stop ghost && docker rm ghost
```

______________________________________________________________________

## 📦 Git Tagging (Version Release)

```bash
# Pull latest
git pull --rebase origin main

# Create tag
git tag v10.2.0

# Push tag to remote
git push origin v10.2.0

# View all tags
git tag -l
```

______________________________________________________________________

## ☁️ GitHub Codespaces Tips

If running in Codespaces:

1. **Start server:** `./start_ghost_live.sh`
2. **Go to PORTS tab** (bottom panel)
3. **Find port 5000**
4. **Right-click → Port Visibility → Public**
5. **Click 🌐 globe icon** to open in browser

______________________________________________________________________

## 🎯 Quick Verification Tests

Run these after startup to verify everything works:

```bash
#!/bin/bash
echo "🔍 Running GHOST verification tests..."

# 1. Health check
echo "1. Health check..."
curl -sf http://127.0.0.1:5000/health > /dev/null && echo "✅ Health: OK" || echo "❌ Health: FAIL"

# 2. Price provider
echo "2. Price provider..."
PROVIDER=$(curl -s http://127.0.0.1:5000/api/price/diagnostics | jq -r '.provider')
echo "   Provider: $PROVIDER"

# 3. Cockpit
echo "3. Cockpit..."
curl -sf http://127.0.0.1:5000/api/cockpit > /dev/null && echo "✅ Cockpit: OK" || echo "❌ Cockpit: FAIL"

# 4. Forecast
echo "4. Forecast..."
FORECAST_COUNT=$(curl -s "http://127.0.0.1:5000/forecast/48h/recent?symbol=WOLF&limit=1" | jq '. | length')
echo "   Forecast rows: $FORECAST_COUNT"

echo "✅ Verification complete!"
```

______________________________________________________________________

## 📞 Support

**Issues?** Check:

1. **Logs:** `tail -f ghost_server.log`
2. **Database:** `ls -lh data/`
3. **Network:** `netstat -tlnp | grep 5000`
4. **Environment:** `env | grep -E "(ALPHA|POLYGON|GHOST)"`

______________________________________________________________________

**Version:** 10.2.0\
**Last Updated:** October 10, 2025\
**Railway URL:** https://web-production-8e9a0.up.railway.app
