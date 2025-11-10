# STAGE 2 ENHANCEMENTS: UI & AUTO-TUNING

**Date**: October 5, 2025\
**Status**: Ready to implement after server verification\
**Prerequisites**: Server running with Stage 1 & Stage 2 enabled

______________________________________________________________________

## 🔍 VERIFICATION CHECKLIST (Do First)

### 1. Server Restart & Health Check

```bash
# Start server
cd /workspaces/GHOST
source .venv/bin/activate
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload
```

**Look for in logs:**

- ✅ `stage1_initialized` - Context Awareness layer active
- ✅ `stage2_initialized` - Self-Evaluation system active

### 2. Stage 1 Cron Verification

```bash
# Check background updater status
curl http://localhost:5000/api/stage1/stats | jq
```

**Expected fields:**

```json
{
  "last_update": "2025-10-05T14:30:00Z",
  "article_count": 47,
  "next_refresh_in": "4m 23s",  // ← CRITICAL: Confirms cron is running
  "update_interval": "5 minutes"
}
```

**If `next_refresh_in` is missing:**

- Wait 5 minutes for first update cycle
- Check logs for `stage1_context_updated` message
- Verify RSS feeds accessible (outbound HTTPS)

### 3. Environment Verification

```bash
# Check container time
date

# Test outbound HTTPS
curl -I https://www.reuters.com

# Verify Python environment
python --version
pip list | grep -E "feedparser|yfinance|vaderSentiment"
```

**Expected:**

- Container clock matches real time (±1 min tolerance)
- HTTPS connections succeed (RSS feeds need this)
- All Stage 1 dependencies installed

______________________________________________________________________

## 🎯 STAGE 2 ENHANCEMENTS (Next Implementation)

Once verification passes, implement these 3 features:

______________________________________________________________________

## 1. DAILY ACCURACY LEDGER

**Goal**: Show forecast accuracy history in UI

### Backend (Already Complete ✅)

- `core/accuracy_tracker.py` - Stores forecasts in SQLite
- API: `GET /api/stage2/forecasts?limit=30`

### Frontend (To Implement)

**Location**: `templates/cockpit.html`

**Add new section** (after Stage 1 widget):

```html
<section class="card" style="grid-column: span 8;">
  <div class="section-title">
    <h2>📊 Daily Accuracy Ledger</h2>
    <button id="btnAccuracyRefresh">Refresh</button>
  </div>
  
  <div id="accuracyLedger">
    <table style="width: 100%; border-collapse: collapse;">
      <thead>
        <tr style="border-bottom: 1px solid var(--border);">
          <th style="text-align: left; padding: 8px;">Date</th>
          <th style="text-align: left;">Symbol</th>
          <th style="text-align: right;">Forecast</th>
          <th style="text-align: right;">Actual</th>
          <th style="text-align: right;">Delta</th>
          <th style="text-align: center;">Result</th>
        </tr>
      </thead>
      <tbody id="ledgerBody">
        <!-- Populated by JavaScript -->
      </tbody>
    </table>
  </div>
  
  <div id="accuracySummary" style="margin-top: 16px; display: flex; gap: 16px;">
    <!-- Summary stats populated by JS -->
  </div>
</section>
```

**JavaScript function**:

```javascript
async function loadAccuracyLedger() {
  const btnId = 'btnAccuracyRefresh';
  setButtonState(btnId, 'loading');
  
  try {
    // Fetch recent forecasts
    const resp = await fetch('/api/stage2/forecasts?limit=30');
    const data = await resp.json();
    const forecasts = data.forecasts || [];
    
    const tbody = document.getElementById('ledgerBody');
    tbody.innerHTML = '';
    
    if (forecasts.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 16px;">No forecasts yet. Start recording predictions!</td></tr>';
      setButtonState(btnId, 'success');
      return;
    }
    
    // Render each forecast
    forecasts.forEach(f => {
      const row = document.createElement('tr');
      row.style.borderBottom = '1px solid var(--border)';
      
      // Date
      const date = new Date(f.timestamp * 1000).toLocaleDateString();
      
      // Delta calculation
      const hasPrediction = f.actual_price !== null;
      const delta = hasPrediction ? (f.actual_price - f.forecast_price) : null;
      const deltaPct = hasPrediction ? (delta / f.forecast_price * 100) : null;
      
      // Result indicator
      let resultIcon = '⏳';
      let resultColor = 'var(--text)';
      if (hasPrediction) {
        const errorPct = Math.abs(deltaPct);
        if (errorPct < 2) {
          resultIcon = '✅';
          resultColor = 'var(--good)';
        } else if (errorPct < 5) {
          resultIcon = '⚠️';
          resultColor = 'var(--warn)';
        } else {
          resultIcon = '❌';
          resultColor = 'var(--bad)';
        }
      }
      
      row.innerHTML = `
        <td style="padding: 8px;">${date}</td>
        <td>${f.symbol}</td>
        <td style="text-align: right;">$${f.forecast_price.toFixed(2)}</td>
        <td style="text-align: right;">${hasPrediction ? '$' + f.actual_price.toFixed(2) : '—'}</td>
        <td style="text-align: right; color: ${delta ? (delta > 0 ? 'var(--good)' : 'var(--bad)') : 'inherit'};">
          ${hasPrediction ? (delta > 0 ? '+' : '') + delta.toFixed(2) + ' (' + deltaPct.toFixed(1) + '%)' : '—'}
        </td>
        <td style="text-align: center; font-size: 20px; color: ${resultColor};">${resultIcon}</td>
      `;
      
      tbody.appendChild(row);
    });
    
    // Summary stats
    const completed = forecasts.filter(f => f.actual_price !== null);
    const correct = completed.filter(f => Math.abs((f.actual_price - f.forecast_price) / f.forecast_price * 100) < 2).length;
    const warning = completed.filter(f => {
      const err = Math.abs((f.actual_price - f.forecast_price) / f.forecast_price * 100);
      return err >= 2 && err < 5;
    }).length;
    const wrong = completed.filter(f => Math.abs((f.actual_price - f.forecast_price) / f.forecast_price * 100) >= 5).length;
    
    const summary = document.getElementById('accuracySummary');
    summary.innerHTML = `
      <div style="flex: 1; padding: 12px; background: var(--bg-alt); border-radius: 4px;">
        <div style="font-size: 24px; color: var(--good);">✅ ${correct}</div>
        <div style="font-size: 12px; opacity: 0.7;">Accurate (<2%)</div>
      </div>
      <div style="flex: 1; padding: 12px; background: var(--bg-alt); border-radius: 4px;">
        <div style="font-size: 24px; color: var(--warn);">⚠️ ${warning}</div>
        <div style="font-size: 12px; opacity: 0.7;">Fair (2-5%)</div>
      </div>
      <div style="flex: 1; padding: 12px; background: var(--bg-alt); border-radius: 4px;">
        <div style="font-size: 24px; color: var(--bad);">❌ ${wrong}</div>
        <div style="font-size: 12px; opacity: 0.7;">Poor (>5%)</div>
      </div>
      <div style="flex: 1; padding: 12px; background: var(--bg-alt); border-radius: 4px;">
        <div style="font-size: 24px;">⏳ ${forecasts.length - completed.length}</div>
        <div style="font-size: 12px; opacity: 0.7;">Pending</div>
      </div>
    `;
    
    setButtonState(btnId, 'success');
  } catch (e) {
    console.error('Failed to load accuracy ledger:', e);
    setButtonState(btnId, 'error');
  }
}
```

**Wire up** (in DOMContentLoaded):

```javascript
// Initial load
loadAccuracyLedger();

// Button
const btnAcc = document.getElementById('btnAccuracyRefresh');
if (btnAcc) btnAcc.onclick = loadAccuracyLedger;

// Auto-refresh every 10 minutes
setInterval(loadAccuracyLedger, 600000);
```

______________________________________________________________________

## 2. AUTO WEIGHT-TUNING

**Goal**: Automatically adjust model parameters based on accuracy

### Backend (Already Complete ✅)

- `core/learning_loop.py` - Analyzes MAP and adjusts config
- API: `POST /api/stage2/tune`

### Automation Options

#### Option A: Scheduled Background Task (Recommended)

Add to `wolf_app.py` startup:

```python
# In startup event handler
if STAGE2_ENABLED:
    # Schedule daily tuning check at 6 PM
    import asyncio
    
    async def daily_tuning_check():
        while True:
            # Wait until 6 PM
            now = datetime.now()
            target = now.replace(hour=18, minute=0, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            wait_seconds = (target - now).total_seconds()
            
            await asyncio.sleep(wait_seconds)
            
            # Run learning cycle
            try:
                from core.learning_loop import run_learning_cycle
                result = run_learning_cycle(symbol='WOLF', days=7, auto_apply=True)
                if result['adjustments_made']:
                    LOGGER.info("auto_tuning_complete", extra={
                        "summary": result['summary'],
                        "changes": len(result['adjustments']['changes'])
                    })
            except Exception as e:
                LOGGER.exception("auto_tuning_failed", extra={"error": str(e)})
    
    # Start background task
    asyncio.create_task(daily_tuning_check())
```

#### Option B: Manual Trigger via UI

Add button to cockpit:

```html
<button id="btnRunTuning" style="margin-top: 16px;">
  🔄 Run Auto-Tuning Check
</button>
```

```javascript
async function runTuning() {
  const btn = document.getElementById('btnRunTuning');
  btn.disabled = true;
  btn.textContent = '⏳ Checking...';
  
  try {
    const resp = await fetch('/api/stage2/tune', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({symbol: 'WOLF', days: 7, auto_apply: true})
    });
    const result = await resp.json();
    
    if (result.tuning_needed && result.adjustments_made) {
      btn.textContent = '✅ Tuned!';
      btn.style.background = 'var(--good)';
      alert(`Auto-tuning complete!\n\n${result.summary}\n\nChanges:\n${result.adjustments.changes.map(c => `• ${c.parameter}: ${c.old_value} → ${c.new_value}`).join('\n')}`);
    } else if (result.tuning_needed) {
      btn.textContent = '⚠️ Needs Review';
      alert('Tuning needed but no clear adjustments. Review metrics manually.');
    } else {
      btn.textContent = '✅ Performance OK';
      btn.style.background = 'var(--good)';
      alert(`Performance is good!\n\nMAPE: ${result.performance.metrics.map}% (threshold: 5%)`);
    }
    
    setTimeout(() => {
      btn.disabled = false;
      btn.textContent = '🔄 Run Auto-Tuning Check';
      btn.style.background = '';
    }, 3000);
  } catch (e) {
    btn.textContent = '❌ Failed';
    btn.disabled = false;
    console.error('Tuning failed:', e);
  }
}

document.getElementById('btnRunTuning').onclick = runTuning;
```

#### Option C: Cron Job (External)

```bash
#!/bin/bash
# /etc/cron.d/ghost-autotuning
# Run daily at 6 PM

0 18 * * * cd /workspaces/GHOST && curl -X POST http://localhost:5000/api/stage2/tune -H "Content-Type: application/json" -d '{"symbol": "WOLF", "days": 7, "auto_apply": true}' >> /var/log/ghost-tuning.log 2>&1
```

______________________________________________________________________

## 3. UI CHIPS: RIGHT/WRONG & DELTA

**Goal**: Visual indicators showing forecast accuracy inline

### Implementation: Badge System

Add CSS for chips:

```css
/* Add to ghost.css */
.accuracy-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  line-height: 1;
}

.accuracy-chip.correct {
  background: rgba(0, 255, 0, 0.1);
  color: var(--good);
  border: 1px solid var(--good);
}

.accuracy-chip.warning {
  background: rgba(255, 255, 0, 0.1);
  color: var(--warn);
  border: 1px solid var(--warn);
}

.accuracy-chip.wrong {
  background: rgba(255, 0, 0, 0.1);
  color: var(--bad);
  border: 1px solid var(--bad);
}

.accuracy-chip.pending {
  background: rgba(128, 128, 128, 0.1);
  color: var(--text);
  border: 1px solid var(--border);
  opacity: 0.7;
}
```

### Usage in Forecast Cards

Modify any forecast display to include chips:

```javascript
function renderForecastWithAccuracy(forecast) {
  const hasPrediction = forecast.actual_price !== null;
  const errorPct = hasPrediction ? Math.abs((forecast.actual_price - forecast.forecast_price) / forecast.forecast_price * 100) : null;
  
  let chipClass = 'pending';
  let chipIcon = '⏳';
  let chipText = 'Pending';
  
  if (hasPrediction) {
    if (errorPct < 2) {
      chipClass = 'correct';
      chipIcon = '✅';
      chipText = `Accurate (${errorPct.toFixed(1)}%)`;
    } else if (errorPct < 5) {
      chipClass = 'warning';
      chipIcon = '⚠️';
      chipText = `Fair (${errorPct.toFixed(1)}%)`;
    } else {
      chipClass = 'wrong';
      chipIcon = '❌';
      chipText = `Off (${errorPct.toFixed(1)}%)`;
    }
  }
  
  return `
    <div class="forecast-card">
      <div class="forecast-header">
        <span>${forecast.symbol}</span>
        <span class="accuracy-chip ${chipClass}">
          ${chipIcon} ${chipText}
        </span>
      </div>
      <div class="forecast-body">
        <div>Forecast: $${forecast.forecast_price.toFixed(2)}</div>
        ${hasPrediction ? `<div>Actual: $${forecast.actual_price.toFixed(2)}</div>` : ''}
        ${hasPrediction ? `<div>Delta: ${(forecast.actual_price - forecast.forecast_price).toFixed(2)}</div>` : ''}
      </div>
    </div>
  `;
}
```

______________________________________________________________________

## 🎯 IMPLEMENTATION ORDER

**After server verification passes:**

1. **Daily Accuracy Ledger** (~45 min)

   - Add HTML table section
   - Implement `loadAccuracyLedger()` JavaScript
   - Wire up refresh button + timer
   - Test with mock data

2. **UI Chips** (~30 min)

   - Add CSS for accuracy chips
   - Update forecast rendering functions
   - Test different accuracy states (✅⚠️❌⏳)

3. **Auto Weight-Tuning** (~45 min)

   - Choose automation approach (Option A/B/C)
   - Implement selected method
   - Test tuning cycle manually
   - Monitor logs for auto-tuning events

**Total Time**: ~2 hours **Result**: Complete Stage 2 UI + automation

______________________________________________________________________

## 📊 TESTING CHECKLIST

### Daily Accuracy Ledger

- [ ] Table displays when no forecasts exist
- [ ] Table shows all forecasts correctly
- [ ] Delta calculations are accurate
- [ ] Colors match accuracy levels (green/yellow/red)
- [ ] Summary stats update correctly
- [ ] Refresh button works
- [ ] Auto-refresh every 10 minutes

### Auto Weight-Tuning

- [ ] Manual trigger button works
- [ ] Tuning runs when MAP > 5%
- [ ] Parameters update in model_memory.json
- [ ] Logs show tuning events
- [ ] No tuning when MAP < 5%
- [ ] Background task (if used) runs daily

### UI Chips

- [ ] ✅ Green chip for \<2% error
- [ ] ⚠️ Yellow chip for 2-5% error
- [ ] ❌ Red chip for >5% error
- [ ] ⏳ Gray chip for pending forecasts
- [ ] Delta displays correctly (+/- and %)
- [ ] Chips render inline without layout issues

______________________________________________________________________

## 🚀 QUICK START (After Verification)

```bash
# 1. Verify server is healthy
curl http://localhost:5000/api/stage1/stats | jq '.next_refresh_in'

# 2. Record a test forecast
curl -X POST http://localhost:5000/api/stage2/forecasts \
  -H "Content-Type: application/json" \
  -d '{"symbol": "WOLF", "forecast_price": 8.50, "forecast_horizon_hours": 24, "confidence": 0.85}'

# 3. Check it appears in ledger
curl http://localhost:5000/api/stage2/forecasts | jq

# 4. Open cockpit and verify UI
open http://localhost:5000/cockpit
```

______________________________________________________________________

**Next**: Let's verify the server is running correctly, then implement these
enhancements! 🚀
