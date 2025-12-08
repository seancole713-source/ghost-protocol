# 🎯 GHOST PROTOCOL FULL SYSTEM AUDIT - PHASE 2B FINDINGS

**Date:**October 11, 2025\**Phase:**Deep Code Scan (Post-Cleanup)\**Status:**🔍 VIOLATIONS IDENTIFIED

______________________________________________________________________

## 📊**AUDIT SUMMARY**| Category | Count | Severity | Action Required |

|----------|-------|----------|-----------------| |**Hardcoded Test Data in UI**| 3
instances | 🟡 HIGH | Replace with real API calls | |**Naming Inconsistencies**| 3 env
vars | 🟠 MEDIUM | Standardize names | |**Legacy Functions Still Active**| 1 function |
🟠 MEDIUM | Document or refactor | |**UI Comments Misleading**| 4 comments | 🟢 LOW |
Update documentation |

______________________________________________________________________

## 🟡**HIGH PRIORITY: Hardcoded Test Data in Production UI**###**Issue 1: Portfolio Optimization Uses Fake Data**

**File:**`/workspaces/GHOST/templates/cockpit.html` (lines 1336-1350)**Problem:**```javascript
// Sample data for demo (in production, fetch real portfolio data)
const sampleAssets = ['WOLF', 'SPY', 'TLT'];
const sampleReturns = {
    'WOLF': [0.01, 0.02, -0.01, 0.015, 0.005],  // ← HARDCODED
    'SPY': [0.008, 0.006, 0.012, -0.003, 0.007],
    'TLT': [-0.002, 0.003, 0.001, 0.004, -0.001]
};

// Sends fake data to API
const optResult = await fetch('/api/stage4/portfolio/optimize', {
    method: 'POST',
    body: JSON.stringify({
        assets: sampleAssets,
        returns_data: sampleReturns  // ← FAKE DATA
    })
})

```text**Impact:**- UI shows optimization results based on**fake returns**- User sees invalid portfolio allocation recommendations

- Sharpe ratio, volatility, expected return all calculated from test data**Fix:**```javascript


// Fetch real portfolio data from Ghost state
async function loadPortfolioOptimization() {
    const btnId = 'btnPortfolioRefresh';
    try {
        setButtonState(btnId, 'loading');

        // Get real portfolio data
        const portfolioData = await fetch('/api/portfolio/data', {
            headers: authHeaders()
        }).then(r => r.json());

        if (!portfolioData || !portfolioData.assets || portfolioData.assets.length === 0) {
            el('portfolioAllocation').innerHTML = '<div class="small muted">No portfolio data available</div>';
            setButtonState(btnId, 'error');
            return;
        }

        // Optimize with real data
        const optResult = await fetch('/api/stage4/portfolio/optimize', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(portfolioData)  // ← REAL DATA
        }).then(r => r.json());

        // ... rest of code
    }
}

```text

______________________________________________________________________

###**Issue 2: Beta Hedge Uses Fake Returns**

**File:**`/workspaces/GHOST/templates/cockpit.html` (lines 1373-1383)**Problem:**```javascript

// Beta hedge (sample)
const hedgeResult = await fetch('/api/stage4/hedging/beta-hedge', {
    method: 'POST',
    body: JSON.stringify({
        portfolio_symbol: 'WOLF',
        portfolio_returns: [0.01, 0.02, -0.01, 0.015, 0.005],  // ← FAKE
        market_returns: [0.008, 0.006, 0.012, -0.003, 0.007],  // ← FAKE
        hedge_symbol: 'SPY'
    })
})

```text**Impact:**- Beta calculation is based on test data, not actual price history

- Hedge recommendations are invalid
- User might make real trades based on fake analysis**Fix:**```javascript


// Calculate beta hedge from real historical data
async function loadBetaHedge() {
    try {
        // Fetch real WOLF and SPY returns from price history
        const wolfHist = await fetch('/api/prices/history?symbol=WOLF&days=90').then(r => r.json());
        const spyHist = await fetch('/api/prices/history?symbol=SPY&days=90').then(r => r.json());

        if (!wolfHist.returns || !spyHist.returns) {
            el('hedgingSuggestion').innerHTML = '<div class="small muted">Insufficient data</div>';
            return;
        }

        const hedgeResult = await fetch('/api/stage4/hedging/beta-hedge', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                portfolio_symbol: 'WOLF',
                portfolio_returns: wolfHist.returns,  // ← REAL DATA
                market_returns: spyHist.returns,      // ← REAL DATA
                hedge_symbol: 'SPY'
            })
        }).then(r => r.json());

        // ... display results
    }
}

```text

______________________________________________________________________

###**Issue 3: Backtest Uses Hardcoded Returns**

**File:**`/workspaces/GHOST/templates/cockpit.html` (lines 1394-1402)**Problem:**```javascript

// Backtest (sample)
const backtestResult = await fetch('/api/stage4/backtest/run', {
    method: 'POST',
    body: JSON.stringify({
        strategy_name: 'Ghost Strategy',
        returns: [0.01, 0.02, -0.01, 0.015, 0.005, 0.008, -0.002],  // ← FAKE
        // ...
    })
})

```text**Impact:**- Backtest results are completely fabricated

- Performance metrics (Sharpe, drawdown) are meaningless
- User cannot trust system validation**Fix:**```javascript


// Run backtest with real historical trades
async function loadBacktest() {
    try {
        // Fetch real decision history from AI memory
        const decisions = await fetch('/api/ai/memory/history?days=90', {
            headers: authHeaders()
        }).then(r => r.json());

        if (!decisions || decisions.length === 0) {
            el('backtestResults').innerHTML = '<div class="small muted">No decision history available</div>';
            return;
        }

        // Run backtest with actual decisions and outcomes
        const backtestResult = await fetch('/api/stage4/backtest/run', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                strategy_name: 'Ghost Strategy',
                decisions: decisions,  // ← REAL DECISIONS
                verify_outcomes: true
            })
        }).then(r => r.json());

        // ... display results
    }
}

```text

______________________________________________________________________

## 🟠**MEDIUM PRIORITY: Naming Inconsistencies**###**Issue 4: AI_ON vs AGENTS_ENABLED Duality**

**File:**`/workspaces/GHOST/wolf_app.py` (line 3424)**Problem:**```python

AI_ON = int(os.getenv("AI_ON", os.getenv("AGENTS_ENABLED", "0")))

```text**Impact:**- Two names for the same config (confusing)

- User doesn't know which one to set
- Documentation uses both names**Recommendation:**```python


# OPTION A: Deprecate AI_ON, use AGENTS_ENABLED only

AGENTS_ENABLED = int(os.getenv("AGENTS_ENABLED", "0"))
AI_ON = AGENTS_ENABLED  # Backwards compatibility alias (deprecated)

# OPTION B: Deprecate AGENTS_ENABLED, use AI_ON only

AI_ON = int(os.getenv("AI_ON", "0"))

# Remove AGENTS_ENABLED fallback

```text**Also affects:**- `llm/agent.py` (uses AGENTS_ENABLED)

- `secrets.env` (documents AGENTS_ENABLED)
- Railway env vars (user has AGENTS_ENABLED set)**Recommended action:** **Standardize on `AGENTS_ENABLED`**(more descriptive)


______________________________________________________________________

###**Issue 5: AI_MODEL vs AGENT_MODEL Duality**

**File:**`/workspaces/GHOST/wolf_app.py` (line 3425-3426)**Problem:**```python

AI_MODEL = os.getenv("AI_MODEL", os.getenv("AGENT_MODEL", "llama3.1:8b"))

```text**Impact:**- Same as AI_ON issue - two names, user confusion**Recommendation:**```python

# Standardize on AGENT_MODEL (matches AGENTS_ENABLED naming pattern)

AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
AI_MODEL = AGENT_MODEL  # Backwards compatibility alias (deprecated)

```text

______________________________________________________________________

###**Issue 6: OPENAI_API_KEY vs OPENAI_AGENT_API_KEY**

**Files:**Multiple**Problem:**- `OPENAI_API_KEY` = General purpose OpenAI calls

- `OPENAI_AGENT_API_KEY` = Dedicated agent key (optional)
- Code prefers `OPENAI_AGENT_API_KEY` if set, falls back to `OPENAI_API_KEY`**Impact:**- Not clear when to use which
- User might set one but not the other**Current behavior (CORRECT):**```python


OPENAI_API_KEY = (
    os.getenv("OPENAI_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY", "")
).strip()

```text**Recommendation:** **Keep as-is**but document clearly:

- Set `OPENAI_API_KEY` for all OpenAI calls (required)
- Set `OPENAI_AGENT_API_KEY` only if you want separate key for agent (optional)


______________________________________________________________________

## 🟢**LOW PRIORITY: Legacy Code Still Active**###**Issue 7: \_legacy_snapshot_to_decision() Still Used**

**File:**`/workspaces/GHOST/wolf_app.py` (line 1624)**Problem:**```python

def _legacy_snapshot_to_decision(row: tuple[Any, ...]) -> dict[str, Any]:
    """Convert legacy AI snapshot row to modern AIMemory decision format."""

    

```text**Impact:**- Function is used for one-time migration from old DB format

- Still in main app code instead of migration script
- Runs on every startup (line 2903: `_migrate_legacy_ai_memory()`)**Current behavior:**- Checks if old DB exists
- If yes, migrates data once
- Safe to keep (doesn't break anything)**Recommendation:**```python


# OPTION A: Move to scripts/one_time_migrations/

# Keep in codebase for historical migrations

# Add comment: "Legacy migration - kept for backwards compatibility"

# OPTION B: Remove after confirming all users have migrated

# Check if AI_LEGACY_DB_PATH exists for any production deployments

# If not, safe to delete

# OPTION C (CURRENT - ACCEPTABLE)

# Leave as-is, it's harmless and handles edge case of old DB

```text**Action:** **Leave as-is**(it's defensive code, doesn't hurt)

______________________________________________________________________

## 📋**DETAILED VIOLATION CHECKLIST**###**UI/API Mismatches (MUST FIX)**| File:Line | Violation | Fix Priority | Estimated Time |

|-----------|-----------|--------------|----------------| |
`templates/cockpit.html:1336-1350` | Portfolio optimization uses fake returns | 🔴 HIGH |
30 min | | `templates/cockpit.html:1373-1383` | Beta hedge uses fake returns | 🔴 HIGH |
20 min | | `templates/cockpit.html:1394-1402` | Backtest uses fake returns | 🔴 HIGH | 20
min |**Total UI fixes:**~70 minutes of work

______________________________________________________________________

###**Naming Standardization (SHOULD FIX)**| Variable Pair | Recommended Standard | Fix Priority | Affected Files |

|---------------|---------------------|--------------|----------------| | `AI_ON` vs
`AGENTS_ENABLED` | Use `AGENTS_ENABLED` | 🟠 MEDIUM | 3 files | | `AI_MODEL` vs
`AGENT_MODEL` | Use `AGENT_MODEL` | 🟠 MEDIUM | 3 files | | `OPENAI_API_KEY` vs
`OPENAI_AGENT_API_KEY` | Keep both (documented) | 🟢 LOW | Documentation only |**Total naming fixes:**~40 minutes of work

______________________________________________________________________

###**Legacy Code (OPTIONAL)**| Function/Code | Action | Priority | Notes |

|---------------|--------|----------|-------| | `_legacy_snapshot_to_decision()` | Keep
| 🟢 LOW | Handles old DB migration safely | | `AI_LEGACY_DB_PATH` | Keep | 🟢 LOW | Used
for migration check | | `_migrate_legacy_ai_memory()` | Keep | 🟢 LOW | Runs once, no
performance impact |**Total legacy cleanup:**0 minutes (keep as-is)

______________________________________________________________________

## 🎯**PHASE 2B COMPLETION STATUS**###**Violations Found:**- ✅**3 UI hardcoded test data issues**(HIGH)

- ✅**3 env var naming inconsistencies**(MEDIUM)
- ✅**1 legacy function**(LOW - acceptable)


###**Violations Fixed (Phase 2A):**- ✅ 7 placeholder comments removed

- ✅ 2 fake endpoints converted to honest 501 errors
- ✅ 3 deployment scripts secured (no hardcoded keys)
- ✅ 3 legacy files archived


###**Total Audit Progress:**-**Phase 1:**Security scan complete (API keys identified)

-**Phase 2A:**Placeholder cleanup complete ✅
-**Phase 2B:**Deep code audit complete ✅
-**Phase 3:**UI/API fixes (IN QUEUE)
-**Phase 4:**Naming standardization (IN QUEUE)
-**Phase 5:**Final validation (PENDING)


______________________________________________________________________

## 🚀**NEXT STEPS**###**Option 1: Fix UI Test Data Now (Recommended)**Apply fixes to `templates/cockpit.html` to replace fake data with real API calls

###**Option 2: Standardize Naming Now**Update env var names to use `AGENTS_ENABLED` and `AGENT_MODEL` consistently

###**Option 3: Continue Audit**Scan remaining areas

- JSON config files
- Test files for hardcoded mock data
- Documentation for outdated examples


###**Option 4: Generate Final Report**Create comprehensive `GHOST_FULL_AUDIT_REPORT.md` with all findings

______________________________________________________________________**User, what would you like to do next?**

1. "Fix UI test data" (Apply Option 1)
2. "Standardize naming" (Apply Option 2)
3. "Continue audit" (Scan remaining files)
4. "Generate final report" (Complete audit)
5. "Show me the UI fix code first" (Review before applying)
