# GHOST PROTOCOL — MASTER TODO LIST
# Goal: Take Ghost from 5.5/10 to 10/10
# Created: 2026-03-21 (Session 5)
# Rule: Check off each item ONLY after tested and verified in production

---

## PHASE 1: AI BRAIN & ACCURACY (Current: 41% → Target: 60%+)
> This is the #1 priority. Nothing else matters if the brain can't predict.

**🔍 ROOT CAUSE DISCOVERED (2026-03-21)**: Performance Gate Death Spiral
- Performance Gate kill threshold was 45% when overall accuracy is 41%
- This killed ALL symbols except LINK → no predictions for 256 minutes  
- Death spiral: No predictions → no learning → accuracy can't improve
- **FIX APPLIED**: Lowered kill threshold 45% → 25% (only kill worse-than-random)
- Predictions should resume within 60 minutes of deploy

- [ ] 1.1 Retrain XGBoost model with updated feature engineering (current: 59 features, 84.8% training accuracy but 41% live)
- [ ] 1.2 Add walk-forward validation to prevent overfitting (training vs live gap is massive)
- [ ] 1.3 Implement proper backtesting framework before any model goes live
- [ ] 1.4 Fix Learning Brain — currently "0 symbols inverted", self-correction loop is not learning
- [ ] 1.5 Fix AI Memory — "0 entries in ring", long-term memory store is empty and not accumulating
- [ ] 1.6 Remove or retrain worst symbols: PANW (5%), DDOG (17%), NET (20%), XPO (23%)
- [ ] 1.7 Tune Confidence Calibrator — avg confidence is 67.9% but win rate is 41%, calibration is way off
- [ ] 1.8 Tune Quality Gate — it should be blocking bad predictions, not letting 41% accuracy through
- [ ] 1.9 Implement dynamic position sizing based on confidence (high confidence = bigger bets)
- [ ] 1.10 Add regime-aware predictions — Regime Detector exists but isn't influencing predictions

## PHASE 2: DATA FEEDS & NEWS (Current: Broken → Target: Fully Operational)
> Ghost needs real-time data to make real-time decisions.

- [ ] 2.1 Fix price feeds — "1/2 feeds responding, partial outage" on health check
- [ ] 2.2 Fix News Brain pipeline — News page shows "No news articles", cache exists but no output
- [ ] 2.3 Wire News Brain sentiment into prediction logic (currently disconnected)
- [ ] 2.4 Add more price feed providers for redundancy (currently alphavantage + polygon, both with failures)
- [ ] 2.5 Fix ALPACA_API_KEY missing warning in integrity audit
- [ ] 2.6 Ensure World Feed Fusion is actually fusing news + sentiment into predictions
- [ ] 2.7 Add market hours awareness — stocks should not predict during off-hours

## PHASE 3: DISPLAY & COCKPIT UI (Current: 6/10 → Target: 10/10)
> The dashboard should look professional and show all data correctly.

- [x] 3.1 Fix P&L chart on Financials page — chart area is blank, API has data (api/v4/history has pnl field) — **FIXED 2026-03-21**
- [x] 3.2 Fix Stocks page — only 3 symbols (AAPL, NVDA, WOLF), all showing HOLD with "--" confidence — **FIXED 2026-03-21**
- [x] 3.3 Fix Crypto page — only 3 symbols (BTC, ETH, SOL), all showing HOLD with "--" confidence — **FIXED 2026-03-21**
- [ ] 3.4 Add confidence scores to watchlist display (currently all show "--")
- [ ] 3.5 Add actual price change data to watchlist (currently all +0.00% — market is closed but should show last close)
- [ ] 3.6 Populate News page with actual articles from News Brain
- [ ] 3.7 Add win/loss streaks and trends to History page
- [ ] 3.8 Add accuracy charts (daily/weekly/monthly trend lines)
- [ ] 3.9 Make pick cards show time remaining until "Done by" date
- [ ] 3.10 Add mobile-responsive layout (cockpit is currently desktop-only)

## PHASE 4: PREDICTION ENGINE HARDENING
> Make predictions reliable, not just frequent.

- [ ] 4.1 Fix prediction staleness — predictions go stale after deploy (214 min gap currently)
- [ ] 4.2 Implement prediction scheduling — ensure cycle runs on consistent intervals
- [ ] 4.3 Add prediction diversity — currently heavy on crypto DOWN predictions
- [ ] 4.4 Validate entry/exit/stop-loss prices are realistic (some show 3% target with 6% stop = bad risk/reward)
- [ ] 4.5 Paper trading win rate should match live accuracy (942 trades tracked, verify consistency)
- [ ] 4.6 Implement A/B testing framework for model changes (test new model alongside old before switching)
- [ ] 4.7 Add prediction explanations — why did Ghost pick this direction?

## PHASE 5: INFRASTRUCTURE & RELIABILITY
> The foundation needs to be rock-solid.

- [ ] 5.1 Fix 3 startup errors: api.cockpit_v2_endpoints, core.position_manager, stage2_init_failed
- [ ] 5.2 Fix Telegram integration — status "never_run", alerts not sending
- [ ] 5.3 Add proper logging dashboard (currently log level set to reduce noise but no structured logging)
- [ ] 5.4 Implement proper error alerting (Telegram or webhook when health drops below 90)
- [ ] 5.5 Add database backup strategy for PostgreSQL predictions table (1481 predictions = valuable data)
- [ ] 5.6 Fix duplicate predictions check — 0 found but verify logic is working
- [ ] 5.7 Reduce memory usage if needed (currently 388MB RSS)
- [ ] 5.8 Add graceful shutdown handling (predictions in-flight during deploy)

## PHASE 6: TESTING & VALIDATION
> No false scores. Everything must be proven.

- [ ] 6.1 Create automated test suite (currently NO tests exist)
- [ ] 6.2 Add integration tests for all API endpoints
- [ ] 6.3 Add accuracy tracking tests — verify accuracy calculation is correct
- [ ] 6.4 Add paper trading validation — confirm P&L calculations match
- [ ] 6.5 Backtest model on 90 days of historical data before deploying
- [ ] 6.6 Set up CI/CD pipeline with tests running before deploy
- [ ] 6.7 Create health score regression tests — health should never drop below 90 after a deploy
- [ ] 6.8 Validate all 40 integrity checks are testing what they claim

## SCORING CRITERIA (must ALL be true for 10/10)
- [ ] Health score stable at 97+ / 100
- [ ] Accuracy above 55% over 30-day rolling window
- [ ] All 9 intelligence subsystems actively contributing to predictions
- [ ] News feed populated with real articles
- [ ] P&L chart rendering correctly
- [ ] All watchlist symbols showing live prices and confidence
- [ ] AI Memory accumulating entries
- [ ] Learning Brain actively inverting poor-performing symbols
- [ ] Zero startup errors
- [ ] Telegram alerts operational
- [ ] Automated test suite passing
- [ ] No stale predictions (cycle running on schedule)

---

## PROGRESS LOG
| Date | Item | Status | Notes |
|------|------|--------|-------|
| 2026-03-21 | File created | Session 5 | Full system audit completed, scored 5.5/10 |
