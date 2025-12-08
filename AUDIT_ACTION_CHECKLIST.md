# 🚀 GHOST PROTOCOL - ACTION CHECKLIST (Quick Reference)

**Generated**: November 27, 2024
**Full Report**: See `COMPREHENSIVE_AUDIT_REPORT.md`

---

## ⚡ IMMEDIATE ACTIONS (DO FIRST)

### 🔴 CRITICAL: Replace Mock Data in Cockpit V2

**File**: `api/cockpit_v2_endpoints.py`

- [ ] **Line 76**- Hunter feed: Use real `turbo_provider` data instead of hardcoded BTC/WEPE
- [ ]**Line 131**- Wolf status: Use real predictions from `ghost_predictions.db`
- [ ]**Line 181-196**- World context: Calculate real change_pct for QQQ/BTC/DXY
- [ ]**Line 234**- News: Integrate `news_sentiment.py` or remove
- [ ]**Line 255**- Risk dashboard: Use real portfolio risk OR mark as "coming soon"
- [ ]**Line 293**- Portfolio: Use real positions OR mark as "unavailable"
- [ ]**Line 313**- Goals: Use real goal tracking OR remove panel
- [ ]**Line 334**- Predictions panel: Use real accuracy from `prediction_outcomes.db`
- [ ]**Line 410**- Health dashboard: Use real system health checks
- [ ]**Line 462**- Provider status: Query actual provider health
- [ ]**Line 489**- Logs: Integrate real logging system OR remove**Estimated Time**: 8-12 hours


---

## 📋 TODO COMMENT CLEANUP

### By File Priority

#### api/cockpit_v2_endpoints.py (13 TODOs)

- [ ] Line 76 - Integrate hunter algorithm
- [ ] Line 131 - Integrate price_quorum.py
- [ ] Lines 181-196 - Add world context calculations
- [ ] Line 234 - Integrate news_sentiment.py
- [ ] Line 255 - Integrate risk management
- [ ] Line 293 - Integrate portfolio system
- [ ] Line 313 - Integrate goal tracking
- [ ] Line 334 - Integrate prediction system
- [ ] Line 410 - Integrate health system
- [ ] Line 462 - Integrate provider health checks
- [ ] Line 489 - Integrate logging system


#### core/social_sentiment.py (3 TODOs)

- [ ] Line 51 - Implement Twitter API v2
- [ ] Line 119 - Implement Reddit API (PRAW)
- [ ] Line 228 - Implement trending detection


#### core/ai_memory.py (1 TODO)

- [ ] Line 193 - Implement FAISS vector store


#### core/economic_calendar.py (2 TODOs)

- [ ] Line 57 - Implement Trading Economics/Fred API
- [ ] Line 120 - Implement earnings calendar API


#### core/providers/unified_provider.py (3 TODOs)

- [ ] Lines 193, 204 - Implement provider chain logic
- [ ] Line 310 - Implement yfinance fallback


#### wolf_app.py (2 TODOs)

- [ ] Line 3641 - Calculate predicted_pct from forecast
- [ ] Line 22781 - Calculate daily_pnl from trades


**Total**: 38 TODOs

---

## 🔧 FIX RECOMMENDATIONS

### Option A: Full Implementation (60-80 hours)

✅ Implement all TODOs
✅ Add Twitter/Reddit APIs
✅ Add economic calendar APIs
✅ Complete FAISS integration
✅ Complete unified provider

### Option B: Clean MVP (10-14 hours) ⭐ RECOMMENDED

✅ Replace Cockpit V2 mock data with real data
✅ Remove unimplemented features from UI
✅ Document TODOs clearly as "coming soon"
✅ Keep core functionality working

### Option C: Hybrid Approach (28-36 hours)

✅ Replace Cockpit V2 mock data
✅ Implement social sentiment APIs
✅ Implement economic calendar APIs
⚠️ Leave FAISS and unified provider as TODOs

---

## 📊 PLACEHOLDER REMOVAL TARGETS

### Production Code (MUST FIX)

- [ ] `api/cockpit_v2_endpoints.py` - All mock responses
- [ ] `core/social_sentiment.py` - Stub functions
- [ ] `core/economic_calendar.py` - Stub functions


### Test Code (OK TO KEEP)

- [x] `generate_simulation_data.py` - Test mock data (acceptable)
- [x] `test_agentkit_integration.py` - Test placeholder keys (acceptable)
- [x] Test files with mock data (acceptable)


---

## 🎯 ACCEPTANCE CRITERIA

### Zero Placeholders Checklist

- [ ] No TODO comments in production endpoints
- [ ] No mock data returned to frontend users
- [ ] No fake success responses (HTTP 200 with dummy data)
- [ ] No hardcoded test values in production code
- [ ] All user-facing features either work OR marked "coming soon"
- [ ] All API endpoints return real data OR HTTP 501 "Not Implemented"
- [ ] Documentation matches actual implementation


---

## 🧪 TESTING CHECKLIST

After fixes, verify:

- [ ] Run `python3 test_endpoints.py` - Should pass for BTC and PACS
- [ ] Run `python3 -m py_compile wolf_app.py` - Should compile clean
- [ ] Start server: `python3 wolf_app.py`
- [ ] Test `/api/v2/cockpit/hunter` - Should return real data
- [ ] Test `/api/v2/cockpit/wolf-status` - Should return real forecast
- [ ] Test `/api/turbo/predict?symbol=BTC` - Should work
- [ ] Check Railway deployment - Should auto-deploy
- [ ] Review UI - Should show real data OR "coming soon" labels


---

## 🚨 BREAKING CHANGES WARNING

### If You Remove Mock Endpoints

**Before removing mock data**, coordinate with frontend:

1. Update UI to handle "coming soon" states
2. Add loading indicators for unavailable features
3. Test that UI doesn't crash on missing data
4. Update API documentation


**Suggested approach**:

- Return HTTP 501 "Not Implemented" instead of mock data
- Add `"available": false` flag in response
- Provide clear error messages


---

## 📞 HUMAN DECISIONS NEEDED

### Product Decisions

- [ ] Keep Cockpit V2 or deprecate in favor of V3?
- [ ] Implement social sentiment or remove from roadmap?
- [ ] Implement economic calendar or mark as future feature?
- [ ] FAISS integration priority? (AI memory semantic search)


### API Key Procurement

- [ ] Twitter API v2 developer account
- [ ] Reddit API credentials (PRAW)
- [ ] Trading Economics API key
- [ ] Fred API key (economic data)


### Architecture Decisions

- [ ] Complete unified provider abstraction or use turbo provider?
- [ ] SQLite-only AI memory vs FAISS vector store?
- [ ] Refactor cockpit_v2_endpoints.py or rewrite?


---

## 📈 SUCCESS METRICS

### Before (Current State)

- ❌ 38 TODO comments
- ❌ 13 mock endpoints
- ❌ 3 unimplemented APIs
- ✅ 0 syntax errors
- ⚠️ 52 lint violations


### After (Target State)

- ✅ 0 TODO comments in production
- ✅ 0 mock endpoints
- ✅ All APIs work OR marked unavailable
- ✅ 0 syntax errors
- ✅ 0 critical lint violations


---

## 🔗 QUICK LINKS

- **Full Audit Report**: `COMPREHENSIVE_AUDIT_REPORT.md`
- **Turbo Provider Status**: `TURBO_PROVIDER_STATUS.md`
- **Test Suite**: `test_endpoints.py`
- **Main App**: `wolf_app.py`
- **Cockpit V2 API**: `api/cockpit_v2_endpoints.py`
- **Railway Dashboard**: <<<<<<https://railway.app>>>>>>


---

## ⚡ NEXT STEPS

1. **Read full audit report**: `COMPREHENSIVE_AUDIT_REPORT.md`
2. **Choose fix strategy**: Option A, B, or C above
3. **Start with Cockpit V2**: Replace mock data with real integrations
4. **Test thoroughly**: Use `test_endpoints.py`
5. **Deploy to Railway**: Push to main branch
6. **Monitor logs**: Check Railway deployment logs
7. **Update docs**: Reflect actual implementation status


---

**Priority**: 🔴 CRITICAL
**Estimated Time**: 10-80 hours (depending on approach)
**Next Review**: After Phase 1 fixes completed
