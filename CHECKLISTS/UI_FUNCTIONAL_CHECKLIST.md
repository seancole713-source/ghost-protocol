# GHOST UI Functional Checklist

**Version**: 1.0
**Date**: October 4, 2025
**Owner**: Frontend Team / QA
**Frequency**: Pre-release + Sprint End

---

## Cockpit Display

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Two-line forecast visible (Ghost vs Live) | ✅ **VISIBLE**| P0 | Verified: purple vs orange lines |
| ✅ Forecast accuracy chips display | ❌**MISSING**| P3 |**GH-AUD-010**: MAP/RMSE not in API |
| ✅ Portfolio value matches backend state | ✅ **CORRECT** | P0 | NAV = cash + positions * price |
| ✅ PnL color coding (green/red) works | ✅ **WORKS**| P1 | CSS classes applied correctly |
| ✅ Real-time updates via SSE stream | ✅**WORKS**| P0 | `/api/cockpit/stream` working |
| ✅ Stale data indicators visible | ⚠️**PARTIAL**| P2 | `_degraded` field not always rendered |**Action Items**:

- [ ] Add forecast accuracy to `/api/cockpit` response (GH-AUD-010)
- [ ] Add UI chips: "MAP: 2.3%" and "RMSE: $45.67"
- [ ] Test overlay: verify Ghost line (purple) vs Live line (orange) distinct
- [ ] Add yellow "⚠ Using cached data" banner when `_degraded: true`
- [ ] Test SSE reconnect: kill server → restart → verify UI reconnects

---

## News Feed

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Reuters feed loads | ⚠️ **FRAGILE**| P1 |**GH-AUD-006**: DNS failure crashes loop |
| ✅ Polygon news loads | ✅ **WORKS**| P1 | Verified with live API |
| ✅ News items have timestamps | ✅**WORKS**| P1 | ISO 8601 format |
| ✅ News items have source attribution | ✅**WORKS**| P2 | "Reuters" or "Polygon" label |
| ✅ Empty state shown when no news | ⚠️**UNKNOWN**| P2 | Test: disable feeds → verify UI message |
| ✅ Degraded mode indicator | ❌**MISSING**| P2 | When cached news shown, no indicator |**Action Items**:

- [ ] Fix Reuters crash on DNS failure (GH-AUD-006)
- [ ] Test degraded mode: kill Reuters → verify cached news shown
- [ ] Add badge: "Last updated 15 minutes ago" for cached news
- [ ] Add empty state: "No news available" with retry button
- [ ] Test mixed sources: verify Reuters + Polygon combined correctly

---

## Portfolio Persistence

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Portfolio survives server restart | ⚠️ **DEFAULT FAIL**| P1 | Default `PORTFOLIO_PERSIST="none"` |
| ✅ Zero-dollar boot prevented | ❌**HAPPENS**| P1 | Must enable persistence manually |
| ✅ Last-known-good snapshot loaded | ⚠️**WHEN ENABLED**| P1 | Works if `persist="auto"` |
| ✅ Autosave thread writes to disk | ✅**WORKS**| P1 | Line 3550: 30-second interval |
| ✅ UI shows persistence status | ❌**MISSING**| P2 | No indicator of save state |**Action Items**:

- [ ] Change default to `PORTFOLIO_PERSIST="auto"` (critical fix)
- [ ] Add UI indicator: "Last saved: 2 minutes ago"
- [ ] Test cold boot: restart → verify positions restored
- [ ] Test autosave: make trade → wait 30s → restart → verify persisted
- [ ] Add alert: "Portfolio persistence disabled" if mode=none

---

## Telegram Integration

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Open message sent at market open | ⚠️ **UNRELIABLE**| P2 | Fire-and-forget, no delivery confirmation |
| ✅ Close message sent at market close | ⚠️**UNRELIABLE**| P2 | Same issue |
| ✅ Day summary includes PnL | ✅**WORKS**| P1 | Verified in logs |
| ✅ Day summary includes win rate | ✅**WORKS**| P2 | Verified in logs |
| ✅ Webhook receives commands | ⚠️**INSECURE**| P2 |**GH-AUD-007**: No signature validation |
| ✅ Message delivery failures logged | ⚠️ **UNKNOWN**| P2 | Check logs for errors |**Action Items**:

- [ ] Add Telegram webhook signature validation (GH-AUD-007)
- [ ] Add delivery confirmation: check HTTP 200 response
- [ ] Log failed message deliveries with alert
- [ ] Test webhook: send `/status` command → verify response
- [ ] Add retry logic: 3 attempts with exp backoff if Telegram down

---

## Markets View

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Watchlist displays current prices | ✅ **WORKS**| P0 | Yahoo/Polygon sources |
| ✅ Price staleness indicated | ⚠️**UNKNOWN**| P2 | Test: use old cached price → verify indicator |
| ✅ Market hours indicator | ⚠️**UNKNOWN**| P2 | "Market open" vs "Market closed" badge |
| ✅ Provider source shown per symbol | ⚠️**UNKNOWN**| P3 | E.g., "AAPL: Yahoo (150.23)" |
| ✅ Add symbol to watchlist works | ⚠️**UNKNOWN**| P1 | Test via UI |**Action Items**:

- [ ] Add market hours badge (green "OPEN" / red "CLOSED")
- [ ] Show provider source per price: "Yahoo", "Polygon", "AlphaVantage"
- [ ] Add stale price indicator: gray text + clock icon if >15 min old
- [ ] Test watchlist: add/remove symbols → verify persisted
- [ ] Test price updates: verify SSE pushes new prices to Markets view

---

## Engine Status

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Circuit breaker states visible | ⚠️ **UNKNOWN**| P2 | Check if exposed in `/api/status` |
| ✅ Background thread health visible | ⚠️**UNKNOWN**| P2 | Autosave/Alerts/Scheduler status |
| ✅ AI memory size shown | ⚠️**UNKNOWN**| P3 | "12 memories stored" |
| ✅ Uptime displayed | ⚠️**UNKNOWN**| P2 | Server start timestamp |
| ✅ Config toggles accessible | ⚠️**UNKNOWN**| P1 | Runtime flags for feeds/alerts/etc |**Action Items**:

- [ ] Add `/admin/engine` page showing circuit breaker states
- [ ] Add background thread heartbeat indicators
- [ ] Add AI memory count + last decision timestamp
- [ ] Add uptime counter (days:hours:minutes)
- [ ] Add toggle switches for runtime flags (requires auth)

---

## Security View

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Protected endpoints require auth | ✅ **WORKS**| P0 | 21 endpoints guarded |
| ✅ Auth failure shows 401/403 | ✅**WORKS**| P0 | Correct HTTP status |
| ✅ Token rotation UI available | ❌**MISSING**| P2 | Need admin page |
| ✅ Audit log accessible | ❌**MISSING**| P3 | View recent admin actions |**Action Items**:

- [ ] Create `/admin/security` page
- [ ] Add "Rotate API token" button (generates new, shows once)
- [ ] Add audit log table: timestamp, user, action, IP
- [ ] Test 401 error: remove token → verify UI shows login prompt
- [ ] Test 403 error: use wrong token → verify error message

---

## Bank View (Cash & Transfers)

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Cash balance displayed | ✅ **WORKS**| P0 | Verified: matches backend |
| ✅ Transfer IN works | ⚠️**UNKNOWN**| P1 | Test via UI |
| ✅ Transfer OUT works | ⚠️**UNKNOWN**| P1 | Test via UI |
| ✅ Transfer history shown | ⚠️**UNKNOWN**| P2 | List of past transfers |
| ✅ Overdraft prevented | ⚠️**UNKNOWN**| P1 | Cannot withdraw more than cash |**Action Items**:

- [ ] Test transfer IN: add $10k → verify cash increases
- [ ] Test transfer OUT: withdraw $5k → verify cash decreases
- [ ] Test overdraft: try withdraw more than cash → verify error
- [ ] Add transfer history table with timestamps
- [ ] Add CSV export of transfer history

---

## Monthly View (Historical)

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Monthly PnL chart | ⚠️ **UNKNOWN**| P2 | Verify chart renders |
| ✅ Win rate per month | ⚠️**UNKNOWN**| P2 | Percentage calculation correct |
| ✅ Trade count per month | ⚠️**UNKNOWN**| P2 | Total trades shown |
| ✅ Date range selector | ⚠️**UNKNOWN**| P3 | Filter by year/month |**Action Items**:

- [ ] Test monthly view: verify chart shows last 12 months
- [ ] Test win rate calculation: manually verify against trades
- [ ] Test empty state: new account → verify "No data" message
- [ ] Add CSV export of monthly data

---

## Responsiveness & Performance

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Mobile layout works | ⚠️ **UNKNOWN**| P2 | Test on iPhone/Android |
| ✅ Tablet layout works | ⚠️**UNKNOWN**| P2 | Test on iPad |
| ✅ Page load <2 seconds | ⚠️**UNKNOWN**| P1 | Use Lighthouse audit |
| ✅ SSE reconnects automatically | ⚠️**UNKNOWN**| P1 | Test: kill server → verify reconnect |
| ✅ No console errors | ⚠️**UNKNOWN**| P1 | Check browser dev console |**Action Items**:

- [ ] Run Lighthouse audit on `/cockpit`
- [ ] Test mobile: iPhone 12 Pro, Pixel 6
- [ ] Test tablet: iPad Air, Samsung Tab
- [ ] Test SSE reconnect: kill server → restart → verify stream resumes
- [ ] Check console for JS errors, fix all

---

## Accessibility (A11Y)

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Keyboard navigation works | ⚠️ **UNKNOWN**| P2 | Tab through UI |
| ✅ Screen reader compatible | ⚠️**UNKNOWN**| P3 | ARIA labels present |
| ✅ Color contrast WCAG AA | ⚠️**UNKNOWN**| P2 | Use contrast checker |
| ✅ Focus indicators visible | ⚠️**UNKNOWN**| P2 | Tab → verify focus ring |**Action Items**:

- [ ] Run axe DevTools on all pages
- [ ] Add ARIA labels to buttons/icons
- [ ] Test keyboard-only navigation
- [ ] Fix color contrast issues (e.g., gray text on white)

---

## Error Handling

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ 500 errors show user-friendly message | ⚠️ **UNKNOWN**| P1 | Test: cause backend crash → verify UI |
| ✅ Network timeout handled | ⚠️**UNKNOWN**| P1 | Test: disconnect WiFi → verify message |
| ✅ Invalid input validated | ⚠️**UNKNOWN**| P1 | Test: negative trade quantity → verify error |
| ✅ Error details sent to logs | ⚠️**UNKNOWN**| P2 | Check Railway logs for JS errors |**Action Items**:

- [ ] Test 500 error: trigger backend crash → verify user sees "Something went wrong"
- [ ] Test network timeout: slow network → verify "Loading..." indicator
- [ ] Test validation: enter invalid data → verify inline error messages
- [ ] Add frontend error logging (Sentry or similar)

---

## Browser Compatibility

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Chrome (latest) | ⚠️ **UNKNOWN**| P0 | Test version 118+ |
| ✅ Firefox (latest) | ⚠️**UNKNOWN**| P1 | Test version 119+ |
| ✅ Safari (latest) | ⚠️**UNKNOWN**| P1 | Test version 17+ |
| ✅ Edge (latest) | ⚠️**UNKNOWN**| P2 | Test version 118+ |**Action Items**:

- [ ] Test all major features in Chrome 118
- [ ] Test all major features in Firefox 119
- [ ] Test all major features in Safari 17 (macOS/iOS)
- [ ] Document unsupported browsers (IE, old Safari)

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Frontend Lead | TBD | YYYY-MM-DD | ____________ |
| QA Lead | TBD | YYYY-MM-DD | ____________ |
| Product Owner | TBD | YYYY-MM-DD | ____________ |

**Next Review**: Sprint End (Every 2 weeks)

---

**Checklist Maintained By**: QA Team
**Last Updated**: October 4, 2025
**Version**: 1.0
**Related Documents**: `GHOST_DEEP_AUDIT.md`, `PASS_FAIL_TABLE.md`
