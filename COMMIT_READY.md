# Commit Ready Summary

This document summarizes the recent enhancements implemented to improve real-time
pricing reliability, feature explainability, and fusion panel insight.

## 1. Real-Time Price Reliability

- Added background live price updater loop (`_auto_refresh_price`) scheduled at startup.
- Reduced `PRICE_TTL_OPEN_S` default to 5s to minimize stale spans during market hours.
- Added `/api/price/refresh` endpoint to force cache invalidation.
- Added `/api/price/diagnostics` endpoint returning:
  - Current price / prev_close / provider
  - Cache age and active TTL
  - `PRICE_DIAG` structure (provider, latency, fallback reason, spread, quorum flags)
  - Recent price-related events (up to 30)


## 2. APEX Trade Card Enrichment

- Injected `numeric_value` for each top feature (Momentum 20d, News Sentiment, RSI 14,


  Volume Surge, Volatility 20d) in `TradeCardGenerator._calculate_top_features`.

- Preserves formatted string `value` while enabling UI gauge/bar rendering.


## 3. Fusion Panel Metrics

- Extended `/fusion/ai` response with:
  - `risk_score` (heuristic = 1 - confidence)
  - `confidence_score` (scaled |macro score| / 3)
  - `drivers` (top 5 outlook reasons)
- Backward compatible: existing `outlook` and `source` unchanged.


## 4. Diagnostics & Observability

- Enhanced `diagnostics_summary` to include price diagnostics snippet.
- New price diagnostics endpoint (see above) for deeper debugging.


## 5. Environment / Config Additions

- `PRICE_AUTO_REFRESH_S` (default 7) controls background refresher cadence.
- Existing keys reused: `PRICE_TTL_OPEN_S`, `PRICE_TTL_S`.


## 6. Testing & Validation Plan

Recommended manual checks:

1. Start server and hit `/api/price/diagnostics` multiple times during market open


   simulation; ensure `cache_age_s` stays < TTL and provider transitions off
   `prev-close`.

1. Hit `/fusion/ai` and confirm presence of `risk_score`, `confidence_score`, and


   `drivers` fields.

1. Generate a trade card (`/api/trade/card` or cockpit path) and verify


   `top_features[*].numeric_value` present.

1. (Optional) Trigger price anomaly to verify `PRICE_DIAG.anomaly` flags appear in


   diagnostics.

## 7. Follow-Up Opportunities

- Add percentile normalization for `numeric_value` fields.
- Incorporate realized volatility into fusion risk heuristic.
- Persist diagnostics snapshots for historical analysis.


## 8. File Touches (Summary)

- `wolf_app.py`: Added background price loop, `/fusion/ai` enrichment,


  `/api/price/diagnostics`.

- `core/trade_card.py`: Added `numeric_value` to top feature objects.
- `COMMIT_READY.md`: This document.


## 9. Rollback Notes

All changes are additive. To rollback:

- Remove `_auto_refresh_price` coroutine + startup scheduling line.
- Revert `/fusion/ai` block to prior minimal return.
- Remove `/api/price/diagnostics` endpoint.
- Remove `numeric_value` entries in trade card feature assembly.


## 10. API Contract Deltas

Endpoint | Change | Notes ---------|--------|------ `/fusion/ai` | Added `risk_score`,
`confidence_score`, `drivers` | Non-breaking. `/api/price/diagnostics` | New | For
internal / UI debugging. `/api/price/refresh` | Added earlier in pricing improvements |
Force refresh.

______________________________________________________________________

Generated on: $(date -u)
