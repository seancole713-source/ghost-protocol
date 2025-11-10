#!/bin/bash
# Ghost Cockpit - Apply Critical Patches to wolf_app.py
# Mission: Add /api/regime/current and enhance SSE with event types
# Date: 2025-11-10

echo "=========================================="
echo "PHASE 2: Critical Patches Summary"
echo "=========================================="
echo ""
echo "This script documents the patches needed for wolf_app.py"
echo "Manual application required due to file size (20,269 lines)"
echo ""

echo "PATCH 1: Add /api/regime/current endpoint"
echo "──────────────────────────────────────────"
echo "Location: After line 10910 (after @APP.get('/api/stage3/regime/history'))"
echo ""
echo "Add this code:"
cat <<'EOF'

@APP.get("/api/regime/current")
async def api_regime_current():
    """Get current market regime (neutral fallback if Stage 3 not enabled)."""
    try:
        if STAGE3_ENABLED:
            regime_detector = get_regime_detector()
            return {
                "regime": regime_detector.current_regime.lower(),
                "ts": int(time.time()),
                "confidence": float(regime_detector.confidence),
                "source": "stage3_detector"
            }
        else:
            return {
                "regime": "neutral",
                "ts": int(time.time()),
                "confidence": 0.5,
                "source": "fallback"
            }
    except Exception as e:
        LOGGER.error(f"regime_current_error: {e}")
        return {
            "regime": "neutral",
            "ts": int(time.time()),
            "confidence": 0.5,
            "source": "error_fallback",
            "error": str(e)
        }
EOF
echo ""

echo "PATCH 2: Enhance SSE /api/cockpit/stream with event types"
echo "─────────────────────────────────────────────────────────"
echo "Location: Lines 11653-11730 (replace current @APP.get('/api/cockpit/stream'))"
echo ""
echo "Key changes:"
echo "  1. Add 'event: status' on connect with {status:'live', ts, sim_mode, focus_wolf_only}"
echo "  2. Change heartbeat from comment ': heartbeat' to 'event: ping' with {ts}"
echo "  3. Prefix all data with 'event: snapshot'"
echo "  4. Reduce ping interval from 15s to 10s for better responsiveness"
echo ""
echo "See SSE_REGIME_PATCHES.md for full implementation"
echo ""

echo "PATCH 3: Verify existing endpoints (NO CHANGES NEEDED)"
echo "──────────────────────────────────────────────────────"
echo "✅ /api/price/{symbol} already uses ensure_price_cached() - returns instantly on cache hit"
echo "✅ /api/portfolio and /api/position should be verified to use cached snapshots"
echo ""

echo "=========================================="
echo "NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Apply patches to wolf_app.py:"
echo "   - Add /api/regime/current endpoint (Patch 1)"
echo "   - Enhance SSE stream with event types (Patch 2)"
echo ""
echo "2. Commit changes:"
echo "   git add wolf_app.py"
echo "   git commit -m 'feat: add /api/regime/current and SSE event types (status/ping/snapshot)'"
echo ""
echo "3. Deploy to Railway:"
echo "   git push"
echo "   railway up"
echo ""
echo "4. Run validation tests:"
echo "   ./PRODUCTION_VALIDATION_TESTS.sh"
echo ""
echo "5. Monitor for 5 minutes:"
echo "   curl -s \"\$GHOST_BASE_URL/api/admin/logs?window=5m\" | grep -E '499|502'"
echo ""
