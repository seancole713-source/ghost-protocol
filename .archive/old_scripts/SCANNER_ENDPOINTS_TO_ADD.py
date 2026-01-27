# Add these endpoints to wolf_app.py before "if __name__ == '__main__':"

# ============================================================================
# GHOST INVESTMENT HUNTER - MARKET SCANNER ENDPOINTS
# ============================================================================

@APP.get("/api/scan/stocks")
async def api_scan_stocks():
    """
    Scan entire stock market for opportunities.
    Returns top 20 high-confidence stock opportunities.
    """
    try:
        from core.market_scanner import scan_stocks
        
        opportunities = await scan_stocks()
        
        return {
            "ok": True,
            "opportunities": opportunities,
            "count": len(opportunities),
            "timestamp": int(time.time())
        }
    except Exception as e:
        LOGGER.error(f"Stock scan failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "opportunities": [],
            "count": 0,
            "timestamp": int(time.time())
        }


@APP.get("/api/scan/crypto")
async def api_scan_crypto():
    """
    Scan crypto market for opportunities.
    Returns high-confidence crypto opportunities.
    """
    try:
        from core.market_scanner import scan_crypto
        
        opportunities = await scan_crypto()
        
        return {
            "ok": True,
            "opportunities": opportunities,
            "count": len(opportunities),
            "timestamp": int(time.time())
        }
    except Exception as e:
        LOGGER.error(f"Crypto scan failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "opportunities": [],
            "count": 0,
            "timestamp": int(time.time())
        }


@APP.get("/api/scan/all")
async def api_scan_all():
    """
    Scan both stocks and crypto for opportunities.
    Returns combined opportunity list.
    """
    try:
        from core.market_scanner import scan_all
        
        results = await scan_all()
        
        return {
            "ok": True,
            **results
        }
    except Exception as e:
        LOGGER.error(f"Full market scan failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "stocks": [],
            "crypto": [],
            "total": 0,
            "timestamp": int(time.time())
        }


@APP.get("/api/opportunities/top")
async def api_opportunities_top(limit: int = 10, min_confidence: float = 0.70):
    """
    Get top-ranked opportunities across all markets.
    
    Query params:
        limit: Max opportunities to return (default 10)
        min_confidence: Minimum confidence threshold (default 0.70)
    """
    try:
        from core.market_scanner import scan_all
        
        # Get all opportunities
        results = await scan_all()
        
        # Combine and sort by confidence
        all_opportunities = results.get("stocks", []) + results.get("crypto", [])
        
        # Filter by confidence
        filtered = [opp for opp in all_opportunities if opp.get("confidence", 0) >= min_confidence]
        
        # Sort by confidence (highest first)
        filtered.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Take top N
        top = filtered[:limit]
        
        return {
            "ok": True,
            "opportunities": top,
            "count": len(top),
            "total_scanned": len(all_opportunities),
            "min_confidence": min_confidence,
            "timestamp": int(time.time())
        }
    except Exception as e:
        LOGGER.error(f"Top opportunities failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "opportunities": [],
            "count": 0,
            "timestamp": int(time.time())
        }
