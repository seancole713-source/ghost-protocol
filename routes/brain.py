"""Routes: brain — extracted from wolf_app.py (Step 12)"""
# fmt: off
# ruff: noqa

import asyncio
import json
import logging
import os
import re
import time
import hashlib
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response, Query, Header, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse, RedirectResponse

try:
    import httpx
except ImportError:
    httpx = None

try:
    from state import APP_STATE, POOL, DB_URL, PREDICTION_HISTORY
except ImportError:
    APP_STATE = {}
    POOL = None
    DB_URL = ""
    PREDICTION_HISTORY = []

try:
    from wolf_helpers import (
        AUTH_DEP, SECURITY_SCHEME, WOLF, WOLF_SQLITE_PATH,
        _is_truthy, _json500, with_cap,
        AlertTemplateBody, AlertToggle, AlertConfigBody,
        RuntimeConfigBody, ControlBody, ModeBody, TrainBody,
        AgentControlBody, CashBody, PositionAddBody, PositionsImportBody,
        WatchlistImportBody, TradeRequest, PredFeedbackBody,
        AddPositionBody, OrderPlaceBody,
        _PredictRunBody, _RecordPriceBody, _ScoreBody, _BacktestBody,
        ChatRequest, AiDecision, TelegramUpdate,
    )
    from fastapi.security import HTTPAuthorizationCredentials
except Exception as _wh_e:
    import logging as _l
    _l.getLogger("ghost").warning(f"wolf_helpers import partial: {_wh_e}")
    AUTH_DEP = None
    WOLF = "WOLF"
    WOLF_SQLITE_PATH = "data/wolf.db"


router = APIRouter()
LOGGER = logging.getLogger("ghost")

# --- 44 endpoints ---

@router.get("/api/v3/research/batch")
async def api_v3_research_batch(symbols: str):
    """
    Get research reports for multiple symbols (comma-separated).
    
    Example: /api/v3/research/batch?symbols=AAPL,TSLA,GOOGL
    
    Returns array of research reports.
    """
    try:
        from core.research import batch_research
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        
        if not symbol_list:
            return {"ok": False, "error": "No symbols provided"}
        
        if len(symbol_list) > 10:
            return {"ok": False, "error": "Maximum 10 symbols per batch"}
        
        results = await batch_research(symbol_list)
        return {"ok": True, "count": len(results), "results": results}
    except ImportError as e:
        LOGGER.error(f"Research module import failed: {e}")
        return {"ok": False, "error": "Research module not available"}
    except Exception as e:
        LOGGER.error(f"Batch research failed: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/research/{symbol}")
async def api_v3_research_symbol(symbol: str):
    """
    Get comprehensive research report for a symbol.
    
    Includes:
    - Earnings calendar (upcoming earnings dates)
    - News sentiment analysis
    - Seasonal patterns (historical performance this time of year)
    - 52-week range position
    - YTD performance
    - Same period last year comparison
    
    Returns confidence adjustments based on research findings.
    """
    try:
        from core.research import deep_research
        result = await deep_research(symbol.upper())
        return {"ok": True, **result}
    except ImportError as e:
        LOGGER.error(f"Research module import failed: {e}")
        return {"ok": False, "error": "Research module not available", "symbol": symbol}
    except Exception as e:
        LOGGER.error(f"Research failed for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/research/earnings/{symbol}")
async def api_v3_research_earnings(symbol: str):
    """Get earnings calendar data for a symbol"""
    try:
        from core.research import check_earnings_risk
        result = await check_earnings_risk(symbol.upper())
        return {"ok": True, "symbol": symbol.upper(), **result}
    except Exception as e:
        LOGGER.error(f"Earnings check failed for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/research/news/{symbol}")
async def api_v3_research_news(symbol: str):
    """Get news sentiment analysis for a symbol"""
    try:
        from core.research import analyze_news
        result = await analyze_news(symbol.upper())
        return {"ok": True, **result}
    except Exception as e:
        LOGGER.error(f"News analysis failed for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/research/seasonal/{symbol}")
async def api_v3_research_seasonal(symbol: str):
    """Get seasonal pattern analysis for a symbol"""
    try:
        from core.research import analyze_seasonal
        result = await analyze_seasonal(symbol.upper())
        return {"ok": True, **result}
    except Exception as e:
        LOGGER.error(f"Seasonal analysis failed for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/research/historical/{symbol}")
async def api_v3_research_historical(symbol: str):
    """Get historical performance analysis for a symbol"""
    try:
        from core.research import analyze_historical
        result = await analyze_historical(symbol.upper())
        return {"ok": True, **result}
    except Exception as e:
        LOGGER.error(f"Historical analysis failed for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/brain/{symbol}")
async def api_v3_brain_analysis(symbol: str):
    """
    Full Ghost Brain analysis - combines ALL intelligence sources.
    
    Returns:
    - Micro signals (insider, whale, options, social, volume)
    - Human behavior (narratives, influencers)
    - Historical patterns (seasonal, events)
    - Overall recommendation
    
    This is the "weatherman" endpoint - tells you WHY, not just WHAT.
    """
    try:
        from core.intelligence.ghost_brain import analyze_with_intelligence
        result = await analyze_with_intelligence(symbol.upper())
        return {"ok": True, **result}
    except ImportError as e:
        LOGGER.error(f"Intelligence module import failed: {e}")
        return {"ok": False, "error": "Intelligence module not available", "symbol": symbol}
    except Exception as e:
        LOGGER.error(f"Brain analysis failed for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/micro/{symbol}")
async def api_v3_micro_signals(symbol: str):
    """
    Micro signal scan - early warning system.
    
    Returns:
    - Alert level (SHADOW, WHISPER, RIPPLE, WAVE)
    - Insider activity (stock only)
    - Whale movements (crypto only)
    - Options flow (stock only)
    - Social velocity
    - Volume anomalies
    """
    try:
        from core.intelligence.micro_signals.micro_aggregator import scan_micro_signals
        result = await scan_micro_signals(symbol.upper())
        return {"ok": True, **result}
    except ImportError as e:
        LOGGER.error(f"Micro signals module import failed: {e}")
        return {"ok": False, "error": "Micro signals module not available", "symbol": symbol}
    except Exception as e:
        LOGGER.error(f"Micro scan failed for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/narrative")
async def api_v3_narrative(symbol: str = None):
    """
    Detect current market narratives.
    
    Returns:
    - Active narratives (AI_REVOLUTION, FED_PIVOT, BITCOIN_HALVING, etc.)
    - Dominant narrative
    - Market mood (BULLISH/BEARISH/MIXED)
    """
    try:
        from core.intelligence.human_behavior.narrative_detector import detect_narratives
        result = await detect_narratives(symbol.upper() if symbol else None)
        return {"ok": True, **result}
    except ImportError as e:
        LOGGER.error(f"Narrative module import failed: {e}")
        return {"ok": False, "error": "Narrative module not available"}
    except Exception as e:
        LOGGER.error(f"Narrative detection failed: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/influencers/{symbol}")
async def api_v3_influencers(symbol: str):
    """
    Check for recent influencer activity.
    
    Tracks: Elon Musk, Michael Saylor, Trump, Buffett, Cramer, etc.
    
    Returns:
    - Recent mentions by key influencers
    - Sentiment analysis
    - Impact assessment
    """
    try:
        from core.intelligence.human_behavior.influencer_tracker import check_influencers
        result = await check_influencers(symbol.upper())
        return {"ok": True, **result}
    except ImportError as e:
        LOGGER.error(f"Influencer module import failed: {e}")
        return {"ok": False, "error": "Influencer module not available", "symbol": symbol}
    except Exception as e:
        LOGGER.error(f"Influencer check failed for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/historical/{event_type}")
async def api_v3_historical_event(event_type: str, symbol: str = None):
    """
    Get historical outcomes for an event type.
    
    Event types:
    - BTC_HALVING - Bitcoin halving history
    - FED_RATE_CUT - Fed rate cut outcomes
    - EARNINGS_BEAT - Earnings beat patterns
    - EARNINGS_MISS - Earnings miss patterns
    - SANTA_RALLY - Holiday rally data
    - COVID_CRASH - Black swan reference
    - JANUARY_EFFECT - January anomaly
    """
    try:
        from core.intelligence.historical.event_outcomes import get_historical_outcomes
        result = await get_historical_outcomes(event_type.upper(), symbol)
        return {"ok": True, **result}
    except ImportError as e:
        LOGGER.error(f"Historical module import failed: {e}")
        return {"ok": False, "error": "Historical module not available"}
    except Exception as e:
        LOGGER.error(f"Historical lookup failed: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/seasonal/{symbol}")
async def api_v3_seasonal_pattern(symbol: str, month: int = None):
    """
    Get seasonal pattern for current period.
    
    Returns:
    - Monthly tendency (BULLISH/BEARISH/NEUTRAL)
    - Special periods (Santa Rally, January Effect, etc.)
    - Historical performance for this time of year
    """
    try:
        from core.intelligence.historical.event_outcomes import get_event_database
        result = await get_event_database().get_seasonal_pattern(symbol.upper(), month)
        return {"ok": True, **result}
    except ImportError as e:
        LOGGER.error(f"Seasonal module import failed: {e}")
        return {"ok": False, "error": "Seasonal module not available", "symbol": symbol}
    except Exception as e:
        LOGGER.error(f"Seasonal pattern failed for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/weather-report/{symbol}")
async def api_v3_weather_report(symbol: str):
    """
    🌤️ THE WEATHERMAN ENDPOINT
    
    Like a weather forecast, but for markets.
    Combines ALL data sources into one comprehensive report.
    
    Returns:
    - Current conditions (price, volume, technicals)
    - Incoming fronts (earnings, news, events)
    - Historical patterns (seasonal, same period last year)
    - Sector health (is it the stock or the whole sector?)
    - Forecast with probability
    - Confidence adjustments with REASONS
    - Final recommendation
    
    This is Ghost thinking like a weatherman! 🐺🌤️
    """
    try:
        # Get brain analysis (combines everything)
        from core.intelligence.ghost_brain import analyze_with_intelligence
        brain = await analyze_with_intelligence(symbol.upper())
        
        # Get research data
        from core.research import deep_research
        research = await deep_research(symbol.upper())
        
        # Combine into weather report format
        report = {
            "ok": True,
            "symbol": symbol.upper(),
            "timestamp": brain.get("timestamp"),
            
            # Current conditions
            "current_conditions": {
                "alert_level": brain.get("alert_level", "SHADOW"),
                "overall_signal": brain.get("overall_signal", "NEUTRAL"),
                "micro_signals": brain.get("micro_signals", {}).get("signals", {}),
            },
            
            # Incoming fronts (events that could move the market)
            "incoming_fronts": {
                "earnings": research.get("earnings", {}),
                "news_sentiment": research.get("news", {}).get("sentiment", "unknown"),
                "news_count": research.get("news", {}).get("article_count", 0),
                "key_headlines": research.get("news", {}).get("key_headlines", [])[:3],
            },
            
            # Historical patterns
            "historical_pattern": {
                "seasonal": brain.get("seasonal", {}),
                "same_period_last_year": research.get("historical", {}).get("same_period_last_year", {}),
                "52_week_position": research.get("historical", {}).get("52_week_range", {}).get("range_position"),
            },
            
            # Market narrative
            "market_narrative": {
                "dominant": brain.get("narratives", {}).get("dominant_narrative", {}),
                "mood": brain.get("narratives", {}).get("market_mood", {}),
            },
            
            # Influencer activity
            "influencer_activity": brain.get("influencers", {}),
            
            # The forecast
            "forecast": {
                "direction": brain.get("overall_signal", "NEUTRAL"),
                "confidence_adjustment": brain.get("confidence_adjustment", 0),
            },
            
            # Why - the adjustments with reasons
            "confidence_adjustments": brain.get("warnings", []) + brain.get("positives", []),
            
            # Final recommendation
            "final_recommendation": brain.get("recommendation", ""),
            
            # Summary
            "summary": brain.get("summary", {}),
        }
        
        return report
        
    except Exception as e:
        LOGGER.error(f"Weather report failed for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/opus/analyze/{symbol}")
async def api_v3_opus_analyze(symbol: str):
    """
    🧠 Claude AI analysis of a symbol.
    
    Combines all available data and asks Claude to THINK about:
    - What's the story driving this asset?
    - What are the risks?
    - What does history say?
    - What will humans do?
    - What's the smart play?
    
    Returns Claude's reasoning and recommendation.
    """
    try:
        from core.intelligence.opus_brain import opus_analyze
        
        # Gather context for Claude
        context = await _gather_opus_context(symbol.upper())
        
        # Get Claude's analysis
        result = await opus_analyze(symbol.upper(), context)
        return result
    except ImportError as e:
        LOGGER.error(f"Opus brain import failed: {e}")
        return {"ok": False, "error": "Opus brain module not available", "symbol": symbol}
    except Exception as e:
        LOGGER.error(f"Opus analyze error for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/opus/research/{symbol}")
async def api_v3_opus_research(symbol: str, question: str = None):
    """
    🔬 Ask Claude to do deep research on a symbol.
    
    Optional question parameter for specific queries like:
    - "What's the bull case right now?"
    - "Why did it crash last week?"
    - "What are the risks before earnings?"
    - "What's the best entry point?"
    
    Without a question, provides comprehensive analysis.
    """
    try:
        from core.intelligence.opus_brain import opus_research
        return await opus_research(symbol.upper(), question)
    except ImportError as e:
        LOGGER.error(f"Opus brain import failed: {e}")
        return {"ok": False, "error": "Opus brain module not available", "symbol": symbol}
    except Exception as e:
        LOGGER.error(f"Opus research error for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/opus/explain/{symbol}")
async def api_v3_opus_explain(symbol: str, move: str):
    """
    📰 Ask Claude to explain a price move.
    
    Example: /api/v3/opus/explain/BTC?move=dropped 5% in 1 hour
    
    Claude will explain:
    - What likely caused this move
    - Whether it's significant or just noise
    - What traders should do now
    """
    try:
        from core.intelligence.opus_brain import opus_explain
        return await opus_explain(symbol.upper(), move)
    except ImportError as e:
        LOGGER.error(f"Opus brain import failed: {e}")
        return {"ok": False, "error": "Opus brain module not available", "symbol": symbol}
    except Exception as e:
        LOGGER.error(f"Opus explain error for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/opus/compare")
async def api_v3_opus_compare(symbols: str, question: str = None):
    """
    ⚖️ Ask Claude to compare multiple assets.
    
    Example: /api/v3/opus/compare?symbols=BTC,ETH,SOL
    
    Optional question for specific comparison:
    - "Which one has better risk/reward?"
    - "Which should I buy for the next month?"
    """
    try:
        from core.intelligence.opus_brain import opus_compare
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if len(symbol_list) < 2:
            return {"ok": False, "error": "Need at least 2 symbols to compare"}
        if len(symbol_list) > 5:
            return {"ok": False, "error": "Maximum 5 symbols for comparison"}
        return await opus_compare(symbol_list, question)
    except ImportError as e:
        LOGGER.error(f"Opus brain import failed: {e}")
        return {"ok": False, "error": "Opus brain module not available"}
    except Exception as e:
        LOGGER.error(f"Opus compare error: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/opus/predict/{symbol}")
async def api_v3_opus_predict(symbol: str, bypass_calendar: bool = False):
    """
    🎯 Enhanced prediction with Claude AI reasoning.
    
    1. Runs technical analysis (existing Ghost logic)
    2. Gathers all intelligence (micro signals, news, etc.)
    3. Asks Claude to THINK about it
    4. Returns enhanced prediction with Claude's reasoning
    
    This is Ghost's brain thinking like a professional trader!
    
    Args:
        bypass_calendar: If true, skip FOMC/CPI/NFP blackout checks (for testing)
    """
    try:
        symbol = symbol.upper().strip()
        
        # FIX (Feb 6, 2026): Crypto should NEVER be blocked by economic calendar
        # The calendar gate (FOMC/CPI/NFP) only applies to stocks
        from core.asset_classification import is_crypto_symbol as _opus_is_crypto
        if _opus_is_crypto(symbol):
            bypass_calendar = True  # Crypto trades 24/7, no economic calendar
        
        # FIX (Jan 27, 2026): Try stock engine directly with bypass_calendar
        # This avoids V2 filter blocking and honors bypass_calendar param
        from core.stock_engine import get_stock_engine
        engine = get_stock_engine()
        
        try:
            stock_pred = await engine.predict(symbol, bypass_calendar=bypass_calendar)
            technical = {
                "direction": stock_pred.direction,
                "confidence": stock_pred.confidence,
                "signals": stock_pred.reasons,
                "entry_price": stock_pred.entry_price,
                "target_price": stock_pred.target_price,
            }
            prediction_ok = stock_pred.direction != "HOLD"
        except Exception as stock_err:
            LOGGER.warning(f"Stock engine failed for {symbol}: {stock_err}, trying legacy")
            # Fallback to original prediction
            prediction_result = await run_single_prediction_async(symbol)
            if not prediction_result.get("ok"):
                return {"ok": False, "error": f"Technical prediction failed: {prediction_result.get('error', 'unknown')}", "symbol": symbol}
            technical = prediction_result.get("prediction", prediction_result)
            prediction_ok = True
        
        # Gather context for Claude
        context = await _gather_opus_context(symbol.upper())
        context["technical_signal"] = technical.get("direction", "UNKNOWN")
        context["technical_confidence"] = round(technical.get("confidence", 0) * 100, 1)
        
        # Get Claude's analysis
        from core.intelligence.opus_brain import opus_analyze
        opus_analysis = await opus_analyze(symbol.upper(), context)
        
        # Calculate adjusted confidence
        original_confidence = technical.get("confidence", 0.5)
        opus_adjustment = opus_analysis.get("confidence_adjustment", 0) / 100
        adjusted_confidence = max(0.1, min(0.85, original_confidence + opus_adjustment))
        
        # Check for signal conflict
        signal_conflict = False
        opus_signal = opus_analysis.get("signal", "NEUTRAL")
        tech_direction = technical.get("direction", "FLAT")
        final_direction = tech_direction  # Start with technical direction
        direction_overridden = False
        
        # Opus Direction Override: When Opus has high conviction, override FLAT
        opus_adj_value = opus_analysis.get("confidence_adjustment", 0)
        if tech_direction == "FLAT" and abs(opus_adj_value) >= 20:
            if opus_signal == "BULLISH" and opus_adj_value >= 20:
                final_direction = "UP"
                direction_overridden = True
                LOGGER.info(f"[OPUS OVERRIDE] {symbol}: FLAT→UP (Opus BULLISH +{opus_adj_value})")
            elif opus_signal == "BEARISH" and opus_adj_value <= -20:
                final_direction = "DOWN"
                direction_overridden = True
                LOGGER.info(f"[OPUS OVERRIDE] {symbol}: FLAT→DOWN (Opus BEARISH {opus_adj_value})")
        
        if (opus_signal == "BEARISH" and tech_direction == "UP") or \
           (opus_signal == "BULLISH" and tech_direction == "DOWN"):
            signal_conflict = True
            adjusted_confidence *= 0.8  # Reduce confidence when signals conflict
        
        return {
            "ok": True,
            "symbol": symbol.upper(),
            
            # Technical analysis (with potential Opus override)
            "direction": final_direction,
            "original_direction": tech_direction if direction_overridden else None,
            "direction_overridden": direction_overridden,
            "original_confidence": round(original_confidence, 3),
            
            # Claude's analysis
            "opus_signal": opus_signal,
            "opus_adjustment": opus_analysis.get("confidence_adjustment", 0),
            "adjusted_confidence": round(adjusted_confidence, 3),
            
            # Reasoning
            "opus_reasoning": opus_analysis.get("reasoning", ""),
            "opus_key_factors": opus_analysis.get("key_factors", []),
            "opus_risks": opus_analysis.get("risks", []),
            "opus_recommendation": opus_analysis.get("recommendation", ""),
            
            # Conflict detection
            "signal_conflict": signal_conflict,
            "conflict_note": "Claude and technicals disagree - reduced confidence" if signal_conflict else None,
            
            # Technical signals
            "technical_signals": technical.get("signals", [])[:5],
            
            # Price info
            "current_price": context.get("current_price"),
            "price_change_24h": context.get("price_change_24h"),
            
            # Timestamp
            "timestamp": datetime.now().isoformat(),
            "model": opus_analysis.get("model", "unknown")
        }
        
    except ImportError as e:
        LOGGER.error(f"Opus brain import failed: {e}")
        return {"ok": False, "error": "Opus brain module not available", "symbol": symbol}
    except Exception as e:
        LOGGER.error(f"Opus predict error for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/stock/predict/{symbol}")
async def api_v3_stock_predict(symbol: str, bypass_calendar: bool = False):
    """
    🏛️ Stock Engine Prediction - Optimized for stocks (not crypto)
    
    Uses stock-tuned parameters:
    - 24h horizon (vs 48h for crypto)
    - 2% target (vs 6% for crypto)
    - RSI 35/65 (vs 30/70 for crypto)
    - 4 confirmations (vs 3 for crypto)
    - VIX < 20 gate
    - SPY regime gate
    - Economic calendar gate (FOMC, CPI, NFP blackouts)
    - Sector momentum gate
    - Multi-timeframe confirmation
    
    Query params:
        bypass_calendar: Set to true to skip FOMC/CPI/NFP blackout (TESTING ONLY)
    
    Target: 40-50% win rate (up from 4.5%)
    """
    try:
        from core.stock_engine import get_stock_engine
        
        engine = get_stock_engine()
        result = await engine.predict(symbol.upper(), bypass_calendar=bypass_calendar)
        
        return {
            "ok": True,
            "engine": "stock_v1",
            "symbol": symbol.upper(),
            "bypass_calendar": bypass_calendar,
            **result.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError as e:
        LOGGER.error(f"Stock engine import failed: {e}")
        return {"ok": False, "error": "Stock engine not available", "symbol": symbol}
    except Exception as e:
        LOGGER.error(f"Stock engine error for {symbol}: {e}")
        return {"ok": False, "error": str(e), "symbol": symbol}


@router.get("/api/v3/stock/batch")
async def api_v3_stock_batch(symbols: str = "AAPL,MSFT,JPM", bypass_calendar: bool = False):
    """
    🏛️ Batch stock predictions for multiple symbols.
    
    Query params:
        symbols: Comma-separated list of stock symbols
        bypass_calendar: Skip FOMC/CPI/NFP blackout checks (for testing)
    
    Returns predictions for all symbols.
    """
    try:
        from core.stock_engine import get_stock_engine
        
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        
        if not symbol_list:
            return {"ok": False, "error": "No symbols provided"}
        
        if len(symbol_list) > 10:
            return {"ok": False, "error": "Max 10 symbols per batch"}
        
        engine = get_stock_engine()
        results = await engine.predict_batch(symbol_list, bypass_calendar=bypass_calendar)
        
        return {
            "ok": True,
            "engine": "stock_v1",
            "count": len(results),
            "predictions": {sym: pred.to_dict() for sym, pred in results.items()},
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        LOGGER.error(f"Stock batch error: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/stock/debug/{symbol}")
async def api_v3_stock_debug(symbol: str):
    """
    🔧 Debug stock data - check what yfinance returns for a symbol
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period="5d")
        
        if hist.empty:
            return {
                "ok": False,
                "symbol": symbol.upper(),
                "error": "No history data from yfinance",
                "ticker_info": str(ticker.info.get('shortName', 'Unknown'))[:50]
            }
        
        return {
            "ok": True,
            "symbol": symbol.upper(),
            "current_price": float(hist['Close'].iloc[-1]),
            "rows": len(hist),
            "date_range": f"{hist.index[0]} to {hist.index[-1]}",
            "last_5_closes": [round(float(x), 2) for x in hist['Close'].tolist()[-5:]],
        }
    except Exception as e:
        return {"ok": False, "symbol": symbol, "error": str(e)}


@router.get("/api/v3/stock/config")
async def api_v3_stock_config():
    """
    🏛️ Get current stock engine configuration.
    
    Shows the tuned parameters for stock predictions.
    """
    try:
        from core.stock_engine import STOCK_CONFIG
        
        return {
            "ok": True,
            "engine": "stock_v1",
            "config": STOCK_CONFIG.to_dict(),
            "description": {
                "horizon_hours": "Prediction horizon (24h for stocks vs 48h for crypto)",
                "target_pct": "Expected move percentage (2% for stocks vs 6% for crypto)",
                "rsi_oversold": "RSI level to consider oversold (35 for stocks vs 30 for crypto)",
                "rsi_overbought": "RSI level to consider overbought (65 for stocks vs 70 for crypto)",
                "min_confirmations": "Minimum confirmations needed (4 for stocks vs 3 for crypto)",
                "vix_max": "Max VIX to allow predictions (20 for stocks vs 25 for crypto)",
                "require_spy_bull": "Require SPY above 20MA (bull market)",
                "market_hours_only": "Only predict during market hours",
                "earnings_blackout_days": "Days around earnings to block predictions",
            },
            "comparison_to_crypto": {
                "horizon": "24h vs 48h (stocks move slower)",
                "target": "2% vs 6% (more realistic)",
                "rsi": "35/65 vs 30/70 (stocks rarely hit extremes)",
                "confirmations": "4 vs 3 (stricter)",
                "vix_gate": "20 vs 25 (stricter fear threshold)",
            }
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/stock/calendar")
async def api_v3_stock_calendar():
    """
    🏛️ Get upcoming economic events that will block stock predictions.
    
    Shows FOMC, CPI, NFP dates - these are blackout days.
    """
    try:
        from core.economic_calendar import (
            get_upcoming_blackout_events,
            is_fomc_day,
            is_cpi_day,
            is_nfp_day
        )
        
        return {
            "ok": True,
            "today_status": {
                "is_fomc_day": is_fomc_day(),
                "is_cpi_day": is_cpi_day(),
                "is_nfp_day": is_nfp_day(),
                "stocks_blocked": is_fomc_day() or is_cpi_day() or is_nfp_day()
            },
            "upcoming_blackout_events": get_upcoming_blackout_events(days_ahead=30),
            "note": "Stock predictions are BLOCKED on these days due to unpredictable binary outcomes"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/research/{symbol}")
async def api_research(
    symbol: str,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Research Blueprint - Multi-source aggregation
    
    Aggregates data from 12 categories:
    
    Fundamentals:
    - P/E ratio, Market Cap, Profit Margins, Revenue Growth
    
    Technicals:
    - RSI, Bollinger Bands, MA20/50/200
    
    News & Sentiment:
    - Recent news headlines with sentiment scores
    
    EDGAR Filings:
    - Recent SEC filings (10-K, 10-Q, 8-K)
    
    Returns:
        Aggregate impact score: -1.0 (very bearish) to +1.0 (very bullish)
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    
    if os.getenv("RESEARCH_BLUEPRINT_ENABLED", "0") != "1":
        raise HTTPException(503, "Research blueprint not enabled. Set RESEARCH_BLUEPRINT_ENABLED=1")
    
    try:
        from core.research_blueprint import aggregate_research
        
        research = aggregate_research(symbol.upper())
        
        return {
            "symbol": symbol.upper(),
            "research": research,
            "timestamp": int(time.time()),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Research aggregation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Research failed: {str(e)[:200]}")


@router.post("/ai/chat")
async def ai_chat(
    req: ChatRequest,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """Natural language Q&A with Ghost AI.

    Example:
    POST /ai/chat
    {"question": "What would a Bitcoin drop do to WOLF stock?"}
    """
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )

    if not req.question or not req.question.strip():
        raise HTTPException(400, "question required")

    try:
        answer = _ask_ghost_ai(req.question.strip())
        ctx = _build_ai_context() if req.include_context else {}

        return {
            "ok": True,
            "question": req.question,
            "answer": answer,
            "context": ctx,
        }
    except Exception as e:
        LOGGER.error(f"AI chat endpoint error: {e}", exc_info=True)
        raise HTTPException(500, f"AI chat failed: {str(e)}")


@router.post("/ai/agent/run")
async def ai_agent_run(
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
    idempotency_key: str | None = Header(
        default=None, convert_underscores=False, alias="Idempotency-Key"
    ),
):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    # Idempotency: early return if cached
    try:
        now_ts = time.time()
        for k, ts in _IDEMP_CACHE_TS.items():
            if now_ts - ts > _IDEMPOTENCY_TTL_S:
                _IDEMP_CACHE.pop(k, None)
                _IDEMP_CACHE_TS.pop(k, None)
        if idempotency_key:
            prior = _IDEMP_CACHE.get(idempotency_key)
            if isinstance(prior, dict):
                return prior
    except Exception:
        pass
    # Safety: require API key present
    if not OPENAI_API_KEY:
        raise HTTPException(400, "AI disabled: OPENAI_API_KEY not set")

    # Local tool router mapping to internal helpers
    def _tool_router(name: str, args: dict):
        name = (name or "").strip()
        if name == "get_price":
            p, prev, prov = get_wolf_price()
            return {"price": p, "prev_close": prev, "provider": prov}
        if name == "get_news":
            lim = int(args.get("limit", 10) if isinstance(args, dict) else 10)
            news = get_wolf_news(limit=min(25, max(1, lim)))
            # keep compact fields
            items = [
                {
                    "ts": it.get("ts"),
                    "headline": it.get("headline"),
                    "url": it.get("url"),
                    "sent": it.get("sent"),
                }
                for it in news.get("items", [])
            ]
            return {"items": items, "news_signal": news.get("news_signal")}
        if name == "get_position":
            return {
                "qty": float(STATE.get("qty", 0.0)),
                "avg_cost": float(STATE.get("avg_cost", 0.0)),
            }
        if name == "dispatch_alert":
            text = str((args or {}).get("text") or "").strip()
            if not text:
                return {"ok": False, "error": "empty"}
            ok = enqueue_alert_text(text)
            return {"ok": bool(ok)}
        return {"error": "unknown_tool"}

    # Build a minimal snapshot to pass implicitly
    snap = _build_ai_context()
    try:
        from llm.agent import run_once  # type: ignore
    except Exception:
        raise HTTPException(500, "llm agent missing")
    
    # CRITICAL: Run LLM agent in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    out = await loop.run_in_executor(None, run_once, _tool_router)
    # Persist agent result to AI memory
    try:
        px = snap.get("prices") or {}
        pos = snap.get("position") or {}
        ns = (snap.get("news_signal") or {}).get("score")
        feats = _extract_features(
            px.get("price"),
            px.get("prev_close"),
            float(pos.get("qty") or 0.0),
            float(pos.get("avg_cost") or 0.0),
            ns,
        )
        _ai_memory_append(
            {
                "ts": int(time.time()),
                "price": px.get("price"),
                "prev": px.get("prev_close"),
                "qty": float(pos.get("qty") or 0.0),
                "avg": float(pos.get("avg_cost") or 0.0),
                "news_score": (ns if isinstance(ns, (int, float)) else 0.0),
                "features": feats,
                "label_next_move": _label_from_action(str((out or {}).get("action"))),
                "advisory": str((out or {}).get("card") or (out or {}).get("rationale") or ""),
                "confidence": int((out or {}).get("confidence") or 0),
            }
        )
    except Exception:
        pass
    try:
        if _C_LLM_CALLS is not None:
            _C_LLM_CALLS.labels(endpoint="ai_agent_run", result="ok").inc()
        if isinstance(out, dict) and _C_LLM_DECISIONS is not None:
            _C_LLM_DECISIONS.labels(
                endpoint="ai_agent_run", action=str(out.get("action") or "?")
            ).inc()
        if isinstance(out, dict) and _G_LLM_CONFIDENCE is not None:
            conf = int(out.get("confidence") or 0)
            _G_LLM_CONFIDENCE.labels(endpoint="ai_agent_run").set(conf)
    except Exception:
        pass
    # Optionally auto-dispatch card (advisory only)
    try:
        if isinstance(out, dict) and out.get("card") and int(os.getenv("AI_AGENT_AUTOSEND", "0")):
            enqueue_alert_text(str(out.get("card")))
    except Exception:
        pass
    resp = {
        "ok": True,
        "result": out,
        "context": snap if int(os.getenv("AI_INCLUDE_CONTEXT", "0")) else {},
    }
    try:
        if idempotency_key:
            _IDEMP_CACHE[idempotency_key] = resp
            _IDEMP_CACHE_TS[idempotency_key] = time.time()
    except Exception:
        pass
    return resp


@router.get("/ai/memory/stats")
async def ai_memory_stats(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    if _is_ai_memory_auth_required():
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    t0 = time.perf_counter()
    count = 0
    last_ts: int | None = None
    try:
        if AI_MEMORY_STORE is not None:
            cur = AI_MEMORY_STORE.conn.execute("SELECT COUNT(1), MAX(ts) FROM ai_memory")
            row = cur.fetchone() or [0, None]
            count = int(row[0] or 0)
            last_raw = row[1]
            last_ts = int(last_raw) if last_raw is not None else None
        else:
            # Fallback to in-memory ring
            mem = list(AI_MEMORY_RING)
            count = len(mem)
            last_ts = int(mem[-1].get("ts") or 0) if mem else None
    except Exception:
        pass
    resp = {"ok": True, "count": count, "last_ts": last_ts}
    try:
        if _H_AI_MEMORY_LAT is not None:
            _H_AI_MEMORY_LAT.labels(endpoint="stats").observe(time.perf_counter() - t0)
        if _C_AI_MEMORY_REQ is not None:
            _C_AI_MEMORY_REQ.labels(endpoint="stats", result="ok").inc()
    except Exception:
        pass
    return resp


@router.get("/ai/memory/recent")
async def ai_memory_recent(
    limit: int = 50,
    offset: int = 0,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    if _is_ai_memory_auth_required():
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    t0 = time.perf_counter()
    try:
        lim = max(1, min(int(limit), 200))
    except Exception:
        lim = 50
    try:
        off = max(0, int(offset))
    except Exception:
        off = 0

    items: list[dict[str, Any]] = []
    total = 0
    try:
        if AI_MEMORY_STORE is not None:
            # total count
            try:
                cur = AI_MEMORY_STORE.conn.execute("SELECT COUNT(1) FROM ai_memory")
                row = cur.fetchone()
                total = int((row[0] if row else 0) or 0)
            except Exception:
                total = 0
            # page of recent items
            cur = AI_MEMORY_STORE.conn.execute(
                "SELECT * FROM ai_memory ORDER BY ts DESC LIMIT ? OFFSET ?",
                (lim, off),
            )
            rows = cur.fetchall() or []
            for r in rows:
                d = _serialize_memory_decision(r)
                # Backfill qty/avg from features for legacy consumers
                feats = d.get("features") or {}
                d_legacy = {
                    "ts": d.get("ts") or 0,
                    "price": d.get("price") or 0.0,
                    "prev": d.get("prev") or 0.0,
                    "qty": float(feats.get("qty") or 0.0),
                    "avg": float(feats.get("avg_cost") or 0.0),
                    "news_score": (d.get("news_score") if d.get("news_score") is not None else 0.0),
                    "features": feats,
                    "label_next_move": d.get("label_next_move") or 0,
                    "action": d.get("action") or "HOLD",
                    "advisory": d.get("reasoning") or "",
                    "confidence": int(round((d.get("confidence_float") or 0.0) * 100)),
                }
                items.append(d_legacy)
        else:
            # Fallback to in-memory ring (newest first)
            mem = list(reversed(list(AI_MEMORY_RING)))
            total = len(mem)
            items = mem[off : off + lim]
    except Exception:
        # As a last resort no items
        items = []
        total = 0

    resp = {"ok": True, "items": items, "total": total, "limit": lim, "offset": off}
    try:
        if _H_AI_MEMORY_LAT is not None:
            _H_AI_MEMORY_LAT.labels(endpoint="recent").observe(time.perf_counter() - t0)
        if _C_AI_MEMORY_REQ is not None:
            _C_AI_MEMORY_REQ.labels(endpoint="recent", result="ok").inc()
    except Exception:
        pass
    return resp


@router.post("/ai/memory/debug/auth")
async def ai_memory_debug_auth(on: int = 1):
    # Only allow in explicit test mode; otherwise 404 to avoid accidental exposure
    if os.getenv("SNAP_TEST_MODE", "0") not in ("1", "true", "yes"):
        raise HTTPException(status_code=404, detail="Not found")
    try:
        global _AI_MEMORY_AUTH_REQUIRED
        _AI_MEMORY_AUTH_REQUIRED = bool(int(on))
    except Exception:
        _AI_MEMORY_AUTH_REQUIRED = True
    return {"ok": True, "memory_auth": _AI_MEMORY_AUTH_REQUIRED}


@router.post("/ai/memory/similar")
async def ai_memory_similar(
    payload: dict[str, Any], credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    if _is_ai_memory_auth_required():
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    t0 = time.perf_counter()
    try:
        if AI_MEMORY_STORE is None:
            return JSONResponse({"ok": False, "error": "memory_unavailable"}, 503)

        k = int(payload.get("k", payload.get("limit", 10)))
        filters = payload.get("filters") or {}

        # Get current price for similarity matching
        price, prev, provider = get_wolf_price()

        current_state = {
            "symbol": payload.get("symbol") or WOLF,
            "price": payload.get("price") or price or 0.0,
            "features": payload.get("features") or {},
        }
        similar = AI_MEMORY_STORE.find_similar_situations(current_state, k=k, filters=filters)
        out = [_serialize_memory_decision(r) for r in similar]
        return {"ok": True, "items": out, "count": len(out)}
    except Exception as e:
        LOGGER.exception("ai_memory_similar_failed", extra={"error": str(e)})
        return JSONResponse({"ok": False, "error": str(e)}, 500)
    finally:
        try:
            if _H_AI_MEMORY_LAT is not None:
                _H_AI_MEMORY_LAT.labels(endpoint="similar").observe(time.perf_counter() - t0)
            if _C_AI_MEMORY_REQ is not None:
                _C_AI_MEMORY_REQ.labels(endpoint="similar", result="ok").inc()
        except Exception:
            pass


@router.get("/research/snapshot")
async def api_research_snapshot(symbol: str = WOLF, asset_type: str = "stock"):
    if not RESEARCH_BLUEPRINT_ON:
        return {"ok": False, "error": "research_blueprint_unavailable"}
    try:
        snap = build_research_snapshot(symbol, asset_type=asset_type)
        return {"ok": True, "snapshot": snap}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/preview")
async def ai_preview():
    import os

    price, prev, provider = get_wolf_price()
    qty = float(STATE.get("qty", 0.0))
    avg = float(STATE.get("avg_cost", 0.0))
    ns = None
    try:
        ns = (get_wolf_news(limit=1).get("news_signal") or {}).get("score")
    except Exception:
        ns = None
    feats = _extract_features(price, prev, qty, avg, ns)
    gps, conf, reasons, analogs = _ai_infer(feats)

    # Return analogs from AI inference
    if not analogs:
        # No analogs available — return empty list instead of fabricated data
        analogs = []

    return {
        "gps": float(f"{gps:.2f}"),
        "confidence": int(conf),
        "reasons": reasons,
        "analogs": analogs,
        "features": feats,
    }


@router.post("/ai/train")
async def ai_train(
    body: TrainBody | None = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    # Manual training workflow required - see docs/AI_TRAINING.md
    raise HTTPException(501, "AI training requires manual workflow - not automated")


@router.post("/ai/backfill")
async def ai_backfill(days: int = 30, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    # Backfill not implemented - AI memory populated in real-time only
    raise HTTPException(501, "Backfill not implemented - memory is real-time only")


@router.get("/api/v3/analyst/status")
async def api_analyst_status():
    """Get GPT-4 Analyst status."""
    try:
        from llm.gpt4_analyst import get_status, is_enabled
        return {
            "ok": True,
            **get_status(),
            "is_enabled": is_enabled(),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/analyst/analyze/{symbol}")
async def api_analyst_analyze(symbol: str):
    """Get GPT-4 analysis for a symbol's prediction."""
    try:
        from llm.gpt4_analyst import analyze_prediction, is_enabled
        
        if not is_enabled():
            return {
                "ok": False,
                "error": "GPT-4 Analyst not enabled. Set ENABLE_GPT4_ANALYST=1 and OPENAI_API_KEY",
            }
        
        # Get latest prediction for this symbol
        pred = _LATEST_PREDICTIONS.get(symbol.upper(), {})
        if not pred:
            return {
                "ok": False,
                "error": f"No prediction found for {symbol}",
            }
        
        direction = pred.get("direction", "FLAT")
        confidence = pred.get("confidence", 0.5)
        current_price = pred.get("price_at_prediction", 0)
        target_price = pred.get("target_price", current_price)
        features = pred.get("feature_status", {})
        
        result = await analyze_prediction(
            symbol=symbol.upper(),
            direction=direction,
            confidence=confidence,
            current_price=current_price,
            target_price=target_price,
            features=features,
        )
        
        return result
        
    except Exception as e:
        LOGGER.error(f"Analyst analysis failed: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/analyst/commentary")
async def api_analyst_commentary():
    """Get general market commentary from GPT-4."""
    try:
        from llm.gpt4_analyst import get_market_commentary, is_enabled
        
        if not is_enabled():
            return {
                "ok": False,
                "error": "GPT-4 Analyst not enabled",
            }
        
        # Get top symbols from latest predictions
        symbols = list(_LATEST_PREDICTIONS.keys())[:5]
        
        return await get_market_commentary(symbols)
        
    except Exception as e:
        LOGGER.error(f"Market commentary failed: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/santiment/status")
async def api_santiment_status():
    """Get Santiment integration status"""
    try:
        from core.santiment_signals import is_enabled, get_santiment
        provider = get_santiment()
        return {
            "ok": True,
            "enabled": is_enabled(),
            "api_key_set": bool(provider.api_key),
            "cache_size": len(provider.cache)
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/santiment/{symbol}")
async def api_santiment_data(symbol: str):
    """Get Santiment social/on-chain data for a symbol"""
    try:
        from core.santiment_signals import get_sentiment_signal, is_enabled
        
        result = get_sentiment_signal(symbol.upper())
        if result:
            return {"ok": True, "enabled": is_enabled(), **result}
        else:
            return {"ok": False, "error": "No Santiment data available"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/vwap/{symbol}")
async def api_vwap_analysis(symbol: str):
    """Get VWAP analysis for a symbol"""
    try:
        from core.vwap_signals import get_vwap_signal
        
        result = get_vwap_signal(symbol.upper())
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/v3/agreement/check")
async def api_check_model_agreement(request: Request):
    """Check agreement across multiple model signals"""
    try:
        from core.model_agreement import check_model_agreement
        
        data = await request.json()
        signals = data.get("signals", {})
        
        if not signals:
            return {"ok": False, "error": "No signals provided"}
        
        result = check_model_agreement(signals)
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/exits/calculate")
async def api_calculate_exit_levels(
    entry_price: float,
    direction: str,
    confidence: float = 0.7
):
    """Calculate dynamic exit levels for a trade"""
    try:
        from core.dynamic_exits import calculate_exits
        
        levels = calculate_exits(entry_price, direction.upper(), confidence)
        return {"ok": True, **levels}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/v3/exits/check")
async def api_check_exit_condition(request: Request):
    """Check if exit condition is met for an open position"""
    try:
        from core.dynamic_exits import check_exit
        
        data = await request.json()
        
        result = check_exit(
            entry_price=data["entry_price"],
            current_price=data["current_price"],
            high_since_entry=data.get("high_since_entry", data["current_price"]),
            low_since_entry=data.get("low_since_entry", data["current_price"]),
            direction=data["direction"],
            hours_held=data.get("hours_held", 0),
            exit_levels=data["exit_levels"]
        )
        
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/mtf/{symbol}")
async def api_multi_timeframe(symbol: str):
    """Get multi-timeframe analysis for a symbol"""
    try:
        from core.multi_timeframe import _generate_timeframe_forecast
        
        results = {}
        for tf, hours in [("1h", 1), ("4h", 4), ("1d", 24)]:
            forecast = _generate_timeframe_forecast(symbol.upper(), tf, hours)
            results[tf] = forecast
        
        # Determine consensus
        directions = [r.get("direction") for r in results.values() if r.get("ok")]
        up_count = sum(1 for d in directions if d == "UP")
        down_count = sum(1 for d in directions if d == "DOWN")
        
        if up_count > down_count:
            consensus = "UP"
            agreement = up_count / len(directions) if directions else 0
        elif down_count > up_count:
            consensus = "DOWN"
            agreement = down_count / len(directions) if directions else 0
        else:
            consensus = "MIXED"
            agreement = 0.5
        
        return {
            "ok": True,
            "symbol": symbol.upper(),
            "consensus_direction": consensus,
            "agreement_pct": round(agreement * 100, 0),
            "timeframes": results
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


