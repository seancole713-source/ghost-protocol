#!/usr/bin/env python3
"""
🎭 GHOST PROTOCOL MASTER ORCHESTRATOR
Unified background service coordinator - wires all autonomous systems at startup

Responsibilities:
1. Start all background tasks in correct dependency order
2. Manage scheduler consolidation (resolve competing systems)
3. Provide system health monitoring
4. Handle graceful shutdown
5. Expose orchestration status via API

Architecture:
- Price Refresh Loop (5-10s interval)
- Movers Scanner (stock: scheduled CT times, crypto: 5min)
- VIP Scanner (60s interval, Cash-App alerts for WEPE, LILPEPE, DORKL, SLOTH, APC)
- SL/TP Monitor (60s interval, conditional on BROKER_ENABLED)
- Scheduled Predictions (market hours only, consolidates beast_scheduler + scheduled_predictions)
- Stage 1 Context Engine (hourly RSS/sentiment refresh)
- Market Scanner (autonomous opportunity detection)
- Daily Reports (07:00 CT + 20:00 CT)
- Outcome Reconciler (60min interval, 48h accuracy measurement)

Usage:
    from core.orchestrator import start_all_background_services, get_system_status
    
    @app.on_event("startup")
    async def startup():
        await start_all_background_services(app, logger, redis_client)
"""

import asyncio
import logging
import os
import time

LOGGER: logging.Logger | None = None

# Background task handles
_TASKS = {}
_START_TIME = 0
_SYSTEM_STATUS = {
    "price_refresh": {"status": "stopped", "last_run": 0, "error": None},
    "movers_scanner": {"status": "stopped", "last_run": 0, "error": None},
    "vip_scanner": {"status": "stopped", "last_run": 0, "error": None},
    "sl_tp_monitor": {"status": "stopped", "last_run": 0, "error": None},
    "scheduled_predictions": {"status": "stopped", "last_run": 0, "error": None},
    "context_engine": {"status": "stopped", "last_run": 0, "error": None},
    "market_scanner": {"status": "stopped", "last_run": 0, "error": None},
    "daily_reports": {"status": "stopped", "last_run": 0, "error": None},
    "outcome_reconciler": {"status": "stopped", "last_run": 0, "error": None},
    "autonomous_execution": {"status": "stopped", "last_run": 0, "error": None},
}


async def start_all_background_services(
    app,
    logger,
    redis_client=None,
    fetch_price_func=None,
    run_prediction_func=None,
):
    """
    Master orchestration: Start all Ghost background services in dependency-safe order
    
    Args:
        app: FastAPI app instance
        logger: Logger instance
        redis_client: Redis client for caching
        fetch_price_func: Price fetching function
        run_prediction_func: Prediction runner function
    """
    global LOGGER, _START_TIME
    LOGGER = logger
    _START_TIME = time.time()
    
    LOGGER.info("🎭 MASTER ORCHESTRATOR: Initializing all background services...")
    
    # ============================================================================
    # PHASE 1: PRICE REFRESH (Core Dependency - Must Start First)
    # ============================================================================
    try:
        from wolf_app import _auto_refresh_price
        _TASKS["price_refresh"] = asyncio.create_task(_auto_refresh_price())
        _SYSTEM_STATUS["price_refresh"]["status"] = "running"
        _SYSTEM_STATUS["price_refresh"]["last_run"] = int(time.time())
        LOGGER.info("✅ Price Refresh Loop: STARTED (5-10s interval)")
    except Exception as e:
        _SYSTEM_STATUS["price_refresh"]["status"] = "failed"
        _SYSTEM_STATUS["price_refresh"]["error"] = str(e)
        LOGGER.error(f"❌ Price Refresh Loop FAILED: {e}", exc_info=True)
    
    # ============================================================================
    # PHASE 2: MOVERS SCANNER (Depends on Price Refresh)
    # ============================================================================
    try:
        from wolf_app import _auto_scan_movers
        _TASKS["movers_scanner"] = asyncio.create_task(_auto_scan_movers())
        _SYSTEM_STATUS["movers_scanner"]["status"] = "running"
        _SYSTEM_STATUS["movers_scanner"]["last_run"] = int(time.time())
        LOGGER.info("✅ Movers Scanner: STARTED (stocks: scheduled CT times, crypto: 5min)")
    except Exception as e:
        _SYSTEM_STATUS["movers_scanner"]["status"] = "failed"
        _SYSTEM_STATUS["movers_scanner"]["error"] = str(e)
        LOGGER.error(f"❌ Movers Scanner FAILED: {e}", exc_info=True)

    # ============================================================================
    # PHASE 2B: VIP MICROCAP SCANNER (Priority 2: Cash-App Alerts for WEPE, LILPEPE, DORKL, SLOTH, APC)
    # ============================================================================
    vip_scanner_enabled = os.getenv("VIP_SCANNER_ENABLED", "1") == "1"

    if vip_scanner_enabled:
        try:
            from core.vip_scanner import scan_vip_coins

            async def _vip_scanner_loop():
                """Background loop for VIP microcap scanning"""
                from core.vip_scanner import VIP_SCAN_INTERVAL_S
                while True:
                    try:
                        result = scan_vip_coins()
                        _SYSTEM_STATUS["vip_scanner"]["last_run"] = int(time.time())
                        LOGGER.info(
                            f"VIP scan: {result['available']}/{result['scanned']} available, "
                            f"{len(result['opportunities'])} opportunities, {result['alerts_sent']} alerts"
                        )
                    except Exception as e:
                        LOGGER.error(f"VIP scanner error: {e}", exc_info=True)
                    await asyncio.sleep(VIP_SCAN_INTERVAL_S)

            _TASKS["vip_scanner"] = asyncio.create_task(_vip_scanner_loop())
            _SYSTEM_STATUS["vip_scanner"]["status"] = "running"
            _SYSTEM_STATUS["vip_scanner"]["last_run"] = int(time.time())
            LOGGER.info("✅ VIP Microcap Scanner: STARTED (60s interval, Cash-App alerts)")
        except Exception as e:
            _SYSTEM_STATUS["vip_scanner"]["status"] = "failed"
            _SYSTEM_STATUS["vip_scanner"]["error"] = str(e)
            LOGGER.error(f"❌ VIP Scanner FAILED: {e}", exc_info=True)
    else:
        _SYSTEM_STATUS["vip_scanner"]["status"] = "disabled"
        LOGGER.info("⚪ VIP Microcap Scanner: DISABLED (VIP_SCANNER_ENABLED=0)")
    
    # ============================================================================
    # PHASE 2.6: PRE-MARKET PREDICTOR
    # ============================================================================
    premarket_enabled = os.getenv("PREMARKET_ENABLED", "1") == "1"
    
    if premarket_enabled:
        try:
            from core.premarket_predictor import should_run_premarket, run_premarket_predictions
            
            _SYSTEM_STATUS["premarket_predictor"] = {
                "status": "checking",
                "enabled": True,
                "last_run": 0,
                "next_run": "7:00 AM CT weekdays"
            }
            
            async def _premarket_check_loop():
                """Check every 5 minutes if it's time to run pre-market predictions"""
                while True:
                    try:
                        should_run, reason = should_run_premarket()
                        
                        if should_run:
                            LOGGER.info("🌅 Starting pre-market predictions...")
                            result = await run_premarket_predictions()
                            _SYSTEM_STATUS["premarket_predictor"]["status"] = "completed"
                            _SYSTEM_STATUS["premarket_predictor"]["last_run"] = result['run_at']
                            LOGGER.info(
                                f"🌅 Pre-market complete: {len(result['predictions'])} predictions, "
                                f"{result['alerts_sent']} alerts sent"
                            )
                        else:
                            _SYSTEM_STATUS["premarket_predictor"]["status"] = f"waiting ({reason})"
                    
                    except Exception as e:
                        _SYSTEM_STATUS["premarket_predictor"]["status"] = "error"
                        _SYSTEM_STATUS["premarket_predictor"]["error"] = str(e)
                        LOGGER.error(f"Pre-market check error: {e}", exc_info=True)
                    
                    await asyncio.sleep(300)  # Check every 5 minutes
            
            _TASKS["premarket_predictor"] = asyncio.create_task(_premarket_check_loop())
            _SYSTEM_STATUS["premarket_predictor"]["status"] = "waiting"
            LOGGER.info("✅ Pre-market Predictor: STARTED (checks every 5min, runs 7:00 AM CT)")
        
        except Exception as e:
            _SYSTEM_STATUS["premarket_predictor"]["status"] = "failed"
            _SYSTEM_STATUS["premarket_predictor"]["error"] = str(e)
            LOGGER.error(f"❌ Pre-market Predictor FAILED: {e}", exc_info=True)
    else:
        _SYSTEM_STATUS["premarket_predictor"] = {"status": "disabled", "enabled": False}
        LOGGER.info("⚪ Pre-market Predictor: DISABLED (PREMARKET_ENABLED=0)")
    
    # ============================================================================
    # PHASE 3: SL/TP MONITOR (Conditional - Only if Broker Enabled)
    # ============================================================================
    broker_enabled = os.getenv("BROKER_ENABLED", "0") == "1"
    sl_tp_enabled = os.getenv("SL_TP_MONITOR_ENABLED", "1") == "1"
    
    if broker_enabled and sl_tp_enabled:
        try:
            from core.sl_tp_monitor import start_sl_tp_monitor
            _TASKS["sl_tp_monitor"] = asyncio.create_task(start_sl_tp_monitor())
            _SYSTEM_STATUS["sl_tp_monitor"]["status"] = "running"
            _SYSTEM_STATUS["sl_tp_monitor"]["last_run"] = int(time.time())
            LOGGER.info("✅ SL/TP Monitor: STARTED (60s interval)")
        except Exception as e:
            _SYSTEM_STATUS["sl_tp_monitor"]["status"] = "failed"
            _SYSTEM_STATUS["sl_tp_monitor"]["error"] = str(e)
            LOGGER.error(f"❌ SL/TP Monitor FAILED: {e}", exc_info=True)
    else:
        _SYSTEM_STATUS["sl_tp_monitor"]["status"] = "disabled"
        LOGGER.info("⚪ SL/TP Monitor: DISABLED (BROKER_ENABLED=0)")
    
    # ============================================================================
    # PHASE 4: SCHEDULED PREDICTIONS (Consolidated Scheduler)
    # ============================================================================
    # DECISION: Use beast_scheduler as primary (more comprehensive)
    # DISABLED: scheduled_predictions.py (redundant multi-symbol variant)
    try:
        from core import beast_scheduler
        
        # Inject dependencies
        beast_scheduler.REDIS_CLIENT = redis_client
        beast_scheduler.LOGGER = logger
        beast_scheduler.GET_PRICE_FUNC = fetch_price_func
        beast_scheduler.RUN_PREDICTION_FUNC = run_prediction_func
        
        # Import telegram_alerts module
        from core import telegram_alerts
        beast_scheduler.TELEGRAM_ALERTS_MODULE = telegram_alerts
        
        # Start scheduler
        beast_scheduler.start_beast_scheduler()
        _SYSTEM_STATUS["scheduled_predictions"]["status"] = "running"
        _SYSTEM_STATUS["scheduled_predictions"]["last_run"] = int(time.time())
        LOGGER.info("✅ Scheduled Predictions (Beast Scheduler): STARTED")
        LOGGER.info("   Stocks: 07:55, 09:35, 12:00, 15:10 CT")
        LOGGER.info("   Crypto: Every 2 hours")
    except Exception as e:
        _SYSTEM_STATUS["scheduled_predictions"]["status"] = "failed"
        _SYSTEM_STATUS["scheduled_predictions"]["error"] = str(e)
        LOGGER.error(f"❌ Scheduled Predictions FAILED: {e}", exc_info=True)
    
    # ============================================================================
    # PHASE 5: STAGE 1 CONTEXT ENGINE (Hourly RSS/Sentiment Refresh)
    # ============================================================================
    context_enabled = os.getenv("STAGE1_CONTEXT_ENABLED", "1") == "1"
    
    if context_enabled:
        try:
            from core.context_engine import start_background_updater
            
            # Start background updater with 60-minute refresh interval
            _TASKS["context_engine"] = asyncio.create_task(
                start_background_updater(refresh_interval_minutes=60)
            )
            _SYSTEM_STATUS["context_engine"]["status"] = "running"
            _SYSTEM_STATUS["context_engine"]["last_run"] = int(time.time())
            LOGGER.info("✅ Stage 1 Context Engine: STARTED (hourly refresh)")
        except Exception as e:
            _SYSTEM_STATUS["context_engine"]["status"] = "failed"
            _SYSTEM_STATUS["context_engine"]["error"] = str(e)
            LOGGER.error(f"❌ Stage 1 Context Engine FAILED: {e}", exc_info=True)
    else:
        _SYSTEM_STATUS["context_engine"]["status"] = "disabled"
        LOGGER.info("⚪ Stage 1 Context Engine: DISABLED (set STAGE1_CONTEXT_ENABLED=1 to enable)")
    
    # ============================================================================
    # PHASE 6: MARKET SCANNER (Autonomous Opportunity Detection)
    # ============================================================================
    # NOTE: market_scanner.py has scan_all() function but no background loop
    # It's designed to be called on-demand via API endpoints
    # If we want autonomous scanning, we'd need to create a loop here
    _SYSTEM_STATUS["market_scanner"]["status"] = "on_demand"
    LOGGER.info("⚪ Market Scanner: ON-DEMAND MODE (API endpoints only)")
    
    # ============================================================================
    # PHASE 7: DAILY REPORTS (07:00 CT + 20:00 CT)
    # ============================================================================
    try:
        from core.telegram_hunter import daily_report_loop
        from core.market_scanner import scan_all
        from core.prediction_tracker import calculate_accuracy
        
        async def get_top_opportunities():
            """Get top opportunities for daily report"""
            results = await scan_all()
            all_opps = results["stocks"] + results["crypto"]
            all_opps.sort(key=lambda x: x.get("score", 0), reverse=True)
            return all_opps[:10]
        
        async def get_accuracy_stats(period="24h"):
            """Get accuracy stats for daily report"""
            return calculate_accuracy(period)
        
        _TASKS["daily_reports"] = asyncio.create_task(
            daily_report_loop(get_top_opportunities, get_accuracy_stats)
        )
        _SYSTEM_STATUS["daily_reports"]["status"] = "running"
        _SYSTEM_STATUS["daily_reports"]["last_run"] = int(time.time())
        LOGGER.info("✅ Daily Reports: STARTED (07:00 CT + 20:00 CT)")
    except Exception as e:
        _SYSTEM_STATUS["daily_reports"]["status"] = "failed"
        _SYSTEM_STATUS["daily_reports"]["error"] = str(e)
        LOGGER.error(f"❌ Daily Reports FAILED: {e}", exc_info=True)
    
    # ============================================================================
    # PHASE 8: OUTCOME RECONCILER (48h Accuracy Measurement)
    # ============================================================================
    reconciler_enabled = os.getenv("OUTCOME_RECONCILER_ENABLED", "1") == "1"
    
    if reconciler_enabled:
        try:
            from services.outcome_reconciler_v2 import reconcile_outcomes_v2
            
            async def _outcome_reconciler_loop():
                """
                Background loop for reconciling prediction outcomes after 48h window closes.
                This is CRITICAL for accuracy tracking and learning loop.
                
                Runs every 60 minutes to:
                1. Find predictions where 48h has elapsed
                2. Fetch actual prices from live providers
                3. Calculate MAE, MAPE, RMSE, direction accuracy
                4. Store in ghost_prediction_outcomes table
                """
                reconciler_interval_s = int(os.getenv("OUTCOME_RECONCILER_INTERVAL_S", "3600"))  # Default 60min
                
                while True:
                    try:
                        result = reconcile_outcomes_v2()
                        _SYSTEM_STATUS["outcome_reconciler"]["last_run"] = int(time.time())
                        
                        if result["success"] > 0:
                            LOGGER.info(
                                f"✅ Reconciler: {result['success']} outcomes processed, "
                                f"{result['no_data']} no data, {result['error']} errors, "
                                f"{result['skipped']} skipped"
                            )
                        else:
                            LOGGER.debug(
                                f"Reconciler: {result['success']} outcomes (no new data this cycle)"
                            )
                    
                    except Exception as e:
                        _SYSTEM_STATUS["outcome_reconciler"]["error"] = str(e)
                        LOGGER.error(f"Outcome reconciler error: {e}", exc_info=True)
                    
                    await asyncio.sleep(reconciler_interval_s)
            
            _TASKS["outcome_reconciler"] = asyncio.create_task(_outcome_reconciler_loop())
            _SYSTEM_STATUS["outcome_reconciler"]["status"] = "running"
            _SYSTEM_STATUS["outcome_reconciler"]["last_run"] = int(time.time())
            
            reconciler_interval_min = int(os.getenv("OUTCOME_RECONCILER_INTERVAL_S", "3600")) // 60
            LOGGER.info(f"✅ Outcome Reconciler: STARTED ({reconciler_interval_min}min interval)")
        
        except Exception as e:
            _SYSTEM_STATUS["outcome_reconciler"]["status"] = "failed"
            _SYSTEM_STATUS["outcome_reconciler"]["error"] = str(e)
            LOGGER.error(f"❌ Outcome Reconciler FAILED: {e}", exc_info=True)
    else:
        _SYSTEM_STATUS["outcome_reconciler"]["status"] = "disabled"
        LOGGER.info("⚪ Outcome Reconciler: DISABLED (OUTCOME_RECONCILER_ENABLED=0)")
    
    # ============================================================================
    # PHASE 5: AUTONOMOUS EXECUTION ENGINE (Trade on Predictions)
    # ============================================================================
    auto_execution_enabled = os.getenv("AUTO_EXECUTION_ENABLED", "0") == "1"
    
    if auto_execution_enabled:
        try:
            from core.autonomous_execution_engine import start_autonomous_execution_loop
            
            # Get execution interval (default: 300s = 5 minutes)
            execution_interval_s = int(os.getenv("AUTO_EXECUTION_INTERVAL_S", "300"))
            
            _TASKS["autonomous_execution"] = asyncio.create_task(
                start_autonomous_execution_loop(interval_s=execution_interval_s)
            )
            _SYSTEM_STATUS["autonomous_execution"]["status"] = "running"
            _SYSTEM_STATUS["autonomous_execution"]["last_run"] = int(time.time())
            
            execution_interval_min = execution_interval_s // 60
            LOGGER.info(f"✅ Autonomous Execution Engine: STARTED ({execution_interval_min}min interval)")
            LOGGER.warning("⚠️  [AUTO-EXEC] LIVE TRADING ENABLED - Monitor positions carefully!")
        except Exception as e:
            _SYSTEM_STATUS["autonomous_execution"]["status"] = "failed"
            _SYSTEM_STATUS["autonomous_execution"]["error"] = str(e)
            LOGGER.error(f"❌ Autonomous Execution FAILED: {e}", exc_info=True)
    else:
        _SYSTEM_STATUS["autonomous_execution"]["status"] = "disabled"
        LOGGER.info("⚪ Autonomous Execution Engine: DISABLED (set AUTO_EXECUTION_ENABLED=1 to enable)")
        LOGGER.info("   ℹ️  Predictions will be generated but not automatically traded")
    
    # ============================================================================
    # PHASE 6: SPIKE DETECTOR (Catch Random Price Spikes & News Catalysts)
    # ============================================================================
    spike_detector_enabled = os.getenv("SPIKE_DETECTOR_ENABLED", "1") == "1"
    
    if spike_detector_enabled:
        try:
            from core.spike_detector import spike_scanner_loop
            from core.beast_scheduler import STOCK_SYMBOLS, CRYPTO_SYMBOLS
            
            # Combine all tracked symbols
            all_symbols = STOCK_SYMBOLS + CRYPTO_SYMBOLS
            
            _TASKS["spike_detector"] = asyncio.create_task(spike_scanner_loop(all_symbols))
            _SYSTEM_STATUS["spike_detector"] = {
                "status": "running",
                "enabled": True,
                "last_run": int(time.time()),
                "symbols_tracked": len(all_symbols)
            }
            
            LOGGER.info(f"🚀 Spike Detector: STARTED (tracking {len(all_symbols)} symbols)")
            LOGGER.info("   ✅ Pre-market spikes (>5%)")
            LOGGER.info("   ✅ Unusual volume (10x+ average)")
            LOGGER.info("   ✅ Breaking news/catalysts")
            LOGGER.info("   ✅ Social sentiment surges")
        except Exception as e:
            _SYSTEM_STATUS["spike_detector"] = {
                "status": "failed",
                "enabled": False,
                "error": str(e)
            }
            LOGGER.error(f"❌ Spike Detector FAILED: {e}", exc_info=True)
    else:
        _SYSTEM_STATUS["spike_detector"] = {"status": "disabled", "enabled": False}
        LOGGER.info("⚪ Spike Detector: DISABLED (set SPIKE_DETECTOR_ENABLED=1 to enable)")
    
    # ============================================================================
    # ORCHESTRATION COMPLETE
    # ============================================================================
    uptime = time.time() - _START_TIME
    active_tasks = sum(1 for s in _SYSTEM_STATUS.values() if s["status"] == "running")
    total_services = len(_SYSTEM_STATUS)
    
    LOGGER.info("=" * 80)
    LOGGER.info(f"🎭 MASTER ORCHESTRATOR: Initialization complete in {uptime:.2f}s")
    LOGGER.info(f"📊 Status: {active_tasks}/{total_services} services running")
    LOGGER.info("=" * 80)
    
    # Log individual service status
    for service, status in _SYSTEM_STATUS.items():
        status_icon = {
            "running": "✅",
            "failed": "❌",
            "disabled": "⚪",
            "stopped": "🔴",
            "on_demand": "🟡",
        }.get(status["status"], "❓")
        
        LOGGER.info(f"{status_icon} {service}: {status['status']}")
        if status.get("error"):
            LOGGER.error(f"   Error: {status['error']}")


def get_system_status() -> dict:
    """
    Get current system orchestration status
    
    Returns:
        Dict with service statuses, uptime, and health metrics
    """
    uptime_seconds = int(time.time() - _START_TIME) if _START_TIME else 0
    
    return {
        "ok": True,
        "uptime_seconds": uptime_seconds,
        "services": _SYSTEM_STATUS.copy(),
        "active_tasks": sum(1 for s in _SYSTEM_STATUS.values() if s["status"] == "running"),
        "total_services": len(_SYSTEM_STATUS),
        "timestamp": int(time.time()),
    }


async def shutdown_all_services():
    """
    Gracefully shutdown all background services
    """
    global _TASKS
    
    if LOGGER:
        LOGGER.info("🛑 MASTER ORCHESTRATOR: Shutting down all background services...")
    
    # Cancel all tasks
    for service_name, task in _TASKS.items():
        try:
            task.cancel()
            await task
        except asyncio.CancelledError:
            if LOGGER:
                LOGGER.info(f"✅ {service_name}: Stopped")
        except Exception as e:
            if LOGGER:
                LOGGER.error(f"❌ {service_name}: Shutdown error: {e}")
    
    # Stop beast scheduler (threading-based)
    try:
        from core import beast_scheduler
        beast_scheduler.stop_beast_scheduler()
        if LOGGER:
            LOGGER.info("✅ Beast Scheduler: Stopped")
    except Exception as e:
        if LOGGER:
            LOGGER.error(f"❌ Beast Scheduler: Shutdown error: {e}")
    
    _TASKS.clear()
    
    for service in _SYSTEM_STATUS:
        _SYSTEM_STATUS[service]["status"] = "stopped"
    
    if LOGGER:
        LOGGER.info("🛑 MASTER ORCHESTRATOR: All services stopped")
