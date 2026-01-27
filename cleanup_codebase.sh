#!/bin/bash
# ============================================
# GHOST PROTOCOL - CODEBASE CLEANUP SCRIPT
# ============================================
# This script organizes the codebase into a clean structure
# Run with: bash cleanup_codebase.sh
# ============================================

set -e
echo "🧹 GHOST PROTOCOL - CODEBASE CLEANUP"
echo "====================================="
echo ""

# Create archive directories
echo "📁 Creating organization directories..."
mkdir -p .archive/old_scripts
mkdir -p .archive/old_diagnostics
mkdir -p .archive/old_migrations
mkdir -p .archive/old_training
mkdir -p .archive/old_fixes
mkdir -p tests/integration
mkdir -p tests/unit
mkdir -p tools/diagnostics
mkdir -p scripts/training
mkdir -p scripts/migrations

# Move test files to tests/
echo "📦 Moving test files..."
TEST_FILES=(
    "test_200_day_demo.py"
    "test_accuracy_fixes.py"
    "test_agentkit_integration.py"
    "test_alpaca_broker.py"
    "test_apex_integration.py"
    "test_autonomous_execution.py"
    "test_comprehensive.py"
    "test_comprehensive_fixes.py"
    "test_concurrency_tools.py"
    "test_context.py"
    "test_crypto_module.py"
    "test_crypto_routing.py"
    "test_direct_fetch.py"
    "test_endpoints.py"
    "test_endpoints_and_accuracy.py"
    "test_features_local.py"
    "test_fixes.py"
    "test_general_queries.py"
    "test_ghost_neural_network.py"
    "test_honest_ghost.py"
    "test_level10.py"
    "test_meta_live.py"
    "test_morning_now.py"
    "test_news_brain_loop.py"
    "test_news_endpoints.py"
    "test_pepe.py"
    "test_polygon_debug.py"
    "test_polygon_historical.py"
    "test_polygon_hourly.py"
    "test_polygon_snapshot.py"
    "test_portfolio_persistence.py"
    "test_postgres_fixes.py"
    "test_prediction_endpoints.py"
    "test_predictions.py"
    "test_predictions_wiring.py"
    "test_price_quorum.py"
    "test_production_features.py"
    "test_production_fixes.py"
    "test_production_watchlist.py"
    "test_reconciler.py"
    "test_rss_fix.py"
    "test_server.py"
    "test_spy_vix_fix.py"
    "test_telegram_direct.py"
    "test_telegram_send.py"
    "test_telegram_watchlist.py"
    "test_ui_endpoints.py"
    "test_v2_strict_mode.py"
    "test_watchlist_endpoints.py"
    "test_wolf_queries.py"
    "test_world_feed.py"
    "test_yfinance_direct.py"
)

for f in "${TEST_FILES[@]}"; do
    if [ -f "$f" ]; then
        mv "$f" tests/ 2>/dev/null || true
        echo "  ✓ $f → tests/"
    fi
done

# Move diagnostic scripts
echo "📦 Moving diagnostic scripts..."
DIAG_FILES=(
    "diagnostic_scanner.py"
    "diagnostics.py"
    "railway_diagnostic.py"
    "repair_diagnostic.py"
    "repair_optimize.py"
    "check_env_vars.py"
    "check_outcomes_status.py"
    "check_production_predictions.py"
    "check_routes.py"
    "check_state.py"
    "check_status.py"
    "check_zec_features.py"
    "deep_dive_audit.py"
    "inspect_routes.py"
    "audit_ghost.py"
    "railway_health_check.py"
)

for f in "${DIAG_FILES[@]}"; do
    if [ -f "$f" ]; then
        mv "$f" tools/diagnostics/ 2>/dev/null || true
        echo "  ✓ $f → tools/diagnostics/"
    fi
done

# Move training scripts
echo "📦 Moving training scripts..."
TRAIN_FILES=(
    "train_ml_models.py"
    "train_ml_models_v2.py"
    "train_ml_models_v3.py"
    "train_ml_models_v3_hourly.py"
    "train_models_now.py"
    "retrain_model.py"
    "retrain_model_old.py"
    "retrain_monthly.py"
)

for f in "${TRAIN_FILES[@]}"; do
    if [ -f "$f" ]; then
        mv "$f" scripts/training/ 2>/dev/null || true
        echo "  ✓ $f → scripts/training/"
    fi
done

# Move migration scripts
echo "📦 Moving migration scripts..."
MIGRATE_FILES=(
    "migrate_ai_memory.py"
    "migrate_paper_trades_schema.py"
    "migrate_portfolio_to_nvda.py"
    "apply_outcome_migration.py"
    "apply_symbol_migration.py"
    "trigger_migration.py"
    "backfill_outcomes.py"
)

for f in "${MIGRATE_FILES[@]}"; do
    if [ -f "$f" ]; then
        mv "$f" scripts/migrations/ 2>/dev/null || true
        echo "  ✓ $f → scripts/migrations/"
    fi
done

# Archive likely dead code
echo "📦 Archiving likely dead code..."
ARCHIVE_FILES=(
    "activate_all_systems.py"
    "activate_to_the_moon.py"
    "add_missing_ui_endpoints.py"
    "add_simulation_endpoint.py"
    "add_stocks_to_watchlist.py"
    "add_test_data.py"
    "add_wolf_to_watchlist.py"
    "apply_comprehensive_fixes.py"
    "autofix_startup.py"
    "chat_with_ghost.py"
    "debug_uvicorn.py"
    "enhanced_price_fetcher.py"
    "enhanced_rate_limiter.py"
    "explain_issue.py"
    "fix_news_order.py"
    "fix_telegram.py"
    "fix_timezone.py"
    "force_price_refresh.py"
    "generate_ops_report.py"
    "generate_simulation_data.py"
    "generate_status_report.py"
    "ghost_agent_loop.py"
    "ghost_bootstrap.py"
    "ghost_brain_enhanced.py"
    "ghost_diagnostic.py"
    "ghost_init.py"
    "ghost_rpc_guard.py"
    "ghost_state.py"
    "initialize_v2_whitelist.py"
    "intel_smoke_test.py"
    "launch_ghost.py"
    "monitor_prediction_accuracy.py"
    "monitor_telegram_bot.py"
    "ops_worker.py"
    "query_production_winrates.py"
    "quick_test.py"
    "quick_validate.py"
    "reconcile_historical_predictions.py"
    "reconcile_with_coingecko.py"
    "regression_audit.py"
    "reset_telegram_bot_name.py"
    "send_audit_summary.py"
    "send_completion_summary.py"
    "send_issue_found.py"
    "send_real_prediction.py"
    "send_real_prediction_api.py"
    "send_telegram_notification.py"
    "show_v2_status.py"
    "state_manager.py"
    "system_verification.py"
    "telegram_bot_security_integration.py"
    "ui_verification_audit.py"
    "update_whitelist_direct.py"
    "validate_ghost_fixes.py"
    "validate_ghost_predictions.py"
    "verify_bug_fixes.py"
    "verify_news_deployment.py"
    "verify_postgres_migration.py"
    "verify_production.py"
    "verify_simulation.py"
    "verify_stage1.py"
    "SCANNER_ENDPOINTS_TO_ADD.py"
    "run_full_system_test.py"
)

for f in "${ARCHIVE_FILES[@]}"; do
    if [ -f "$f" ]; then
        mv "$f" .archive/old_scripts/ 2>/dev/null || true
        echo "  ✓ $f → .archive/old_scripts/"
    fi
done

echo ""
echo "====================================="
echo "✅ CLEANUP COMPLETE"
echo "====================================="
echo ""
echo "📊 Summary:"
echo "  - Test files → tests/"
echo "  - Diagnostics → tools/diagnostics/"
echo "  - Training scripts → scripts/training/"
echo "  - Migration scripts → scripts/migrations/"
echo "  - Dead code → .archive/old_scripts/"
echo ""
echo "⚠️  Review .archive/ before deleting permanently"
echo ""

# Count remaining root files
echo "📁 Remaining root Python files:"
ls -1 *.py 2>/dev/null | wc -l

echo ""
echo "🎯 Active files that should remain in root:"
echo "  - wolf_app.py (main app)"
echo "  - main.py (entry point)"
echo "  - db.py (database)"
echo "  - signals.py (signals)"
echo "  - universe.py (asset universe)"
echo "  - dispatch.py (message dispatch)"
echo "  - sitecustomize.py (Python startup)"
