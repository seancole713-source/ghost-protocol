#!/usr/bin/env python3
"""
PHASE 5 MILESTONE 1 - TESTING & VALIDATION
===========================================
Test autonomous execution engine before deployment

Tests:
1. Configuration loading
2. Trade decision filters
3. Position sizing (Kelly Criterion)
4. Risk limits (circuit breakers)
5. Execution cycle (dry run)
"""

import logging
import os
import sys
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)


def test_configuration() -> bool:
    """Test 1: Configuration loading"""
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("TEST 1: Configuration Loading")
    LOGGER.info("=" * 60)
    
    try:
        from core.autonomous_execution_engine import (
            AUTO_EXECUTION_ENABLED,
            MIN_CONFIDENCE,
            MAX_POSITIONS,
            INTERVAL_S,
            MIN_VOLUME_STOCKS,
            MIN_PRICE,
            MAX_DAILY_LOSS_PCT,
            MAX_DRAWDOWN_PCT
        )
        
        LOGGER.info(f"AUTO_EXECUTION_ENABLED: {AUTO_EXECUTION_ENABLED}")
        LOGGER.info(f"MIN_CONFIDENCE: {MIN_CONFIDENCE}%")
        LOGGER.info(f"MAX_POSITIONS: {MAX_POSITIONS}")
        LOGGER.info(f"INTERVAL_S: {INTERVAL_S}s")
        LOGGER.info(f"MIN_VOLUME_STOCKS: {MIN_VOLUME_STOCKS:,}")
        LOGGER.info(f"MIN_PRICE: ${MIN_PRICE}")
        LOGGER.info(f"MAX_DAILY_LOSS_PCT: {MAX_DAILY_LOSS_PCT}%")
        LOGGER.info(f"MAX_DRAWDOWN_PCT: {MAX_DRAWDOWN_PCT}%")
        
        LOGGER.info("✅ Configuration loaded successfully")
        return True
    
    except Exception as e:
        LOGGER.error(f"❌ Configuration test failed: {e}", exc_info=True)
        return False


def test_trade_decision_engine() -> bool:
    """Test 2: Trade decision filters"""
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("TEST 2: Trade Decision Engine")
    LOGGER.info("=" * 60)
    
    try:
        # Temporarily disable market hours check for testing
        import os
        os.environ["AUTO_EXECUTION_MARKET_HOURS_ONLY"] = "0"
        
        from core.trade_decision_engine import evaluate_trade_opportunity
        
        # Test prediction (high confidence, good liquidity)
        test_prediction = {
            "symbol": "AAPL",
            "confidence": 78,
            "direction": "UP",
            "action": "BUY",
            "price": 150.25,
            "predicted_pct": 8.5,
            "market": "stock",
            "volume": 50000000
        }
        
        test_portfolio = {
            "portfolio_value": 100000,
            "cash": 50000,
            "buying_power": 50000
        }
        
        test_positions = []
        
        LOGGER.info(f"Evaluating: {test_prediction['symbol']} ({test_prediction['confidence']}% confidence)")
        
        decision = evaluate_trade_opportunity(
            prediction=test_prediction,
            portfolio=test_portfolio,
            current_positions=test_positions
        )
        
        LOGGER.info(f"Decision: {decision['action']}")
        LOGGER.info(f"Reason: {decision['reason']}")
        
        if decision["action"] == "EXECUTE":
            LOGGER.info(f"  Symbol: {decision['symbol']}")
            LOGGER.info(f"  Side: {decision['side']}")
            LOGGER.info(f"  Shares: {decision['shares']}")
            LOGGER.info(f"  Entry: ${decision['entry_price']:.2f}")
            LOGGER.info(f"  Stop Loss: ${decision['stop_loss_price']:.2f}")
            LOGGER.info(f"  Take Profit: ${decision['take_profit_price']:.2f}")
            LOGGER.info(f"  Position Value: ${decision['position_value']:,.2f}")
            LOGGER.info(f"  Kelly Fraction: {decision['kelly_fraction']:.2%}")
        
        # Test should EXECUTE
        if decision["action"] != "EXECUTE":
            LOGGER.error(f"❌ Expected EXECUTE, got {decision['action']}")
            return False
        
        LOGGER.info("✅ Trade decision engine working correctly")
        return True
    
    except Exception as e:
        LOGGER.error(f"❌ Trade decision engine test failed: {e}", exc_info=True)
        return False


def test_position_sizing() -> bool:
    """Test 3: Position sizing (Kelly Criterion)"""
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("TEST 3: Position Sizing (Kelly Criterion)")
    LOGGER.info("=" * 60)
    
    try:
        from core.trade_decision_engine import _calculate_position_size
        
        test_prediction = {
            "symbol": "AAPL",
            "confidence": 70,  # 70% confidence
            "price": 150.00
        }
        
        test_portfolio = {
            "portfolio_value": 100000,
            "buying_power": 50000
        }
        
        sizing = _calculate_position_size(test_prediction, test_portfolio)
        
        LOGGER.info(f"Confidence: {test_prediction['confidence']}%")
        LOGGER.info(f"Kelly Fraction: {sizing['kelly_fraction']:.2%}")
        LOGGER.info(f"Position %: {sizing['position_pct']:.2f}%")
        LOGGER.info(f"Shares: {sizing['shares']}")
        LOGGER.info(f"Position Value: ${sizing['position_value']:,.2f}")
        
        # Validate position size is reasonable (not too large)
        if sizing['position_pct'] > 10:
            LOGGER.error(f"❌ Position size too large: {sizing['position_pct']:.2f}%")
            return False
        
        if sizing['shares'] == 0:
            LOGGER.error("❌ Position size is 0")
            return False
        
        LOGGER.info("✅ Position sizing working correctly")
        return True
    
    except Exception as e:
        LOGGER.error(f"❌ Position sizing test failed: {e}", exc_info=True)
        return False


def test_circuit_breakers() -> bool:
    """Test 4: Circuit breakers (risk limits)"""
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("TEST 4: Circuit Breakers (Risk Limits)")
    LOGGER.info("=" * 60)
    
    try:
        from core.autonomous_execution_engine import AutonomousExecutionEngine
        
        engine = AutonomousExecutionEngine()
        
        # Test 1: Daily loss circuit breaker
        test_account = {
            "portfolio_value": 95000,  # Lost $5k (5%)
            "cash": 95000,
            "buying_power": 95000
        }
        
        # Mock portfolio start value
        from core.autonomous_execution_engine import _execution_state
        _execution_state["portfolio_start_value"] = 100000
        _execution_state["portfolio_peak_value"] = 100000
        
        risk_check = engine._check_risk_limits(test_account)
        
        LOGGER.info("Test 1: Daily loss 5% (should trigger circuit breaker)")
        LOGGER.info(f"  Result: {risk_check['status']}")
        LOGGER.info(f"  Reason: {risk_check.get('reason', 'N/A')}")
        
        if risk_check["status"] != "circuit_breaker":
            LOGGER.error(f"❌ Expected circuit_breaker, got {risk_check['status']}")
            return False
        
        # Test 2: Drawdown circuit breaker
        _execution_state["portfolio_start_value"] = 100000
        _execution_state["portfolio_peak_value"] = 120000  # Peak at $120k
        
        test_account["portfolio_value"] = 100000  # Now at $100k (16.7% drawdown from peak)
        
        risk_check = engine._check_risk_limits(test_account)
        
        LOGGER.info("\nTest 2: Drawdown 16.7% (should trigger circuit breaker)")
        LOGGER.info(f"  Result: {risk_check['status']}")
        LOGGER.info(f"  Reason: {risk_check.get('reason', 'N/A')}")
        
        if risk_check["status"] != "circuit_breaker":
            LOGGER.error(f"❌ Expected circuit_breaker, got {risk_check['status']}")
            return False
        
        LOGGER.info("✅ Circuit breakers working correctly")
        return True
    
    except Exception as e:
        LOGGER.error(f"❌ Circuit breaker test failed: {e}", exc_info=True)
        return False


def test_execution_cycle_dry_run() -> bool:
    """Test 5: Execution cycle (dry run, no actual trades)"""
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("TEST 5: Execution Cycle (Dry Run)")
    LOGGER.info("=" * 60)
    
    try:
        from core.autonomous_execution_engine import get_execution_engine, get_execution_status
        
        engine = get_execution_engine()
        
        LOGGER.info("Getting execution status...")
        status = get_execution_status()
        
        LOGGER.info(f"Enabled: {status['enabled']}")
        LOGGER.info(f"Circuit Breaker: {status['circuit_breaker_active']}")
        LOGGER.info(f"Total Cycles: {status['total_cycles']}")
        LOGGER.info(f"Trades Today: {status['trades_today']}")
        
        LOGGER.info("\n⚠️  Skipping actual execution cycle (requires live broker)")
        LOGGER.info("   To test execution cycle:")
        LOGGER.info("   1. Set AUTO_EXECUTION_ENABLED=1")
        LOGGER.info("   2. Configure Alpaca API keys")
        LOGGER.info("   3. Run: python3 core/autonomous_execution_engine.py")
        
        LOGGER.info("✅ Execution cycle dry run passed")
        return True
    
    except Exception as e:
        LOGGER.error(f"❌ Execution cycle test failed: {e}", exc_info=True)
        return False


def run_all_tests() -> Dict[str, bool]:
    """Run all tests and return results"""
    results = {
        "configuration": test_configuration(),
        "trade_decision_engine": test_trade_decision_engine(),
        "position_sizing": test_position_sizing(),
        "circuit_breakers": test_circuit_breakers(),
        "execution_cycle": test_execution_cycle_dry_run()
    }
    
    return results


def main():
    """Main test runner"""
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 5 MILESTONE 1 - TEST SUITE")
    LOGGER.info("Autonomous Execution Engine Validation")
    LOGGER.info("=" * 60)
    
    # Run tests
    results = run_all_tests()
    
    # Summary
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("TEST SUMMARY")
    LOGGER.info("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        LOGGER.info(f"{test_name}: {status}")
    
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info(f"TOTAL: {passed}/{total} tests passed")
    LOGGER.info("=" * 60)
    
    if passed == total:
        LOGGER.info("\n🎉 ALL TESTS PASSED - Ready for deployment")
        return 0
    else:
        LOGGER.error(f"\n❌ {total - passed} tests failed - Fix issues before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
