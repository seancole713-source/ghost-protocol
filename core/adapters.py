"""
Adapter Layer: Convert production prediction formats to core.models
====================================================================

This module bridges the gap between production prediction outputs
and the clean core.models types used by V3 filter and formatters.

Production formats vary depending on source:
1. run_single_prediction() - turbo engine output
2. stock_engine results - 24h horizon with gates
3. _LATEST_PREDICTIONS cache - stored predictions

Core format:
- core.models.Prediction - normalized input for V3 filter
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from core.models import Prediction, Direction


def _parse_direction(raw: str) -> Direction:
    """
    Convert production direction string to Direction enum.
    
    Production uses: 'UP', 'DOWN', 'BUY', 'SELL', 'HOLD', 'ERROR'
    Core uses: Direction.UP, Direction.DOWN
    
    BUY → UP, SELL → DOWN (for backwards compatibility)
    """
    raw_upper = (raw or "").upper().strip()
    
    if raw_upper in ("UP", "BUY"):
        return Direction.UP
    elif raw_upper in ("DOWN", "SELL"):
        return Direction.DOWN
    else:
        # HOLD, ERROR, or unknown → default to HOLD-like behavior
        # Caller should filter these out
        raise ValueError(f"Cannot convert direction '{raw}' to UP/DOWN")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_datetime(value: Any) -> datetime:
    """Safely parse datetime from various formats."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        # Unix timestamp
        return datetime.fromtimestamp(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            pass
    return datetime.now()


def production_to_prediction(raw: Dict[str, Any]) -> Optional[Prediction]:
    """
    Convert a single production prediction to core.models.Prediction.
    
    Production format (from run_single_prediction):
        {
            'ok': True,
            'symbol': 'ETH',
            'direction': 'DOWN',
            'confidence': 0.72,
            'current_price': 2310.0,
            'target_price': 2200.0,
            'stop_loss': 2350.0,
            'run_at': 1706990400,  # Unix timestamp
            ...
        }
    
    Returns:
        Prediction object, or None if conversion fails (e.g., HOLD direction)
    """
    # Skip failed predictions
    if not raw.get('ok', True):
        return None
    
    # Skip HOLD/ERROR directions
    direction_raw = raw.get('direction', 'HOLD')
    if direction_raw in ('HOLD', 'ERROR', None, ''):
        return None
    
    try:
        direction = _parse_direction(direction_raw)
    except ValueError:
        return None
    
    symbol = raw.get('symbol', '').upper().strip()
    if not symbol:
        return None
    
    confidence = _safe_float(raw.get('confidence'), 0.0)
    if confidence <= 0:
        return None
    
    current_price = _safe_float(raw.get('current_price') or raw.get('price_current'))
    target_price = _safe_float(raw.get('target_price') or raw.get('price_pred_mid'))
    stop_loss = _safe_float(raw.get('stop_loss'))
    
    # Derive stop_loss if missing (2% from entry)
    if stop_loss == 0.0 and current_price > 0:
        if direction == Direction.UP:
            stop_loss = round(current_price * 0.98, 4)
        else:
            stop_loss = round(current_price * 1.02, 4)
    
    # Derive target_price if missing (3% expected move)
    if target_price == 0.0 and current_price > 0:
        if direction == Direction.UP:
            target_price = round(current_price * 1.03, 4)
        else:
            target_price = round(current_price * 0.97, 4)
    
    timestamp = _safe_datetime(raw.get('run_at') or raw.get('timestamp'))
    
    return Prediction(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        current_price=current_price,
        target_price=target_price,
        stop_loss=stop_loss,
        timestamp=timestamp,
        news_influenced=raw.get('news_influenced', False),
    )


def batch_convert(raw_predictions: List[Dict[str, Any]]) -> List[Prediction]:
    """
    Convert a list of production predictions to core format.
    
    Filters out:
    - Failed predictions (ok=False)
    - HOLD/ERROR directions
    - Missing required fields
    
    Args:
        raw_predictions: List of dicts from production prediction system
        
    Returns:
        List of valid Prediction objects
    """
    results = []
    for raw in raw_predictions:
        try:
            pred = production_to_prediction(raw)
            if pred is not None:
                results.append(pred)
        except Exception:
            # Skip any prediction that fails conversion
            continue
    return results


def from_latest_predictions(latest_predictions: Dict[str, Dict[str, Any]]) -> List[Prediction]:
    """
    Convert from _LATEST_PREDICTIONS cache format.
    
    Production stores predictions in a dict keyed by symbol:
        _LATEST_PREDICTIONS = {
            'ETH': {'symbol': 'ETH', 'direction': 'DOWN', ...},
            'BTC': {'symbol': 'BTC', 'direction': 'UP', ...},
        }
    
    Args:
        latest_predictions: Dict mapping symbol -> prediction dict
        
    Returns:
        List of valid Prediction objects
    """
    return batch_convert(list(latest_predictions.values()))


# =============================================================================
# Output Adapters: core.models → formatter dict format
# =============================================================================

from core.models import ScoredPrediction
from config.symbols import V3_VALIDATED_STRATEGIES, is_crypto, ValidatedStrategy

def scored_to_formatter_dict(scored: ScoredPrediction) -> Dict[str, Any]:
    """
    Convert ScoredPrediction to dict format expected by formatters.
    
    Formatter TradePick.from_dict() expects:
        {
            'symbol': str,
            'direction': 'UP' | 'DOWN',
            'confidence': float,
            'current': float,
            'target': float,
            'stop': float,
            'is_inverse': bool,
            'hold_days': int,
            'win_rate': float,
            'sample_size': int,
        }
    """
    # Get V3 strategy info (ValidatedStrategy dataclass or None)
    strategy: Optional[ValidatedStrategy] = V3_VALIDATED_STRATEGIES.get(scored.symbol)
    
    # Extract strategy attributes with defaults
    hold_hours = strategy.hold_hours if strategy else 72
    win_rate = strategy.backtest_win_rate if strategy else 0.0
    sample_size = strategy.backtest_trades if strategy else 0
    strategy_name = strategy.strategy if strategy else 'unknown'
    
    return {
        'symbol': scored.symbol,
        'direction': scored.direction.value,  # Direction enum → 'UP'/'DOWN'
        'confidence': scored.confidence,
        'current': scored.current_price,
        'current_price': scored.current_price,  # Some formatters use this key
        'target': scored.target_price,
        'target_price': scored.target_price,
        'prediction_48h': scored.target_price,  # Alias for paper tracker
        'stop': scored.stop_loss,
        'stop_price': scored.stop_loss,
        # V3 fields - use v3_ prefix as expected by TradePick.from_dict
        'v3_validated': True,  # All V3 filtered predictions are validated
        'v3_is_inverse': scored.is_inverse,
        'v3_original_direction': 'DOWN' if scored.is_inverse else '',  # ETH inverse flips DOWN→UP
        'v3_strategy': strategy_name,
        'v3_hold_hours': hold_hours,
        'v3_historical_win_rate': win_rate,
        'v3_backtest_win_rate': win_rate,  # Alias
        'v3_sample_size': sample_size,
        'v3_is_whitelisted': True,  # All V3 validated symbols are "whitelisted"
        'v3_score': scored.score,
        # Legacy fields
        'is_inverse': scored.is_inverse,
        'hold_days': hold_hours // 24,  # Convert hours to days
        'win_rate': win_rate,
        'sample_size': sample_size,
        'hold_hours': hold_hours,
        'strategy': strategy_name,
        'backtest_win_rate': win_rate,
    }


def scored_list_to_formatter(
    scored_predictions: List[ScoredPrediction]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert list of ScoredPrediction to (stocks, crypto) lists for formatter.
    
    Returns:
        (stocks_list, crypto_list) - both in dict format for TradePick.from_dict()
    """
    stocks = []
    crypto = []
    
    for scored in scored_predictions:
        d = scored_to_formatter_dict(scored)
        if is_crypto(scored.symbol):
            crypto.append(d)
        else:
            stocks.append(d)
    
    return stocks, crypto


def format_v3_alert_from_scored(scored: ScoredPrediction) -> str:
    """
    Format a V3 alert from a ScoredPrediction object.
    
    This bridges ScoredPrediction to the format_v3_alert function.
    """
    from notifications.formatters import format_v3_alert
    
    strategy: Optional[ValidatedStrategy] = V3_VALIDATED_STRATEGIES.get(scored.symbol)
    
    return format_v3_alert(
        symbol=scored.symbol,
        direction=scored.direction.value,
        confidence=scored.confidence,
        strategy=strategy.strategy if strategy else 'unknown',
        is_inverse=scored.is_inverse,
        hold_hours=strategy.hold_hours if strategy else 72,
        win_rate=strategy.backtest_win_rate if strategy else 0.0,
    )


# =============================================================================
# Production Bridge: Complete V3 Pipeline
# =============================================================================

def process_v3_predictions(
    raw_predictions: List[Dict[str, Any]],
    min_confidence: float = 0.70
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Complete V3 pipeline for production use.
    
    This is the main entry point for wolf_app.py to use the clean architecture.
    
    Args:
        raw_predictions: List of dicts from wolf_app prediction engine
                         OR dict mapping symbol -> prediction (from _LATEST_PREDICTIONS)
        min_confidence: Minimum confidence threshold (default 0.70)
        
    Returns:
        (stocks, crypto) tuple ready for format_top10_message()
        
    Example:
        from core.adapters import process_v3_predictions
        
        # From _LATEST_PREDICTIONS dict
        stocks, crypto = process_v3_predictions(list(_LATEST_PREDICTIONS.values()))
        
        # Or from list
        stocks, crypto = process_v3_predictions(raw_list)
    """
    from core.v3_filter import V3Filter
    
    # Convert production format to core models
    predictions = batch_convert(raw_predictions)
    
    # Filter through V3
    v3_filter = V3Filter(min_confidence=min_confidence)
    scored = v3_filter.filter_and_score(predictions)
    
    # Convert to formatter format
    stocks, crypto = scored_list_to_formatter(scored)
    
    return stocks, crypto


def process_v3_from_cache(
    latest_predictions: Dict[str, Dict[str, Any]],
    min_confidence: float = 0.70
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process V3 predictions from _LATEST_PREDICTIONS cache format.
    
    This is specifically for the wolf_app notification loop which
    uses _LATEST_PREDICTIONS dict (symbol -> prediction).
    
    EDGE WHITELIST (Feb 10, 2026): Filters to edge symbols BEFORE
    V3 scoring. Previously this was a backdoor that bypassed the
    edge filter in get_top10_predictions() — LINK, XRP, ETH leaked through.
    
    Args:
        latest_predictions: Dict mapping symbol -> prediction dict
        min_confidence: Minimum confidence threshold
        
    Returns:
        (stocks, crypto) tuple ready for format_top10_message()
    """
    import os
    import logging
    _logger = logging.getLogger("ghost")
    
    # EDGE WHITELIST: Filter predictions to proven edge symbols only
    _edge_enabled = os.getenv("EDGE_WHITELIST_ENABLED", "1") == "1"
    if _edge_enabled:
        _edge_csv = os.getenv("EDGE_SYMBOLS",
            "T,GME,TURBO,RNDR,ENJ,JUP,BAND,HOOD,IQ,BMBL,HBAR,XPO,"
            "PEPE,IOTX,GIGA,COIN,ILV,BCH,CHZ,ALICE,YFI,ITRI,ICP,BRETT"
        )
        _edge_set = set(s.strip().upper() for s in _edge_csv.split(",") if s.strip())
        filtered = {sym: pred for sym, pred in latest_predictions.items() if sym.upper() in _edge_set}
        blocked = len(latest_predictions) - len(filtered)
        _logger.info(f"[V3-CLEAN] 🎯 EDGE WHITELIST: {len(filtered)} edge kept, {blocked} non-edge blocked")
        latest_predictions = filtered
    
    raw_list = list(latest_predictions.values())
    return process_v3_predictions(raw_list, min_confidence)
