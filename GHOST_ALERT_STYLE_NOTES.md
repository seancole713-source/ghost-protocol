# Ghost Alert Style Implementation - System Inventory

## Phase 1: Current Alert Generators (Inventory)

### 1. Telegram Alert Infrastructure

| Source | Function | File | Data Available | Channels Used | Status |
|--------|----------|------|----------------|---------------|--------|
| **Prediction Alerts**| `render_alert()` | `core/telegram_alerts.py:28` | symbol, market, horizon_bucket,
prediction{action, confidence, direction, factors}, price_meta{price, prev_close, provider, after_hours} | Telegram | ✅
ACTIVE |
|**Mover Alerts**| `send_mover_alert()` | `core/telegram_alerts.py:293` | symbol, price, pct_1h, pct_24h, vol_mult,
age_s, provider, tier | Telegram | ✅ ACTIVE |
|**Hunter Opportunities**| `format_opportunity_alert()` | `core/telegram_hunter.py:163` | symbol, confidence,
predicted_pct, timeframe_hours, action, volume_ratio, sentiment, score | Telegram | ✅ ACTIVE |
|**Daily Reports**| `format_daily_report()` | `core/telegram_hunter.py:248` | opportunities[], accuracy_stats{} |
Telegram | ✅ ACTIVE |
|**Instant Alerts**| `send_instant_alert()` | `core/telegram_hunter.py:363` | opportunity{score >= 80} | Telegram | ✅
ACTIVE |
|**Pre-Market Predictions**| `_generate_multi_symbol_predictions()` | `wolf_app.py:5904` | HUNTER_STOCK_SYMBOLS,
HUNTER_CRYPTO_SYMBOLS | Scheduled (8am, 12pm, 4pm ET) | ✅ ACTIVE |
|**Morning Report**| `daily_report_loop()` | `core/telegram_hunter.py:429` | opportunities, accuracy_stats | Telegram
(7am, 8pm) | ✅ ACTIVE |

### 2. Data Flow Summary**Prediction Pipeline**

```text
api_predict_run() → price_quorum → forecast_engine → predictor.create_prediction()
                 → _LATEST_PREDICTIONS[symbol] = {...}
                 → telegram_alerts.render_alert() → Telegram

```text

**Hunter Pipeline**:

```text

market_scanner.scan_stocks_full() → opportunity_scorer.calculate_opportunity_score()
                                  → telegram_hunter.format_opportunity_alert()
                                  → telegram_hunter.send_instant_alert() → Telegram

```text

**Data Available for Alerts**:

- **From Prediction**: symbol, confidence (0-1), direction (UP/DOWN/FLAT), action (BUY/SELL/HOLD), factors[], price, prev_close, provider, horizon_h
- **From Hunter**: symbol, score (0-100), predicted_pct, timeframe_hours, volume_ratio, sentiment
- **From Scanner**: price, pct_1h, pct_24h, vol_mult, age_s, tier


### 3. Current Alert Formats

**Prediction Alert (Verbose)**:

```text

🌅 STOCK PREDICTION
⏰ 2025-01-20 07:55 AM CT | 2025-01-20 13:55 UTC

📈 STOCK • WOLF
Price: $17.51  Prev: $17.45  Δ: +0.06 (+0.34%)
Provider: yahoo

🎯 Short-term (2h-30d)
Action: BUY   Confidence: 60%   Direction: UP

📈 Factors:
• Recent momentum trending positive
• Volume above average
• Price breaking resistance

🔁 Next check: 09:55 AM CT

```text

**Hunter Alert (Verbose)**:

```text

🔥 GHOST HUNTER ALERT 🔥

**WOLF**— Grade: B+ (Score: 85/100)

📈**Prediction:**BUY +5.2% in 6h
🎯**AI Confidence:**78%**Signals:**• Volume: 3.2x average
• Sentiment: Positive**Recommendation:**Strong buy opportunity with high confidence

⏱️ Timeframe: 6h
📊 Ghost Score: 85/100

_Track this on your watchlist. Ghost accuracy: 85%+_

```text**Mover Alert (Current)**:

```text

📈 STOCKS Mover • WOLF
Price $17.5100 (+5.20%, 1h +2.10%) • Vol×3.20
Provider polygon • Age 180s
Tier 📊6+
Short-term: 48h window • Long-term: 30–180d

```text

## Phase 2: Proposed Simple Alert Format

### Cash-App Style Target (140-180 chars)

**Option A - Ultra Compact**:

```text

Ghost 🔮 WOLF — BUY (78%) | $17.51 (+5.2%)

```text

**Option B - Balanced**:

```text

WOLF up 5.2% to $17.51 — Ghost BUY (78% confidence)

```text

**Option C - Context**:

```text

Ghost detected: WOLF +5.2% to $17.51 | BUY signal | 78% confidence | 6h window

```text

### Alert DTO Structure

```python

@dataclass
class Alert:
    """Unified alert payload for all Ghost signals"""

    # Core identification

    symbol: str                          # e.g., "WOLF"
    market: str                          # "stock" or "crypto"

    # Signal data

    direction: Literal["BUY", "SELL", "HOLD", "WATCH"]
    confidence: float                    # 0.0-1.0 (internally) or 0-100 (for display)

    # Price information

    price_now: float                     # Current price
    price_prev: float                    # Previous close / reference price
    change_pct: float                    # Day % change (or recent change)

    # Prediction context

    predicted_pct: float | None = None   # Expected % move
    horizon_h: int | None = None         # Time window (hours)

    # Metadata

    source: str = "hunter"               # "hunter", "pre_market", "open_check", "prediction"
    score: int | None = None             # 0-100 opportunity score (if hunter)
    volume_ratio: float | None = None    # Volume multiplier
    provider: str = "polygon"            # Price data source

    # Factors

    factors: list[str] = field(default_factory=list)

    # Timestamps

    timestamp: float = field(default_factory=time.time)

```text

### Formatter Function Signature

```python

def format_simple_alert(
    alert: Alert,
    style: Literal["compact", "balanced", "context"] = "balanced"
) -> str:
    """
    Format alert in Cash-App style (1-2 lines, max 180 chars)

    Args:
        alert: Unified alert payload
        style: Output format variant

    Returns:
        Formatted alert string (Markdown compatible)

    Examples:
        compact:  Ghost 🔮 WOLF — BUY (78%) | $17.51 (+5.2%)
        balanced: WOLF up 5.2% to $17.51 — Ghost BUY (78% confidence)
        context:  Ghost detected: WOLF +5.2% to $17.51 | BUY | 78% | 6h
    """

```text

## Phase 3: Configuration

### Environment Variables

```bash

# Alert style control (default: "simple")

ALERT_STYLE=simple              # "simple" or "verbose"
ALERT_SIMPLE_FORMAT=balanced    # "compact", "balanced", "context"
MIN_ALERT_CONFIDENCE=0.60       # Minimum confidence to send alert (default 60%)
PRICE_SOURCE_PRIMARY=polygon    # "polygon" or "yahoo"
PRICE_SOURCE_SECONDARY=yahoo    # Fallback provider

```text

### Implementation Plan

**Files to Modify**:

1. `core/telegram_alerts.py` - Add `format_simple_alert()` and Alert DTO
2. `core/telegram_hunter.py` - Wire simple formatter into `send_instant_alert()`
3. `wolf_app.py` - Wire simple formatter into prediction alerts
4. New: `core/alert_dto.py` - Alert dataclass definition (optional separate file)


**Backward Compatibility**:

- Keep all existing verbose templates
- Add `ALERT_STYLE` env check at send time
- Default to "verbose" initially, allow opt-in to "simple"


## Phase 4: 0% Confidence Policy (Already Implemented ✅)

Current implementation in `core/telegram_alerts.py`:

```python

# Filter out 0% confidence (diagnostic only, not real predictions)

confidence = prediction.get("confidence", 0)
if confidence < 0.10:
    if LOGGER:
        LOGGER.info(f"Skipping 0% confidence alert: {market}/{symbol}/{horizon_bucket}")
    return False

```text

**Status**: ✅ Already enforced at line 200-205

## Phase 5: Next Steps

1. ✅ Inventory complete (this document)
2. Create `Alert` dataclass in `core/telegram_alerts.py`
3. Implement `format_simple_alert()` with 3 style variants
4. Add `ALERT_STYLE` env check to `send_alert()`
5. Wire into `telegram_hunter.py` instant alerts
6. Add unit tests for formatter
7. Deploy and A/B test simple vs verbose


---

**Status**: Phase 1 Complete - System Mapped
**Next**: Implement Alert DTO and simple formatter
**Safety**: No execution code touched ✅
