-- Migration: Personal Watchlist Schema
-- Version: 001
-- Description: Single-owner persistent watchlist for stocks and crypto with prediction tracking

-- ====================================================================
-- GHOST_WATCHLIST_ITEMS
-- ====================================================================
-- Stores user's manually curated watchlist (single owner, no multi-tenant)
CREATE TABLE IF NOT EXISTS ghost_watchlist_items (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    asset_type TEXT NOT NULL CHECK (asset_type IN ('crypto', 'stock')),
    owns_position BOOLEAN DEFAULT FALSE,
    notes TEXT DEFAULT '',
    added_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    active BOOLEAN DEFAULT TRUE,
    
    -- Metadata for tracking
    price_at_add REAL,
    alert_threshold_pct REAL DEFAULT 5.0,  -- Alert if price moves ±5% by default
    priority INTEGER DEFAULT 1,  -- 1=normal, 2=high, 3=critical
    
    -- Constraints
    UNIQUE (symbol, asset_type) WHERE active = TRUE,
    CHECK (LENGTH(symbol) > 0 AND LENGTH(symbol) <= 20)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_watchlist_symbol ON ghost_watchlist_items(symbol) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_watchlist_asset_type ON ghost_watchlist_items(asset_type) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_watchlist_active ON ghost_watchlist_items(active, priority DESC, added_at DESC);
CREATE INDEX IF NOT EXISTS idx_watchlist_owns_position ON ghost_watchlist_items(owns_position) WHERE owns_position = TRUE AND active = TRUE;

-- ====================================================================
-- WATCHLIST_PREDICTION_TRACKING
-- ====================================================================
-- Tracks prediction generation for watchlist symbols (daily/intraday)
CREATE TABLE IF NOT EXISTS watchlist_prediction_tracking (
    id BIGSERIAL PRIMARY KEY,
    watchlist_item_id BIGINT NOT NULL REFERENCES ghost_watchlist_items(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    prediction_id BIGINT,  -- References ghost_predictions(id)
    
    -- Prediction snapshot (denormalized for performance)
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    expected_move_pct REAL NOT NULL,
    horizon_h INTEGER NOT NULL DEFAULT 48,
    
    -- Context
    price_at_prediction REAL,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    reason TEXT,  -- 'market_open', 'market_close', 'big_move', 'manual'
    
    -- Alert tracking
    alert_sent BOOLEAN DEFAULT FALSE,
    alert_sent_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_watchlist_pred_item ON watchlist_prediction_tracking(watchlist_item_id, generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_watchlist_pred_symbol ON watchlist_prediction_tracking(symbol, generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_watchlist_pred_alerts ON watchlist_prediction_tracking(alert_sent, generated_at DESC);

-- ====================================================================
-- WATCHLIST_PRICE_SNAPSHOTS
-- ====================================================================
-- High-frequency price tracking for big-move detection
CREATE TABLE IF NOT EXISTS watchlist_price_snapshots (
    id BIGSERIAL PRIMARY KEY,
    watchlist_item_id BIGINT NOT NULL REFERENCES ghost_watchlist_items(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    price REAL NOT NULL,
    change_pct_24h REAL,
    volume_24h REAL,
    snapshot_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_watchlist_prices_item ON watchlist_price_snapshots(watchlist_item_id, snapshot_at DESC);
CREATE INDEX IF NOT EXISTS idx_watchlist_prices_symbol ON watchlist_price_snapshots(symbol, snapshot_at DESC);

-- Retention policy: Keep only last 7 days of snapshots
-- (Manual cleanup job or TTL extension for Postgres 15+)

-- ====================================================================
-- WATCHLIST_ALERTS_LOG
-- ====================================================================
-- Historical log of all alerts sent (for cooldown enforcement and debugging)
CREATE TABLE IF NOT EXISTS watchlist_alerts_log (
    id BIGSERIAL PRIMARY KEY,
    watchlist_item_id BIGINT REFERENCES ghost_watchlist_items(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    alert_type TEXT NOT NULL,  -- 'open', 'close', 'big_move', 'target_hit'
    
    -- Alert content
    direction TEXT,
    confidence REAL,
    expected_move_pct REAL,
    current_price REAL,
    change_pct REAL,
    message TEXT,
    
    -- Telegram delivery
    telegram_sent BOOLEAN DEFAULT FALSE,
    telegram_sent_at TIMESTAMPTZ,
    telegram_chat_id BIGINT,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_watchlist_alerts_symbol ON watchlist_alerts_log(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_watchlist_alerts_type ON watchlist_alerts_log(alert_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_watchlist_alerts_cooldown ON watchlist_alerts_log(symbol, alert_type, created_at DESC);

-- ====================================================================
-- SEED DATA (Optional)
-- ====================================================================
-- Insert default watchlist if empty (remove if not needed)
INSERT INTO ghost_watchlist_items (symbol, asset_type, owns_position, priority, notes)
SELECT * FROM (VALUES
    -- Top crypto (priority 2)
    ('BTC', 'crypto', FALSE, 2, 'Bitcoin - flagship crypto asset'),
    ('ETH', 'crypto', FALSE, 2, 'Ethereum - smart contract platform'),
    
    -- Top stocks (priority 2)
    ('AAPL', 'stock', FALSE, 2, 'Apple Inc. - mega cap tech'),
    ('TSLA', 'stock', FALSE, 2, 'Tesla - EV and energy'),
    
    -- Additional tracking (priority 1)
    ('XRP', 'crypto', FALSE, 1, 'Ripple - payment network'),
    ('NVDA', 'stock', FALSE, 1, 'NVIDIA - AI chips'),
    ('MSFT', 'stock', FALSE, 1, 'Microsoft - cloud + software')
) AS seed_data (symbol, asset_type, owns_position, priority, notes)
WHERE NOT EXISTS (SELECT 1 FROM ghost_watchlist_items WHERE active = TRUE LIMIT 1);

-- ====================================================================
-- COMMENTS & NOTES
-- ====================================================================
COMMENT ON TABLE ghost_watchlist_items IS 'Single-owner persistent watchlist for stocks and crypto';
COMMENT ON COLUMN ghost_watchlist_items.owns_position IS 'TRUE if user currently holds this asset in broker account';
COMMENT ON COLUMN ghost_watchlist_items.active IS 'FALSE = soft deleted, allows re-activation';
COMMENT ON COLUMN ghost_watchlist_items.alert_threshold_pct IS 'Price move % to trigger big-move alert';
COMMENT ON COLUMN ghost_watchlist_items.priority IS '1=normal, 2=high, 3=critical (affects alert frequency)';

COMMENT ON TABLE watchlist_prediction_tracking IS 'Tracks prediction generation for watchlist symbols';
COMMENT ON COLUMN watchlist_prediction_tracking.reason IS 'Why prediction was generated: market_open, market_close, big_move, manual';

COMMENT ON TABLE watchlist_price_snapshots IS 'High-frequency price snapshots for intraday big-move detection';

COMMENT ON TABLE watchlist_alerts_log IS 'Historical log of all Telegram alerts sent for watchlist symbols';
COMMENT ON COLUMN watchlist_alerts_log.telegram_sent IS 'TRUE if successfully delivered to Telegram';
