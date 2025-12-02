-- =====================================================
-- Ghost Protocol - Prediction Outcomes Migration
-- =====================================================
-- Migration: 002
-- Purpose: Add outcome tracking table for 48h prediction accuracy
-- Date: December 2, 2025
-- 
-- This schema stores the actual vs predicted comparison for each prediction
-- after the 48-hour horizon window closes. Used to calculate Ghost's accuracy.

-- =====================================================
-- 1. PREDICTION OUTCOMES TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS ghost_prediction_outcomes (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- Foreign Key to prediction
    prediction_id INTEGER NOT NULL,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMP NOT NULL,  -- When outcome was resolved (run_at + 48h)
    
    -- Price Data
    price_at_prediction NUMERIC(20, 8) NOT NULL,  -- Price when prediction made (t0)
    price_at_resolution NUMERIC(20, 8),           -- Price at t+48h (t1) - NULL if unavailable
    
    -- Movement Metrics
    realized_move_pct NUMERIC(10, 4),  -- ((t1 - t0) / t0) * 100 - NULL if price unavailable
    predicted_direction VARCHAR(10),    -- UP, DOWN, FLAT
    actual_direction VARCHAR(10),       -- UP, DOWN, FLAT - based on realized_move_pct
    
    -- Accuracy Result
    hit_direction INTEGER,  -- 1=correct, 0=wrong, NULL=no_data (couldn't get price)
    direction_threshold_pct NUMERIC(5, 2) DEFAULT 0.25,  -- Threshold used (e.g., 0.25%)
    
    -- Statistical Metrics
    mae NUMERIC(20, 8),   -- Mean Absolute Error (for multi-point forecasts)
    mape NUMERIC(10, 4),  -- Mean Absolute Percentage Error
    rmse NUMERIC(20, 8),  -- Root Mean Squared Error
    
    -- Confidence Tracking
    predicted_confidence NUMERIC(5, 4),  -- Original prediction confidence (0.0-1.0)
    confidence_calibration NUMERIC(10, 4),  -- Did confidence match accuracy? (future use)
    
    -- Data Quality
    resolution_method VARCHAR(50),  -- 'live_provider', 'fallback', 'no_data'
    resolution_provider VARCHAR(50), -- 'polygon', 'binance', 'coingecko', etc.
    notes TEXT,  -- Any issues during resolution
    
    -- Status
    status VARCHAR(20) DEFAULT 'completed',  -- 'completed', 'failed', 'no_data'
    
    -- Audit Trail
    reconciled_by VARCHAR(50) DEFAULT 'outcome_reconciler',
    reconciliation_version VARCHAR(20) DEFAULT '1.0'
);

-- =====================================================
-- 2. INDEXES FOR PERFORMANCE
-- =====================================================

-- Primary lookup: Find outcome by prediction_id
CREATE INDEX IF NOT EXISTS idx_outcomes_prediction_id 
    ON ghost_prediction_outcomes(prediction_id);

-- Accuracy queries: Filter by hit_direction (correct vs wrong)
CREATE INDEX IF NOT EXISTS idx_outcomes_hit_direction 
    ON ghost_prediction_outcomes(hit_direction);

-- Time-based queries: Filter by closed_at for daily/weekly/monthly accuracy
CREATE INDEX IF NOT EXISTS idx_outcomes_closed_at 
    ON ghost_prediction_outcomes(closed_at);

-- Status filtering: Exclude 'no_data' from accuracy calculations
CREATE INDEX IF NOT EXISTS idx_outcomes_status 
    ON ghost_prediction_outcomes(status);

-- Composite index: Common query pattern (status + closed_at + hit_direction)
CREATE INDEX IF NOT EXISTS idx_outcomes_accuracy_calc 
    ON ghost_prediction_outcomes(status, closed_at, hit_direction);

-- =====================================================
-- 3. CONSTRAINTS
-- =====================================================

-- Ensure each prediction only has one outcome
CREATE UNIQUE INDEX IF NOT EXISTS idx_outcomes_unique_prediction 
    ON ghost_prediction_outcomes(prediction_id);

-- hit_direction can only be 0, 1, or NULL
ALTER TABLE ghost_prediction_outcomes 
    ADD CONSTRAINT chk_hit_direction 
    CHECK (hit_direction IS NULL OR hit_direction IN (0, 1));

-- predicted_confidence must be between 0 and 1
ALTER TABLE ghost_prediction_outcomes 
    ADD CONSTRAINT chk_confidence_range 
    CHECK (predicted_confidence >= 0.0 AND predicted_confidence <= 1.0);

-- status must be valid
ALTER TABLE ghost_prediction_outcomes 
    ADD CONSTRAINT chk_status_valid 
    CHECK (status IN ('completed', 'failed', 'no_data'));

-- =====================================================
-- 4. HELPFUL VIEWS
-- =====================================================

-- View: Global Accuracy Summary (all-time)
CREATE OR REPLACE VIEW v_global_accuracy AS
SELECT
    COUNT(*) AS total_predictions,
    SUM(CASE WHEN status = 'completed' AND hit_direction IS NOT NULL THEN 1 ELSE 0 END) AS evaluated_predictions,
    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) AS correct_predictions,
    SUM(CASE WHEN hit_direction = 0 THEN 1 ELSE 0 END) AS wrong_predictions,
    SUM(CASE WHEN status = 'no_data' OR hit_direction IS NULL THEN 1 ELSE 0 END) AS no_data_predictions,
    ROUND(
        (SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END)::NUMERIC / 
         NULLIF(SUM(CASE WHEN hit_direction IS NOT NULL THEN 1 ELSE 0 END), 0)) * 100, 
        2
    ) AS accuracy_pct,
    AVG(CASE WHEN mae IS NOT NULL THEN mae END) AS avg_mae,
    AVG(CASE WHEN mape IS NOT NULL THEN mape END) AS avg_mape,
    AVG(CASE WHEN rmse IS NOT NULL THEN rmse END) AS avg_rmse
FROM ghost_prediction_outcomes;

-- View: Last 24 Hours Accuracy
CREATE OR REPLACE VIEW v_accuracy_24h AS
SELECT
    COUNT(*) AS total_predictions,
    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) AS correct_predictions,
    SUM(CASE WHEN hit_direction = 0 THEN 1 ELSE 0 END) AS wrong_predictions,
    ROUND(
        (SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END)::NUMERIC / 
         NULLIF(SUM(CASE WHEN hit_direction IS NOT NULL THEN 1 ELSE 0 END), 0)) * 100, 
        2
    ) AS accuracy_pct
FROM ghost_prediction_outcomes
WHERE closed_at >= NOW() - INTERVAL '24 hours'
  AND hit_direction IS NOT NULL;

-- View: Last 7 Days Accuracy
CREATE OR REPLACE VIEW v_accuracy_7d AS
SELECT
    COUNT(*) AS total_predictions,
    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) AS correct_predictions,
    SUM(CASE WHEN hit_direction = 0 THEN 1 ELSE 0 END) AS wrong_predictions,
    ROUND(
        (SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END)::NUMERIC / 
         NULLIF(SUM(CASE WHEN hit_direction IS NOT NULL THEN 1 ELSE 0 END), 0)) * 100, 
        2
    ) AS accuracy_pct
FROM ghost_prediction_outcomes
WHERE closed_at >= NOW() - INTERVAL '7 days'
  AND hit_direction IS NOT NULL;

-- View: Last 30 Days Accuracy
CREATE OR REPLACE VIEW v_accuracy_30d AS
SELECT
    COUNT(*) AS total_predictions,
    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) AS correct_predictions,
    SUM(CASE WHEN hit_direction = 0 THEN 1 ELSE 0 END) AS wrong_predictions,
    ROUND(
        (SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END)::NUMERIC / 
         NULLIF(SUM(CASE WHEN hit_direction IS NOT NULL THEN 1 ELSE 0 END), 0)) * 100, 
        2
    ) AS accuracy_pct
FROM ghost_prediction_outcomes
WHERE closed_at >= NOW() - INTERVAL '30 days'
  AND hit_direction IS NOT NULL;

-- =====================================================
-- 5. SAMPLE QUERIES (DOCUMENTATION)
-- =====================================================

-- Query 1: Check if 70% threshold is met (last 30 days)
-- SELECT accuracy_pct >= 70.0 AS meets_threshold, accuracy_pct
-- FROM v_accuracy_30d;

-- Query 2: Get accuracy by symbol (requires join to ghost_predictions)
-- SELECT 
--     gp.symbol,
--     COUNT(*) AS total,
--     SUM(CASE WHEN gpo.hit_direction = 1 THEN 1 ELSE 0 END) AS correct,
--     ROUND((SUM(CASE WHEN gpo.hit_direction = 1 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)) * 100, 2) AS accuracy_pct
-- FROM ghost_prediction_outcomes gpo
-- JOIN ghost_predictions gp ON gpo.prediction_id = gp.id
-- WHERE gpo.hit_direction IS NOT NULL
-- GROUP BY gp.symbol
-- ORDER BY accuracy_pct DESC;

-- Query 3: Find predictions without outcomes (need reconciliation)
-- SELECT 
--     gp.id, 
--     gp.symbol, 
--     gp.run_at,
--     gp.run_at + (gp.horizon_h * INTERVAL '1 hour') AS resolve_at,
--     NOW() AS current_time
-- FROM ghost_predictions gp
-- LEFT JOIN ghost_prediction_outcomes gpo ON gp.id = gpo.prediction_id
-- WHERE gpo.id IS NULL
--   AND (gp.run_at + (gp.horizon_h * INTERVAL '1 hour')) <= NOW()
-- ORDER BY gp.run_at;

-- Query 4: Confidence calibration (is higher confidence = higher accuracy?)
-- SELECT 
--     ROUND(predicted_confidence * 10) / 10.0 AS confidence_bucket,
--     COUNT(*) AS total,
--     SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) AS correct,
--     ROUND((SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)) * 100, 2) AS accuracy_pct
-- FROM ghost_prediction_outcomes
-- WHERE hit_direction IS NOT NULL
-- GROUP BY confidence_bucket
-- ORDER BY confidence_bucket;

-- =====================================================
-- 6. MIGRATION NOTES
-- =====================================================

-- This migration creates the foundation for Ghost Protocol's accuracy tracking.
-- 
-- IMPORTANT: This table must be populated by the outcome_reconciler service.
-- The reconciler runs as a background task and:
--   1. Finds predictions where (run_at + 48h) <= NOW()
--   2. Fetches actual price at t+48h from live providers
--   3. Compares predicted vs actual direction
--   4. Stores outcome in this table
--
-- FAIL CLOSED: If actual price cannot be obtained (API down, symbol delisted, etc.),
-- the outcome should have:
--   - status = 'no_data'
--   - hit_direction = NULL
--   - price_at_resolution = NULL
--   - notes = '<reason for failure>'
-- These outcomes are EXCLUDED from accuracy calculations.
--
-- 70% THRESHOLD: Ghost is considered "ACCURATE" when:
--   SELECT accuracy_pct FROM v_accuracy_30d
-- returns >= 70.0
--
-- If below threshold, Ghost should display "IN TRAINING" or "BELOW TARGET" status.

-- =====================================================
-- 7. ROLLBACK (IF NEEDED)
-- =====================================================

-- To rollback this migration:
-- DROP VIEW IF EXISTS v_accuracy_30d;
-- DROP VIEW IF EXISTS v_accuracy_7d;
-- DROP VIEW IF EXISTS v_accuracy_24h;
-- DROP VIEW IF EXISTS v_global_accuracy;
-- DROP TABLE IF EXISTS ghost_prediction_outcomes CASCADE;
