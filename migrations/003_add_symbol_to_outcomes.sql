-- =====================================================
-- Ghost Protocol - Add Symbol Column to Outcomes
-- =====================================================
-- Migration: 003
-- Purpose: Add symbol column to ghost_prediction_outcomes for by-symbol analytics
-- Date: December 14, 2025

-- Add symbol column to outcomes table
ALTER TABLE ghost_prediction_outcomes 
    ADD COLUMN IF NOT EXISTS symbol VARCHAR(20);

-- Create index for symbol-based queries (accuracy by symbol)
CREATE INDEX IF NOT EXISTS idx_outcomes_symbol 
    ON ghost_prediction_outcomes(symbol);

-- Update comment
COMMENT ON COLUMN ghost_prediction_outcomes.symbol IS 'Trading symbol (BTC, ETH, AAPL, etc.) - enables by-symbol accuracy tracking';
