#!/bin/bash
# Check production predictions via Railway CLI
# Usage: ./railway_check_predictions.sh

echo "Checking production predictions via Railway..."
echo "This will run in Railway's production environment"
echo ""

railway shell ghost-protocol <<'EOSHELL'
python3 - <<'EOPYSHELL'
import os
from sqlalchemy import create_engine, text
from datetime import datetime
import time

database_url = os.getenv('DATABASE_URL')
engine = create_engine(database_url)

with engine.connect() as conn:
    now = time.time()
    cutoff_48h = now - (48 * 3600)
    
    # Count total predictions
    total = conn.execute(text("SELECT COUNT(*) FROM ghost_predictions")).scalar()
    print(f"Total Predictions: {total}")
    
    # Count predictions ready for reconciliation (>48h old)
    ready_48h = conn.execute(text(
        "SELECT COUNT(*) FROM ghost_predictions WHERE run_at < :cutoff"
    ), {"cutoff": cutoff_48h}).scalar()
    print(f"Ready for Reconciliation (>48h): {ready_48h}")
    
    # Get date range
    oldest = conn.execute(text("SELECT MIN(run_at) FROM ghost_predictions")).scalar()
    newest = conn.execute(text("SELECT MAX(run_at) FROM ghost_predictions")).scalar()
    
    if oldest:
        oldest_dt = datetime.fromtimestamp(oldest)
        newest_dt = datetime.fromtimestamp(newest)
        age_days = (now - oldest) / 86400
        
        print(f"Oldest: {oldest_dt.isoformat()} ({age_days:.1f} days ago)")
        print(f"Newest: {newest_dt.isoformat()}")
    
    # Count outcomes
    outcomes_total = conn.execute(text("SELECT COUNT(*) FROM ghost_prediction_outcomes")).scalar()
    print(f"Reconciled Outcomes: {outcomes_total}")
    
    if outcomes_total == 0 and ready_48h > 0:
        print(f"WARNING: {ready_48h} predictions ready but 0 outcomes!")
EOPYSHELL
EOSHELL
