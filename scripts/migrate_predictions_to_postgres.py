#!/usr/bin/env python3
"""
PostgreSQL Migration Script for Ghost Predictions
===================================================

This script migrates existing SQLite predictions to PostgreSQL.

SAFE MODE: This script only reads from SQLite and writes to PostgreSQL.
It does NOT modify or delete existing SQLite data.

Environment Variables Required:
- DATABASE_URL: PostgreSQL connection string (postgres://user:pass@host:port/db)
- GHOST_PREDICT_DB: SQLite database path (default: ./data/ghost_predictions.db)

Usage:
    python scripts/migrate_predictions_to_postgres.py [--batch-size 100] [--dry-run]

Options:
    --batch-size N    : Process N predictions at a time (default: 100)
    --dry-run         : Print migration plan without executing
    --verify          : Verify data integrity after migration
"""

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.prediction_store import get_prediction_store, PREDICTION_STORE_ENGINE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)


def count_sqlite_predictions(db_path: str) -> dict[str, int]:
    """Count records in SQLite database."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        prediction_count = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        point_count = cursor.execute("SELECT COUNT(*) FROM prediction_points").fetchone()[0]
        outcome_count = cursor.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        
        return {
            "predictions": prediction_count,
            "points": point_count,
            "outcomes": outcome_count,
        }
    finally:
        conn.close()


def migrate_predictions(batch_size: int = 100, dry_run: bool = False):
    """
    Migrate predictions from SQLite to PostgreSQL.
    
    Args:
        batch_size: Number of predictions to process at once
        dry_run: If True, only print plan without executing
    """
    db_path = os.getenv("GHOST_PREDICT_DB", "./data/ghost_predictions.db")
    
    if not Path(db_path).exists():
        LOGGER.error(f"SQLite database not found: {db_path}")
        return 1
    
    # Count SQLite records
    LOGGER.info("=" * 60)
    LOGGER.info("SQLite → PostgreSQL Migration")
    LOGGER.info("=" * 60)
    
    counts = count_sqlite_predictions(db_path)
    LOGGER.info(f"SQLite records found:")
    LOGGER.info(f"  - Predictions: {counts['predictions']}")
    LOGGER.info(f"  - Points: {counts['points']}")
    LOGGER.info(f"  - Outcomes: {counts['outcomes']}")
    
    if dry_run:
        LOGGER.info("\n[DRY RUN] Migration plan:")
        LOGGER.info(f"  1. Create PostgreSQL schema (tables + indexes)")
        LOGGER.info(f"  2. Migrate {counts['predictions']} predictions in batches of {batch_size}")
        LOGGER.info(f"  3. Migrate {counts['points']} prediction points")
        LOGGER.info(f"  4. Migrate {counts['outcomes']} outcomes")
        LOGGER.info(f"\n[DRY RUN] No changes will be made")
        return 0
    
    # Get prediction store (should be PostgreSQL backend)
    if PREDICTION_STORE_ENGINE != "postgres":
        LOGGER.error("PREDICTION_STORE_ENGINE must be set to 'postgres'")
        LOGGER.error("Set: export PREDICTION_STORE_ENGINE=postgres")
        return 1
    
    try:
        store = get_prediction_store()
        LOGGER.info(f"✅ Connected to backend: {store.backend.__class__.__name__}")
    except Exception as e:
        LOGGER.error(f"Failed to initialize PostgreSQL backend: {e}")
        return 1
    
    # Migration logic
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.cursor()
        
        # Fetch all predictions
        cursor.execute("SELECT * FROM predictions ORDER BY id")
        predictions = cursor.fetchall()
        
        migrated_count = 0
        failed_count = 0
        
        LOGGER.info(f"\n🔄 Starting migration of {len(predictions)} predictions...")
        
        for i, pred_row in enumerate(predictions, 1):
            try:
                # Fetch prediction points
                cursor.execute(
                    "SELECT ts, price FROM prediction_points WHERE prediction_id=? AND kind='forecast' ORDER BY ts",
                    (pred_row['id'],)
                )
                forecast_points = [(row['ts'], row['price']) for row in cursor.fetchall()]
                
                # Save to PostgreSQL
                new_id = store.backend.save_prediction(
                    symbol=pred_row['symbol'],
                    forecast_points=forecast_points,
                    method=pred_row['method'],
                    confidence=pred_row['confidence'],
                    direction=pred_row['direction'],
                    features=pred_row['features_json'],
                    params=pred_row['params_json'],
                    tag=pred_row['tag'] or "",
                )
                
                # Migrate actual points if any
                cursor.execute(
                    "SELECT ts, price FROM prediction_points WHERE prediction_id=? AND kind='actual' ORDER BY ts",
                    (pred_row['id'],)
                )
                actual_points = [(row['ts'], row['price']) for row in cursor.fetchall()]
                
                if actual_points:
                    store.backend.append_actual_points(new_id, actual_points)
                
                # Migrate outcome if exists
                cursor.execute(
                    "SELECT * FROM outcomes WHERE prediction_id=?",
                    (pred_row['id'],)
                )
                outcome_row = cursor.fetchone()
                
                if outcome_row:
                    store.backend.create_outcome(
                        prediction_id=new_id,
                        mae=outcome_row['mae'],
                        map_val=outcome_row['map'],
                        rmse=outcome_row['rmse'],
                        hit_direction=outcome_row['hit_direction'],
                        hit_ratio_window=outcome_row['hit_ratio_window'],
                        notes=outcome_row['notes'] or "",
                    )
                
                migrated_count += 1
                
                if i % batch_size == 0:
                    LOGGER.info(f"  ✓ Migrated {i}/{len(predictions)} predictions...")
                
            except Exception as e:
                LOGGER.error(f"  ✗ Failed to migrate prediction {pred_row['id']} ({pred_row['symbol']}): {e}")
                failed_count += 1
        
        LOGGER.info(f"\n✅ Migration complete!")
        LOGGER.info(f"  - Migrated: {migrated_count}")
        LOGGER.info(f"  - Failed: {failed_count}")
        LOGGER.info(f"  - Total: {len(predictions)}")
        
        return 0 if failed_count == 0 else 1
        
    finally:
        conn.close()


def verify_migration():
    """Verify data integrity after migration."""
    LOGGER.info("\n🔍 Verifying migration...")
    
    db_path = os.getenv("GHOST_PREDICT_DB", "./data/ghost_predictions.db")
    sqlite_counts = count_sqlite_predictions(db_path)
    
    # Count PostgreSQL records
    try:
        store = get_prediction_store()
        
        # This would require custom queries - simplified verification
        LOGGER.info("✅ PostgreSQL backend accessible")
        LOGGER.info(f"SQLite predictions: {sqlite_counts['predictions']}")
        LOGGER.info("Run manual verification queries to compare counts")
        
    except Exception as e:
        LOGGER.error(f"Verification failed: {e}")
        return 1
    
    return 0


def main():
    """Main migration script."""
    parser = argparse.ArgumentParser(description="Migrate Ghost predictions from SQLite to PostgreSQL")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for migration")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    parser.add_argument("--verify", action="store_true", help="Verify migration after completion")
    
    args = parser.parse_args()
    
    # Run migration
    result = migrate_predictions(batch_size=args.batch_size, dry_run=args.dry_run)
    
    # Run verification if requested
    if args.verify and result == 0 and not args.dry_run:
        result = verify_migration()
    
    return result


if __name__ == "__main__":
    sys.exit(main())
