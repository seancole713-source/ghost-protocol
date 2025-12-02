#!/usr/bin/env python3
"""
PostgreSQL Migration Readiness Check
Validates connectivity, schema, and prepares for historical data migration.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dns_resolution(hostname: str) -> bool:
    """Check if hostname resolves."""
    import socket
    try:
        socket.gethostbyname(hostname)
        logger.info(f"✅ DNS resolution successful: {hostname}")
        return True
    except socket.gaierror as e:
        logger.error(f"❌ DNS resolution failed for {hostname}: {e}")
        return False


def check_postgres_connection(database_url: str) -> bool:
    """Test PostgreSQL connection."""
    try:
        import psycopg2
        from urllib.parse import urlparse
        
        parsed = urlparse(database_url)
        logger.info(f"Testing connection to: {parsed.hostname}:{parsed.port}")
        
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        logger.info(f"✅ PostgreSQL connected: {version[:50]}...")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return False


def check_postgres_schema(database_url: str) -> dict:
    """Check if PredictionStore schema exists."""
    try:
        import psycopg2
        
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Check tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('ghost_predictions', 'ghost_prediction_points', 'ghost_prediction_outcomes')
        """)
        tables = [row[0] for row in cur.fetchall()]
        
        # Check row counts
        counts = {}
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cur.fetchone()[0]
        
        # Check indexes
        cur.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'public' 
            AND tablename IN ('ghost_predictions', 'ghost_prediction_points', 'ghost_prediction_outcomes')
        """)
        indexes = [row[0] for row in cur.fetchall()]
        
        cur.close()
        conn.close()
        
        return {
            'tables': tables,
            'counts': counts,
            'indexes': indexes
        }
    except Exception as e:
        logger.error(f"❌ Schema check failed: {e}")
        return {'tables': [], 'counts': {}, 'indexes': []}


def check_sqlite_data(db_path: str) -> dict:
    """Check SQLite data to migrate."""
    try:
        import sqlite3
        
        if not os.path.exists(db_path):
            logger.error(f"❌ SQLite database not found: {db_path}")
            return {'predictions': 0, 'points': 0, 'outcomes': 0}
        
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Count records
        cur.execute("SELECT COUNT(*) FROM ghost_predictions")
        predictions = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM ghost_prediction_points")
        points = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM ghost_prediction_outcomes")
        outcomes = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        logger.info(f"✅ SQLite data found:")
        logger.info(f"   - Predictions: {predictions}")
        logger.info(f"   - Points: {points}")
        logger.info(f"   - Outcomes: {outcomes}")
        
        return {
            'predictions': predictions,
            'points': points,
            'outcomes': outcomes
        }
    except Exception as e:
        logger.error(f"❌ SQLite check failed: {e}")
        return {'predictions': 0, 'points': 0, 'outcomes': 0}


def validate_prediction_store_config() -> dict:
    """Validate PredictionStore environment configuration."""
    config = {
        'PREDICTION_STORE_ENGINE': os.getenv('PREDICTION_STORE_ENGINE', 'sqlite'),
        'PREDICTION_DUAL_WRITE': os.getenv('PREDICTION_DUAL_WRITE', '0'),
        'DATABASE_URL': os.getenv('DATABASE_URL', ''),
        'GHOST_PREDICT_DB': os.getenv('GHOST_PREDICT_DB', './data/ghost_predictions.db')
    }
    
    logger.info("Configuration:")
    for key, value in config.items():
        if 'URL' in key and value:
            # Mask password in URL
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(value)
            masked = parsed._replace(netloc=f"{parsed.username}:****@{parsed.hostname}:{parsed.port}")
            logger.info(f"   {key}: {urlunparse(masked)}")
        else:
            logger.info(f"   {key}: {value}")
    
    return config


def main():
    logger.info("=" * 60)
    logger.info("PostgreSQL Migration Readiness Check")
    logger.info("=" * 60)
    
    # Step 1: Validate configuration
    config = validate_prediction_store_config()
    
    database_url = config['DATABASE_URL']
    sqlite_path = config['GHOST_PREDICT_DB']
    
    if not database_url:
        logger.error("❌ DATABASE_URL not set")
        return False
    
    # Step 2: Check DNS resolution
    from urllib.parse import urlparse
    parsed = urlparse(database_url)
    if not check_dns_resolution(parsed.hostname):
        return False
    
    # Step 3: Check PostgreSQL connection
    if not check_postgres_connection(database_url):
        return False
    
    # Step 4: Check PostgreSQL schema
    logger.info("\n" + "=" * 60)
    logger.info("PostgreSQL Schema Check")
    logger.info("=" * 60)
    schema = check_postgres_schema(database_url)
    
    required_tables = ['ghost_predictions', 'ghost_prediction_points', 'ghost_prediction_outcomes']
    missing_tables = [t for t in required_tables if t not in schema['tables']]
    
    if missing_tables:
        logger.warning(f"⚠️  Missing tables: {missing_tables}")
        logger.info("Schema will be created during migration")
    else:
        logger.info(f"✅ All required tables exist")
        for table, count in schema['counts'].items():
            logger.info(f"   - {table}: {count} rows")
    
    if schema['indexes']:
        logger.info(f"✅ Indexes found: {len(schema['indexes'])}")
    
    # Step 5: Check SQLite data
    logger.info("\n" + "=" * 60)
    logger.info("SQLite Data Check")
    logger.info("=" * 60)
    sqlite_data = check_sqlite_data(sqlite_path)
    
    # Step 6: Final readiness verdict
    logger.info("\n" + "=" * 60)
    logger.info("Migration Readiness Verdict")
    logger.info("=" * 60)
    
    ready = True
    
    if not database_url:
        logger.error("❌ DATABASE_URL not configured")
        ready = False
    
    if sqlite_data['predictions'] == 0:
        logger.warning("⚠️  No SQLite predictions to migrate")
    
    if schema['counts'].get('ghost_predictions', 0) > 0:
        logger.info(f"✅ PostgreSQL already has {schema['counts']['ghost_predictions']} predictions (from dual-write)")
    
    if ready:
        logger.info("\n✅ READY FOR MIGRATION")
        logger.info("\nRecommended migration command:")
        logger.info(f"   export PREDICTION_STORE_ENGINE=postgres")
        logger.info(f"   export DATABASE_URL='{database_url}'")
        logger.info(f"   export GHOST_PREDICT_DB='{sqlite_path}'")
        logger.info(f"   python3 scripts/migrate_predictions_to_postgres.py --batch-size 100 --verify")
    else:
        logger.error("\n❌ NOT READY FOR MIGRATION")
        logger.error("Fix configuration issues above before proceeding")
    
    return ready


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
