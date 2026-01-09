"""
Model persistence layer for Ghost Protocol.

Stores trained models in PostgreSQL to survive Railway restarts.
Uses BYTEA column for binary pickle data + metadata JSON column.
"""

import os
import pickle
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from urllib.parse import urlparse

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    raise ImportError("psycopg2 is required for ModelStore. Install with: pip install psycopg2-binary")

LOGGER = logging.getLogger(__name__)


class ModelStore:
    """PostgreSQL-backed model storage."""
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize model store.
        
        Args:
            db_url: PostgreSQL connection URL (defaults to DATABASE_URL env var)
        """
        self.db_url = db_url or os.getenv("DATABASE_URL")
        
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        # Convert postgres:// to postgresql://
        if self.db_url.startswith("postgres://"):
            self.db_url = self.db_url.replace("postgres://", "postgresql://", 1)
        
        self._ensure_table_exists()
    
    def _get_connection(self):
        """Get PostgreSQL connection."""
        try:
            result = urlparse(self.db_url)
            conn = psycopg2.connect(
                database=result.path[1:],
                user=result.username,
                password=result.password,
                host=result.hostname,
                port=result.port
            )
            return conn
        except Exception as e:
            LOGGER.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def _ensure_table_exists(self):
        """Create model_store table if it doesn't exist."""
        conn = self._get_connection()
        
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_store (
                    model_id SERIAL PRIMARY KEY,
                    model_name VARCHAR(100) NOT NULL UNIQUE,
                    model_version VARCHAR(50),
                    model_data BYTEA NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for fast lookups
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_store_name 
                ON model_store(model_name)
            """)
            
            conn.commit()
            LOGGER.info("✅ model_store table ready")
        
        except Exception as e:
            LOGGER.error(f"Failed to create model_store table: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        model_version: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save trained model to PostgreSQL.
        
        Args:
            model: Trained model object (will be pickled)
            model_name: Unique model identifier (e.g., "ghost_xgboost_v2")
            model_version: Version string (e.g., "2026-01-09_balanced")
            metadata: Additional metadata (features, accuracy, etc.)
        
        Returns:
            True if saved successfully, False otherwise
        """
        conn = self._get_connection()
        
        try:
            # Serialize model
            model_bytes = pickle.dumps(model)
            
            # Prepare metadata
            meta = metadata or {}
            meta.update({
                "saved_at": datetime.now().isoformat(),
                "size_bytes": len(model_bytes)
            })
            
            # Upsert (insert or update if exists)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO model_store (model_name, model_version, model_data, metadata, updated_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (model_name) 
                DO UPDATE SET 
                    model_version = EXCLUDED.model_version,
                    model_data = EXCLUDED.model_data,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                model_name,
                model_version or "latest",
                psycopg2.Binary(model_bytes),
                psycopg2.extras.Json(meta)
            ))
            
            conn.commit()
            
            LOGGER.info(
                f"✅ Model saved: {model_name} v{model_version} "
                f"({len(model_bytes):,} bytes)"
            )
            return True
        
        except Exception as e:
            LOGGER.error(f"Failed to save model: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load trained model from PostgreSQL.
        
        Args:
            model_name: Model identifier (e.g., "ghost_xgboost_v2")
        
        Returns:
            Deserialized model object, or None if not found
        """
        conn = self._get_connection()
        
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT model_data, model_version, metadata
                FROM model_store
                WHERE model_name = %s
            """, (model_name,))
            
            row = cur.fetchone()
            
            if not row:
                LOGGER.warning(f"Model not found: {model_name}")
                return None
            
            model_bytes, version, metadata = row
            
            # Deserialize model
            model = pickle.loads(bytes(model_bytes))
            
            LOGGER.info(
                f"✅ Model loaded: {model_name} v{version} "
                f"({len(model_bytes):,} bytes)"
            )
            
            return model
        
        except Exception as e:
            LOGGER.error(f"Failed to load model: {e}")
            return None
        finally:
            conn.close()
    
    def get_metadata(self, model_name: str) -> Optional[Dict]:
        """
        Get model metadata without loading the model itself.
        
        Args:
            model_name: Model identifier
        
        Returns:
            Metadata dict, or None if not found
        """
        conn = self._get_connection()
        
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT model_version, metadata, updated_at
                FROM model_store
                WHERE model_name = %s
            """, (model_name,))
            
            row = cur.fetchone()
            
            if not row:
                return None
            
            version, metadata, updated_at = row
            
            return {
                "model_name": model_name,
                "version": version,
                "updated_at": updated_at.isoformat() if updated_at else None,
                **metadata
            }
        
        except Exception as e:
            LOGGER.error(f"Failed to get metadata: {e}")
            return None
        finally:
            conn.close()
    
    def list_models(self) -> list:
        """
        List all stored models.
        
        Returns:
            List of model metadata dicts
        """
        conn = self._get_connection()
        
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT model_name, model_version, metadata, updated_at,
                       pg_size_pretty(LENGTH(model_data)::bigint) as size
                FROM model_store
                ORDER BY updated_at DESC
            """)
            
            rows = cur.fetchall()
            
            models = []
            for row in rows:
                name, version, metadata, updated_at, size = row
                models.append({
                    "model_name": name,
                    "version": version,
                    "size": size,
                    "updated_at": updated_at.isoformat() if updated_at else None,
                    "metadata": metadata
                })
            
            return models
        
        except Exception as e:
            LOGGER.error(f"Failed to list models: {e}")
            return []
        finally:
            conn.close()


# Singleton instance
_MODEL_STORE = None

def get_model_store() -> ModelStore:
    """Get or create singleton ModelStore instance."""
    global _MODEL_STORE
    if _MODEL_STORE is None:
        _MODEL_STORE = ModelStore()
    return _MODEL_STORE
