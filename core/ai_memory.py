"""
GHOST AI Memory System
======================
Long-term memory with vector embeddings for semantic search.
Replaces 100-sample ring buffer with unlimited persistent storage.

CRITICAL FIX (Mar 18, 2026):
  Previous version used SQLite on Railway's ephemeral filesystem.
  Every deploy wiped all AI memory - Ghost couldn't learn from mistakes.
  Now reads/writes PostgreSQL via DATABASE_URL so memories persist.
  Falls back to SQLite for local development when DATABASE_URL is absent.

Features:
- PostgreSQL for structured queries (persistent across deploys)
- Vector embeddings for semantic similarity (FAISS/ChromaDB)
- Confidence calibration (predicted vs actual outcomes)
- Episodic memory (significant trades, market events)
- Memory consolidation (summarize old experiences)

Author: Ghost AI
Date: 2025-10-03
Updated: 2026-03-18 - PostgreSQL migration
"""

import json
import logging
import os
import sqlite3
import time
from collections import defaultdict
from typing import Any

import numpy as np

# =============================================================================
# Vector Store Configuration (from environment)
# =============================================================================
VECTOR_SOURCE = os.getenv("VECTOR_SOURCE", "chromadb").lower()
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID", "")
MEMORY_TTL_DAYS = int(os.getenv("MEMORY_TTL_DAYS", "90"))
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Optional: ChromaDB for vector storage
try:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    import importlib.util
    HAS_FAISS = importlib.util.find_spec("faiss") is not None
except ImportError:
    HAS_FAISS = False

LOGGER = logging.getLogger(__name__)


class AIMemory:
    """
    Ghost AI long-term memory system.

    Architecture:
    - PostgreSQL: Structured queries (persistent on Railway)
    - SQLite: Fallback for local development
    - Vector Store: Semantic similarity search
    - In-Memory Cache: Recent 1000 decisions for fast access
    """

    def __init__(self, db_path: str = "data/ai_memory.db", vector_store: str = None):
        """Initialize AI Memory. Uses PostgreSQL if DATABASE_URL is set."""
        self.db_path = db_path
        self.vector_store_type = vector_store or VECTOR_SOURCE or "chromadb"
        self.vector_store_id = VECTOR_STORE_ID
        self._use_pg = bool(DATABASE_URL)

        if self._use_pg:
            self._sqlite_conn = None
            self._init_pg_tables()
            LOGGER.info("AIMemory using PostgreSQL (persistent across deploys)")
        else:
            os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
            self._sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
            self._sqlite_conn.row_factory = sqlite3.Row
            self._init_sqlite_tables()
            LOGGER.info(f"AIMemory using SQLite: {db_path} (local dev mode)")

        # Initialize vector store
        self.vector_store = None
        if self.vector_store_type == "openai" and VECTOR_STORE_ID:
            LOGGER.info(f"Using OpenAI vector store: {VECTOR_STORE_ID}")
        elif self.vector_store_type == "chromadb" and HAS_CHROMADB:
            self._init_chromadb()
        elif self.vector_store_type == "faiss" and HAS_FAISS:
            self._init_faiss()
        else:
            LOGGER.warning(f"Vector store '{self.vector_store_type}' not available")

        # In-memory cache for recent decisions
        self.cache: list[dict] = []
        self.cache_size = 1000
        self._load_cache()

        LOGGER.info(
            f"AIMemory initialized: backend={'pg' if self._use_pg else 'sqlite'}, "
            f"vector_store={self.vector_store_type}, cache_size={len(self.cache)}"
        )

    # -- Connection helpers --------------------------------------------------

    def _get_pg_conn(self):
        """Get a PostgreSQL connection context manager from the pool."""
        from core.db_pool import get_sync_connection
        return get_sync_connection()

    def _execute(self, query, params=(), fetch="none"):
        """
        Execute a query on the active backend.
        query uses %s placeholders (auto-converted to ? for SQLite).
        fetch: "none", "one", "all", or "lastrowid".
        """
        if self._use_pg:
            with self._get_pg_conn() as conn:
                cur = conn.cursor()
                cur.execute(query, params)
                if fetch == "all":
                    cols = [desc[0] for desc in cur.description] if cur.description else []
                    return [dict(zip(cols, row)) for row in cur.fetchall()]
                elif fetch == "one":
                    row = cur.fetchone()
                    if row and cur.description:
                        cols = [desc[0] for desc in cur.description]
                        return dict(zip(cols, row))
                    return row
                elif fetch == "lastrowid":
                    conn.commit()
                    return cur.fetchone()[0] if cur.description else 0
                else:
                    conn.commit()
                    return cur.rowcount
        else:
            sqlite_query = query.replace("%s", "?")
            cur = self._sqlite_conn.execute(sqlite_query, params)
            if fetch == "all":
                return [dict(row) for row in cur.fetchall()]
            elif fetch == "one":
                row = cur.fetchone()
                return dict(row) if row else None
            elif fetch == "lastrowid":
                self._sqlite_conn.commit()
                return cur.lastrowid or 0
            else:
                self._sqlite_conn.commit()
                return cur.rowcount

    # -- Table initialization ------------------------------------------------

    def _init_pg_tables(self):
        """Create PostgreSQL tables if not exist."""
        with self._get_pg_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_memory (
                    id SERIAL PRIMARY KEY,
                    ts BIGINT NOT NULL,
                    symbol TEXT NOT NULL,
                    price DOUBLE PRECISION,
                    prev_close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    volatility DOUBLE PRECISION,
                    news_score DOUBLE PRECISION,
                    sentiment DOUBLE PRECISION,
                    features TEXT,
                    action TEXT,
                    confidence DOUBLE PRECISION,
                    reasoning TEXT,
                    model_version TEXT,
                    model_type TEXT,
                    outcome_1h DOUBLE PRECISION,
                    outcome_24h DOUBLE PRECISION,
                    outcome_7d DOUBLE PRECISION,
                    realized_pnl DOUBLE PRECISION,
                    user_feedback TEXT,
                    executed BOOLEAN DEFAULT FALSE
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_aimem_ts ON ai_memory(ts)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_aimem_symbol ON ai_memory(symbol)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_aimem_action ON ai_memory(action)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_aimem_executed ON ai_memory(executed)")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS calibration_metrics (
                    id SERIAL PRIMARY KEY,
                    computed_at BIGINT,
                    model_type TEXT,
                    confidence_bucket TEXT,
                    predicted_prob DOUBLE PRECISION,
                    actual_success_rate DOUBLE PRECISION,
                    sample_count INTEGER
                )
            """)
            conn.commit()
        LOGGER.info("AI memory PostgreSQL tables initialized")

    def _init_sqlite_tables(self):
        """Create SQLite tables if not exist (local dev fallback)."""
        with self._sqlite_conn:
            self._sqlite_conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts BIGINT NOT NULL, symbol TEXT NOT NULL,
                    price REAL, prev_close REAL, volume REAL, volatility REAL,
                    news_score REAL, sentiment REAL, features TEXT,
                    action TEXT, confidence REAL, reasoning TEXT,
                    model_version TEXT, model_type TEXT,
                    outcome_1h REAL, outcome_24h REAL, outcome_7d REAL,
                    realized_pnl REAL, user_feedback TEXT,
                    executed BOOLEAN DEFAULT 0
                )
            """)
            self._sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON ai_memory(ts)")
            self._sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON ai_memory(symbol)")
            self._sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_action ON ai_memory(action)")
            self._sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_executed ON ai_memory(executed)")
            self._sqlite_conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    computed_at BIGINT, model_type TEXT, confidence_bucket TEXT,
                    predicted_prob REAL, actual_success_rate REAL, sample_count INTEGER
                )
            """)
        LOGGER.info("AI memory SQLite tables initialized")

    # -- Vector store initialization -----------------------------------------

    def _init_chromadb(self):
        """Initialize ChromaDB vector store."""
        try:
            client = chromadb.PersistentClient(
                path="data/chromadb", settings=Settings(anonymized_telemetry=False)
            )
            self.vector_store = client.get_or_create_collection(
                name="ghost_decisions", metadata={"hnsw:space": "cosine"}
            )
            LOGGER.info("ChromaDB initialized")
        except Exception as e:
            LOGGER.error(f"ChromaDB init failed: {e}")
            self.vector_store = None

    def _init_faiss(self):
        """Initialize FAISS vector index."""
        try:
            import faiss
            import pickle
            from pathlib import Path

            index_path = Path("data/faiss_index.bin")
            metadata_path = Path("data/faiss_metadata.pkl")
            vector_dim = 512

            if index_path.exists() and metadata_path.exists():
                self.vector_store = faiss.read_index(str(index_path))
                with open(metadata_path, "rb") as f:
                    self.faiss_metadata = pickle.load(f)
                LOGGER.info(f"Loaded FAISS index with {self.vector_store.ntotal} vectors")
            else:
                self.vector_store = faiss.IndexFlatL2(vector_dim)
                self.faiss_metadata = []
                LOGGER.info(f"Created new FAISS index (dimension={vector_dim})")

            self.index_path = index_path
            self.metadata_path = metadata_path
        except ImportError:
            LOGGER.warning("FAISS not installed, using DB-only mode")
            self.vector_store = None
        except Exception as e:
            LOGGER.error(f"FAISS init failed: {e}")
            self.vector_store = None

    # -- Cache ---------------------------------------------------------------

    def _load_cache(self):
        """Load recent decisions into cache."""
        try:
            rows = self._execute(
                "SELECT * FROM ai_memory ORDER BY ts DESC LIMIT %s",
                (self.cache_size,), fetch="all",
            )
            self.cache = rows if rows else []
        except Exception as e:
            LOGGER.warning(f"Cache load failed: {e}")
            self.cache = []
        LOGGER.info(f"Loaded {len(self.cache)} decisions into cache")

    # -- Core operations -----------------------------------------------------

    def store_decision(self, decision: dict[str, Any]) -> int:
        """Store AI decision with context. Returns row ID."""
        features_json = json.dumps(decision.get("features", {}))

        params = (
            decision.get("ts", int(time.time())),
            decision.get("symbol", "WOLF"),
            decision.get("price"),
            decision.get("prev_close"),
            decision.get("volume"),
            decision.get("volatility"),
            decision.get("news_score"),
            decision.get("sentiment"),
            features_json,
            decision.get("action", "HOLD"),
            decision.get("confidence", 0.5),
            decision.get("reasoning", ""),
            decision.get("model_version", "unknown"),
            decision.get("model_type", "knn"),
            decision.get("executed", False),
        )

        if self._use_pg:
            row_id = self._execute(
                """INSERT INTO ai_memory (
                    ts, symbol, price, prev_close, volume, volatility,
                    news_score, sentiment, features, action, confidence,
                    reasoning, model_version, model_type, executed
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id""",
                params, fetch="lastrowid",
            )
        else:
            row_id = self._execute(
                """INSERT INTO ai_memory (
                    ts, symbol, price, prev_close, volume, volatility,
                    news_score, sentiment, features, action, confidence,
                    reasoning, model_version, model_type, executed
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                params, fetch="lastrowid",
            )

        # Add to vector store
        if self.vector_store and decision.get("features") and row_id > 0:
            try:
                self._add_to_vector_store(row_id, decision)
            except Exception as e:
                LOGGER.error(f"Vector store add failed: {e}")

        # Update cache
        decision_copy = decision.copy()
        decision_copy["id"] = row_id
        self.cache.insert(0, decision_copy)
        if len(self.cache) > self.cache_size:
            self.cache.pop()

        LOGGER.debug(
            f"Stored decision {row_id}: {decision.get('action')} "
            f"{decision.get('symbol')} @ {decision.get('confidence', 0):.2f}"
        )
        return row_id

    def _add_to_vector_store(self, row_id: int, decision: dict):
        """Add decision to vector store for semantic search."""
        if not self.vector_store or self.vector_store_type != "chromadb":
            return
        features = decision.get("features", {})
        embedding = self._features_to_vector(features)
        document = decision.get("reasoning", "")
        metadata = {
            "symbol": decision.get("symbol", ""),
            "action": decision.get("action", ""),
            "confidence": float(decision.get("confidence", 0.5)),
            "model_type": decision.get("model_type", ""),
            "ts": int(decision.get("ts", 0)),
        }
        self.vector_store.add(
            ids=[str(row_id)], embeddings=[embedding],
            documents=[document], metadatas=[metadata],
        )

    def _features_to_vector(self, features: dict) -> list[float]:
        """Convert feature dict to fixed-length vector."""
        standard_keys = [
            "ret_1d", "ret_7d", "ret_30d", "vol_20d", "vol_60d",
            "news_score", "sentiment", "rsi_14", "macd", "macd_signal",
            "bb_width", "bb_position", "pos_pct", "pnl_pct",
            "momentum_1d", "momentum_7d",
        ]
        return [float(features.get(key, 0.0)) for key in standard_keys]

    # -- Search & retrieval --------------------------------------------------

    def find_similar_situations(self, current_state: dict, k: int = 10,
                                filters: dict | None = None) -> list[dict]:
        """Find similar past scenarios using vector similarity."""
        if not self.vector_store:
            return self._find_similar_sql(current_state, k, filters)

        if self.vector_store_type == "chromadb":
            embedding = self._features_to_vector(current_state.get("features", {}))
            where = {}
            if filters:
                if "symbol" in filters: where["symbol"] = filters["symbol"]
                if "action" in filters: where["action"] = filters["action"]
            results = self.vector_store.query(
                query_embeddings=[embedding], n_results=k,
                where=where if where else None,
            )
            ids = results["ids"][0] if results["ids"] else []
            similar = []
            for doc_id in ids:
                row = self._execute("SELECT * FROM ai_memory WHERE id=%s",
                                    (int(doc_id),), fetch="one")
                if row: similar.append(row)
            return similar
        return []

    def _find_similar_sql(self, current_state: dict, k: int,
                          filters: dict | None) -> list[dict]:
        """Fallback similarity search using SQL."""
        symbol = current_state.get("symbol", "WOLF")
        price = current_state.get("price", 0)
        price_range = 0.1
        rows = self._execute(
            "SELECT * FROM ai_memory WHERE symbol=%s AND price BETWEEN %s AND %s "
            "ORDER BY ts DESC LIMIT %s",
            (symbol, price * (1 - price_range), price * (1 + price_range), k),
            fetch="all",
        )
        return rows or []

    def get_outcomes_for_action(self, action: str, symbol: str | None = None,
                                horizon: str = "24h") -> list[dict]:
        """Get historical outcomes for a specific action."""
        outcome_col = f"outcome_{horizon}"
        if outcome_col not in ("outcome_1h", "outcome_24h", "outcome_7d"):
            outcome_col = "outcome_24h"
        if symbol:
            rows = self._execute(
                f"SELECT * FROM ai_memory WHERE action=%s AND {outcome_col} IS NOT NULL "
                f"AND symbol=%s ORDER BY ts DESC LIMIT 100",
                (action, symbol), fetch="all")
        else:
            rows = self._execute(
                f"SELECT * FROM ai_memory WHERE action=%s AND {outcome_col} IS NOT NULL "
                f"ORDER BY ts DESC LIMIT 100",
                (action,), fetch="all")
        return rows or []

    def update_outcome(self, decision_id: int, horizon: str, outcome: float):
        """Update outcome for a past decision."""
        outcome_col = f"outcome_{horizon}"
        if outcome_col not in ("outcome_1h", "outcome_24h", "outcome_7d"):
            outcome_col = "outcome_24h"
        self._execute(f"UPDATE ai_memory SET {outcome_col} = %s WHERE id = %s",
                      (outcome, decision_id))
        LOGGER.debug(f"Updated outcome for decision {decision_id}: {outcome_col}={outcome}")

    # -- Calibration ---------------------------------------------------------

    def compute_calibration_metrics(self, model_type: str | None = None) -> dict[str, Any]:
        """Compute confidence calibration metrics."""
        if model_type:
            rows = self._execute(
                "SELECT confidence, outcome_24h FROM ai_memory "
                "WHERE outcome_24h IS NOT NULL AND model_type=%s",
                (model_type,), fetch="all")
        else:
            rows = self._execute(
                "SELECT confidence, outcome_24h FROM ai_memory "
                "WHERE outcome_24h IS NOT NULL", fetch="all")

        if not rows or len(rows) < 10:
            return {"error": "Insufficient data (need 10+ outcomes)",
                    "sample_count": len(rows or [])}

        buckets = defaultdict(lambda: {"predicted": [], "actual": []})
        for row in rows:
            conf = row["confidence"] or 0
            outcome = row["outcome_24h"] or 0
            bucket = int(conf * 5) / 5
            bucket_label = f"{bucket:.1f}-{bucket + 0.2:.1f}"
            buckets[bucket_label]["predicted"].append(conf)
            buckets[bucket_label]["actual"].append(1 if outcome > 0 else 0)

        calibration_data = []
        all_errors = []
        for bucket_label in sorted(buckets.keys()):
            bd = buckets[bucket_label]
            pp = np.mean(bd["predicted"])
            asr = np.mean(bd["actual"])
            cnt = len(bd["actual"])
            calibration_data.append({
                "confidence_bucket": bucket_label,
                "predicted_prob": round(pp, 3),
                "actual_success_rate": round(asr, 3),
                "sample_count": cnt,
                "error": abs(pp - asr),
            })
            all_errors.append(abs(pp - asr))

        overall_error = np.mean(all_errors) if all_errors else 0
        r_squared = 0
        if len(calibration_data) >= 2:
            predicted = [d["predicted_prob"] for d in calibration_data]
            actual = [d["actual_success_rate"] for d in calibration_data]
            corr = np.corrcoef(predicted, actual)[0, 1]
            r_squared = corr ** 2

        ts = int(time.time())
        for b in calibration_data:
            self._execute(
                "INSERT INTO calibration_metrics "
                "(computed_at, model_type, confidence_bucket, "
                "predicted_prob, actual_success_rate, sample_count) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (ts, model_type or "all", b["confidence_bucket"],
                 b["predicted_prob"], b["actual_success_rate"], b["sample_count"]))

        return {
            "buckets": calibration_data,
            "overall_error": round(overall_error, 3),
            "r_squared": round(r_squared, 3),
            "total_samples": len(rows),
            "computed_at": ts,
        }

    # -- Stats ---------------------------------------------------------------

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        try:
            row = self._execute("SELECT COUNT(*) as cnt FROM ai_memory", fetch="one")
            total_count = (row["cnt"] if isinstance(row, dict) else row[0]) if row else 0

            row = self._execute("SELECT MIN(ts) as min_ts, MAX(ts) as max_ts FROM ai_memory", fetch="one")
            if row and isinstance(row, dict):
                min_ts, max_ts = row.get("min_ts"), row.get("max_ts")
            elif row:
                min_ts, max_ts = row[0], row[1]
            else:
                min_ts, max_ts = None, None

            action_rows = self._execute(
                "SELECT action, COUNT(*) as cnt FROM ai_memory GROUP BY action", fetch="all")
            action_counts = {r["action"]: r["cnt"] for r in (action_rows or [])}

            row = self._execute(
                "SELECT COUNT(*) as cnt FROM ai_memory WHERE outcome_24h IS NOT NULL", fetch="one")
            outcomes_count = (row["cnt"] if isinstance(row, dict) else row[0]) if row else 0

            row = self._execute("SELECT AVG(confidence) as avg_conf FROM ai_memory", fetch="one")
            avg_confidence = 0
            if row:
                avg_confidence = (row["avg_conf"] if isinstance(row, dict) else row[0]) or 0

            return {
                "total_decisions": total_count,
                "time_range": {
                    "start": min_ts, "end": max_ts,
                    "span_days": (max_ts - min_ts) / 86400 if min_ts and max_ts else 0,
                },
                "action_distribution": action_counts,
                "outcomes_tracked": outcomes_count,
                "avg_confidence": round(float(avg_confidence), 3),
                "cache_size": len(self.cache),
                "vector_store": self.vector_store_type if self.vector_store else "none",
                "backend": "postgresql" if self._use_pg else "sqlite",
            }
        except Exception as e:
            LOGGER.error(f"get_memory_stats failed: {e}")
            return {
                "total_decisions": 0,
                "time_range": {"start": None, "end": None, "span_days": 0},
                "action_distribution": {}, "outcomes_tracked": 0,
                "avg_confidence": 0, "cache_size": len(self.cache),
                "vector_store": "none",
                "backend": "postgresql" if self._use_pg else "sqlite",
                "error": str(e),
            }

    def search_by_reasoning(self, query: str, k: int = 10) -> list[dict]:
        """Semantic search over decision reasoning text."""
        if self.vector_store and self.vector_store_type == "chromadb":
            results = self.vector_store.query(query_texts=[query], n_results=k)
            ids = results["ids"][0] if results["ids"] else []
            similar = []
            for doc_id in ids:
                row = self._execute("SELECT * FROM ai_memory WHERE id=%s",
                                    (int(doc_id),), fetch="one")
                if row: similar.append(row)
            return similar
        rows = self._execute(
            "SELECT * FROM ai_memory WHERE reasoning LIKE %s ORDER BY ts DESC LIMIT %s",
            (f"%{query}%", k), fetch="all")
        return rows or []

    def export_for_training(self, symbol: str | None = None,
                            min_samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Export data for ML model training. Returns (X, y)."""
        if symbol:
            rows = self._execute(
                "SELECT features, outcome_24h FROM ai_memory "
                "WHERE outcome_24h IS NOT NULL AND symbol=%s",
                (symbol,), fetch="all")
        else:
            rows = self._execute(
                "SELECT features, outcome_24h FROM ai_memory "
                "WHERE outcome_24h IS NOT NULL", fetch="all")

        if not rows or len(rows) < min_samples:
            raise ValueError(f"Insufficient training data: {len(rows or [])} < {min_samples}")

        X, y = [], []
        for row in rows:
            feat = json.loads(row["features"]) if isinstance(row["features"], str) else row["features"]
            X.append(self._features_to_vector(feat))
            y.append(1 if row["outcome_24h"] > 0 else 0)
        return np.array(X), np.array(y)

    def prune_old_memories(self, keep_days: int = 365):
        """Prune memories older than keep_days. Keep significant events."""
        cutoff_ts = int(time.time()) - (keep_days * 86400)
        deleted = self._execute(
            "DELETE FROM ai_memory WHERE ts < %s AND user_feedback IS NULL "
            "AND ABS(COALESCE(outcome_24h, 0)) < 0.05", (cutoff_ts,))
        self._load_cache()
        LOGGER.info(f"Pruned {deleted} old memories (kept significant events)")
        return deleted

    def close(self):
        """Close database connections."""
        if self._sqlite_conn:
            self._sqlite_conn.close()
        LOGGER.info("AI Memory closed")


# -- Singleton ---------------------------------------------------------------

_memory_instance: AIMemory | None = None


def get_memory(db_path: str = "data/ai_memory.db", vector_store: str = "chromadb") -> AIMemory:
    """Get or create singleton AI memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = AIMemory(db_path, vector_store)
    return _memory_instance
