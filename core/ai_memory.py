"""
GHOST AI Memory System
======================
Long-term memory with vector embeddings for semantic search.
Replaces 100-sample ring buffer with unlimited persistent storage.

Features:
- SQLite for structured queries (timestamp, symbol, action, outcome)
- Vector embeddings for semantic similarity (FAISS/ChromaDB)
- Confidence calibration (predicted vs actual outcomes)
- Episodic memory (significant trades, market events)
- Memory consolidation (summarize old experiences)

Author: Ghost AI
Date: 2025-10-03
"""

import json
import logging
import sqlite3
import time
from collections import defaultdict
from typing import Any

import numpy as np

# Optional: ChromaDB for vector storage (fallback to FAISS if not available)
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
    - SQLite: Structured queries (filters, aggregations)
    - Vector Store: Semantic similarity search
    - In-Memory Cache: Recent 1000 decisions for fast access

    Schema:
    -------
    ai_memory table:
        - id: INTEGER PRIMARY KEY
        - ts: BIGINT (Unix timestamp)
        - symbol: TEXT
        - price: REAL
        - prev_close: REAL
        - volume: REAL (optional)
        - volatility: REAL (optional)
        - news_score: REAL
        - sentiment: REAL (optional)
        - features: TEXT (JSON blob, 100+ dimensions)
        - action: TEXT (BUY, SELL, HOLD)
        - confidence: REAL (0-1)
        - reasoning: TEXT
        - model_version: TEXT
        - model_type: TEXT (knn, rl, ensemble)
        - outcome_1h: REAL (PnL 1h later)
        - outcome_24h: REAL (PnL 24h later)
        - outcome_7d: REAL (PnL 7d later)
        - realized_pnl: REAL (actual PnL if executed)
        - user_feedback: TEXT (optional)
        - executed: BOOLEAN
    """

    def __init__(self, db_path: str = "data/ai_memory.db", vector_store: str = "chromadb"):
        """
        Initialize AI Memory.

        Args:
            db_path: Path to SQLite database
            vector_store: "chromadb", "faiss", or "none"
        """
        self.db_path = db_path
        self.vector_store_type = vector_store

        # Connect to SQLite
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

        # Initialize vector store
        self.vector_store = None
        if vector_store == "chromadb" and HAS_CHROMADB:
            self._init_chromadb()
        elif vector_store == "faiss" and HAS_FAISS:
            self._init_faiss()
        else:
            LOGGER.warning(f"Vector store '{vector_store}' not available, using SQLite only")

        # In-memory cache for recent decisions (fast access)
        self.cache: list[dict] = []
        self.cache_size = 1000
        self._load_cache()

        LOGGER.info(
            f"AIMemory initialized: db={db_path}, vector_store={vector_store}, cache_size={len(self.cache)}"
        )

    def _init_tables(self):
        """Create SQLite tables if not exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts BIGINT NOT NULL,
                    symbol TEXT NOT NULL,

                    -- Market Context
                    price REAL,
                    prev_close REAL,
                    volume REAL,
                    volatility REAL,
                    news_score REAL,
                    sentiment REAL,

                    -- Features (JSON blob)
                    features TEXT,

                    -- Decision
                    action TEXT,
                    confidence REAL,
                    reasoning TEXT,

                    -- Model Info
                    model_version TEXT,
                    model_type TEXT,

                    -- Outcomes (filled later)
                    outcome_1h REAL,
                    outcome_24h REAL,
                    outcome_7d REAL,
                    realized_pnl REAL,

                    -- Metadata
                    user_feedback TEXT,
                    executed BOOLEAN DEFAULT 0
                )
            """)

            # Indexes for common queries
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON ai_memory(ts)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON ai_memory(symbol)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_action ON ai_memory(action)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_executed ON ai_memory(executed)")

            # Calibration metrics table (confidence vs actual outcomes)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    computed_at BIGINT,
                    model_type TEXT,
                    confidence_bucket TEXT,
                    predicted_prob REAL,
                    actual_success_rate REAL,
                    sample_count INTEGER
                )
            """)

        LOGGER.info("AI memory tables initialized")

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
        # TODO: Implement FAISS initialization
        # Load existing index or create new one
        LOGGER.warning("FAISS support not yet implemented, falling back to SQLite")
        self.vector_store = None

    def _load_cache(self):
        """Load recent decisions into cache."""
        cur = self.conn.execute(
            "SELECT * FROM ai_memory ORDER BY ts DESC LIMIT ?", (self.cache_size,)
        )
        self.cache = [dict(row) for row in cur.fetchall()]
        LOGGER.info(f"Loaded {len(self.cache)} decisions into cache")

    def store_decision(self, decision: dict[str, Any]) -> int:
        """
        Store AI decision with context.

        Args:
            decision: Dictionary with keys:
                - ts (int): Timestamp
                - symbol (str): Ticker
                - price (float): Current price
                - prev_close (float): Previous close
                - news_score (float): News sentiment
                - features (dict): Feature vector
                - action (str): BUY/SELL/HOLD
                - confidence (float): 0-1
                - reasoning (str): Explanation
                - model_version (str): Model ID
                - model_type (str): knn/rl/ensemble

        Returns:
            int: Row ID of stored decision
        """
        # Store in SQLite
        features_json = json.dumps(decision.get("features", {}))

        cur = self.conn.execute(
            """
            INSERT INTO ai_memory (
                ts, symbol, price, prev_close, volume, volatility,
                news_score, sentiment, features, action, confidence,
                reasoning, model_version, model_type, executed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
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
            ),
        )
        self.conn.commit()
        row_id = cur.lastrowid if cur.lastrowid is not None else 0

        # Add to vector store (if available)
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
            f"Stored decision {row_id}: {decision.get('action')} {decision.get('symbol')} @ {decision.get('confidence'):.2f}"
        )

        return row_id

    def _add_to_vector_store(self, row_id: int, decision: dict):
        """Add decision to vector store for semantic search."""
        if not self.vector_store:
            return

        if self.vector_store_type == "chromadb":
            # Convert features dict to list
            features = decision.get("features", {})
            embedding = self._features_to_vector(features)

            # Create document (reasoning text)
            document = decision.get("reasoning", "")

            # Metadata
            metadata = {
                "symbol": decision.get("symbol", ""),
                "action": decision.get("action", ""),
                "confidence": float(decision.get("confidence", 0.5)),
                "model_type": decision.get("model_type", ""),
                "ts": int(decision.get("ts", 0)),
            }

            self.vector_store.add(
                ids=[str(row_id)],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata],
            )

    def _features_to_vector(self, features: dict) -> list[float]:
        """
        Convert feature dict to fixed-length vector.

        Standard feature order:
        [ret_1d, ret_7d, vol_20d, news_score, sentiment, pos_pct, ...]
        """
        # Define standard feature keys (extend as needed)
        standard_keys = [
            "ret_1d",
            "ret_7d",
            "ret_30d",
            "vol_20d",
            "vol_60d",
            "news_score",
            "sentiment",
            "rsi_14",
            "macd",
            "macd_signal",
            "bb_width",
            "bb_position",
            "pos_pct",
            "pnl_pct",
            "momentum_1d",
            "momentum_7d",
        ]

        vector = []
        for key in standard_keys:
            vector.append(float(features.get(key, 0.0)))

        return vector

    def find_similar_situations(
        self, current_state: dict, k: int = 10, filters: dict | None = None
    ) -> list[dict]:
        """
        Find similar past scenarios using vector similarity.

        Args:
            current_state: Dict with 'features', 'symbol', etc.
            k: Number of similar situations to return
            filters: Optional filters (symbol, action, date range)

        Returns:
            List of similar decisions with outcomes
        """
        if not self.vector_store:
            # Fallback to SQLite-only search (less sophisticated)
            return self._find_similar_sql(current_state, k, filters)

        if self.vector_store_type == "chromadb":
            embedding = self._features_to_vector(current_state.get("features", {}))

            # Build where clause from filters
            where = {}
            if filters:
                if "symbol" in filters:
                    where["symbol"] = filters["symbol"]
                if "action" in filters:
                    where["action"] = filters["action"]

            results = self.vector_store.query(
                query_embeddings=[embedding], n_results=k, where=where if where else None
            )

            # Fetch full records from SQLite
            ids = results["ids"][0] if results["ids"] else []
            similar = []
            for doc_id in ids:
                cur = self.conn.execute("SELECT * FROM ai_memory WHERE id=?", (int(doc_id),))
                row = cur.fetchone()
                if row:
                    similar.append(dict(row))

            return similar

        return []

    def _find_similar_sql(self, current_state: dict, k: int, filters: dict | None) -> list[dict]:
        """Fallback similarity search using SQL (no vector DB)."""
        # Simple heuristic: Match on symbol, similar price range
        symbol = current_state.get("symbol", "WOLF")
        price = current_state.get("price", 0)

        query = """
            SELECT * FROM ai_memory
            WHERE symbol=?
            AND price BETWEEN ? AND ?
            ORDER BY ts DESC
            LIMIT ?
        """

        price_range = 0.1  # ±10%
        cur = self.conn.execute(
            query, (symbol, price * (1 - price_range), price * (1 + price_range), k)
        )

        return [dict(row) for row in cur.fetchall()]

    def get_outcomes_for_action(
        self, action: str, symbol: str | None = None, horizon: str = "24h"
    ) -> list[dict]:
        """
        Get historical outcomes for a specific action.

        Args:
            action: BUY, SELL, or HOLD
            symbol: Optional symbol filter
            horizon: '1h', '24h', or '7d'

        Returns:
            List of decisions with realized outcomes
        """
        outcome_col = f"outcome_{horizon}"

        query = f"""
            SELECT * FROM ai_memory
            WHERE action=?
            AND {outcome_col} IS NOT NULL
        """
        params = [action]

        if symbol:
            query += " AND symbol=?"
            params.append(symbol)

        query += " ORDER BY ts DESC LIMIT 100"

        cur = self.conn.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    def update_outcome(self, decision_id: int, horizon: str, outcome: float):
        """
        Update outcome for a past decision.

        Args:
            decision_id: Row ID of decision
            horizon: '1h', '24h', or '7d'
            outcome: PnL or return %
        """
        outcome_col = f"outcome_{horizon}"

        self.conn.execute(
            f"""
            UPDATE ai_memory
            SET {outcome_col} = ?
            WHERE id = ?
        """,
            (outcome, decision_id),
        )
        self.conn.commit()

        LOGGER.debug(f"Updated outcome for decision {decision_id}: {outcome_col}={outcome}")

    def compute_calibration_metrics(self, model_type: str | None = None) -> dict[str, Any]:
        """
        Compute confidence calibration metrics.

        Calibration: "If model says 70% confidence, it should be right 70% of the time"

        Returns:
            Dict with calibration data:
            - buckets: List of {confidence_range, predicted_prob, actual_success_rate, count}
            - overall_error: Mean absolute calibration error
            - r_squared: R² of calibration plot
        """
        # Get all decisions with outcomes
        query = """
            SELECT confidence, outcome_24h
            FROM ai_memory
            WHERE outcome_24h IS NOT NULL
        """
        params = []

        if model_type:
            query += " AND model_type=?"
            params.append(model_type)

        cur = self.conn.execute(query, params)
        rows = cur.fetchall()

        if len(rows) < 10:
            return {"error": "Insufficient data (need 10+ outcomes)", "sample_count": len(rows)}

        # Bin by confidence (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
        buckets = defaultdict(lambda: {"predicted": [], "actual": []})

        for row in rows:
            conf = row[0]
            outcome = row[1]

            # Determine bucket
            bucket = int(conf * 5) / 5  # 0.0, 0.2, 0.4, 0.6, 0.8
            bucket_label = f"{bucket:.1f}-{bucket + 0.2:.1f}"

            buckets[bucket_label]["predicted"].append(conf)
            buckets[bucket_label]["actual"].append(1 if outcome > 0 else 0)

        # Compute calibration for each bucket
        calibration_data = []
        all_errors = []

        for bucket_label in sorted(buckets.keys()):
            bucket_data = buckets[bucket_label]
            predicted_prob = np.mean(bucket_data["predicted"])
            actual_success_rate = np.mean(bucket_data["actual"])
            count = len(bucket_data["actual"])

            calibration_data.append(
                {
                    "confidence_bucket": bucket_label,
                    "predicted_prob": round(predicted_prob, 3),
                    "actual_success_rate": round(actual_success_rate, 3),
                    "sample_count": count,
                    "error": abs(predicted_prob - actual_success_rate),
                }
            )

            all_errors.append(abs(predicted_prob - actual_success_rate))

        # Overall metrics
        overall_error = np.mean(all_errors) if all_errors else 0

        # R² (how well calibrated)
        if len(calibration_data) >= 2:
            predicted = [d["predicted_prob"] for d in calibration_data]
            actual = [d["actual_success_rate"] for d in calibration_data]
            correlation = np.corrcoef(predicted, actual)[0, 1]
            r_squared = correlation**2
        else:
            r_squared = 0

        # Store in database
        ts = int(time.time())
        for bucket in calibration_data:
            self.conn.execute(
                """
                INSERT INTO calibration_metrics (
                    computed_at, model_type, confidence_bucket,
                    predicted_prob, actual_success_rate, sample_count
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    ts,
                    model_type or "all",
                    bucket["confidence_bucket"],
                    bucket["predicted_prob"],
                    bucket["actual_success_rate"],
                    bucket["sample_count"],
                ),
            )
        self.conn.commit()

        return {
            "buckets": calibration_data,
            "overall_error": round(overall_error, 3),
            "r_squared": round(r_squared, 3),
            "total_samples": len(rows),
            "computed_at": ts,
        }

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        cur = self.conn.execute("SELECT COUNT(*) FROM ai_memory")
        total_count = cur.fetchone()[0]

        cur = self.conn.execute("SELECT MIN(ts), MAX(ts) FROM ai_memory")
        row = cur.fetchone()
        min_ts, max_ts = row[0], row[1]

        cur = self.conn.execute("""
            SELECT action, COUNT(*) FROM ai_memory
            GROUP BY action
        """)
        action_counts = {row[0]: row[1] for row in cur.fetchall()}

        cur = self.conn.execute("""
            SELECT COUNT(*) FROM ai_memory
            WHERE outcome_24h IS NOT NULL
        """)
        outcomes_count = cur.fetchone()[0]

        cur = self.conn.execute("""
            SELECT AVG(confidence) FROM ai_memory
        """)
        avg_confidence = cur.fetchone()[0] or 0

        return {
            "total_decisions": total_count,
            "time_range": {
                "start": min_ts,
                "end": max_ts,
                "span_days": (max_ts - min_ts) / 86400 if min_ts and max_ts else 0,
            },
            "action_distribution": action_counts,
            "outcomes_tracked": outcomes_count,
            "avg_confidence": round(avg_confidence, 3),
            "cache_size": len(self.cache),
            "vector_store": self.vector_store_type if self.vector_store else "none",
        }

    def search_by_reasoning(self, query: str, k: int = 10) -> list[dict]:
        """
        Semantic search over decision reasoning text.

        Args:
            query: Natural language query (e.g., "high volatility bull market")
            k: Number of results

        Returns:
            List of decisions matching query
        """
        if not self.vector_store or self.vector_store_type != "chromadb":
            # Fallback to SQL LIKE search
            cur = self.conn.execute(
                """
                SELECT * FROM ai_memory
                WHERE reasoning LIKE ?
                ORDER BY ts DESC
                LIMIT ?
            """,
                (f"%{query}%", k),
            )
            return [dict(row) for row in cur.fetchall()]

        # ChromaDB semantic search
        results = self.vector_store.query(query_texts=[query], n_results=k)

        ids = results["ids"][0] if results["ids"] else []
        similar = []
        for doc_id in ids:
            cur = self.conn.execute("SELECT * FROM ai_memory WHERE id=?", (int(doc_id),))
            row = cur.fetchone()
            if row:
                similar.append(dict(row))

        return similar

    def export_for_training(
        self, symbol: str | None = None, min_samples: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Export data for ML model training.

        Returns:
            (X, y) where:
            - X: Feature matrix (N x D)
            - y: Labels (N,) - 1 for positive outcome, 0 for negative
        """
        query = """
            SELECT features, outcome_24h FROM ai_memory
            WHERE outcome_24h IS NOT NULL
        """
        params = []

        if symbol:
            query += " AND symbol=?"
            params.append(symbol)

        cur = self.conn.execute(query, params)
        rows = cur.fetchall()

        if len(rows) < min_samples:
            raise ValueError(f"Insufficient training data: {len(rows)} < {min_samples}")

        # Parse features and labels
        X = []
        y = []
        for row in rows:
            features = json.loads(row[0])
            outcome = row[1]

            X.append(self._features_to_vector(features))
            y.append(1 if outcome > 0 else 0)

        return np.array(X), np.array(y)

    def prune_old_memories(self, keep_days: int = 365):
        """
        Prune memories older than keep_days (default 1 year).
        Keep significant events (large outcomes, user feedback).
        """
        cutoff_ts = int(time.time()) - (keep_days * 86400)

        # Delete non-significant old memories
        cur = self.conn.execute(
            """
            DELETE FROM ai_memory
            WHERE ts < ?
            AND user_feedback IS NULL
            AND ABS(COALESCE(outcome_24h, 0)) < 0.05
        """,
            (cutoff_ts,),
        )

        deleted_count = cur.rowcount
        self.conn.commit()

        # Reload cache
        self._load_cache()

        LOGGER.info(f"Pruned {deleted_count} old memories (kept significant events)")

        return deleted_count

    def close(self):
        """Close database connections."""
        self.conn.close()
        LOGGER.info("AI Memory closed")


# Singleton instance (optional, for convenience)
_memory_instance: AIMemory | None = None


def get_memory(db_path: str = "data/ai_memory.db", vector_store: str = "chromadb") -> AIMemory:
    """Get or create singleton AI memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = AIMemory(db_path, vector_store)
    return _memory_instance
