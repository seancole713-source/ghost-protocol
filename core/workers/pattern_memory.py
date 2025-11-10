"""
Historical Pattern Memory (Long-Term Memory)
Stores market episodes and enables cosine similarity queries.
Backed by DuckDB (ghost.duckdb) with an optional FAISS-like cosine search in Python.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from typing import Any

import duckdb
import numpy as np

DB_SQLITE = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
DB_DUCK = os.getenv("GHOST_DUCKDB", "ghost.duckdb")
REFRESH_S = int(os.getenv("PATTERN_REFRESH_S", "900"))  # 15 min

# DuckDB table for vector storage
DUCK_SQL = """
CREATE TABLE IF NOT EXISTS analog_memory (
  ts INTEGER,
  label TEXT,
  vector BLOB
);
"""

# SQLite index for quick snapshots
SQLITE_SQL = """
CREATE TABLE IF NOT EXISTS analog_index (
  ts INTEGER PRIMARY KEY,
  label TEXT,
  meta_json TEXT
);
"""


def _vec(x: dict[str, float]) -> np.ndarray:
    keys = ["ret_1d", "dist_avg", "news", "qty"]
    return np.array([float(x.get(k, 0.0)) for k in keys], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


async def ensure_tables() -> None:
    os.makedirs(os.path.dirname(DB_SQLITE), exist_ok=True)
    sqlite3.connect(DB_SQLITE).execute(SQLITE_SQL).close()
    duckdb.connect(DB_DUCK).execute(DUCK_SQL).close()


async def add_episode(
    ts: int, label: str, features: dict[str, float], meta: dict[str, Any]
) -> None:
    v = _vec(features)
    con = duckdb.connect(DB_DUCK)
    try:
        con.execute(
            "INSERT INTO analog_memory (ts,label,vector) VALUES (?,?,?)", (ts, label, v.tobytes())
        )
    finally:
        con.close()
    conn = sqlite3.connect(DB_SQLITE)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO analog_index (ts,label,meta_json) VALUES (?,?,?)",
            (ts, label, json.dumps(meta)),
        )
        conn.commit()
    finally:
        conn.close()


async def find_similar(features: dict[str, float], k: int = 20) -> list[dict[str, Any]]:
    q = _vec(features)
    con = duckdb.connect(DB_DUCK)
    try:
        rows = con.execute("SELECT ts,label,vector FROM analog_memory").fetchall()
    finally:
        con.close()
    scored: list[tuple[float, int, str]] = []
    for ts, label, vec_blob in rows:
        v = np.frombuffer(vec_blob, dtype=np.float32)
        sc = cosine(q, v)
        scored.append((sc, int(ts), str(label)))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    conn = sqlite3.connect(DB_SQLITE)
    try:
        for sc, ts, label in scored[:k]:
            r = conn.execute("SELECT meta_json FROM analog_index WHERE ts=?", (ts,)).fetchone()
            meta = json.loads(r[0]) if r else {}
            out.append({"ts": ts, "label": label, "score": float(sc), "meta": meta})
    finally:
        conn.close()
    return out


async def run_forever() -> None:
    await ensure_tables()
    # This worker does not generate episodes on its own; it maintains the store.
    # Episodes should be added from API flows (e.g., post-forecast, post-decision).
    while True:
        await asyncio.sleep(max(60, REFRESH_S))
