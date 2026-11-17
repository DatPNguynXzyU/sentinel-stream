"""PostgreSQL connection pool and automatic schema initialisation."""

import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger("sentinel.db")

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel"
)

# Arbitrary constant used to serialise schema creation across processes
# (api and worker both call init_db() at start-up).
_SCHEMA_LOCK_ID = 7272_0001

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS flagged_transactions (
    tx_id            TEXT PRIMARY KEY,
    user_id          TEXT             NOT NULL,
    amount           DOUBLE PRECISION NOT NULL,
    tx_count_60s     INTEGER          NOT NULL,
    total_spend_60s  DOUBLE PRECISION NOT NULL,
    latency_ms       DOUBLE PRECISION NOT NULL,
    created_at       TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_flagged_transactions_created_at
    ON flagged_transactions (created_at DESC);

CREATE TABLE IF NOT EXISTS throughput_metrics (
    id               BIGSERIAL PRIMARY KEY,
    consumer         TEXT             NOT NULL,
    window_seconds   DOUBLE PRECISION NOT NULL,
    processed        INTEGER          NOT NULL,
    tps              DOUBLE PRECISION NOT NULL,
    p50_latency_ms   DOUBLE PRECISION NOT NULL,
    p95_latency_ms   DOUBLE PRECISION NOT NULL,
    p99_latency_ms   DOUBLE PRECISION NOT NULL,
    created_at       TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_throughput_metrics_created_at
    ON throughput_metrics (created_at DESC);
"""

_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_pool(max_wait_seconds: float = 60.0) -> ConnectionPool:
    """Return the process-wide connection pool, creating it on first use.

    Retries until PostgreSQL accepts connections (or the deadline passes).
    """
    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:
            return _pool

        deadline = time.monotonic() + max_wait_seconds
        attempt = 0
        while True:
            attempt += 1
            pool = ConnectionPool(
                conninfo=DATABASE_URL,
                min_size=1,
                max_size=int(os.getenv("DB_POOL_MAX", "5")),
                kwargs={"autocommit": False},
                open=False,
            )
            try:
                pool.open(wait=True, timeout=5.0)
                _pool = pool
                logger.info("PostgreSQL pool ready (attempt %d)", attempt)
                return _pool
            except Exception as exc:  # pool.PoolTimeout, OperationalError, ...
                pool.close()
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"PostgreSQL not reachable after {attempt} attempts: {exc}"
                    ) from exc
                logger.warning("PostgreSQL not ready (%s); retrying...", exc)
                time.sleep(1.5)


def init_db() -> None:
    """Create all tables/indexes if they do not exist (safe to call concurrently)."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK_ID,))
            cur.execute(SCHEMA_SQL)
        conn.commit()
    logger.info("Database schema initialised")


def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def insert_flagged_batch(rows: Sequence[Dict[str, Any]]) -> int:
    """Insert a batch of flagged transactions in one round trip-friendly transaction.

    Duplicate tx_ids (e.g. redelivered stream messages) are ignored.
    """
    if not rows:
        return 0
    params = [
        (
            r["tx_id"],
            r["user_id"],
            r["amount"],
            r["tx_count_60s"],
            r["total_spend_60s"],
            r["latency_ms"],
            r["created_at"],
        )
        for r in rows
    ]
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO flagged_transactions
                    (tx_id, user_id, amount, tx_count_60s, total_spend_60s,
                     latency_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tx_id) DO NOTHING
                """,
                params,
            )
        conn.commit()
    return len(params)


def insert_throughput_metric(
    consumer: str,
    window_seconds: float,
    processed: int,
    tps: float,
    p50: float,
    p95: float,
    p99: float,
) -> None:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO throughput_metrics
                    (consumer, window_seconds, processed, tps,
                     p50_latency_ms, p95_latency_ms, p99_latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (consumer, window_seconds, processed, tps, p50, p95, p99),
            )
        conn.commit()


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def fetch_latest_anomalies(limit: int = 25) -> List[Dict[str, Any]]:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT tx_id, user_id, amount, tx_count_60s, total_spend_60s,
                       latency_ms, created_at
                FROM flagged_transactions
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return [{k: _iso(v) for k, v in row.items()} for row in rows]


def count_flagged() -> int:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM flagged_transactions")
            row = cur.fetchone()
    return int(row[0]) if row else 0


def ping() -> bool:
    try:
        pool = get_pool(max_wait_seconds=3.0)
        with pool.connection() as conn:
            conn.execute("SELECT 1")
        return True
    except (psycopg.Error, RuntimeError):
        return False
