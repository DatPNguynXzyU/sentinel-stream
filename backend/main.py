"""FastAPI server: REST API, SSE alert stream, metrics and the live dashboard."""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

import numpy as np
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse

import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("sentinel.api")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "static" / "index.html"

ALERT_CHANNEL = "alerts_channel"
LATENCY_KEY = "metrics:latencies"
TOTAL_KEY = "metrics:total_tx"
FLAGGED_KEY = "metrics:flagged_tx"
THROUGHPUT_PREFIX = "metrics:tput:"
BURST_KEY = "sim:burst_mode"
BURST_TTL_SECONDS = 10
THROUGHPUT_WINDOW_SECONDS = 10
SSE_KEEPALIVE_SECONDS = 15


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.redis = aioredis.Redis.from_url(
        REDIS_URL, decode_responses=True, health_check_interval=30
    )
    # Wait for PostgreSQL and create tables (runs in a thread: psycopg pool is sync).
    await run_in_threadpool(db.init_db)
    logger.info("API ready")
    try:
        yield
    finally:
        await app.state.redis.aclose()
        await run_in_threadpool(db.close_pool)


app = FastAPI(title="sentinel-stream", version="1.0.0", lifespan=lifespan)


def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    p50, p95, p99 = np.percentile(np.asarray(values, dtype=np.float64), [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99)}


async def _throughput_tps(r: aioredis.Redis) -> float:
    """Average tx/s over the last N *complete* seconds (per-second counters)."""
    now = int(time.time())
    keys = [f"{THROUGHPUT_PREFIX}{now - i}" for i in range(1, THROUGHPUT_WINDOW_SECONDS + 1)]
    counts = await r.mget(keys)
    return sum(int(c) for c in counts if c is not None) / THROUGHPUT_WINDOW_SECONDS


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/api/health")
async def health(request: Request) -> Dict[str, Any]:
    try:
        redis_ok = bool(await request.app.state.redis.ping())
    except Exception:
        redis_ok = False
    postgres_ok = await run_in_threadpool(db.ping)
    return {"status": "ok" if redis_ok and postgres_ok else "degraded",
            "redis": redis_ok, "postgres": postgres_ok}


@app.get("/api/metrics")
@app.get("/metrics")
async def metrics(request: Request) -> Dict[str, Any]:
    r: aioredis.Redis = request.app.state.redis
    pipe = r.pipeline(transaction=False)
    pipe.get(TOTAL_KEY)
    pipe.get(FLAGGED_KEY)
    pipe.lrange(LATENCY_KEY, 0, -1)
    pipe.ttl(BURST_KEY)
    total_raw, flagged_raw, latencies_raw, burst_ttl = await pipe.execute()

    try:
        flagged_count = await run_in_threadpool(db.count_flagged)
    except Exception:
        logger.warning("Postgres count failed; falling back to Redis counter")
        flagged_count = int(flagged_raw or 0)

    pct = _percentiles([float(v) for v in latencies_raw])
    return {
        "total_processed": int(total_raw or 0),
        "flagged_count": flagged_count,
        "p50_latency_ms": round(pct["p50"], 4),
        "p95_latency_ms": round(pct["p95"], 4),
        "p99_latency_ms": round(pct["p99"], 4),
        "throughput_tps": round(await _throughput_tps(r), 2),
        "latency_samples": len(latencies_raw),
        "burst_active": burst_ttl is not None and burst_ttl > 0,
        "burst_seconds_remaining": max(int(burst_ttl or 0), 0),
    }


@app.get("/api/anomalies")
def anomalies() -> List[Dict[str, Any]]:
    try:
        return db.fetch_latest_anomalies(25)
    except Exception as exc:
        logger.exception("Failed to query anomalies")
        raise HTTPException(status_code=503, detail="database unavailable") from exc


@app.post("/api/trigger-burst")
async def trigger_burst(request: Request) -> Dict[str, Any]:
    await request.app.state.redis.set(BURST_KEY, "1", ex=BURST_TTL_SECONDS)
    return {"status": "burst_started", "duration_seconds": BURST_TTL_SECONDS}


@app.get("/api/stream")
@app.get("/stream")
async def stream(request: Request) -> StreamingResponse:
    """Server-Sent Events feed of anomaly alerts published on `alerts_channel`."""

    async def event_source() -> AsyncIterator[str]:
        pubsub = request.app.state.redis.pubsub(ignore_subscribe_messages=True)
        await pubsub.subscribe(ALERT_CHANNEL)
        try:
            yield "retry: 3000\n\n"
            yield ": connected\n\n"
            last_sent = time.monotonic()
            while True:
                if await request.is_disconnected():
                    break
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message.get("type") == "message":
                    payload = message["data"]
                    if not isinstance(payload, str):
                        payload = json.dumps(payload)
                    yield f"event: anomaly\ndata: {payload}\n\n"
                    last_sent = time.monotonic()
                elif time.monotonic() - last_sent >= SSE_KEEPALIVE_SECONDS:
                    yield ": keepalive\n\n"
                    last_sent = time.monotonic()
                else:
                    await asyncio.sleep(0)
        finally:
            try:
                await pubsub.unsubscribe(ALERT_CHANNEL)
                await pubsub.aclose()
            except Exception:
                logger.debug("pubsub cleanup failed", exc_info=True)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
