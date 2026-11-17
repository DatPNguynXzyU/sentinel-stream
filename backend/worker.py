"""Stream worker: Redis Streams consumer group -> ZSET features -> ONNX scoring.

Pipeline per transaction
  1. XREADGROUP from `tx_stream` (consumer group `tx_group`)
  2. Single Redis pipeline on `user_window:{user_id}` (ZADD / ZREMRANGEBYSCORE /
     ZRANGE / EXPIRE) -> tx_count_60s, total_spend_60s
  3. ONNX Runtime IsolationForest inference (label -1 == anomaly)
  4. Metrics (LPUSH/LTRIM latencies, INCR counters), Pub/Sub alert on anomalies,
     XACK -- all in one pipeline
  5. Anomalies are handed to a background thread which batches them into PostgreSQL
"""

import json
import logging
import os
import queue
import signal
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import redis

import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("sentinel.worker")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_PATH = os.getenv("MODEL_PATH", "model.onnx")
CONSUMER_NAME = os.getenv("CONSUMER_NAME", socket.gethostname())

STREAM = "tx_stream"
GROUP = "tx_group"
ALERT_CHANNEL = "alerts_channel"
LATENCY_KEY = "metrics:latencies"
TOTAL_KEY = "metrics:total_tx"
FLAGGED_KEY = "metrics:flagged_tx"
THROUGHPUT_PREFIX = "metrics:tput:"

WINDOW_SECONDS = 60.0
WINDOW_TTL_SECONDS = 120
LATENCY_HISTORY = 1000

READ_COUNT = int(os.getenv("READ_COUNT", "100"))
READ_BLOCK_MS = 1000
CLAIM_IDLE_MS = 60_000
CLAIM_INTERVAL_S = 30.0

DB_BATCH_SIZE = int(os.getenv("DB_BATCH_SIZE", "100"))
DB_FLUSH_INTERVAL_S = float(os.getenv("DB_FLUSH_INTERVAL_S", "0.5"))
THROUGHPUT_LOG_INTERVAL_S = float(os.getenv("THROUGHPUT_LOG_INTERVAL_S", "10"))


class StatsWindow:
    """Thread-safe accumulator of latencies since the last snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies: List[float] = []
        self._started = time.monotonic()

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._latencies.append(latency_ms)

    def snapshot_and_reset(self) -> Tuple[float, List[float]]:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._started
            latencies = self._latencies
            self._latencies = []
            self._started = now
        return elapsed, latencies


class DBWriter(threading.Thread):
    """Batches flagged transactions into PostgreSQL off the hot path and
    periodically persists throughput metrics."""

    def __init__(self, consumer: str, stats: StatsWindow) -> None:
        super().__init__(name="db-writer")
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=200_000)
        self._stop_event = threading.Event()
        self._consumer = consumer
        self._stats = stats

    def submit(self, row: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            logger.error("DB queue full; dropping flagged tx %s", row.get("tx_id"))

    def stop(self) -> None:
        self._stop_event.set()

    def _drain(self) -> List[Dict[str, Any]]:
        batch: List[Dict[str, Any]] = []
        try:
            batch.append(self._queue.get(timeout=DB_FLUSH_INTERVAL_S))
        except queue.Empty:
            return batch
        while len(batch) < DB_BATCH_SIZE:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _flush(self, batch: List[Dict[str, Any]]) -> None:
        for attempt in range(1, 4):
            try:
                db.insert_flagged_batch(batch)
                return
            except Exception as exc:
                logger.warning("Postgres batch insert failed (attempt %d/3): %s", attempt, exc)
                time.sleep(0.5 * attempt)
        logger.error("Dropping batch of %d flagged transactions after 3 failures", len(batch))

    def _log_throughput(self) -> None:
        elapsed, latencies = self._stats.snapshot_and_reset()
        if not latencies or elapsed <= 0:
            return
        arr = np.asarray(latencies, dtype=np.float64)
        p50, p95, p99 = (float(v) for v in np.percentile(arr, [50, 95, 99]))
        tps = len(latencies) / elapsed
        logger.info(
            "throughput=%.1f tx/s processed=%d p50=%.3fms p95=%.3fms p99=%.3fms",
            tps, len(latencies), p50, p95, p99,
        )
        try:
            db.insert_throughput_metric(
                self._consumer, elapsed, len(latencies), tps, p50, p95, p99
            )
        except Exception as exc:
            logger.warning("Failed to persist throughput metric: %s", exc)

    def run(self) -> None:
        last_metric = time.monotonic()
        while True:
            batch = self._drain()
            if batch:
                self._flush(batch)
            if time.monotonic() - last_metric >= THROUGHPUT_LOG_INTERVAL_S:
                self._log_throughput()
                last_metric = time.monotonic()
            if self._stop_event.is_set() and self._queue.empty():
                break
        self._log_throughput()


def connect_redis() -> redis.Redis:
    while True:
        try:
            client = redis.Redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                health_check_interval=30,
            )
            client.ping()
            logger.info("Connected to Redis at %s", REDIS_URL)
            return client
        except redis.exceptions.RedisError as exc:
            logger.warning("Redis not ready (%s); retrying...", exc)
            time.sleep(1.5)


def ensure_group(client: redis.Redis) -> None:
    try:
        client.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        logger.info("Created consumer group %s on %s", GROUP, STREAM)
    except redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def load_session() -> Tuple[ort.InferenceSession, str]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        MODEL_PATH, sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    # Warm-up so the first real transaction does not pay lazy-initialisation cost.
    warm = np.array([[50.0, 1.0, 50.0]], dtype=np.float32)
    for _ in range(20):
        session.run(None, {input_name: warm})
    logger.info("Loaded ONNX model %s (input=%s)", MODEL_PATH, input_name)
    return session, input_name


class Worker:
    def __init__(self) -> None:
        self.redis = connect_redis()
        ensure_group(self.redis)
        db.init_db()
        self.session, self.input_name = load_session()
        self.stats = StatsWindow()
        self.writer = DBWriter(CONSUMER_NAME, self.stats)
        self.stop_event = threading.Event()

    # ------------------------------------------------------------------ scoring
    def process(self, msg_id: str, fields: Optional[Dict[str, str]]) -> None:
        r = self.redis
        try:
            if not fields:
                raise ValueError("empty message")
            tx_id = str(fields["tx_id"])
            user_id = str(fields["user_id"])
            amount = round(float(fields["amount"]), 2)
            timestamp = float(fields["timestamp"])
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Malformed message %s (%s); acknowledging", msg_id, exc)
            r.xack(STREAM, GROUP, msg_id)
            return

        bench_run = fields.get("bench_run")

        try:
            started = time.perf_counter()

            # --- Feature engineering: one pipeline round trip -----------------
            key = f"user_window:{user_id}"
            pipe = r.pipeline()
            pipe.zadd(key, {f"{tx_id}:{amount:.2f}": timestamp})
            pipe.zremrangebyscore(key, "-inf", timestamp - WINDOW_SECONDS)
            pipe.zrange(key, 0, -1)
            pipe.expire(key, WINDOW_TTL_SECONDS)
            _, _, members, _ = pipe.execute()

            tx_count_60s = len(members)
            total_spend_60s = 0.0
            for member in members:
                total_spend_60s += float(member.rpartition(":")[2])

            features = np.array(
                [[amount, tx_count_60s, total_spend_60s]], dtype=np.float32
            )

            # --- ONNX inference ----------------------------------------------
            outputs = self.session.run(None, {self.input_name: features})
            label = int(np.asarray(outputs[0]).ravel()[0])
            latency_ms = (time.perf_counter() - started) * 1000.0
        except redis.exceptions.RedisError:
            raise  # leave the message pending; it is retried / claimed later
        except Exception:
            logger.exception("Failed to score message %s; acknowledging", msg_id)
            r.xack(STREAM, GROUP, msg_id)
            return

        is_anomaly = label == -1
        self.stats.record(latency_ms)

        second = int(time.time())
        tput_key = f"{THROUGHPUT_PREFIX}{second}"
        mp = r.pipeline(transaction=False)
        mp.lpush(LATENCY_KEY, f"{latency_ms:.4f}")
        mp.ltrim(LATENCY_KEY, 0, LATENCY_HISTORY - 1)
        mp.incr(TOTAL_KEY)
        mp.incr(tput_key)
        mp.expire(tput_key, 60)

        if bench_run:
            bench_lat = f"bench:latencies:{bench_run}"
            mp.rpush(bench_lat, f"{latency_ms:.4f}")
            mp.expire(bench_lat, 900)
            if is_anomaly:
                bench_flag = f"bench:flagged:{bench_run}"
                mp.incr(bench_flag)
                mp.expire(bench_flag, 900)

        if is_anomaly:
            created_at = datetime.now(timezone.utc)
            row = {
                "tx_id": tx_id,
                "user_id": user_id,
                "amount": amount,
                "tx_count_60s": tx_count_60s,
                "total_spend_60s": round(total_spend_60s, 2),
                "latency_ms": round(latency_ms, 4),
                "created_at": created_at,
            }
            score = float(np.asarray(outputs[1]).ravel()[0]) if len(outputs) > 1 else 0.0
            alert = dict(row, created_at=created_at.isoformat(), anomaly_score=round(score, 6))
            mp.incr(FLAGGED_KEY)
            mp.publish(ALERT_CHANNEL, json.dumps(alert))
            self.writer.submit(row)

        mp.xack(STREAM, GROUP, msg_id)
        mp.execute()

    # --------------------------------------------------------------- consumption
    def _handle(self, messages: List[Tuple[str, Optional[Dict[str, str]]]]) -> None:
        for msg_id, fields in messages:
            self.process(msg_id, fields)

    def _claim_stale(self) -> None:
        """Take over messages left pending by crashed/replaced consumers."""
        start_id = "0-0"
        for _ in range(10):
            result = self.redis.xautoclaim(
                STREAM, GROUP, CONSUMER_NAME,
                min_idle_time=CLAIM_IDLE_MS, start_id=start_id, count=READ_COUNT,
            )
            next_id, messages = result[0], result[1]
            if messages:
                logger.info("Claimed %d stale pending messages", len(messages))
                self._handle(messages)
            if next_id == "0-0":
                break
            start_id = next_id

    def run(self) -> None:
        self.writer.start()
        logger.info("Worker %s consuming %s/%s", CONSUMER_NAME, STREAM, GROUP)

        recovering = True  # first replay this consumer's own pending entries
        last_claim = time.monotonic()

        while not self.stop_event.is_set():
            try:
                if recovering:
                    resp = self.redis.xreadgroup(
                        GROUP, CONSUMER_NAME, {STREAM: "0"}, count=READ_COUNT
                    )
                else:
                    resp = self.redis.xreadgroup(
                        GROUP, CONSUMER_NAME, {STREAM: ">"},
                        count=READ_COUNT, block=READ_BLOCK_MS,
                    )
                messages = resp[0][1] if resp else []

                if recovering:
                    if not messages:
                        recovering = False
                        logger.info("Pending replay complete; switching to new messages")
                        continue
                    logger.info("Replaying %d pending messages", len(messages))

                self._handle(messages)

                if time.monotonic() - last_claim >= CLAIM_INTERVAL_S:
                    self._claim_stale()
                    last_claim = time.monotonic()

            except redis.exceptions.ResponseError as exc:
                if "NOGROUP" in str(exc):
                    logger.warning("Consumer group missing; recreating")
                    ensure_group(self.redis)
                    recovering = False
                else:
                    logger.exception("Redis error")
                    time.sleep(1.0)
            except redis.exceptions.RedisError as exc:
                logger.warning("Redis connection problem (%s); retrying", exc)
                time.sleep(1.0)

        logger.info("Shutting down; flushing PostgreSQL writer")
        self.writer.stop()
        self.writer.join(timeout=15)
        db.close_pool()


def main() -> None:
    worker = Worker()

    def _shutdown(signum: int, _frame: Any) -> None:
        logger.info("Received signal %s", signum)
        worker.stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    worker.run()


if __name__ == "__main__":
    main()
