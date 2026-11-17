"""Synthetic transaction producer pushing into the `tx_stream` Redis Stream.

Normal mode : ~NORMAL_RATE_TPS legitimate-looking card swipes per second across
              thousands of users, plus very rare isolated high-value outliers.
Burst mode  : while the Redis key `sim:burst_mode` exists (set for 10 s by
              POST /api/trigger-burst) the simulator floods the stream at
              BURST_RATE_TPS with high-amount swipes from a handful of users,
              producing the high-velocity pattern the model flags.
"""

import logging
import os
import random
import signal
import threading
import time
import uuid

import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("sentinel.simulator")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STREAM = "tx_stream"
BURST_KEY = "sim:burst_mode"

NORMAL_RATE_TPS = float(os.getenv("NORMAL_RATE_TPS", "15"))
BURST_RATE_TPS = float(os.getenv("BURST_RATE_TPS", "60"))
NORMAL_USER_POOL = int(os.getenv("NORMAL_USER_POOL", "2000"))
BURST_USER_POOL = int(os.getenv("BURST_USER_POOL", "8"))
RARE_OUTLIER_PROBABILITY = float(os.getenv("RARE_OUTLIER_PROBABILITY", "0.002"))
STREAM_MAXLEN = int(os.getenv("STREAM_MAXLEN", "100000"))
BURST_POLL_INTERVAL_S = 0.1

stop_event = threading.Event()


def connect() -> redis.Redis:
    while True:
        try:
            client = redis.Redis.from_url(
                REDIS_URL, decode_responses=True, socket_connect_timeout=5,
                health_check_interval=30,
            )
            client.ping()
            logger.info("Connected to Redis at %s", REDIS_URL)
            return client
        except redis.exceptions.RedisError as exc:
            logger.warning("Redis not ready (%s); retrying...", exc)
            time.sleep(1.5)


def make_transaction(burst: bool) -> dict:
    if burst:
        user_id = f"fraud_user_{random.randint(1, BURST_USER_POOL)}"
        amount = random.uniform(800.0, 5000.0)
    elif random.random() < RARE_OUTLIER_PROBABILITY:
        user_id = f"user_{random.randint(1, NORMAL_USER_POOL)}"
        amount = random.uniform(800.0, 5000.0)
    else:
        user_id = f"user_{random.randint(1, NORMAL_USER_POOL)}"
        amount = random.uniform(5.0, 200.0)
    return {
        "tx_id": uuid.uuid4().hex,
        "user_id": user_id,
        "amount": f"{amount:.2f}",
        "timestamp": f"{time.time():.6f}",
    }


def main() -> None:
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    client = connect()
    burst = False
    last_burst_check = 0.0
    next_send = time.monotonic()
    sent_total = 0
    sent_window = 0
    last_report = time.monotonic()

    logger.info(
        "Simulator started: normal=%.1f tx/s, burst=%.1f tx/s", NORMAL_RATE_TPS, BURST_RATE_TPS
    )

    while not stop_event.is_set():
        now = time.monotonic()

        try:
            if now - last_burst_check >= BURST_POLL_INTERVAL_S:
                last_burst_check = now
                active = bool(client.exists(BURST_KEY))
                if active != burst:
                    burst = active
                    next_send = now
                    logger.info("Fraud burst mode %s", "ON" if burst else "OFF")

            tx = make_transaction(burst)
            client.xadd(STREAM, tx, maxlen=STREAM_MAXLEN, approximate=True)
            sent_total += 1
            sent_window += 1
        except redis.exceptions.RedisError as exc:
            logger.warning("Redis error (%s); retrying", exc)
            time.sleep(1.0)
            next_send = time.monotonic()
            continue

        rate = BURST_RATE_TPS if burst else NORMAL_RATE_TPS
        next_send += 1.0 / rate
        delay = next_send - time.monotonic()
        if delay > 0:
            stop_event.wait(delay)
        elif delay < -1.0:
            next_send = time.monotonic()  # fell behind; do not try to catch up

        if time.monotonic() - last_report >= 10.0:
            elapsed = time.monotonic() - last_report
            logger.info(
                "sent=%d (%.1f tx/s) mode=%s", sent_total, sent_window / elapsed,
                "BURST" if burst else "normal",
            )
            sent_window = 0
            last_report = time.monotonic()

    logger.info("Simulator stopped after %d transactions", sent_total)


if __name__ == "__main__":
    main()
