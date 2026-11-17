#!/usr/bin/env python3
"""sentinel-stream benchmark: push N events into Redis Streams and report the
end-to-end throughput plus exact per-event inference latency percentiles.

The worker measures, with time.perf_counter(), the time to build features
(Redis ZSET pipeline) + run ONNX inference for every event. For events tagged
with a `bench_run` field it additionally appends that latency to the Redis list
`bench:latencies:{run}`, which this script reads back so the percentiles cover
*every* benchmark event (not just the rolling 1,000-item window).

Requirements: `pip install redis` and a running stack (`docker compose up --build`).
For the cleanest numbers, pause the background traffic first:
    docker compose stop simulator

Usage:
    python benchmark.py [--events 2000] [--redis-url redis://localhost:6379/0]
"""

import argparse
import math
import random
import sys
import time
import uuid
from typing import Dict, List

try:
    import redis
except ImportError:  # pragma: no cover
    sys.exit("The 'redis' package is required: pip install redis")

STREAM = "tx_stream"
GROUP = "tx_group"
FRAUD_FRACTION = 0.05
FRAUD_USERS = 10
PUSH_BATCH = 200


def percentile(sorted_values: List[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_values:
        return float("nan")
    k = (len(sorted_values) - 1) * pct / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_values[int(k)]
    return sorted_values[lo] * (hi - k) + sorted_values[hi] * (k - lo)


def build_event(run_id: str, index: int, fraud: bool, fraud_user: int) -> Dict[str, str]:
    if fraud:
        user_id = f"bench_{run_id}_fraud_{fraud_user}"
        amount = random.uniform(800.0, 5000.0)
    else:
        user_id = f"bench_{run_id}_user_{index}"
        amount = random.uniform(5.0, 200.0)
    return {
        "tx_id": f"bench-{run_id}-{index}",
        "user_id": user_id,
        "amount": f"{amount:.2f}",
        "timestamp": f"{time.time():.6f}",
        "bench_run": run_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="sentinel-stream throughput/latency benchmark")
    parser.add_argument("--events", type=int, default=2000, help="number of events (default 2000)")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="seconds to wait for the workers to drain the events")
    args = parser.parse_args()

    client = redis.Redis.from_url(args.redis_url, decode_responses=True, socket_connect_timeout=5)
    try:
        client.ping()
    except redis.exceptions.RedisError as exc:
        print(f"Cannot reach Redis at {args.redis_url}: {exc}\n"
              "Start the stack first with: docker compose up --build", file=sys.stderr)
        return 1

    try:
        groups = client.xinfo_groups(STREAM)
        if not any(g["name"] == GROUP for g in groups):
            raise redis.exceptions.ResponseError("group missing")
    except redis.exceptions.ResponseError:
        print("Consumer group not found - is the worker container running?", file=sys.stderr)
        return 1

    n = args.events
    run_id = uuid.uuid4().hex[:10]
    lat_key = f"bench:latencies:{run_id}"
    flag_key = f"bench:flagged:{run_id}"

    # Pre-build every event so generation cost is excluded from the measurement
    # (timestamps are refreshed right before pushing so ZSET windows are correct).
    fraud_total = int(n * FRAUD_FRACTION)
    fraud_slots = set(random.sample(range(n), fraud_total)) if fraud_total else set()
    plan = [(i, i in fraud_slots, i % FRAUD_USERS) for i in range(n)]

    backlog = client.xinfo_groups(STREAM)
    pending_before = sum(g.get("lag") or 0 for g in backlog if g["name"] == GROUP)

    print(f"sentinel-stream benchmark | run={run_id} events={n} "
          f"(~{fraud_total} fraud-profile) backlog_before={pending_before}")

    push_started = time.perf_counter()
    for offset in range(0, n, PUSH_BATCH):
        pipe = client.pipeline(transaction=False)
        for index, fraud, fraud_user in plan[offset:offset + PUSH_BATCH]:
            pipe.xadd(STREAM, build_event(run_id, index, fraud, fraud_user))
        pipe.execute()
    push_elapsed = time.perf_counter() - push_started

    deadline = push_started + args.timeout
    done = 0
    while True:
        done = client.llen(lat_key)
        if done >= n or time.perf_counter() >= deadline:
            break
        time.sleep(0.005)
    total_elapsed = time.perf_counter() - push_started

    latencies = sorted(float(v) for v in client.lrange(lat_key, 0, -1))
    flagged = int(client.get(flag_key) or 0)
    client.delete(lat_key, flag_key)

    if not latencies:
        print("No events were processed - check that the worker is running "
              "(docker compose logs worker).", file=sys.stderr)
        return 1

    complete = done >= n
    print()
    print("=" * 60)
    print(" SENTINEL-STREAM BENCHMARK RESULTS")
    print("=" * 60)
    print(f" Events pushed              : {n}")
    print(f" Events processed           : {len(latencies)}{'' if complete else '  (TIMED OUT)'}")
    print(f" Anomalies flagged          : {flagged}")
    print(f" Producer push rate         : {n / push_elapsed:,.0f} events/sec")
    print(f" End-to-end throughput      : {len(latencies) / total_elapsed:,.1f} req/sec "
          f"({total_elapsed:.3f}s wall clock)")
    print(" Inference latency (feature engineering + ONNX scoring, per event):")
    print(f"   mean : {sum(latencies) / len(latencies):.3f} ms")
    print(f"   p50  : {percentile(latencies, 50):.3f} ms")
    print(f"   p95  : {percentile(latencies, 95):.3f} ms")
    print(f"   p99  : {percentile(latencies, 99):.3f} ms")
    print(f"   max  : {latencies[-1]:.3f} ms")
    print("=" * 60)
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
