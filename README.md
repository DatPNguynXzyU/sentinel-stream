# sentinel-stream

A Real-Time Transaction Anomaly & Fraud Detection Pipeline. Synthetic card
transactions flow through **Redis Streams**, are enriched with rolling 60-second
velocity features in **Redis Sorted Sets**, scored by an **IsolationForest
exported to ONNX** and run in **ONNX Runtime**, persisted to **PostgreSQL**, and
pushed live to a **dark-mode dashboard** over **Server-Sent Events**.

## Architecture

```
                         +---------------------+
                         |     simulator.py    |   normal traffic + toggleable
                         |  (synthetic tx gen) |   fraud burst (sim:burst_mode)
                         +----------+----------+
                                    | XADD
                                    v
 +-----------------------------------------------------------------------+
 |                               Redis 7                                 |
 |                                                                       |
 |  Stream  tx_stream  --(consumer group tx_group)-->                    |
 |  ZSET    user_window:{user_id}   rolling 60 s feature window          |
 |  LIST    metrics:latencies       last 1,000 latencies (LPUSH/LTRIM)   |
 |  STRING  metrics:total_tx        processed counter (INCR)             |
 |  PubSub  alerts_channel          live anomaly alerts                  |
 +-----------+-----------------------------------------------^-----------+
             | XREADGROUP                                    | PUBLISH
             v                                               |
   +---------------------------------------------------------+---+
   |                         worker.py                           |
   |  1. ZADD / ZREMRANGEBYSCORE / ZRANGE / EXPIRE (1 pipeline)  |
   |  2. features = [amount, tx_count_60s, total_spend_60s]      |
   |  3. onnxruntime IsolationForest  (-1 => anomaly)            |
   |  4. metrics + alert + XACK                                  |
   +-------------------------------+-----------------------------+
                                   | batched async inserts (thread)
                                   v
                      +--------------------------+
                      |    PostgreSQL 15         |
                      |  flagged_transactions    |
                      |  throughput_metrics      |
                      +------------+-------------+
                                   ^
                                   | SELECT / COUNT
 +-------------------------+  +----+---------------------------------+
 |  Browser dashboard      |<-|            main.py (FastAPI)         |
 |  Tailwind + vanilla JS  |  |  GET  /                  dashboard   |
 |  EventSource (SSE)      |  |  GET  /api/metrics       p50/95/99   |
 +-------------------------+  |  GET  /api/anomalies     last 25     |
                              |  POST /api/trigger-burst 10 s burst  |
                              |  GET  /api/stream        SSE alerts  |
                              +--------------------------------------+
```

## Quickstart

Requires Docker with Compose v2.

```bash
docker compose up --build
```

This single command:

1. Builds one backend image and **trains the IsolationForest, exporting
   `backend/model.onnx`** during the build (5,000 normal + 250 fraud rows).
2. Boots Redis 7, PostgreSQL 15 (tables are created automatically), the
   `worker`, the `simulator` and the `api`.
3. Streams synthetic transactions (~15 tx/s), scores them in real time and
   stores anomalies in PostgreSQL.

Open **http://localhost:8000** and click **Trigger Fraud Attack Burst** to flood
the stream with high-velocity fraudulent swipes for 10 seconds.

Stop with `Ctrl+C`; remove all data with `docker compose down -v`.

### Scale the workers

Consumers in the same Redis consumer group share the stream:

```bash
docker compose up --build --scale worker=3
```

## Services

| Service     | Role                                                                 |
|-------------|----------------------------------------------------------------------|
| `redis`     | Streams broker, sorted-set feature windows, metrics, Pub/Sub alerts  |
| `postgres`  | Cold storage: `flagged_transactions`, `throughput_metrics`           |
| `worker`    | Consumer group worker: features -> ONNX scoring -> alerts -> XACK    |
| `simulator` | Synthetic producer with fraud burst mode                             |
| `api`       | FastAPI REST + SSE + dashboard on port 8000                          |

## API

| Method | Path                  | Description                                                              |
|--------|-----------------------|--------------------------------------------------------------------------|
| GET    | `/`                   | Dashboard                                                                |
| GET    | `/api/metrics`        | `total_processed`, `flagged_count`, `p50/p95/p99_latency_ms`, `throughput_tps` (alias: `/metrics`) |
| GET    | `/api/anomalies`      | Latest 25 rows of `flagged_transactions`                                 |
| POST   | `/api/trigger-burst`  | Sets `sim:burst_mode` with a 10 s TTL                                    |
| GET    | `/api/stream`         | Server-Sent Events of anomaly alerts (alias: `/stream`)                  |
| GET    | `/api/health`         | Redis / PostgreSQL connectivity                                          |

```bash
curl -s http://localhost:8000/api/metrics
curl -s -X POST http://localhost:8000/api/trigger-burst
curl -N http://localhost:8000/api/stream
```

## Feature engineering & model

Per transaction `{tx_id, user_id, amount, timestamp}` the worker runs one Redis
pipeline on `user_window:{user_id}`: `ZADD` -> `ZREMRANGEBYSCORE -inf
(timestamp-60)` -> `ZRANGE 0 -1` -> `EXPIRE 120`, then derives `tx_count_60s` and
`total_spend_60s` from the returned members. The `float32` vector
`[amount, tx_count_60s, total_spend_60s]` is scored by the ONNX model; a
prediction of `-1` is an anomaly: it is queued for batched insertion into
PostgreSQL and published on `alerts_channel`. Every message is `XACK`ed.

Reported latency = feature engineering + ONNX scoring, measured with
`time.perf_counter()`.

## Benchmark

`benchmark.py` pushes 2,000 events into `tx_stream`, waits for the workers to
process them and prints exact throughput and p50/p95/p99 latency over **all**
events (workers record per-event latency for tagged benchmark events).

```bash
pip install redis
docker compose up --build -d
docker compose stop simulator        # optional: remove background traffic
python benchmark.py                  # default 2,000 events
python benchmark.py --events 5000    # custom size
docker compose start simulator
```

Redis is published on `localhost:6379` for this purpose. With `--scale worker=N`
the end-to-end throughput scales across consumers.

### Verified Hardware Benchmarks (2,000-Event Stress Test, Single Worker)

| Metric | Measured Result |
|---|---|
| End-to-End Throughput | **572.9 req/sec** (3.49s wall clock) |
| p50 Latency | **1.095 ms** |
| p95 Latency | **2.954 ms** |
| p99 Latency | **3.929 ms** |
| Mean / Max Latency | **1.372 ms / 6.118 ms** |
| Fraud Recall / False-Positive Rate | **100% Recall / 0.26% FPR** (100% label parity between scikit-learn and ONNX Runtime) |

## Engineering Notes

- **ONNX Tree Attribute Serialization (`backend/train_model.py`):** `skl2onnx 1.17` emits certain `IsolationForest` `TreeEnsemble` attributes (e.g., `nodes_missing_value_tracks_true`) as Python/NumPy booleans, which `onnx 1.16.2` rejects for `INTS` protobuf attributes (`TypeError: Field onnx.AttributeProto.ints: Expected an int, got a boolean`). `train_model.py` patches `onnx.helper.make_attribute` to cast boolean sequences to `int` prior to serialization, preserving 100% label parity with scikit-learn.

## Configuration

| Variable                          | Default                  | Used by    |
|-----------------------------------|--------------------------|------------|
| `REDIS_URL`                       | `redis://redis:6379/0`   | all        |
| `DATABASE_URL`                    | `postgresql://sentinel:sentinel@postgres:5432/sentinel` | api, worker |
| `NORMAL_RATE_TPS`                 | `15`                     | simulator  |
| `BURST_RATE_TPS`                  | `60`                     | simulator  |
| `RARE_OUTLIER_PROBABILITY`        | `0.002`                  | simulator  |
| `DB_BATCH_SIZE` / `DB_FLUSH_INTERVAL_S` | `100` / `0.5`      | worker     |

## Repository layout

```
docker-compose.yml
benchmark.py
README.md
backend/
  Dockerfile
  requirements.txt
  db.py            PostgreSQL pool + schema init
  train_model.py   IsolationForest -> model.onnx
  worker.py        Streams consumer, features, ONNX scoring, logging
  simulator.py     Synthetic producer with burst mode
  main.py          FastAPI app (REST, SSE, metrics, dashboard host)
  static/index.html
```

## Notes

- Tailwind is loaded from its CDN, so the browser needs internet access to
  style the dashboard; the backend itself has no external runtime dependency.
- Training runs at image build time. To retrain locally:
  `cd backend && python train_model.py`.
