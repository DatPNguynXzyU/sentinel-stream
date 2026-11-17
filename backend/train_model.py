"""Train an IsolationForest on synthetic transaction data and export it to ONNX.

Features (float32, in this order): [amount, tx_count_60s, total_spend_60s]

Usage:  python train_model.py          ->  writes model.onnx next to this file
"""

import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import IsolationForest

def _patch_onnx_bool_attributes() -> None:
    """skl2onnx's IsolationForest converter emits some TreeEnsemble attributes as
    Python/NumPy bools, which recent `onnx` releases reject for INTS attributes
    ("Expected an int, got a boolean"). Cast bool sequences to ints before the
    attribute protobuf is built; every other attribute is passed through untouched."""
    import onnx.helper as onnx_helper

    original = onnx_helper.make_attribute

    def make_attribute(key, value, *args, **kwargs):
        if isinstance(value, np.ndarray) and value.dtype == np.bool_:
            value = value.astype(np.int64).tolist()
        elif isinstance(value, (list, tuple)) and value and all(
            isinstance(v, (bool, np.bool_)) for v in value
        ):
            value = [int(v) for v in value]
        return original(key, value, *args, **kwargs)

    onnx_helper.make_attribute = make_attribute


_patch_onnx_bool_attributes()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sentinel.train")

SEED = 42
N_NORMAL = 5000
N_FRAUD = 250
OUTPUT_PATH = Path(
    os.getenv("MODEL_OUTPUT", str(Path(__file__).resolve().parent / "model.onnx"))
)


def generate_normal(rng: np.random.Generator, n: int) -> np.ndarray:
    """amount $5-$200, tx_count_60s 1-3, total_spend_60s $5-$400."""
    amount = rng.uniform(5.0, 200.0, n)
    count = rng.integers(1, 4, n)  # 1..3 inclusive
    # Total spend = this amount + the other (count-1) purchases in the window.
    others = rng.uniform(5.0, 200.0, (n, 2))
    mask = np.arange(2)[None, :] < (count - 1)[:, None]
    total = np.clip(amount + (others * mask).sum(axis=1), 5.0, 400.0)
    return np.column_stack([amount, count, total]).astype(np.float32)


def generate_fraud(rng: np.random.Generator, n: int) -> np.ndarray:
    """amount $800-$5000, tx_count_60s 5-15, total_spend_60s $2000-$15000."""
    amount = rng.uniform(800.0, 5000.0, n)
    count = rng.integers(5, 16, n)  # 5..15 inclusive
    total = np.maximum(rng.uniform(2000.0, 15000.0, n), amount)
    return np.column_stack([amount, count, total]).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(SEED)
    normal = generate_normal(rng, N_NORMAL)
    fraud = generate_fraud(rng, N_FRAUD)
    X = np.vstack([normal, fraud]).astype(np.float32)
    logger.info("Training data: %d normal + %d fraud rows", len(normal), len(fraud))

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=SEED)
    model.fit(X)

    onnx_model = convert_sklearn(
        model,
        initial_types=[("float_input", FloatTensorType([None, 3]))],
        target_opset={"": 15, "ai.onnx.ml": 3},
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(onnx_model.SerializeToString())
    logger.info("Saved ONNX model to %s (%d bytes)", OUTPUT_PATH, OUTPUT_PATH.stat().st_size)

    # Validate the exported artefact with ONNX Runtime (the same engine the worker uses).
    session = ort.InferenceSession(str(OUTPUT_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    labels = session.run(None, {input_name: X})[0].ravel()
    normal_labels, fraud_labels = labels[:N_NORMAL], labels[N_NORMAL:]
    fraud_recall = float(np.mean(fraud_labels == -1))
    normal_fp_rate = float(np.mean(normal_labels == -1))
    logger.info(
        "Validation on training data: fraud recall=%.1f%%, normal false-positive rate=%.2f%%",
        fraud_recall * 100,
        normal_fp_rate * 100,
    )

    # Sanity-check that the exported model agrees with scikit-learn.
    sk_labels = model.predict(X)
    agreement = float(np.mean(sk_labels == labels))
    logger.info("ONNX vs scikit-learn label agreement: %.2f%%", agreement * 100)

    probes = np.array(
        [[45.0, 1, 45.0], [120.0, 2, 210.0], [2500.0, 9, 8000.0]], dtype=np.float32
    )
    logger.info(
        "Probe predictions [normal, normal, fraud] -> %s",
        session.run(None, {input_name: probes})[0].ravel().tolist(),
    )

    if fraud_recall < 0.80:
        raise SystemExit(
            f"Model sanity check failed: fraud recall {fraud_recall:.2%} < 80%"
        )


if __name__ == "__main__":
    main()
