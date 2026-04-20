#!/usr/bin/env python3
"""Measure local API prediction latency and export summary JSON."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local latency check for /predict endpoint.")
    parser.add_argument("--requests", type=int, default=200, help="Number of requests to issue.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/tracking/latency_report.json"),
        help="Path for latency summary JSON output.",
    )
    return parser.parse_args()


def _sample_payload() -> dict[str, float | int]:
    return {
        "age": 53,
        "sex": 1,
        "chest_pain_type": 2,
        "resting_bp": 132.0,
        "cholesterol": 246.0,
        "fasting_blood_sugar": 0,
        "restecg": 1,
        "max_heart_rate": 151.0,
        "exercise_angina": 1,
        "oldpeak": 1.3,
        "slope": 1,
        "ca": 0,
        "thal": 2,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = pos - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    args = parse_args()
    client = TestClient(app)
    payload = _sample_payload()
    timings_ms: list[float] = []

    for _ in range(max(1, args.requests)):
        start = time.perf_counter()
        response = client.post("/predict", json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if response.status_code != 200:
            raise RuntimeError(f"/predict failed with status {response.status_code}: {response.text}")
        timings_ms.append(elapsed_ms)

    summary = {
        "requests": len(timings_ms),
        "mean_latency_ms": float(statistics.fmean(timings_ms)),
        "p50_latency_ms": float(_percentile(timings_ms, 0.50)),
        "p95_latency_ms": float(_percentile(timings_ms, 0.95)),
        "max_latency_ms": float(max(timings_ms)),
        "min_latency_ms": float(min(timings_ms)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved latency report to {args.output}")
    print(summary)


if __name__ == "__main__":
    main()
