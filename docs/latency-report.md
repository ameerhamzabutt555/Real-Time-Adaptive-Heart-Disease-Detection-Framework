# Latency Report

This report tracks API prediction latency for `/predict` endpoint.

## Procedure

1. Start API server:
   - `make run-api`
2. Run latency script:
   - `python scripts/run_latency_check.py --requests 200 --output experiments/tracking/latency_report.json`
3. Collect:
   - mean latency (ms)
   - p50 latency (ms)
   - p95 latency (ms)
   - max latency (ms)

## Notes

- This measurement is local, single-machine benchmark.
- For production claims, test under controlled load and representative hardware.

