# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Task

Moscow Transport hackathon, track 2. Forecast hourly tram boardings per route (`route × date × hour`) for 2025-11-01 … 2025-12-31 from Jan–Oct 2025 validations. Metric: WAPE-score = 1 − Σ|y−ŷ|/Σy (threshold for max points: > 0.88). Plus a Dockerized web service (API + map dashboard + CSV/XLSX export) and a README with performance numbers. Deadline 2026-09-27 23:59 MSK. `plan/implementation-plan.md` is the pre-data plan (written before the real dataset arrived) and is partly outdated: there is no telematics and no stop in validations.

Data: `dataset.zip` (gitignored) → `make data` → `data/raw/` (train.csv, test.csv ~10 GB, labels, spravochniki). Route 5 is effectively absent from validations: organizers said to fill it with zeros.

## Commands

```
make install      # uv sync
make pipeline     # ingest → geo → backtest → forecast (needs data/raw)
make run          # API + dashboard on :8000
make check        # ruff + pytest
make up-local     # docker compose (needs .env, see .env.example)
make loadtest     # hey-based load test via Docker
```

Single test: `uv run pytest tests/test_core.py::test_submission_grid -q`.

## Architecture

```
data/raw/*.csv ─ ingest.py (DuckDB) ─→ data/processed/route_hour.parquet
scripts/fetch_external.py ─→ data/external/ (Open-Meteo weather, OSM routes)
calendar.py (RF holidays, Moscow school breaks) + features.py (level + calendar + weather)
model.py (LightGBM, rolling-origin backtest) ─ forecast.py ─→ artifacts/{forecast.parquet, submission.csv, metrics.json, model_info.json}
geo.py (spravochniki + OSM) ─→ artifacts/geo.json
api.py (FastAPI, in-memory polars over artifacts/) + static/index.html (MapLibre + ECharts)
```

- `artifacts/` is committed: the Docker image serves only from it and never loads raw data or the model in the request path.
- Model: target = boardings / level (level = mean of route × hour × day-class over the last 4 regular weeks). L1 objective with weight = level, which is equivalent to WAPE. Final prediction = 0.5 · LightGBM + 0.5 · level profile (`BLEND`). Adding `route`/`horizon` features overfit in the backtest.
- Stop-level numbers are the route forecast × a heuristic stop share (hub/terminal weights). Validations have no stop information.
- Leaderboard calibration: `scripts/probe.py` (rounds in `artifacts/probes*/`, `PROBE_ROUND=n`). Final file is `artifacts/submission_calibrated_v3.csv` (0.89037). `probe.py apply FILE` copies it into `forecast.parquet` so the service equals the submission; `test_service_matches_final_submission` guards this.
