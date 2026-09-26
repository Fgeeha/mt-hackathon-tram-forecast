"""Пути проекта. Переопределяются переменными окружения для Docker."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("TRAM_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / "data" / "raw"
EXTERNAL = ROOT / "data" / "external"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = Path(os.environ.get("TRAM_ARTIFACTS", ROOT / "artifacts"))

ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
FORECAST_START = "2025-11-01"
FORECAST_END = "2025-12-31"
