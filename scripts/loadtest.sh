#!/usr/bin/env bash
# Нагрузочный тест через hey (в Docker): по 20 с на эндпоинт, 50 соединений.
# Использование: scripts/loadtest.sh [URL] [DURATION] [CONCURRENCY]
set -euo pipefail
URL=${1:-http://localhost:8000}
DUR=${2:-20s}
CONC=${3:-50}
PATHS=(
  "/api/forecast?horizon=day&route=17&agg=hour"
  "/api/forecast?horizon=month&route=17&route=11&agg=day&k_weather=0.95"
  "/api/forecast?horizon=year&agg=month"
  "/api/forecast?horizon=day&route=1&stop_id=2594&hour_from=7&hour_to=10"
  "/api/export?format=csv&horizon=month&route=12&agg=day"
)
for p in "${PATHS[@]}"; do
  echo "=== $p"
  docker run --rm --network host williamyeh/hey -z "$DUR" -c "$CONC" "$URL$p" \
    | grep -E "Requests/sec|Average|50%|95%|99%|\[2|\[4|\[5"
done
