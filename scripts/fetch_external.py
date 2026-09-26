"""Загрузка внешних данных: погода (Open-Meteo) и геометрия маршрутов (OpenStreetMap).

Источники:
- https://open-meteo.com/en/docs/historical-weather-api (архив ERA5, без ключа)
- https://api.open-meteo.com/v1/forecast (прогноз на ближайшие дни)
- https://maps.mail.ru/osm/tools/overpass/ и https://overpass.private.coffee
  (зеркала Overpass API, OpenStreetMap relation route=tram)
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("data/external")
UA = {"User-Agent": "tram-forecast/0.1"}
ROUTES = ["1", "5", "7", "11", "12", "17", "25", "26", "28", "50"]
OVERPASS = "https://overpass.private.coffee/api/interpreter"
OVERPASS_FAST = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
HOURLY = "temperature_2m,precipitation,snowfall,snow_depth,wind_speed_10m,cloud_cover"


def get(url: str, data: bytes | None = None, attempts: int = 3) -> bytes:
    """GET/POST with retries: public Overpass mirrors often answer 429/504."""
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            print("retry", i + 1, e, flush=True)
            time.sleep(15 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def weather() -> None:
    q = urllib.parse.urlencode({
        "latitude": 55.75, "longitude": 37.62, "timezone": "Europe/Moscow",
        "start_date": "2025-01-01", "end_date": "2025-12-31", "hourly": HOURLY,
    })
    raw = json.loads(get(f"https://archive-api.open-meteo.com/v1/archive?{q}"))["hourly"]
    cols = list(raw)
    lines = [",".join(cols)] + [",".join(str(raw[c][i]) for c in cols) for i in range(len(raw["time"]))]
    (OUT / "weather_moscow_2025.csv").write_text("\n".join(lines) + "\n")


def overpass(url: str, query: str) -> list[dict]:
    body = urllib.parse.urlencode({"data": "[out:json][timeout:180];" + query}).encode()
    return json.loads(get(url, body))["elements"]


# id relation route=tram для маршрутов хакатона (по одному на направление),
# найдены запросом rel["route"="tram"]["ref"~"^(1|5|7|...)$"] в границах Москвы.
OSM_RELATIONS = [
    540033, 540139, 543080, 556900, 918052, 920053, 1283310, 1538169, 1538170, 1689026,
    1689064, 3184022, 3184023, 3186264, 3186265, 3299879, 7556406, 14258874, 14258875, 15840810,
]


def osm_routes() -> None:
    """Relation geometry one by one (public mirrors time out on bulk queries)."""
    path = OUT / "osm_tram_routes.json"
    done = json.loads(path.read_text())["elements"] if path.exists() else []
    have = {e["id"] for e in done}
    for rid in OSM_RELATIONS:
        if rid in have:
            continue
        for url in (OVERPASS_FAST, OVERPASS):
            try:
                done += overpass(url, f"rel({rid});out geom;")
            except RuntimeError as e:
                print(e, flush=True)
                continue
            path.write_text(json.dumps({"elements": done}, ensure_ascii=False))
            print("ok", rid, flush=True)
            break
    # Повторный запуск докачивает только недостающие relation.


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or ["weather", "osm_routes"]:
        globals()[name]()
        print("ok", name)
