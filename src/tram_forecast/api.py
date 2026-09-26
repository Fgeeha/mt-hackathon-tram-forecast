"""REST API и раздача дашборда. Прогнозы посчитаны заранее (artifacts/),
в запросе только фильтрация и агрегация в памяти — модель не загружается."""

import io
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Literal

import orjson
import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from tram_forecast.calendar import calendar_frame
from tram_forecast.settings import ARTIFACTS, ROUTES

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"

FC = pl.read_parquet(ARTIFACTS / "forecast.parquet")
HIST = pl.read_parquet(ARTIFACTS / "history.parquet")
GEO_BYTES = (ARTIFACTS / "geo.json").read_bytes()
GEO = json.loads(GEO_BYTES)
METRICS = json.loads((ARTIFACTS / "metrics.json").read_text())
MODEL_INFO = json.loads((ARTIFACTS / "model_info.json").read_text())
FC_START, FC_END = FC["date"].min(), FC["date"].max()
H_START, H_END = HIST["date"].min(), HIST["date"].max()
HORIZON_DAYS = {"day": 1, "month": 30, "year": 365}

# Типичное число трамваев на линии (медиана октября) → посадок на один вагон.
_cal = calendar_frame(H_START, FC_END).select("date", "day_class")
_veh = (
    HIST.filter(pl.col("date") >= date(2025, 10, 1)).join(_cal, on="date")
    .group_by("route", "hour", "day_class").agg(pl.col("vehicles").median().alias("vehicles"))
)
FC = FC.join(_cal, on="date").join(_veh, on=["route", "hour", "day_class"], how="left").drop("day_class")

# Ключи агрегации считаем один раз при старте, а не в каждом запросе.
def _with_periods(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("date").dt.truncate("1w").alias("week"),
        pl.col("date").dt.strftime("%Y-%m").cast(pl.Categorical).alias("month"),
    )


FC, HIST = _with_periods(FC), _with_periods(HIST)

Agg = Literal["hour", "day", "week", "month", "hour_of_day", "route", "total"]
Coef = Annotated[float, Query(ge=0.1, le=3.0)]

app = FastAPI(
    title="Прогноз пассажиропотока трамваев Москвы",
    description="Прогноз посадок по маршрутам, остановкам и интервалам времени на горизонты день/месяц/год.",
    version="1.0.0",
)


class ORJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


@app.exception_handler(RequestValidationError)
async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
    errs = "; ".join(f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
    return JSONResponse({"error": f"Некорректные параметры запроса — {errs}"}, status_code=422)


@app.exception_handler(HTTPException)
async def _http(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled: %s", exc)
    return JSONResponse({"error": "Внутренняя ошибка сервиса, попробуйте позже"}, status_code=500)


def _stop_share(route: list[int] | None, stop_id: str | None) -> tuple[list[int] | None, float]:
    if not stop_id:
        return route, 1.0
    for r, g in GEO.items():
        for s in g["stops"]:
            if s["stop_id"] == stop_id and (not route or int(r) in route):
                return [int(r)], s["share"]
    raise HTTPException(404, f"Остановка {stop_id} не найдена на выбранных маршрутах")


def _window(start: date, horizon: str, date_from: date | None, date_to: date | None,
            lo: date, hi: date) -> tuple[date, date]:
    a = date_from or start
    b = date_to or a + timedelta(days=HORIZON_DAYS[horizon] - 1)
    if a > b:
        raise HTTPException(400, "Дата начала позже даты окончания")
    if a < lo or b > hi:
        raise HTTPException(400, f"Период вне диапазона данных: доступно {lo} … {hi}")
    return a, b


def _aggregate(df: pl.DataFrame, agg: str, cols: list[str]) -> pl.DataFrame:
    key = {
        "hour": [pl.col("route"), pl.col("date"), pl.col("hour")],
        "day": [pl.col("route"), pl.col("date")],
        "week": [pl.col("route"), pl.col("week")],
        "month": [pl.col("route"), pl.col("month")],
        "hour_of_day": [pl.col("hour")],
        "route": [pl.col("route")],
        "total": [],
    }[agg]
    sums = [pl.col(c).sum().round(1) for c in cols]
    if not key:
        return df.select(sums)
    return df.group_by(key).agg(sums).sort([k.meta.output_name() for k in key])


def _forecast(horizon, start, date_from, date_to, route, stop_id, hour_from, hour_to, agg,
              k_weather, k_event, k_season) -> tuple[pl.DataFrame, dict]:
    if hour_from > hour_to:
        raise HTTPException(400, "hour_from должен быть не больше hour_to")
    route, share = _stop_share(route, stop_id)
    if route and (bad := set(route) - set(ROUTES)):
        raise HTTPException(404, f"Неизвестные маршруты: {sorted(bad)}; доступны {ROUTES}")
    a, b = _window(start, horizon, date_from, date_to, FC_START, FC_END)
    k = share * k_weather * k_event * k_season
    df = FC.filter(pl.col("date").is_between(a, b), pl.col("hour").is_between(hour_from, hour_to))
    if route:
        df = df.filter(pl.col("route").is_in(route))
    df = df.with_columns(
        (pl.col("prediction") * k).alias("prediction"),
        (pl.col("baseline") * share).alias("baseline"),
        (pl.col("prediction") * k / pl.col("vehicles").clip(1)).alias("per_vehicle"),
    )
    out = _aggregate(df, agg, ["prediction", "baseline"])
    if agg == "hour":
        out = out.join(df.select("route", "date", "hour", pl.col("per_vehicle").round(1)), on=["route", "date", "hour"])
    meta = {"date_from": a, "date_to": b, "horizon": horizon, "routes": route or ROUTES,
            "stop_id": stop_id, "stop_share": share, "coefficient": round(k / share, 4), "agg": agg}
    return out, meta


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "forecast_rows": FC.height}


@app.get("/api/meta", response_class=ORJSONResponse)
def meta() -> dict:
    return {
        "routes": [{"route": r, "name": GEO.get(str(r), {}).get("name", ""), "has_geo": str(r) in GEO} for r in ROUTES],
        "forecast_range": [FC_START, FC_END], "history_range": [H_START, H_END],
        "horizons": HORIZON_DAYS, "season_index": MODEL_INFO["season_index"],
    }


@app.get("/api/geo")
def geo() -> Response:
    return Response(GEO_BYTES, media_type="application/json")


@app.get("/api/metrics", response_class=ORJSONResponse)
def metrics() -> dict:
    return {**METRICS, "feature_importance_gain": MODEL_INFO["feature_importance_gain"]}


def _params(
    horizon: Literal["day", "month", "year"] = "day",
    start: date = date(2025, 11, 1),
    date_from: date | None = None,
    date_to: date | None = None,
    route: Annotated[list[int] | None, Query()] = None,
    stop_id: str | None = None,
    hour_from: Annotated[int, Query(ge=0, le=23)] = 0,
    hour_to: Annotated[int, Query(ge=0, le=23)] = 23,
    agg: Agg = "hour",
    k_weather: Coef = 1.0, k_event: Coef = 1.0, k_season: Coef = 1.0,
) -> tuple:
    return (horizon, start, date_from, date_to, route, stop_id, hour_from, hour_to, agg,
            k_weather, k_event, k_season)


@app.get("/api/forecast", response_class=ORJSONResponse)
def forecast(q: tuple = Depends(_params)) -> dict:
    """Прогноз посадок. k_* — корректирующие коэффициенты (погода/событие/сезон)."""
    out, m = _forecast(*q)
    return {"meta": m, "rows": out.to_dicts()}


@app.get("/api/export")
def export(
    fmt: Literal["csv", "xlsx"] = Query("csv", alias="format"),
    q: tuple = Depends(_params),
) -> StreamingResponse:
    """Выгрузка прогноза в CSV или XLSX с теми же фильтрами, что /api/forecast."""
    out, m = _forecast(*q)
    buf = io.BytesIO()
    name = f"forecast_{m['date_from']}_{m['date_to']}_{m['agg']}"
    if fmt == "csv":
        out.write_csv(buf, separator=";")
        media = "text/csv; charset=utf-8"
    else:
        out.write_excel(buf, worksheet="forecast", autofit=True)
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    buf.seek(0)
    return StreamingResponse(buf, media_type=media,
                             headers={"Content-Disposition": f'attachment; filename="{name}.{fmt}"'})


@app.get("/api/history", response_class=ORJSONResponse)
def history(
    date_from: date = date(2025, 10, 1),
    date_to: date = date(2025, 10, 31),
    route: Annotated[list[int] | None, Query()] = None,
    agg: Agg = "day",
) -> dict:
    """Фактические посадки (валидации) за январь–октябрь 2025."""
    a, b = _window(date_from, "day", date_from, date_to, H_START, H_END)
    df = HIST.filter(pl.col("date").is_between(a, b))
    if route:
        df = df.filter(pl.col("route").is_in(route))
    return {"rows": _aggregate(df, agg, ["boardings", "unique_cards"]).to_dicts()}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
