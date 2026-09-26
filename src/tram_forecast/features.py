"""Признаки для прямой многогоризонтной модели.

Для точки прогноза `origin` (первый прогнозируемый день) используется только
история до origin - 1: «уровень» маршрут × час × тип дня за последние 4 недели.
К нему добавляются календарь, погода и горизонт (дней от origin).
"""

from datetime import date, timedelta

import polars as pl

from tram_forecast.calendar import calendar_frame
from tram_forecast.settings import EXTERNAL, PROCESSED

KEYS = ["route", "date", "hour"]
WEATHER_COLS = ["temp", "precip", "snowfall", "snow_depth", "wind"]
# route и horizon в бэктесте ухудшают качество (переобучение на сбои отдельных
# маршрутов), поэтому модель глобальная и опирается на уровень маршрута.
FEATURES = [
    "hour", "dow", "day_class", "is_holiday", "is_working_weekend", "pre_off",
    "post_off", "school_break", "off_run", "lvl", "lvl_share", "lvl_dow",
    *WEATHER_COLS, "temp_day", "precip_day", "snowfall_day",
]


def load_history(path=PROCESSED / "route_hour.parquet", until: date = date(2025, 10, 31)) -> pl.DataFrame:
    """Dense route × date × hour grid of boardings (missing hours = 0)."""
    df = pl.read_parquet(path).filter(pl.col("date") <= until, pl.col("route") != 5)
    grid = (
        df.select("route").unique()
        .join(pl.DataFrame({"date": pl.date_range(df["date"].min(), until, eager=True)}), how="cross")
        .join(pl.DataFrame({"hour": list(range(24))}, schema={"hour": pl.Int64}), how="cross")
    )
    return (
        grid.join(df.select(*KEYS, "boardings"), on=KEYS, how="left")
        .with_columns(pl.col("boardings").fill_null(0).cast(pl.Float64), pl.col("route").cast(pl.Int32))
        .sort(KEYS)
    )


def load_weather() -> pl.DataFrame:
    """Hourly Moscow weather (Open-Meteo archive) with daily aggregates."""
    w = pl.read_csv(EXTERNAL / "weather_moscow_2025.csv", try_parse_dates=True).rename({
        "temperature_2m": "temp", "precipitation": "precip", "wind_speed_10m": "wind",
    })
    w = w.with_columns(pl.col("time").dt.date().alias("date"), pl.col("time").dt.hour().cast(pl.Int64).alias("hour"))
    day = w.group_by("date").agg(
        pl.col("temp").mean().alias("temp_day"),
        pl.col("precip").sum().alias("precip_day"),
        pl.col("snowfall").sum().alias("snowfall_day"),
    )
    return w.select("date", "hour", *WEATHER_COLS).join(day, on="date")


def level(hist: pl.DataFrame, origin: date, days: int = 28) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Recent level per route × hour × day_class (and × weekday) before origin."""
    cal = calendar_frame(origin - timedelta(days=days * 2), origin - timedelta(days=1))
    h = hist.join(cal.select("date", "day_class", "is_holiday", "school_break", "pre_off"), on="date")
    regular = ~pl.col("is_holiday") & ~pl.col("pre_off")

    def agg(frame: pl.DataFrame, name: str) -> pl.DataFrame:
        return frame.group_by("route", "hour", "day_class").agg(pl.col("boardings").mean().alias(name))

    recent = h.filter(pl.col("date") >= origin - timedelta(days=days))
    # Школьные каникулы понижают уровень; берём их, только если других дней нет.
    lvl = agg(recent.filter(regular & ~pl.col("school_break")), "lvl_a")
    lvl_b = agg(recent.filter(regular), "lvl_b")
    lvl_c = agg(h.filter(regular), "lvl_c")
    prev = agg(h.filter(regular, pl.col("date") < origin - timedelta(days=days)), "lvl_prev")
    out = (
        lvl_c.join(lvl_b, on=["route", "hour", "day_class"], how="left")
        .join(lvl, on=["route", "hour", "day_class"], how="left")
        .join(prev, on=["route", "hour", "day_class"], how="left")
        .with_columns(pl.coalesce("lvl_a", "lvl_b", "lvl_c").alias("lvl"))
    )
    dow = (
        recent.filter(regular & ~pl.col("school_break"))
        .with_columns(pl.col("date").dt.weekday().sub(1).alias("dow"))
        .group_by("route", "hour", "dow").agg(pl.col("boardings").median().alias("lvl_dow"))
    )
    day = out.group_by("route", "day_class").agg(
        pl.col("lvl").sum().alias("lvl_day"), pl.col("lvl_prev").sum().alias("prev_day")
    )
    return (
        out.join(day, on=["route", "day_class"])
        .with_columns(
            (pl.col("lvl") / pl.col("lvl_day").clip(1)).alias("lvl_share"),
            (pl.col("lvl_day") / pl.col("prev_day").clip(1)).fill_null(1.0).alias("lvl_trend"),
        )
        .select("route", "hour", "day_class", "lvl", "lvl_day", "lvl_share", "lvl_trend")
    ), dow


def build(hist: pl.DataFrame, origin: date, horizon_days: int, weather: pl.DataFrame,
          routes: list[int] | None = None) -> pl.DataFrame:
    """Feature rows for all routes × [origin, origin + horizon_days) × 24 h."""
    end = origin + timedelta(days=horizon_days - 1)
    cal = calendar_frame(origin, end).with_columns(
        ((pl.col("date") - origin).dt.total_days()).alias("horizon")
    )
    routes = routes or hist["route"].unique().sort().to_list()
    lv, dow = level(hist, origin)
    frame = (
        pl.DataFrame({"route": routes}, schema={"route": pl.Int32})
        .join(cal, how="cross")
        .join(pl.DataFrame({"hour": list(range(24))}, schema={"hour": pl.Int64}), how="cross")
        .join(lv, on=["route", "hour", "day_class"], how="left")
        .join(dow, on=["route", "hour", "dow"], how="left")
        .join(weather, on=["date", "hour"], how="left")
        .join(hist.select(*KEYS, "boardings"), on=KEYS, how="left")
        .with_columns(pl.col("lvl", "lvl_day", "lvl_share").fill_null(0.0),
            pl.coalesce("lvl_dow", "lvl").fill_null(0.0).alias("lvl_dow"), pl.col("lvl_trend").fill_null(1.0))
    )
    return frame.with_columns(pl.col(c).cast(pl.Int32) for c in ["is_holiday", "is_working_weekend", "pre_off", "post_off", "school_break"])
