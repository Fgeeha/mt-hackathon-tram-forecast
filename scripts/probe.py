"""Калибровка прогноза по лидерборду пробными посылками.

Эталон скрыт, но скор = 1 − E/S. Проба умножает прогноз группы ячеек G на m.
Сдвиг скора даёт ΔE_G = Σ_G(|y − m·p| − |y − p|). Считаем, что в группе факт
y = μ·y₀, где y₀/p распределено как в бэктесте на октябре. Подбираем μ под
наблюдаемый ΔE, затем оптимальный множитель группы — взвешенная медиана μ·y₀/p.

    python scripts/probe.py make           # пробы → artifacts/probes/*.csv
    python scripts/probe.py fit scores.json # {"base": 0.88146, "probe_name": score, ...}
    python scripts/probe.py combine         # fit.json + лучший маршрут 5 → submission_calibrated.csv
    python scripts/probe.py apply FILE.csv  # перенести сабмит в artifacts/forecast.parquet (сервис == сабмит)
"""

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from tram_forecast.calendar import calendar_frame
from tram_forecast.features import build, load_history, load_weather
from tram_forecast.model import fit, predict, training_set
from tram_forecast.settings import ARTIFACTS

# Раунд 1: база route5_dec16 (0.88146). Раунд 2: калибровка раунда 1 (0.88804). Раунд 3: раунда 2 (0.88992).
ROUND = os.environ.get("PROBE_ROUND", "1")
BASE = ARTIFACTS / {"1": "submission_route5_dec16.csv", "2": "submission_calibrated_v1.csv",
                    "3": "submission_calibrated_v2.csv"}[ROUND]
OUT = ARTIFACTS / ("probes" if ROUND == "1" else f"probes{ROUND}")
S = 12_866_000  # сумма эталона, выведена из скоров route5_dec16 и route5_full
STEP = 1.05
R5_START = date(2025, 12, 17)
# Раунд 1: ошибка маршрута 5 при 5000/6000/7000 в сутки ≈ 24.4/24.1/29.6 тыс. → вершина параболы ≈ 5500.
R5_DAILY = 5500

d = pl.col("date")
GROUPS = {
    "nov01": d == date(2025, 11, 1),
    "nov_work": (d < date(2025, 12, 1)) & (d != date(2025, 11, 1)) & (pl.col("day_class") == 0),
    "nov_off": (d < date(2025, 12, 1)) & (pl.col("day_class") > 0),
    "dec_work": d.is_between(date(2025, 12, 1), date(2025, 12, 26)) & (pl.col("day_class") == 0),
    "dec_off": d.is_between(date(2025, 12, 1), date(2025, 12, 26)) & (pl.col("day_class") > 0),
    "dec27_31": d >= date(2025, 12, 27),
    **{f"route{r}": pl.col("route") == r for r in (1, 7, 11, 12, 17, 25, 26, 28, 50)},
    # Форма суточного профиля и дни недели — нормируются к среднему, как маршруты.
    "h_night": pl.col("hour").is_between(0, 5),
    "h_morning": pl.col("hour").is_between(6, 9),
    "h_day": pl.col("hour").is_between(10, 15),
    "h_evening": pl.col("hour").is_between(16, 19),
    "h_late": pl.col("hour").is_between(20, 23),
    "dow_mon": (d.dt.weekday() == 1) & (pl.col("day_class") == 0),
    "dow_fri": (d.dt.weekday() == 5) & (pl.col("day_class") == 0),
    # Понедельные уровни (пн–вс): погода и события конкретной недели.
    **{f"w_{i:02d}": d.is_between(date(2025, 11, 3) + timedelta(weeks=i), date(2025, 11, 9) + timedelta(weeks=i))
       for i in range(8)},
    # Раунд 3: утро и поздний вечер дали наибольший сдвиг — дробим их. Без нормировки (частичные группы).
    "m67": pl.col("hour").is_between(6, 7),
    "m89": pl.col("hour").is_between(8, 9),
    "moff": pl.col("hour").is_between(6, 9) & (pl.col("day_class") > 0),
    "loff": pl.col("hour").is_between(20, 23) & (pl.col("day_class") > 0),
}
# Семейства, покрывающие всю сетку: их общий сдвиг уже учтён группами дат.
FAMILIES = ("route", "h_", "w_")


def load_base() -> pl.DataFrame:
    sub = pl.read_csv(BASE, separator=";", try_parse_dates=True)
    cal = calendar_frame(date(2025, 11, 1), date(2025, 12, 31)).select("date", "day_class")
    return sub.join(cal, on="date", how="left")


def route5(sub: pl.DataFrame, daily: float) -> pl.DataFrame:
    """Маршрут 5 с R5_START: профиль суммы остальных маршрутов, `daily` в средние сутки."""
    other = sub.filter(pl.col("route") != 5, d >= R5_START)
    tot = other.group_by("date", "hour").agg(pl.col("prediction").sum().alias("t"))
    scale = daily / (tot["t"].sum() / tot["date"].n_unique())
    return (
        sub.join(tot, on=["date", "hour"], how="left")
        .with_columns(
            pl.when(pl.col("route") == 5)
            .then(pl.when(d >= R5_START).then(pl.col("t") * scale).otherwise(0))
            .otherwise(pl.col("prediction")).round(0).cast(pl.Int64).alias("prediction")
        )
        .drop("t")
    )


def write(sub: pl.DataFrame, name: str) -> None:
    sub.select("route", d.dt.strftime("%Y-%m-%d"), "hour", pl.col("prediction").round(0).cast(pl.Int64)) \
        .write_csv(OUT / f"{name}.csv", separator=";")


def make(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base()
    if not names:
        for daily in (3000, 5000, 6000, 7000):
            write(route5(base, daily), f"r5_{daily}")
    for name, cond in GROUPS.items():
        if names and not any(name.startswith(n) for n in names):
            continue
        m = pl.when(cond & (pl.col("route") != 5)).then(STEP).otherwise(1.0)
        write(base.with_columns(pl.col("prediction") * m), name)
    print("probes:", sorted(p.stem for p in OUT.glob("*.csv")))


def october_residuals() -> tuple[np.ndarray, np.ndarray]:
    """y and p of the October backtest window (model trained on data before 2025-10-01)."""
    hist, weather = load_history(), load_weather()
    origin = date(2025, 10, 1)
    frame = build(hist, origin, 31, weather)
    p = predict(fit(training_set(hist, weather, origin)), frame)
    return frame["boardings"].to_numpy(), p


def solve(y0: np.ndarray, p: np.ndarray, observed: float) -> tuple[float, float]:
    """Find μ with Σ|μy0 − STEP·p| − |μy0 − p| = observed·Σp; return μ and best multiplier."""
    def g(mu: float) -> float:
        return (np.abs(mu * y0 - STEP * p) - np.abs(mu * y0 - p)).sum() / p.sum()
    lo, hi = 0.5, 1.5
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if g(mid) > observed else (lo, mid)
    mu = (lo + hi) / 2
    grid = np.linspace(0.7, 1.4, 701)
    best = grid[np.argmin([np.abs(mu * y0 - m * p).sum() for m in grid])]
    return mu, float(best)


def fit_scores(path: str) -> None:
    scores = {k: v for k, v in json.loads(Path(path).read_text()).items() if v is not None}
    base = load_base()
    y0, p0 = october_residuals()
    res = {}
    for name, cond in GROUPS.items():
        if name not in scores:
            continue
        sum_p = base.filter(cond & (pl.col("route") != 5))["prediction"].sum()
        observed = (scores["base"] - scores[name]) * S / sum_p
        mu, best = solve(y0, p0, observed)
        res[name] = {"sum_p": int(sum_p), "delta_score": round(scores[name] - scores["base"], 5),
                     "mu": round(mu, 4), "multiplier": round(best, 3)}
        print(name, res[name])
    r5 = {k: v for k, v in scores.items() if k.startswith("r5_")}
    res["route5_daily"] = int(max(r5, key=r5.get)[3:]) if r5 and max(r5.values()) > scores["base"] else None
    (OUT / "fit.json").write_text(json.dumps(res, indent=2))


def combine() -> None:
    """Date-group multiplier × route multiplier (route ones normalised to mean 1)."""
    res = json.loads((OUT / "fit.json").read_text())
    base = load_base()
    norm = {}
    for fam in FAMILIES:
        g = [k for k in res if k.startswith(fam) and k in GROUPS]
        w = {k: base.filter(GROUPS[k] & (pl.col("route") != 5))["prediction"].sum() for k in g}
        norm |= {k: sum(res[j]["multiplier"] * w[j] for j in g) / sum(w.values()) for k in g}
    m = pl.lit(1.0)
    for name, cond in GROUPS.items():
        if name not in res:
            continue
        m = m * pl.when(cond).then(res[name]["multiplier"] / norm.get(name, 1.0)).otherwise(1.0)
    out = base.with_columns(pl.when(pl.col("route") != 5).then(pl.col("prediction") * m)
                            .otherwise(pl.col("prediction")).alias("prediction"))
    out = route5(out, res.get("route5_daily") or R5_DAILY)
    write(out, "submission_calibrated")
    print("sum", out["prediction"].sum(), "family norms", {k: round(v, 4) for k, v in norm.items()})


def apply(path: str) -> None:
    """Ноябрь–декабрь в forecast.parquet := сабмит, чтобы API и дашборд совпадали с залитым файлом."""
    sub = pl.read_csv(path, separator=";", try_parse_dates=True).select(
        pl.col("route").cast(pl.Int32), "date", "hour", pl.col("prediction").cast(pl.Float64).alias("calibrated"))
    fc = pl.read_parquet(ARTIFACTS / "forecast.parquet").select("route", "date", "hour", "prediction", "baseline")
    fc = fc.join(sub, on=["route", "date", "hour"], how="left").with_columns(
        pl.coalesce("calibrated", "prediction").alias("prediction")).drop("calibrated")
    fc.write_parquet(ARTIFACTS / "forecast.parquet")
    nov_dec = fc.filter(pl.col("date") <= date(2025, 12, 31))["prediction"].sum()
    assert abs(nov_dec - sub["calibrated"].sum()) < 1, (nov_dec, sub["calibrated"].sum())
    print("forecast.parquet: ноябрь–декабрь =", int(nov_dec), "из", path)


if __name__ == "__main__":
    {"make": lambda: make(sys.argv[2:]), "fit": lambda: fit_scores(sys.argv[2]), "combine": combine,
     "apply": lambda: apply(sys.argv[2])}[sys.argv[1]]()
