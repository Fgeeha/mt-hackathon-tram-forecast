"""Обучение LightGBM, rolling-origin бэктест и итоговый прогноз.

CLI:
    python -m tram_forecast.model backtest   # метрики модели и бейзлайнов
    python -m tram_forecast.model forecast   # артефакты для API + submission.csv
"""

import argparse
import json
import logging
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import polars as pl

from tram_forecast.features import FEATURES, build, load_history, load_weather

log = logging.getLogger(__name__)

HORIZON = 61
FIRST_TRAIN_ORIGIN = date(2025, 2, 3)
BACKTEST_ORIGINS = [date(2025, 3, 1), date(2025, 4, 1), date(2025, 10, 1)]
DATA_END = date(2025, 10, 31)
PARAMS = {
    "objective": "l1", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 500,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "verbose": -1, "seed": 42, "num_threads": 8,
}
ROUNDS = 400
# Доля LightGBM в итоговом прогнозе; остальное — профиль за 4 недели (lvl).
BLEND = 0.5
# Абляции подтверждают вклад внешних источников: модель без погоды / без календаря.
ABLATIONS = {
    "no_weather": {"temp", "precip", "snowfall", "snow_depth", "wind", "temp_day", "precip_day", "snowfall_day"},
    "no_calendar": {"is_holiday", "is_working_weekend", "pre_off", "post_off", "school_break", "off_run"},
}


def wape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).sum() / max(y.sum(), 1e-9))


def training_set(hist: pl.DataFrame, weather: pl.DataFrame, until: date, step: int = 7) -> pl.DataFrame:
    """Stack feature frames for weekly origins whose targets end before `until`."""
    frames, o = [], FIRST_TRAIN_ORIGIN
    while o < until:
        days = min(HORIZON, (until - o).days)
        frames.append(build(hist, o, days, weather))
        o += timedelta(days=step)
    return pl.concat(frames).filter(pl.col("boardings").is_not_null())


def _xy(frame: pl.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = frame.select(features).to_numpy().astype(np.float64)
    base = np.maximum(frame["lvl"].to_numpy(), 1.0)
    return x, base, frame["boardings"].to_numpy() if "boardings" in frame.columns else None


def fit(train: pl.DataFrame, features: list[str] = FEATURES) -> lgb.Booster:
    """Target = boardings / level, weight = level → L1 on ratio ≡ L1 on boardings (WAPE)."""
    x, base, y = _xy(train, features)
    ds = lgb.Dataset(x, y / base, weight=base, feature_name=features)
    return lgb.train(PARAMS, ds, num_boost_round=ROUNDS)


def predict(model: lgb.Booster, frame: pl.DataFrame) -> np.ndarray:
    x, base, _ = _xy(frame, model.feature_name())
    ml = np.clip(model.predict(x) * base, 0, None)
    return BLEND * ml + (1 - BLEND) * frame["lvl"].to_numpy()


def naive_week(hist: pl.DataFrame, frame: pl.DataFrame, origin: date) -> np.ndarray:
    """Baseline: last week before origin repeated (same weekday and hour)."""
    last = hist.filter(pl.col("date") >= origin - timedelta(days=7), pl.col("date") < origin)
    last = last.with_columns(pl.col("date").dt.weekday().alias("wd")).select("route", "hour", "wd", pl.col("boardings").alias("nv"))
    f = frame.with_columns(pl.col("date").dt.weekday().alias("wd")).join(last, on=["route", "hour", "wd"], how="left")
    return f["nv"].fill_null(0).to_numpy()


def backtest() -> dict:
    hist, weather = load_history(), load_weather()
    rows = []
    for origin in BACKTEST_ORIGINS:
        train = training_set(hist, weather, origin)
        test = build(hist, origin, min(HORIZON, (DATA_END - origin).days + 1), weather)
        y = test["boardings"].to_numpy()
        preds = {
            "blend_lgbm_profile": predict(fit(train), test),
            **{f"ablation_{name}": predict(fit(train, [f for f in FEATURES if f not in drop]), test)
               for name, drop in ABLATIONS.items()},
            "naive_last_week": naive_week(hist, test, origin),
            "profile_4w": test["lvl"].to_numpy(),
            # Тот же профиль, но по дню недели без производственного календаря.
            "profile_4w_no_calendar": test["lvl_dow"].to_numpy(),
        }
        for name, p in preds.items():
            peak = test["hour"].is_in([7, 8, 9, 17, 18, 19]).to_numpy()
            rows.append({
                "origin": str(origin), "model": name,
                "wape_score": round(1 - wape(y, p), 4),
                "mae": round(float(np.abs(y - p).mean()), 2),
                "wape_score_peak": round(1 - wape(y[peak], p[peak]), 4),
                "wape_score_day": round(1 - _daily_wape(test, p), 4),
            })
            log.info("%s %-16s %s", origin, name, rows[-1])
    return {"windows": rows, "horizon_days": HORIZON}


def _daily_wape(frame: pl.DataFrame, p: np.ndarray) -> float:
    d = frame.with_columns(pl.Series("p", p)).group_by("route", "date").agg(pl.col("boardings", "p").sum())
    return wape(d["boardings"].to_numpy(), d["p"].to_numpy())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest", "forecast"])
    args = ap.parse_args()
    if args.cmd == "backtest":
        res = backtest()
        from tram_forecast.settings import ARTIFACTS
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        from tram_forecast.forecast import run
        run()


if __name__ == "__main__":
    main()
