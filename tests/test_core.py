from datetime import date

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from tram_forecast.api import app
from tram_forecast.calendar import calendar_frame, day_class
from tram_forecast.model import wape
from tram_forecast.settings import ARTIFACTS, ROUTES

client = TestClient(app)


def test_calendar_november_2025():
    assert day_class(date(2025, 11, 1)) == 0  # рабочая суббота
    assert day_class(date(2025, 11, 3)) == 2 and day_class(date(2025, 11, 4)) == 2
    assert day_class(date(2025, 12, 31)) == 2
    cal = calendar_frame(date(2025, 11, 1), date(2025, 11, 5))
    assert cal.filter(pl.col("date") == date(2025, 11, 5))["post_off"][0]


def test_wape():
    assert wape(np.array([10.0, 0.0]), np.array([10.0, 0.0])) == 0.0
    assert wape(np.array([10.0, 10.0]), np.array([5.0, 15.0])) == pytest.approx(0.5)


@pytest.mark.parametrize("name", ["submission", "submission_route5_dec16", "submission_route5_full"])
def test_submission_grid(name):
    sub = pl.read_csv(ARTIFACTS / f"{name}.csv", separator=";")
    sample = pl.read_csv(ARTIFACTS / "submission.csv", separator=";")
    assert sub.select("route", "date", "hour").equals(sample.select("route", "date", "hour"))
    assert sub.columns == ["route", "date", "hour", "prediction"]
    assert sub.height == len(ROUTES) * 61 * 24
    assert sub.select(pl.struct("route", "date", "hour").n_unique()).item() == sub.height
    assert sub["prediction"].min() >= 0


def test_forecast_aggregation_consistent():
    hourly = client.get("/api/forecast", params={"horizon": "month", "route": 17}).json()["rows"]
    total = client.get("/api/forecast", params={"horizon": "month", "route": 17, "agg": "total"}).json()
    assert total["rows"][0]["prediction"] == pytest.approx(sum(r["prediction"] for r in hourly), rel=1e-3)
    k = client.get("/api/forecast", params={"horizon": "month", "route": 17, "agg": "total", "k_event": 1.2}).json()
    assert k["rows"][0]["prediction"] == pytest.approx(total["rows"][0]["prediction"] * 1.2, rel=1e-3)


@pytest.mark.parametrize("params,code", [
    ({"date_from": "2024-01-01"}, 400),
    ({"hour_from": 30}, 422),
    ({"route": 99}, 404),
    ({"route": 1, "stop_id": "nope"}, 404),
])
def test_errors_are_readable(params, code):
    r = client.get("/api/forecast", params=params)
    assert r.status_code == code and r.json()["error"]


def test_export_xlsx():
    r = client.get("/api/export", params={"format": "xlsx", "agg": "day", "horizon": "month"})
    assert r.status_code == 200 and r.content[:2] == b"PK"
