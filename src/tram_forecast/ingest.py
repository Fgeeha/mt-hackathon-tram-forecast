"""Приём и нормализация сырых валидаций: train.csv/test.csv → route_hour.parquet.

Посадка = успешная валидация (validation_result == 1), маршрут берётся из
ngpt_route, время — из tran_date_time (input_date_time ненадёжен).
Кроме целевой величины считаются уникальные карты, число отказов и число
трамваев (garage_number) на линии за час.
"""

import argparse
import logging

import duckdb
import polars as pl

from tram_forecast.settings import PROCESSED, RAW

log = logging.getLogger(__name__)

COLUMNS = [
    "tran_no", "device_no", "tran_date_time", "begin_date_time", "input_date_time",
    "crd_hashcode", "validation_result", "tran_type_id", "place_id", "good_type",
    "pass_route", "ngpt_route", "bus_exit_no", "garage_number",
]

QUERY = """
with v as (
  select
    try_cast(regexp_extract(ngpt_route, '^(\\d+)', 1) as int) as route,
    try_strptime(tran_date_time, '%Y-%m-%d %H:%M:%S') as ts,
    validation_result = '1' as ok,
    crd_hashcode, garage_number
  from read_csv({files}, delim=';', header=true, quote='', escape='',
                columns={columns}, strict_mode=false, null_padding=true, parallel=false)
)
select route, cast(ts as date) as date, hour(ts) as hour,
       count(*) filter (where ok) as boardings,
       count(distinct crd_hashcode) filter (where ok) as unique_cards,
       count(*) filter (where not ok) as rejected,
       count(distinct garage_number) filter (where ok) as vehicles
from v
where route is not null and ts is not null
group by all
order by all
"""


def ingest(files: list[str]) -> pl.DataFrame:
    """Aggregate raw validation CSVs to route × date × hour."""
    cols = "{" + ",".join(f"'{c}':'VARCHAR'" for c in COLUMNS) + "}"
    sql = QUERY.format(files=[str(f) for f in files], columns=cols)
    return duckdb.sql(sql).pl()


def check_against_labels(df: pl.DataFrame) -> float:
    """Share of label rows matched exactly by our aggregation."""
    labels = pl.concat([
        pl.read_csv(RAW / f"labels/labels_day_{s}.csv", separator=";", try_parse_dates=True)
        for s in ("train", "test")
    ])
    j = labels.join(df, on=["route", "date", "hour"], how="left", suffix="_our")
    return (j["boardings"] == j["boardings_our"]).mean()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", default=[RAW / "train.csv", RAW / "test.csv"])
    args = ap.parse_args()
    df = ingest(args.files)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.write_parquet(PROCESSED / "route_hour.parquet")
    log.info("route_hour: %d строк, %s … %s", len(df), df["date"].min(), df["date"].max())
    log.info("совпадение с labels организаторов: %.4f", check_against_labels(df))


if __name__ == "__main__":
    main()
