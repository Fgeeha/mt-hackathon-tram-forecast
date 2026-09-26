"""Календарь РФ/Москвы на 2025 год.

Источники:
- Производственный календарь 2025, постановление Правительства РФ № 1335
  от 04.10.2024: https://www.consultant.ru/law/ref/calendar/proizvodstvennye/2025/
- Школьные каникулы Москвы (ДОНМ, модульный/четвертной график):
  https://1517.mskobr.ru/edu-news/11020 (2025/26),
  https://www.mos.ru/city/projects/kanikuly/ (общий календарь)
"""

from datetime import date, timedelta

import polars as pl


def _span(a: str, b: str) -> list[date]:
    d0, d1 = date.fromisoformat(a), date.fromisoformat(b)
    return [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]


HOLIDAYS = set(
    _span("2025-01-01", "2025-01-08") + _span("2025-05-01", "2025-05-04")
    + _span("2025-05-08", "2025-05-11") + _span("2025-06-12", "2025-06-15")
    + _span("2025-11-02", "2025-11-04") + _span("2025-12-31", "2026-01-11")
    + [date(2025, 2, 23), date(2025, 3, 8)]
    # 2026: https://www.consultant.ru/law/ref/calendar/proizvodstvennye/2026/
    + [date(2026, 2, 23), date(2026, 3, 9), date(2026, 5, 1), date(2026, 5, 11),
       date(2026, 6, 12), date(2026, 11, 4), date(2026, 12, 31)]
)
WORKING_WEEKENDS = {date(2025, 11, 1)}
SCHOOL_BREAKS = set(
    _span("2025-01-01", "2025-01-08") + _span("2025-03-22", "2025-03-30")
    + _span("2025-05-27", "2025-08-31") + _span("2025-10-25", "2025-11-02")
    + _span("2025-12-31", "2026-01-11") + _span("2026-02-21", "2026-03-01")
    + _span("2026-04-11", "2026-04-19") + _span("2026-05-30", "2026-08-31")
    + _span("2026-10-24", "2026-11-01")
)


def is_off(d: date) -> bool:
    """Non-working day: holiday or weekend that was not moved to a working day."""
    return d in HOLIDAYS or (d.weekday() >= 5 and d not in WORKING_WEEKENDS)


def day_class(d: date) -> int:
    """0 = рабочий, 1 = суббота, 2 = воскресенье/праздник."""
    if d in HOLIDAYS or (d.weekday() == 6 and d not in WORKING_WEEKENDS):
        return 2
    if d.weekday() == 5 and d not in WORKING_WEEKENDS:
        return 1
    return 0


def calendar_frame(start: date, end: date) -> pl.DataFrame:
    """Daily calendar features for [start, end]."""
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    rows = []
    for d in days:
        prev, nxt = d - timedelta(days=1), d + timedelta(days=1)
        rows.append({
            "date": d,
            "dow": d.weekday(),
            "day_class": day_class(d),
            "is_holiday": d in HOLIDAYS,
            "is_working_weekend": d in WORKING_WEEKENDS,
            "pre_off": not is_off(d) and is_off(nxt),
            "post_off": not is_off(d) and is_off(prev),
            "school_break": d in SCHOOL_BREAKS,
            # Длинные праздничные серии (Новый год, май) ведут себя иначе одиночных.
            "off_run": sum(is_off(d + timedelta(days=k)) for k in range(-3, 4)) if is_off(d) else 0,
        })
    return pl.DataFrame(rows)
