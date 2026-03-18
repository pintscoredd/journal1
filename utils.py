import pandas as pd
import pytz
from datetime import datetime

DEFAULT_TZ = "America/Los_Angeles"

def to_local_time(dt_utc: datetime) -> datetime:
    if dt_utc.tzinfo is None:
        dt_utc = pytz.utc.localize(dt_utc)
    return dt_utc.astimezone(pytz.timezone(DEFAULT_TZ))

def safely_divide(a, b, default=0.0):
    try:
        if b == 0 or pd.isna(b):
            return default
        return a / b
    except (ZeroDivisionError, TypeError, ValueError):
        return default
