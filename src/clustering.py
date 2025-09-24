import pandas as pd

def season_of(ts: pd.Timestamp) -> str:
    m = ts.month
    if m in (12, 1, 2): return "Winter"
    if m in (3, 4, 5): return "Spring"
    if m in (6, 7, 8): return "Summer"
    return "Autumn"

def daytype_of(ts: pd.Timestamp) -> str:
    wd = ts.weekday()
    if wd == 5: return "Saturday"
    if wd == 6: return "Holiday"
    return "Weekday"

def hh_shop_cluster(ts: pd.Timestamp) -> str:
    return f"{season_of(ts)}-{daytype_of(ts)}"
