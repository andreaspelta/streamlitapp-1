import pandas as pd

# Deterministic templates now use 12 season/day-type combinations.
CLUSTERS = [
    "Autumn-Sunday",
    "Autumn-Saturday",
    "Autumn-Weekday",
    "Spring-Sunday",
    "Spring-Saturday",
    "Spring-Weekday",
    "Summer-Sunday",
    "Summer-Saturday",
    "Summer-Weekday",
    "Winter-Sunday",
    "Winter-Saturday",
    "Winter-Weekday",
]

_SEASON_RANGES = {
    "Winter": ((12, 21), (3, 20)),
    "Spring": ((3, 21), (6, 20)),
    "Summer": ((6, 21), (9, 20)),
    "Autumn": ((9, 21), (12, 20)),
}


def _in_range(month: int, day: int, start: tuple[int, int], end: tuple[int, int]) -> bool:
    """Return True if (month, day) falls within the inclusive seasonal window."""

    if start <= end:
        return (month, day) >= start and (month, day) <= end
    return (month, day) >= start or (month, day) <= end


def season_of(ts: pd.Timestamp) -> str:
    """Map a timestamp to one of the four meteorological seasons used by the app."""

    month_day = (ts.month, ts.day)
    for season, (start, end) in _SEASON_RANGES.items():
        if _in_range(month_day[0], month_day[1], start, end):
            return season
    # Fallback: default to Winter to keep behaviour deterministic.
    return "Winter"


def daytype_of(ts: pd.Timestamp) -> str:
    """Return the day-type label: Sunday, Saturday, or Weekday."""

    weekday = ts.weekday()
    if weekday == 6:
        return "Sunday"
    if weekday == 5:
        return "Saturday"
    return "Weekday"


def hh_shop_cluster(ts: pd.Timestamp) -> str:
    """Return one of the twelve deterministic clusters for the given timestamp."""

    cluster = f"{season_of(ts)}-{daytype_of(ts)}"
    if cluster not in CLUSTERS:
        # Fall back gracefully if future edits change CLUSTERS
        return CLUSTERS[0]
    return cluster


def assign_clusters(hours: pd.DatetimeIndex) -> pd.Series:
    """Vectorised helper returning the cluster label for every timestamp."""

    return pd.Series([hh_shop_cluster(ts) for ts in hours], index=hours, name="cluster")
