import pandas as pd

# Updated clustering scheme — deterministic templates rely on six clusters only.
CLUSTERS = [
    "Winter-Weekday",
    "Winter-Weekend",
    "Shoulder-Weekday",
    "Shoulder-Weekend",
    "Summer-Weekday",
    "Summer-Weekend",
]


def season_of(ts: pd.Timestamp) -> str:
    """Collapse the calendar into Winter/Summer/Shoulder seasons."""

    m = ts.month
    if m in (12, 1, 2):
        return "Winter"
    if m in (6, 7, 8):
        return "Summer"
    return "Shoulder"


def daytype_of(ts: pd.Timestamp) -> str:
    """Two day-types only: Weekday vs Weekend/Holiday."""

    return "Weekend" if ts.weekday() >= 5 else "Weekday"


def hh_shop_cluster(ts: pd.Timestamp) -> str:
    """Return one of the six deterministic clusters for the given timestamp."""

    cluster = f"{season_of(ts)}-{daytype_of(ts)}"
    if cluster not in CLUSTERS:
        # Fall back gracefully if future edits change CLUSTERS
        return CLUSTERS[0]
    return cluster


def assign_clusters(hours: pd.DatetimeIndex) -> pd.Series:
    """Vectorised helper returning the cluster label for every timestamp."""

    return pd.Series([hh_shop_cluster(ts) for ts in hours], index=hours, name="cluster")
