from __future__ import annotations

from typing import Iterable, List

import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import is_datetime64tz_dtype


WH_PER_KWH = 1000.0
MONTH_LABELS = [
    "Gen",
    "Feb",
    "Mar",
    "Apr",
    "Mag",
    "Giu",
    "Lug",
    "Ago",
    "Set",
    "Ott",
    "Nov",
    "Dic",
]


def _normalize_timestamps(series: pd.Series) -> pd.Series:
    """Return timezone-naive timestamps for downstream processing."""

    if is_datetime64tz_dtype(series.dtype):
        # Preserve the original wall-clock time when removing timezone information
        # so that chart filtering/labeling stays aligned with exported tables.
        return series.dt.tz_localize(None)
    return series


def _hour_labels(timestamps: pd.Series) -> List[str]:
    normalized = _normalize_timestamps(timestamps)
    return normalized.dt.strftime("%H:%M").tolist()


def _month_group(
    timestamps: pd.Series,
    columns: Iterable[str],
    data: pd.DataFrame,
) -> pd.DataFrame:
    normalized = _normalize_timestamps(timestamps)
    month_data = data.copy()
    month_data["month"] = normalized.dt.month
    grouped = (
        month_data.groupby("month")[list(columns)].sum().reindex(range(1, 13), fill_value=0.0)
    )
    grouped.index.name = "month"
    return grouped


def build_profile_chart(community_df: pd.DataFrame, day) -> go.Figure:
    if community_df.empty:
        raise ValueError("Community profile data is empty.")

    timestamps = community_df["timestamp"]
    normalized = _normalize_timestamps(timestamps)
    mask = normalized.dt.date == day
    day_df = community_df.loc[mask].copy()
    if day_df.empty:
        raise ValueError("No community data found for the selected day.")

    day_df = day_df.sort_values("timestamp")
    labels = _hour_labels(day_df["timestamp"])

    pv_wh = day_df["pv_generation_kWh"] * WH_PER_KWH
    surplus_wh = (
        day_df["matched_hh_kWh"] + day_df["matched_shop_kWh"] + day_df["exports_kWh"]
    ) * WH_PER_KWH
    hh_load_wh = day_df["hh_load_kWh"] * WH_PER_KWH
    shop_load_wh = day_df["shop_load_kWh"] * WH_PER_KWH

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="PV generation",
            x=labels,
            y=pv_wh,
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Prosumer surplus",
            x=labels,
            y=surplus_wh,
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Household load",
            x=labels,
            y=hh_load_wh,
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Shop load",
            x=labels,
            y=shop_load_wh,
            mode="lines+markers",
        )
    )

    fig.update_layout(
        title="Profile",
        hovermode="x unified",
        xaxis=dict(title="Hour of day"),
        yaxis=dict(title="Energy (Wh)"),
        legend=dict(title=""),
    )
    return fig


def build_surplus_matching_chart(
    prosumer_df: pd.DataFrame,
    *,
    prosumer_id: str,
    view: str,
    day=None,
) -> go.Figure:
    if prosumer_df.empty:
        raise ValueError("Prosumer hourly data is empty.")

    filtered = prosumer_df[prosumer_df["prosumer_id"] == prosumer_id]
    if filtered.empty:
        raise ValueError("The selected prosumer has no data.")

    fig = go.Figure()
    if view == "monthly":
        month_stats = _month_group(
            filtered["timestamp"],
            ["surplus_kWh", "matched_hh_kWh", "matched_shop_kWh", "exports_kWh"],
            filtered,
        )
        labels = [MONTH_LABELS[m - 1] for m in month_stats.index]
        surplus_wh = month_stats["surplus_kWh"] * WH_PER_KWH
        hh_match_wh = month_stats["matched_hh_kWh"] * WH_PER_KWH
        shop_match_wh = month_stats["matched_shop_kWh"] * WH_PER_KWH
        export_wh = month_stats["exports_kWh"] * WH_PER_KWH
        xaxis_title = "Month"
    else:
        timestamps = filtered["timestamp"]
        normalized = _normalize_timestamps(timestamps)
        if day is None:
            raise ValueError("A day must be provided for daily view.")
        mask = normalized.dt.date == day
        day_df = filtered.loc[mask].copy()
        if day_df.empty:
            raise ValueError("No data found for the selected day.")
        day_df = day_df.sort_values("timestamp")
        labels = _hour_labels(day_df["timestamp"])
        surplus_wh = day_df["surplus_kWh"] * WH_PER_KWH
        hh_match_wh = day_df["matched_hh_kWh"] * WH_PER_KWH
        shop_match_wh = day_df["matched_shop_kWh"] * WH_PER_KWH
        export_wh = day_df["exports_kWh"] * WH_PER_KWH
        xaxis_title = "Hour of day"

    fig.add_trace(
        go.Bar(
            name="Total surplus",
            x=labels,
            y=surplus_wh,
            marker_color="#1f77b4",
            offsetgroup="surplus_total",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Shared with connected consumers",
            x=labels,
            y=hh_match_wh,
            marker_color="#2ca02c",
            offsetgroup="surplus_breakdown",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Shared with shops",
            x=labels,
            y=shop_match_wh,
            marker_color="#ff7f0e",
            offsetgroup="surplus_breakdown",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Export",
            x=labels,
            y=export_wh,
            marker_color="#9467bd",
            offsetgroup="surplus_breakdown",
        )
    )

    fig.update_layout(
        title="Surplus matching",
        barmode="relative",
        xaxis=dict(title=xaxis_title),
        yaxis=dict(title="Energy (Wh)"),
        legend=dict(title=""),
    )
    return fig


def build_household_coverage_chart(
    household_df: pd.DataFrame,
    *,
    household_id: str,
    view: str,
    day=None,
) -> go.Figure:
    if household_df.empty:
        raise ValueError("Household hourly data is empty.")

    filtered = household_df[household_df["household_id"] == household_id]
    if filtered.empty:
        raise ValueError("The selected household has no data.")

    fig = go.Figure()
    if view == "monthly":
        month_stats = _month_group(filtered["timestamp"], ["load_kWh", "matched_kWh"], filtered)
        labels = [MONTH_LABELS[m - 1] for m in month_stats.index]
        load_wh = month_stats["load_kWh"] * WH_PER_KWH
        matched_wh = month_stats["matched_kWh"] * WH_PER_KWH
        xaxis_title = "Month"
    else:
        timestamps = filtered["timestamp"]
        normalized = _normalize_timestamps(timestamps)
        if day is None:
            raise ValueError("A day must be provided for daily view.")
        mask = normalized.dt.date == day
        day_df = filtered.loc[mask].copy()
        if day_df.empty:
            raise ValueError("No data found for the selected day.")
        day_df = day_df.sort_values("timestamp")
        labels = _hour_labels(day_df["timestamp"])
        load_wh = day_df["load_kWh"] * WH_PER_KWH
        matched_wh = day_df["matched_kWh"] * WH_PER_KWH
        xaxis_title = "Hour of day"

    fig.add_trace(
        go.Bar(
            name="Load",
            x=labels,
            y=load_wh,
            marker_color="#1f77b4",
            offsetgroup="load",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Shared energy",
            x=labels,
            y=matched_wh,
            marker_color="#2ca02c",
            offsetgroup="matched",
        )
    )

    fig.update_layout(
        title="Households Demand Coverage",
        barmode="group",
        xaxis=dict(title=xaxis_title),
        yaxis=dict(title="Energy (Wh)"),
        legend=dict(title=""),
    )
    return fig


def build_shop_coverage_chart(
    shop_df: pd.DataFrame,
    *,
    shop_id: str,
    view: str,
    day=None,
) -> go.Figure:
    if shop_df.empty:
        raise ValueError("Shop hourly data is empty.")

    filtered = shop_df[shop_df["shop_id"] == shop_id]
    if filtered.empty:
        raise ValueError("The selected shop has no data.")

    fig = go.Figure()
    if view == "monthly":
        month_stats = _month_group(filtered["timestamp"], ["load_kWh", "matched_kWh"], filtered)
        labels = [MONTH_LABELS[m - 1] for m in month_stats.index]
        load_wh = month_stats["load_kWh"] * WH_PER_KWH
        matched_wh = month_stats["matched_kWh"] * WH_PER_KWH
        xaxis_title = "Month"
    else:
        timestamps = filtered["timestamp"]
        normalized = _normalize_timestamps(timestamps)
        if day is None:
            raise ValueError("A day must be provided for daily view.")
        mask = normalized.dt.date == day
        day_df = filtered.loc[mask].copy()
        if day_df.empty:
            raise ValueError("No data found for the selected day.")
        day_df = day_df.sort_values("timestamp")
        labels = _hour_labels(day_df["timestamp"])
        load_wh = day_df["load_kWh"] * WH_PER_KWH
        matched_wh = day_df["matched_kWh"] * WH_PER_KWH
        xaxis_title = "Hour of day"

    fig.add_trace(
        go.Bar(
            name="Load",
            x=labels,
            y=load_wh,
            marker_color="#1f77b4",
            offsetgroup="load",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Shared energy",
            x=labels,
            y=matched_wh,
            marker_color="#ff7f0e",
            offsetgroup="matched",
        )
    )

    fig.update_layout(
        title="Shop Demand Coverage",
        barmode="group",
        xaxis=dict(title=xaxis_title),
        yaxis=dict(title="Energy (Wh)"),
        legend=dict(title=""),
    )
    return fig
