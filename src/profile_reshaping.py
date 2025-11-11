"""Household profile reshaping helpers.

This module keeps the reshaping logic isolated from the deterministic
simulation pipeline so that the existing workflow is unaffected.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .clustering import CLUSTERS, assign_clusters
from .simulation import build_hourly_profile


@dataclass
class ReshapingParameters:
    """User-controlled constraints for profile reshaping."""

    monthly_targets: Mapping[int, float]
    pv_capacity_kwp: float
    min_hourly_fraction: float
    max_hourly_fraction: float
    absolute_min_kwh: float
    absolute_max_kwh: float
    monthly_scale_limit: float


@dataclass
class ReshapingResult:
    reshaped_series: pd.Series
    reshaped_medians: pd.DataFrame
    monthly_summary: pd.DataFrame
    warnings: List[str]


def reshape_household_profile(
    *,
    hours: pd.DatetimeIndex,
    medians: pd.DataFrame,
    pv_series: pd.Series,
    params: ReshapingParameters,
) -> ReshapingResult:
    """Return a reshaped household profile that meets monthly targets when possible."""

    if hours is None or len(hours) == 0:
        raise ValueError("Calendar hours are missing; set the simulation year first.")
    if medians is None or medians.empty:
        raise ValueError("Household template is empty; upload medians in Templates & Inputs.")
    if pv_series is None or pv_series.empty:
        raise ValueError("PV template is missing; upload PV data in Templates & Inputs.")

    pv_scaled = _ensure_series(pv_series, hours) * float(params.pv_capacity_kwp)
    hh_series = build_hourly_profile(hours, medians)
    hh_series = _ensure_series(hh_series, hours)

    reshaped = hh_series.copy().astype(float)
    monthly_records: List[Dict[str, object]] = []
    warnings: List[str] = []

    for month in range(1, 13):
        mask = reshaped.index.month == month
        if not mask.any():
            continue
        base_month = reshaped.loc[mask]
        original_month = hh_series.loc[mask]
        pv_month = pv_scaled.loc[mask]
        target = float(params.monthly_targets.get(month, 0.0))
        (
            new_values,
            summary,
        ) = _reshape_month(original_month, pv_month, target, params)
        reshaped.loc[mask] = new_values
        monthly_records.append(summary)
        if not summary["feasible"] and summary["note"]:
            warnings.append(f"Month {month:02d}: {summary['note']}")

    medians_out = series_to_cluster_medians(hours, reshaped)
    monthly_df = pd.DataFrame(monthly_records)

    return ReshapingResult(
        reshaped_series=reshaped,
        reshaped_medians=medians_out,
        monthly_summary=monthly_df,
        warnings=warnings,
    )


def series_to_cluster_medians(hours: pd.DatetimeIndex, series: pd.Series) -> pd.DataFrame:
    """Aggregate an hourly series back to the 24×12 deterministic median table."""

    aligned = _ensure_series(series, hours)
    clusters = assign_clusters(hours)
    df = pd.DataFrame(
        {
            "timestamp": hours,
            "cluster": clusters.values,
            "hour": hours.hour,
            "load_kWh": aligned.values,
        }
    )
    grouped = (
        df.groupby(["hour", "cluster"], as_index=False)["load_kWh"]
        .median()
        .pivot(index="hour", columns="cluster", values="load_kWh")
    )
    grouped = grouped.reindex(index=range(24))
    for cluster in CLUSTERS:
        if cluster not in grouped.columns:
            grouped[cluster] = 0.0
    grouped = grouped[CLUSTERS]
    grouped = grouped.fillna(0.0)
    grouped.index.name = "hour"
    return grouped.reset_index()


def build_excel_payload(medians: pd.DataFrame, *, year: int) -> bytes:
    """Create an Excel file compatible with the HH template (24×12 grid)."""

    ordered = medians.copy()
    if "hour" not in ordered.columns:
        ordered.insert(0, "hour", np.arange(len(ordered), dtype=int))
    ordered = ordered.sort_values("hour").reset_index(drop=True)
    ordered = ordered[["hour", *CLUSTERS]]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as xl:
        sheet = "HH_median"
        ordered.to_excel(xl, sheet_name=sheet, index=False, startrow=3)
        ws = xl.sheets[sheet]
        ws.write(0, 0, f"Deterministic HH template — year {year}")
        ws.write(1, 0, "Reshaped median hourly demand (kWh) for each cluster.")
        ws.write(2, 0, "Hour 0 corresponds to 00:00-01:00 local time.")
    return buffer.getvalue()


def _ensure_series(series: pd.Series, hours: pd.DatetimeIndex) -> pd.Series:
    aligned = series.reindex(hours)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index[:5]
        raise ValueError(
            "Series has missing hours compared to the calendar: "
            f"{list(missing)}"
        )
    return aligned.astype(float)


def _reshape_month(
    base_load: pd.Series,
    pv: pd.Series,
    target: float,
    params: ReshapingParameters,
) -> Tuple[pd.Series, Dict[str, object]]:
    """Return reshaped hourly load for a single month plus summary data."""

    hours = base_load.index
    original_values = base_load.to_numpy(dtype=float)
    pv_values = np.maximum(pv.to_numpy(dtype=float), 0.0)
    load = original_values.copy()

    total_original = float(load.sum())
    pv_total = float(pv_values.sum())

    if total_original <= 0.0:
        summary = _build_summary(
            month=int(hours[0].month),
            target=target,
            original_load=0.0,
            reshaped_load=0.0,
            pv_energy=pv_total,
            matched_energy=0.0,
            feasible=target <= 0.0,
            note="No household demand available for this month.",
        )
        return base_load, summary

    min_fraction = np.clip(params.min_hourly_fraction, 0.0, 1.0)
    max_fraction = max(params.max_hourly_fraction, 1.0)

    min_load = np.maximum(min_fraction * original_values, params.absolute_min_kwh)
    min_load = np.minimum(min_load, original_values)
    max_load = np.maximum(max_fraction * original_values, original_values)
    if params.absolute_max_kwh > 0:
        max_load = np.minimum(max_load, params.absolute_max_kwh)
    max_load = np.maximum(max_load, min_load)

    load = np.clip(load, min_load, max_load)

    min_total = total_original * (1.0 - params.monthly_scale_limit)
    max_total = total_original * (1.0 + params.monthly_scale_limit)

    load = _enforce_monthly_total(load, min_load, max_load, pv_values, min_total, max_total)
    load = _redistribute_to_pv(load, min_load, max_load, pv_values)

    matched = np.minimum(load, pv_values)
    total_load = float(load.sum())
    matched_energy = float(matched.sum())
    coverage = matched_energy / total_load if total_load > 1e-9 else 0.0

    feasible = True
    note = ""

    if target > 0:
        load, matched_energy, coverage = _lift_coverage(
            load,
            min_load,
            max_load,
            pv_values,
            min_total,
            target,
        )
        total_load = float(load.sum())
        matched_energy = float(np.minimum(load, pv_values).sum())
        coverage = matched_energy / total_load if total_load > 1e-9 else 0.0

    if target > 0 and coverage + 1e-4 < target:
        feasible = False
        if pv_total <= 1e-9:
            note = "PV energy is zero; coverage target cannot be met."
        else:
            note = (
                "Coverage target could not be met within the provided constraints."
            )

    summary = _build_summary(
        month=int(hours[0].month),
        target=target,
        original_load=total_original,
        reshaped_load=total_load,
        pv_energy=pv_total,
        matched_energy=matched_energy,
        feasible=feasible,
        note=note,
    )

    return pd.Series(load, index=hours), summary


def _enforce_monthly_total(
    load: np.ndarray,
    min_load: np.ndarray,
    max_load: np.ndarray,
    pv: np.ndarray,
    min_total: float,
    max_total: float,
) -> np.ndarray:
    total = float(load.sum())

    if total < min_total:
        deficit = min_total - total
        load = load.copy()
        load = _add_energy(load, max_load, pv, deficit)
    elif total > max_total:
        excess = total - max_total
        load = load.copy()
        load = _remove_energy(load, min_load, pv, excess)
    return load


def _redistribute_to_pv(
    load: np.ndarray,
    min_load: np.ndarray,
    max_load: np.ndarray,
    pv: np.ndarray,
) -> np.ndarray:
    load = load.copy()
    receivers = np.argsort(-pv)
    donors = np.argsort(pv)

    for r in receivers:
        capacity = max_load[r] - load[r]
        if capacity <= 1e-9:
            continue
        for d in donors:
            if d == r or pv[d] >= pv[r] - 1e-9:
                continue
            transferable = load[d] - min_load[d]
            if transferable <= 1e-9:
                continue
            delta = min(capacity, transferable)
            if delta <= 1e-9:
                continue
            load[d] -= delta
            load[r] += delta
            capacity -= delta
            if capacity <= 1e-9:
                break
    return load


def _lift_coverage(
    load: np.ndarray,
    min_load: np.ndarray,
    max_load: np.ndarray,
    pv: np.ndarray,
    min_total: float,
    target: float,
) -> Tuple[np.ndarray, float, float]:
    load = load.copy()

    for _ in range(2):
        load = _redistribute_to_pv(load, min_load, max_load, pv)
        matched = np.minimum(load, pv)
        total = float(load.sum())
        matched_energy = float(matched.sum())
        coverage = matched_energy / total if total > 1e-9 else 0.0
        if coverage >= target - 1e-4:
            return load, matched_energy, coverage

        # Attempt to reduce low-PV hours while staying above min_total
        desired_total = matched_energy / target if target > 1e-6 else total
        desired_total = max(desired_total, min_total)
        if desired_total < total - 1e-6:
            removable = total - desired_total
            load = _remove_energy(load, min_load, pv, removable)
            continue
        break

    matched = np.minimum(load, pv)
    total = float(load.sum())
    matched_energy = float(matched.sum())
    coverage = matched_energy / total if total > 1e-9 else 0.0
    return load, matched_energy, coverage


def _add_energy(
    load: np.ndarray,
    max_load: np.ndarray,
    pv: np.ndarray,
    energy: float,
) -> np.ndarray:
    load = load.copy()
    receivers = np.argsort(-pv)
    remaining = float(energy)
    for idx in receivers:
        if remaining <= 1e-9:
            break
        capacity = max_load[idx] - load[idx]
        if capacity <= 1e-9:
            continue
        delta = min(capacity, remaining)
        load[idx] += delta
        remaining -= delta
    return load


def _remove_energy(
    load: np.ndarray,
    min_load: np.ndarray,
    pv: np.ndarray,
    energy: float,
) -> np.ndarray:
    load = load.copy()
    donors = np.argsort(pv)
    remaining = float(energy)
    for idx in donors:
        if remaining <= 1e-9:
            break
        removable = load[idx] - min_load[idx]
        if removable <= 1e-9:
            continue
        delta = min(removable, remaining)
        load[idx] -= delta
        remaining -= delta
    return load


def _build_summary(
    *,
    month: int,
    target: float,
    original_load: float,
    reshaped_load: float,
    pv_energy: float,
    matched_energy: float,
    feasible: bool,
    note: str,
) -> Dict[str, object]:
    coverage = matched_energy / reshaped_load if reshaped_load > 1e-9 else 0.0
    return {
        "month": month,
        "target_coverage": target,
        "achieved_coverage": coverage,
        "original_load_kWh": original_load,
        "reshaped_load_kWh": reshaped_load,
        "pv_energy_kWh": pv_energy,
        "matched_energy_kWh": matched_energy,
        "feasible": feasible,
        "note": note,
    }

