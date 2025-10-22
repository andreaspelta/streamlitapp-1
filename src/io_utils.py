import io
from typing import IO, Iterable

import numpy as np
import pandas as pd

from pandas.api import types as pdt

from .clustering import CLUSTERS


def _to_positive_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric, keeping only strictly positive values."""

    values = pd.to_numeric(series, errors="coerce")
    values = values.where(values > 0, np.nan)
    return values


def _to_non_negative_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric, keeping zeros while dropping negatives."""

    values = pd.to_numeric(series, errors="coerce")
    values = values.where(values >= 0, np.nan)
    return values

TZ = "Europe/Rome"


def _reset_file_like(file: IO) -> IO:
    """Return a seekable buffer for file-like uploads (Streamlit, Flask, etc.)."""

    if hasattr(file, "seek"):
        try:
            file.seek(0)
            return file
        except (OSError, io.UnsupportedOperation):
            pass

    data = file.read()
    if isinstance(data, (bytes, bytearray)):
        return io.BytesIO(data)
    return io.BytesIO(bytes(data))


def _ensure_ts_local(df: pd.DataFrame, col="timestamp") -> pd.DataFrame:
    """Ensure a timestamp column is timezone-aware in Europe/Rome.

    Supports three input shapes:
    1. Already datetimelike values (naive or tz-aware).
    2. Strings with explicit timezone offsets (e.g. ``+0100``).
    3. Plain strings / Excel datetimes without timezone.

    For naive timestamps we tolerate DST transitions by inferring
    ambiguous clocks and shifting nonexistent times forward by one hour.
    """

    series = df[col]

    if pdt.is_datetime64_any_dtype(series):
        ts = series
    else:
        # Detect explicit timezone offsets (e.g. +0100, +02:00, Z)
        as_str = series.astype(str)
        has_tz_info = as_str.str.contains(r"(?:[+-][0-9]{2}:?[0-9]{2}|Z)$", regex=True, na=False)
        if has_tz_info.any():
            ts = pd.to_datetime(series, utc=True, errors="coerce")
            if ts.isna().any():
                raise ValueError("Invalid timestamp with timezone offset detected.")
            ts = ts.dt.tz_convert(TZ)
            df[col] = ts
            return df
        ts = pd.to_datetime(series, errors="coerce")
        if ts.isna().any():
            raise ValueError("Invalid timestamp detected.")

    if ts.dt.tz is None:
        dup_first = ts.duplicated(keep="first")
        dup_last = ts.duplicated(keep="last")
        if dup_first.any() or dup_last.any():
            ambiguous_flags = np.zeros(len(ts), dtype=bool)
            ambiguous_flags[dup_last.to_numpy()] = True  # first occurrence → DST
        else:
            ambiguous_flags = False
        ts = ts.dt.tz_localize(
            TZ,
            nonexistent="shift_forward",
            ambiguous=ambiguous_flags,
        )
    else:
        ts = ts.dt.tz_convert(TZ)

    df[col] = ts
    return df

def read_households_excel(file: IO) -> pd.DataFrame:
    buffer = _reset_file_like(file)
    xls = pd.ExcelFile(buffer)
    frames = []

    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        df.columns = [c.strip() for c in df.columns]
        if "timestamp" not in df.columns:
            raise ValueError(f"[HH] Missing 'timestamp' in sheet {sheet}")
        power_cols = [c for c in df.columns if c.lower().startswith("power") or "kw" in c.lower()]
        if not power_cols:
            raise ValueError(f"[HH] Missing power (kW) column in sheet {sheet}")
        pcol = power_cols[0]
        df = df[["timestamp", pcol]].rename(columns={pcol: "power_kW_15min"})
        df["household_id"] = sheet
        frames.append(df)

    if not frames:
        raise ValueError("[HH] Excel workbook is empty")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = _ensure_ts_local(df_all, "timestamp")
    df_all["kWh_15"] = _to_positive_numeric(df_all["power_kW_15min"]) * 0.25
    df_all = df_all.dropna(subset=["kWh_15"])
    df_all["timestamp"] = df_all["timestamp"].dt.tz_convert("UTC").dt.floor("h").dt.tz_convert(TZ)

    hourly = (
        df_all.groupby(["timestamp", "household_id"], as_index=False)["kWh_15"]
        .sum(min_count=1)
        .rename(columns={"kWh_15": "kWh"})
    )
    hourly = hourly[hourly["kWh"] > 0]
    return hourly.sort_values(["timestamp", "household_id"]).reset_index(drop=True)

def read_shops_excel(file: IO) -> pd.DataFrame:
    buffer = _reset_file_like(file)
    xls = pd.ExcelFile(buffer)
    frames = []

    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        df.columns = [c.strip() for c in df.columns]
        if "timestamp" not in df.columns:
            raise ValueError(f"[SHOP] Missing 'timestamp' in sheet {sheet}")
        cands = [c for c in df.columns if c.strip().lower() == "activeenergy_generale"]
        if not cands:
            raise ValueError(f"[SHOP] Missing 'ActiveEnergy_Generale' in sheet {sheet}")
        ecol = cands[0]
        df = df[["timestamp", ecol]].rename(columns={ecol: "kWh_15"})
        df["shop_id"] = sheet
        frames.append(df)

    if not frames:
        raise ValueError("[SHOP] Excel workbook is empty")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = _ensure_ts_local(df_all, "timestamp")
    df_all["kWh_15"] = _to_positive_numeric(df_all["kWh_15"])
    df_all = df_all.dropna(subset=["kWh_15"])
    df_all["timestamp"] = df_all["timestamp"].dt.tz_convert("UTC").dt.floor("h").dt.tz_convert(TZ)

    hourly = (
        df_all.groupby(["timestamp", "shop_id"], as_index=False)["kWh_15"]
        .sum(min_count=1)
        .rename(columns={"kWh_15": "kWh"})
    )
    hourly = hourly[hourly["kWh"] > 0]
    return hourly.sort_values(["timestamp", "shop_id"]).reset_index(drop=True)


def _clean_columns(columns: Iterable) -> list[str]:
    cleaned = []
    for col in columns:
        if isinstance(col, str):
            cleaned.append(col.strip())
        elif pd.isna(col):
            cleaned.append("")
        else:
            cleaned.append(str(col).strip())
    return cleaned


def _read_excel_with_flexible_header(
    buffer: IO, required_cols: set[str], headers: Iterable[int] = (0, 1, 2, 3, 4)
) -> pd.DataFrame:
    """Attempt to read an Excel file trying multiple header rows."""

    last_df = None
    for header in headers:
        try:
            buffer.seek(0)
        except (OSError, io.UnsupportedOperation):
            pass
        df = pd.read_excel(buffer, header=header)
        df.columns = _clean_columns(df.columns)
        last_df = df
        if required_cols.issubset(df.columns):
            return df
    return last_df if last_df is not None else pd.DataFrame()


def read_pv_excel(file: IO) -> pd.DataFrame:
    """Read deterministic PV template (timestamp, kWh_per_kWp)."""

    buffer = _reset_file_like(file)
    required_cols = {"timestamp", "kWh_per_kWp"}
    df = _read_excel_with_flexible_header(buffer, required_cols)
    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        raise ValueError(
            f"[PV] Excel must contain columns: {', '.join(sorted(missing))}"
        )
    df = df[["timestamp", "kWh_per_kWp"]]
    df = _ensure_ts_local(df, "timestamp").sort_values("timestamp")
    df["kWh_per_kWp"] = _to_non_negative_numeric(df["kWh_per_kWp"])
    if df["kWh_per_kWp"].isna().any():
        raise ValueError("[PV] Some hourly kWh_per_kWp entries are missing or invalid.")
    return df.reset_index(drop=True)


def _read_cluster_template(file: IO, label: str) -> pd.DataFrame:
    buffer = _reset_file_like(file)
    required_cols = {"hour", *CLUSTERS}
    df = _read_excel_with_flexible_header(buffer, required_cols)
    if not required_cols.issubset(df.columns):
        missing = sorted(required_cols.difference(df.columns))
        raise ValueError(
            f"[{label}] Missing columns: {', '.join(missing)}"
        )
    if "hour" not in df.columns:
        raise ValueError(f"[{label}] Missing 'hour' column")
    missing = [c for c in CLUSTERS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{label}] Missing cluster columns: {', '.join(missing)}"
        )
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype(pd.Int64Dtype())
    if df["hour"].isna().any():
        raise ValueError(f"[{label}] Hour column must contain integers 0-23")
    df = df.dropna(subset=["hour"])
    hours = df["hour"].astype(int)
    if set(hours) != set(range(24)):
        raise ValueError(f"[{label}] Provide all 24 hours (0-23) exactly once")
    med = (
        df.set_index("hour")[CLUSTERS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    return med.reindex(range(24))


def read_household_template(file: IO) -> pd.DataFrame:
    """Read deterministic household cluster medians (24×12)."""

    return _read_cluster_template(file, "HH")


def read_shop_template(file: IO) -> pd.DataFrame:
    """Read deterministic shop cluster medians (24×12)."""

    return _read_cluster_template(file, "SHOP")

def read_zonal_excel(file: IO) -> pd.DataFrame:
    buffer = _reset_file_like(file)
    df = _read_excel_with_flexible_header(buffer, {"timestamp"})
    if "timestamp" not in df.columns:
        raise ValueError("[ZONAL] Missing 'timestamp' column")

    price_col = None
    for col in df.columns:
        norm = col.strip().lower()
        if norm in {
            "zonal_price (eur_per_mwh)",
            "zonal_price",
            "price (eur_per_mwh)",
            "price_eur_per_mwh",
        }:
            price_col = col
            break
    if price_col is None:
        raise ValueError("[ZONAL] Missing 'zonal_price (EUR_per_MWh)' column")

    df = df[["timestamp", price_col]].rename(
        columns={price_col: "zonal_price (EUR_per_MWh)"}
    )
    df = _ensure_ts_local(df, "timestamp")
    df["zonal_price (EUR_per_MWh)"] = _to_non_negative_numeric(
        df["zonal_price (EUR_per_MWh)"]
    )
    return df.dropna(subset=["zonal_price (EUR_per_MWh)"]).sort_values("timestamp")

def read_zonal_csv(file: IO) -> pd.DataFrame:
    """Backward compatible wrapper for CSV uploads (deprecated)."""

    return read_zonal_excel(file)


def read_pun_monthly_excel(file: IO) -> pd.DataFrame:
    """
    Expect monthly PUN in €/kWh.
    Columns:
      - timestamp: any date within the month (e.g., '2024-01-01' ... '2024-12-01')
      - PUN (EUR_per_kWh): numeric
    """
    buffer = _reset_file_like(file)
    df = _read_excel_with_flexible_header(buffer, {"timestamp"})
    if "timestamp" not in df.columns:
        raise ValueError("[PUN] Missing 'timestamp' column")

    pun_col = None
    for c in df.columns:
        if c.strip().lower() in {"pun (eur_per_kwh)", "pun", "pun_eur_per_kwh"}:
            pun_col = c
            break
    if pun_col is None:
        raise ValueError("[PUN] Missing 'PUN (EUR_per_kWh)' column")

    df = df[["timestamp", pun_col]].rename(columns={pun_col: "PUN (EUR_per_kWh)"})
    df = _ensure_ts_local(df, "timestamp")
    # Reduce to one row per (year, month) — if multiple entries per month, take the last
    df["ym"] = df["timestamp"].dt.to_period("M")
    df = df.sort_values("timestamp").groupby("ym", as_index=False).last()
    # Rebuild month representative timestamp as first of month at 00:00
    df["timestamp"] = df["ym"].dt.to_timestamp().dt.tz_localize(TZ)
    df = df[["timestamp", "PUN (EUR_per_kWh)"]].sort_values("timestamp")
    return df


def read_pun_monthly_csv(file: IO) -> pd.DataFrame:
    """Backward compatible wrapper for CSV uploads (deprecated)."""

    return read_pun_monthly_excel(file)

def ensure_price_hours(zonal: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Build the hourly calendar from hourly zonal timestamps. Hard-fail on missing hours.
    """
    h = pd.DatetimeIndex(zonal["timestamp"]).tz_convert(TZ).sort_values()
    # Expected to be hourly continuous (DST handled by local tz)
    idx = pd.date_range(start=h[0], end=h[-1], freq="h", tz=TZ)
    missing = idx.difference(h)
    if len(missing) > 0:
        raise ValueError(f"[Calendar] Zonal has missing hours (hard fail). Example: {list(missing[:5])} ...")
    return h

def expand_monthly_pun_to_hours(pun_monthly: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Expand monthly PUN (€/kWh) to an hourly dataframe aligned to 'hours'.
    Output columns: timestamp, PUN (EUR_per_kWh)
    """
    df = pd.DataFrame({"timestamp": hours})
    month_map = pun_monthly.set_index(pun_monthly["timestamp"].dt.to_period("M"))["PUN (EUR_per_kWh)"]
    df["ym"] = df["timestamp"].dt.to_period("M")
    df["PUN (EUR_per_kWh)"] = df["ym"].map(month_map).astype(float)
    if df["PUN (EUR_per_kWh)"].isna().any():
        # Fill by forward/backward fill (if edges missing), else hard fail
        df["PUN (EUR_per_kWh)"] = df["PUN (EUR_per_kWh)"].ffill().bfill()
        if df["PUN (EUR_per_kWh)"].isna().any():
            raise ValueError("[PUN] Some months have no PUN (€/kWh) defined; cannot expand to hours.")
    return df[["timestamp", "PUN (EUR_per_kWh)"]]
