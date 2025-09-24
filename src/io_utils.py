import io
import pandas as pd
import numpy as np
from typing import IO, Dict
from datetime import timedelta

TZ = "Europe/Rome"

def _ensure_ts_local(df: pd.DataFrame, col="timestamp") -> pd.DataFrame:
    ts = pd.to_datetime(df[col])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(TZ)
    df[col] = ts.dt.tz_convert(TZ)
    return df

def read_households_excel(file: IO) -> pd.DataFrame:
    x = pd.ExcelFile(file)
    recs = []
    for sheet in x.sheet_names:
        df = x.parse(sheet)
        df.columns = [c.strip() for c in df.columns]
        # Expect columns: timestamp, power_kW_15min
        if "timestamp" not in df.columns:
            raise ValueError(f"[HH] Missing 'timestamp' in sheet {sheet}")
        # find power col
        power_cols = [c for c in df.columns if c.lower().startswith("power") or "kW" in c]
        if not power_cols:
            raise ValueError(f"[HH] Missing power (kW) column in sheet {sheet}")
        pcol = power_cols[0]
        df = df[["timestamp", pcol]].rename(columns={pcol: "power_kW_15min"})
        df = _ensure_ts_local(df, "timestamp")
        # 15-min → kWh
        df["kWh"] = df["power_kW_15min"].astype(float) * 0.25
        # hourly aggregate
        df["ts_h"] = df["timestamp"].dt.floor("H")
        hourly = df.groupby("ts_h", as_index=False)["kWh"].sum()
        hourly = hourly.rename(columns={"ts_h": "timestamp"})
        hourly["household_id"] = sheet
        recs.append(hourly[["timestamp", "household_id", "kWh"]])
    out = pd.concat(recs, ignore_index=True).sort_values(["timestamp", "household_id"])
    return out

def read_shops_excel(file: IO) -> pd.DataFrame:
    x = pd.ExcelFile(file)
    recs = []
    for sheet in x.sheet_names:
        df = x.parse(sheet)
        df.columns = [c.strip() for c in df.columns]
        if "timestamp" not in df.columns:
            raise ValueError(f"[SHOP] Missing 'timestamp' in sheet {sheet}")
        # kWh per 15-min in 'ActiveEnergy_Generale' (case-insensitive accepted)
        cands = [c for c in df.columns if c.lower() == "activeenergy_generale".lower()]
        if not cands:
            raise ValueError(f"[SHOP] Missing 'ActiveEnergy_Generale' in sheet {sheet}")
        ecol = cands[0]
        df = df[["timestamp", ecol]].rename(columns={ecol: "kWh_15"})
        df = _ensure_ts_local(df, "timestamp")
        df["ts_h"] = df["timestamp"].dt.floor("H")
        hourly = df.groupby("ts_h", as_index=False)["kWh_15"].sum()
        hourly = hourly.rename(columns={"ts_h": "timestamp", "kWh_15": "kWh"})
        hourly["shop_id"] = sheet
        recs.append(hourly[["timestamp", "shop_id", "kWh"]])
    out = pd.concat(recs, ignore_index=True).sort_values(["timestamp", "shop_id"])
    return out

def read_pv_json(file: IO) -> pd.DataFrame:
    df = pd.read_json(file)
    if "records" in df.columns:
        df = pd.DataFrame(df["records"].tolist())
    if "timestamp" not in df.columns or "energy_kWh_per_kWp" not in df.columns:
        raise ValueError("[PV] JSON must contain 'records' with 'timestamp' and 'energy_kWh_per_kWp'")
    df = df[["timestamp", "energy_kWh_per_kWp"]].rename(columns={"energy_kWh_per_kWp": "kWh_per_kWp"})
    df = _ensure_ts_local(df, "timestamp")
    df = df.sort_values("timestamp")
    return df

def read_zonal_csv(file: IO) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "timestamp" not in df.columns or "zonal_price (EUR_per_MWh)" not in df.columns:
        # accept friendly names too
        if "zonal_price" in df.columns:
            df = df.rename(columns={"zonal_price": "zonal_price (EUR_per_MWh)"})
    df = _ensure_ts_local(df, "timestamp")
    return df[["timestamp", "zonal_price (EUR_per_MWh)"]].sort_values("timestamp")

def read_pun_csv(file: IO) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "timestamp" not in df.columns or "PUN (EUR_per_MWh)" not in df.columns:
        if "PUN" in df.columns:
            df = df.rename(columns={"PUN": "PUN (EUR_per_MWh)"})
    df = _ensure_ts_local(df, "timestamp")
    return df[["timestamp", "PUN (EUR_per_MWh)"]].sort_values("timestamp")

def ensure_price_calendar(zonal: pd.DataFrame, pun: pd.DataFrame) -> pd.DatetimeIndex:
    # Union of timestamps; ensure hourly monotonic with no missing hours
    h = pd.DatetimeIndex(sorted(set(zonal["timestamp"]) | set(pun["timestamp"]))).tz_convert(TZ)
    # Check hourly continuity
    diffs = (h[1:] - h[:-1]).unique()
    if len(diffs) > 2 or not all(d in [pd.Timedelta(hours=1)] for d in diffs):
        # Allow DST +/−1h anomalies by checking .freq via reindex fill
        idx = pd.date_range(start=h[0], end=h[-1], freq="H", tz=TZ)
        if len(idx.symmetric_difference(h)) > 0:
            missing = idx.difference(h)
            raise ValueError(f"Price calendar has missing hours (hard fail). Example: {list(missing[:5])} ...")
    return h
