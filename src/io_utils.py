import pandas as pd
import numpy as np
from typing import IO

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
        if "timestamp" not in df.columns:
            raise ValueError(f"[HH] Missing 'timestamp' in sheet {sheet}")
        power_cols = [c for c in df.columns if c.lower().startswith("power") or "kW" in c]
        if not power_cols:
            raise ValueError(f"[HH] Missing power (kW) column in sheet {sheet}")
        pcol = power_cols[0]
        df = df[["timestamp", pcol]].rename(columns={pcol: "power_kW_15min"})
        df = _ensure_ts_local(df, "timestamp")
        df["kWh"] = df["power_kW_15min"].astype(float) * 0.25
        df["ts_h"] = df["timestamp"].dt.floor("h")
        hourly = df.groupby("ts_h", as_index=False)["kWh"].sum().rename(columns={"ts_h": "timestamp"})
        hourly["household_id"] = sheet
        recs.append(hourly[["timestamp", "household_id", "kWh"]])
    return pd.concat(recs, ignore_index=True).sort_values(["timestamp", "household_id"])

def read_shops_excel(file: IO) -> pd.DataFrame:
    x = pd.ExcelFile(file)
    recs = []
    for sheet in x.sheet_names:
        df = x.parse(sheet)
        df.columns = [c.strip() for c in df.columns]
        if "timestamp" not in df.columns:
            raise ValueError(f"[SHOP] Missing 'timestamp' in sheet {sheet}")
        cands = [c for c in df.columns if c.lower() == "activeenergy_generale".lower()]
        if not cands:
            raise ValueError(f"[SHOP] Missing 'ActiveEnergy_Generale' in sheet {sheet}")
        ecol = cands[0]
        df = df[["timestamp", ecol]].rename(columns={ecol: "kWh_15"})
        df = _ensure_ts_local(df, "timestamp")
        df["ts_h"] = df["timestamp"].dt.floor("h")
        hourly = df.groupby("ts_h", as_index=False)["kWh_15"].sum().rename(columns={"ts_h": "timestamp", "kWh_15": "kWh"})
        hourly["shop_id"] = sheet
        recs.append(hourly[["timestamp", "shop_id", "kWh"]])
    return pd.concat(recs, ignore_index=True).sort_values(["timestamp", "shop_id"])

def read_pv_json(file: IO) -> pd.DataFrame:
    df = pd.read_json(file)
    if "records" in df.columns:
        df = pd.DataFrame(df["records"].tolist())
    if "timestamp" not in df.columns or "energy_kWh_per_kWp" not in df.columns:
        raise ValueError("[PV] JSON must contain 'records' with 'timestamp' and 'energy_kWh_per_kWp'")
    df = df[["timestamp", "energy_kWh_per_kWp"]].rename(columns={"energy_kWh_per_kWp": "kWh_per_kWp"})
    df = _ensure_ts_local(df, "timestamp").sort_values("timestamp")
    return df

def read_zonal_csv(file: IO) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "timestamp" not in df.columns:
        raise ValueError("[ZONAL] Missing 'timestamp' column")
    # Accept 'zonal_price' or 'zonal_price (EUR_per_MWh)'
    if "zonal_price (EUR_per_MWh)" not in df.columns:
        if "zonal_price" in df.columns:
            df = df.rename(columns={"zonal_price": "zonal_price (EUR_per_MWh)"})
        else:
            raise ValueError("[ZONAL] Missing 'zonal_price (EUR_per_MWh)' column")
    df = _ensure_ts_local(df, "timestamp")
    return df[["timestamp", "zonal_price (EUR_per_MWh)"]].sort_values("timestamp")

def read_pun_monthly_csv(file: IO) -> pd.DataFrame:
    """
    Expect monthly PUN in €/kWh.
    Columns:
      - timestamp: any date within the month (e.g., '2024-01-01' ... '2024-12-01')
      - PUN (EUR_per_kWh): numeric
    """
    df = pd.read_csv(file)
    if "timestamp" not in df.columns:
        raise ValueError("[PUN] Missing 'timestamp' column")
    # Accept 'PUN (EUR_per_kWh)' or 'PUN' or 'pun_eur_per_kwh'
    pun_col = None
    for c in df.columns:
        if c.strip().lower() in ["pun (eur_per_kwh)", "pun", "pun_eur_per_kwh"]:
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
