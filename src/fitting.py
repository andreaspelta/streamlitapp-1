import numpy as np
import pandas as pd
from typing import Tuple, Dict
from .clustering import hh_shop_cluster
from scipy import stats

# ---------- HELPERS

def _hourly_tables(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """ df: [timestamp, id_col, kWh] hourly. """
    out = df.copy()
    out["cluster"] = out["timestamp"].apply(hh_shop_cluster)
    out["hour"] = out["timestamp"].dt.hour
    out["date"] = out["timestamp"].dt.date
    return out

def _fit_baseline_and_spreads(df: pd.DataFrame, id_col: str):
    """
    Returns:
      mu: DataFrame index=cluster, columns=0..23 (μ_{c,h})
      sigma_lnD: Series index=cluster (σ_lnD(c))
      sigma_resid: DataFrame index=cluster, columns=0..23 (σ_η(c,h))
      diag: dict with diagnostic frames
    """
    df = _hourly_tables(df, id_col)
    # Daily totals per (id,date)
    daily = df.groupby([id_col, "date", "cluster"], as_index=False)["kWh"].sum().rename(columns={"kWh": "E_day"})
    # Median daily per cluster
    medday = daily.groupby("cluster")["E_day"].median()
    medday = medday.replace(0, np.nan)
    # Hour shares
    tmp = df.merge(daily[[id_col, "date", "E_day", "cluster"]], on=[id_col, "date", "cluster"], how="left")
    tmp["share"] = tmp["kWh"] / tmp["E_day"].replace(0, np.nan)
    shares = tmp.groupby(["cluster", "hour"])["share"].median().unstack().fillna(0.0)
    shares = shares.div(shares.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # Baseline μ_{c,h}
    mu = shares.mul(medday, axis=0).fillna(0.0)

    # Day-scalers ln D
    daily2 = tmp.groupby([id_col, "date", "cluster"], as_index=False).agg(E_day=("E_day", "first"))
    daily2["MedDay_c"] = daily2["cluster"].map(medday)
    eps = 1e-9
    daily2["lnD"] = np.log((daily2["E_day"] + eps) / (daily2["MedDay_c"] + eps))
    sigma_lnD = daily2.groupby("cluster")["lnD"].std(ddof=1).fillna(0.25)

    # Hourly residuals
    tmp["MedDay_c"] = tmp["cluster"].map(medday)
    tmp = tmp.merge(mu.reset_index().melt(id_vars="cluster", var_name="hour", value_name="mu_ch"),
                    on=["cluster", "hour"], how="left")
    tmp["D_d"] = (tmp["E_day"] + eps) / (tmp["MedDay_c"] + eps)
    tmp["resid"] = np.log((tmp["kWh"] + eps) / (tmp["mu_ch"] * tmp["D_d"] + eps))
    sigma_resid = tmp.groupby(["cluster", "hour"])["resid"].std(ddof=1).unstack().fillna(0.25)

    diag = {
        "medday": medday.reset_index().rename(columns={"E_day": "MedDay"}),
        "shares": shares.reset_index(),
        "lnD": daily2[["cluster", "lnD"]],
        "resid": tmp[["cluster", "hour", "resid"]],
    }
    return mu, sigma_lnD, sigma_resid, diag

# ---------- PUBLIC FIT FUNCTIONS

def fit_households(hh_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    mu, sigma_lnD, sigma_resid, diag = _fit_baseline_and_spreads(hh_df, "household_id")
    fit = {"mu": mu, "sigma_lnD": sigma_lnD, "sigma_resid": sigma_resid}
    return fit, diag

def fit_shops(shop_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    mu, sigma_lnD, sigma_resid, diag = _fit_baseline_and_spreads(shop_df, "shop_id")
    # zero probability per hour per cluster
    df = _hourly_tables(shop_df, "shop_id")
    df["is_zero"] = (df["kWh"] <= 1e-12).astype(int)
    p_zero = df.groupby(["cluster", df["timestamp"].dt.hour])["is_zero"].mean().unstack().fillna(0.0)
    fit = {"mu": mu, "sigma_lnD": sigma_lnD, "sigma_resid": sigma_resid, "p_zero": p_zero}
    return fit, diag
