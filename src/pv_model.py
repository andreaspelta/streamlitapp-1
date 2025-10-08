import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy import stats
from .clustering import season_of

def fit_pv_optionB_v3(pv_perkwp: pd.DataFrame) -> Tuple[Dict, Dict]:
    df = pv_perkwp.copy()
    df = df[df["kWh_per_kWp"].notna() & (df["kWh_per_kWp"] > 0)]
    if df.empty:
        raise ValueError("No positive PV per-kWp hours available after filtering.")
    df["season"] = df["timestamp"].apply(season_of)
    df["hour"] = df["timestamp"].dt.hour
    # Envelope per season
    env = df.groupby(["season", "hour"])["kWh_per_kWp"].mean().unstack().fillna(0.0)
    # Zero out nights
    def norm_env(row):
        m = row.max()
        thr = 0.1 * m if m > 0 else 0
        row = row.where(row >= thr, 0.0)
        s = row.sum()
        return row / s if s > 0 else row
    S = env.apply(norm_env, axis=1)

    # Daily totals per season
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby(["season", "date"])["kWh_per_kWp"].sum().reset_index()
    diag = {"daily_totals": daily.copy()}

    # For each season, fit log-logistic (Fisk) to normalized M_d
    LL = {}
    for s, grp in daily.groupby("season"):
        Y = grp["kWh_per_kWp"].values
        med = np.median(Y) if np.isfinite(Y).any() else 1.0
        M = Y / (med if med > 0 else 1.0)
        # Fix loc=0 for stability
        c, loc, scale = stats.fisk.fit(M, floc=0)
        LL[s] = {"c": float(max(c, 0.1)), "scale": float(max(scale, 1e-6))}

    # Simple 2-state (Clear/Cloud) Markov estimated via threshold on daily totals
    markov = {}
    for s, grp in daily.groupby("season"):
        Y = grp["kWh_per_kWp"].values
        med = np.median(Y)
        states = (Y >= med).astype(int)  # 1=Clear, 0=Cloud
        if len(states) < 2:
            P = np.array([[0.7, 0.3], [0.3, 0.7]])
        else:
            c2c = 0; c2l=0; l2c=0; l2l=0
            for a,b in zip(states[:-1], states[1:]):
                if a==1 and b==1: c2c+=1
                if a==1 and b==0: c2l+=1
                if a==0 and b==1: l2c+=1
                if a==0 and b==0: l2l+=1
            def frac(x,y): return x/(y if y>0 else 1)
            P = np.array([[frac(l2l, l2l+l2c), frac(l2c, l2l+l2c)],
                          [frac(c2l, c2l+c2c), frac(c2c, c2l+c2c)]])
        # Beta clearness from normalized hourly profile ratio
        diag_hour = df[df["season"]==s].copy()
        diag_hour["date"] = diag_hour["timestamp"].dt.date
        env_sum = S.loc[s].sum()
        if env_sum <= 0: env_sum = 1.0
        # approximate intra-day clearness; keep in (0,1)
        def clearness(x):
            h = x["hour"]; y = x["kWh_per_kWp"]; base = S.loc[s, h]
            denom = base * (y*0 + 1.0)  # base
            r = y / (denom if denom>0 else 1.0)
            return min(max(r, 1e-6), 1-1e-6)
        diag_hour["rho"] = diag_hour.apply(clearness, axis=1)
        rho = diag_hour["rho"].clip(1e-6, 1-1e-6)
        m = float(np.mean(rho))
        v = float(np.var(rho))
        if v <= 0 or m*(1-m) <= v:
            a, b = 5.0, 5.0
        else:
            common = m*(1-m)/v - 1
            a = max(m*common, 0.5)
            b = max((1-m)*common, 0.5)
        markov[s] = {"P": P.tolist(), "beta": {"alpha": float(a), "beta": float(b)}}

    fit = {"S": S, "loglogistic": LL, "markov": markov}
    diag["envelope"] = S.reset_index()
    diag["LL"] = LL
    diag["markov"] = markov
    return fit, diag
