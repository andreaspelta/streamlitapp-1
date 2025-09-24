import numpy as np
import pandas as pd
from typing import Dict, List, Any
from .clustering import hh_shop_cluster, season_of
from scipy import stats

# ---------- Sampling helpers (unchanged) ----------
def sample_hh_like(hours: pd.DatetimeIndex, fit: Dict, rng: np.random.Generator) -> np.ndarray:
    clusters = pd.Series([hh_shop_cluster(ts) for ts in hours], index=hours)
    hours_of_day = pd.Series([ts.hour for ts in hours], index=hours)

    mu = fit["mu"]; s_lnD = fit["sigma_lnD"]; s_res = fit["sigma_resid"]
    dates = pd.Series([ts.date() for ts in hours], index=hours)
    days = pd.unique(dates)
    day_cluster = pd.Series([hh_shop_cluster(pd.Timestamp(str(d))) for d in days], index=days)
    lnD = {d: rng.normal(loc=0.0, scale=float(s_lnD.get(day_cluster[d], 0.25))) for d in days}
    lnD = pd.Series(lnD)

    out = []
    for ts in hours:
        c = clusters[ts]; h = int(hours_of_day[ts])
        mu_ch = float(mu.loc[c, h]) if (c in mu.index and h in mu.columns) else 0.0
        s = float(s_res.loc[c, h]) if (c in s_res.index and h in s_res.columns) else 0.25
        D = float(np.exp(lnD[ts.date()]))
        e = rng.normal(scale=s)
        out.append(max(0.0, mu_ch * D * np.exp(e)))
    return np.array(out, dtype=float)

def sample_shop_like(hours: pd.DatetimeIndex, fit: Dict, rng: np.random.Generator) -> np.ndarray:
    clusters = pd.Series([hh_shop_cluster(ts) for ts in hours], index=hours)
    hours_of_day = pd.Series([ts.hour for ts in hours], index=hours)
    mu = fit["mu"]; s_lnD = fit["sigma_lnD"]; s_res = fit["sigma_resid"]; p0 = fit["p_zero"]
    dates = pd.Series([ts.date() for ts in hours], index=hours)
    days = pd.unique(dates)
    day_cluster = pd.Series([hh_shop_cluster(pd.Timestamp(str(d))) for d in days], index=days)
    lnD = {d: rng.normal(loc=0.0, scale=float(s_lnD.get(day_cluster[d], 0.25))) for d in days}
    lnD = pd.Series(lnD)

    out = []
    for ts in hours:
        c = clusters[ts]; h = int(hours_of_day[ts])
        pz = float(p0.loc[c, h]) if (c in p0.index and h in p0.columns) else 0.0
        if rng.uniform() < pz:
            out.append(0.0); continue
        mu_ch = float(mu.loc[c, h]) if (c in mu.index and h in mu.columns) else 0.0
        s = float(s_res.loc[c, h]) if (c in s_res.index and h in s_res.columns) else 0.25
        D = float(np.exp(lnD[ts.date()])) ; e = rng.normal(scale=s)
        out.append(max(0.0, mu_ch * D * np.exp(e)))
    return np.array(out, dtype=float)

def sample_pv(hours: pd.DatetimeIndex, pv_fit: Dict, kWp: float, rng: np.random.Generator) -> np.ndarray:
    S = pv_fit["S"]; LL = pv_fit["loglogistic"]; MK = pv_fit["markov"]
    seasons = pd.Series([season_of(ts) for ts in hours], index=hours)
    dates = pd.Series([ts.date() for ts in hours], index=hours)
    uniq_dates = pd.unique(dates)
    day_state = {}
    M_day = {}
    prev_state = {}
    for d in uniq_dates:
        s = season_of(pd.Timestamp(str(d)))
        P = np.array(MK.get(s, {}).get("P", [[0.7,0.3],[0.3,0.7]]))
        if s not in prev_state:
            p01 = P[0,1]; p10 = P[1,0]
            pi_clear = p10/(p01+p10+1e-9)
            stt = 1 if np.random.rand() < pi_clear else 0
        else:
            stt = 1 if np.random.rand() < P[prev_state[s], 1] else 0
        prev_state[s] = stt
        day_state[d] = stt
        pars = LL.get(s, {"c":2.0,"scale":1.0})
        c = max(float(pars.get("c",2.0)), 0.1)
        sc = max(float(pars.get("scale",1.0)), 1e-6)
        M_day[d] = stats.fisk.rvs(c, loc=0, scale=sc, random_state=np.random.default_rng())
    out = []
    for ts in hours:
        s = seasons[ts]; h = ts.hour
        base = float(S.loc[s, h]) if (s in S.index and h in S.columns) else 0.0
        d = ts.date()
        beta = MK.get(s, {}).get("beta", {"alpha":5.0,"beta":5.0})
        a = max(float(beta.get("alpha",5.0)), 0.5); b = max(float(beta.get("beta",5.0)), 0.5)
        cl = np.random.default_rng().beta(a, b)
        eps = np.random.default_rng().normal(scale=0.1)
        per_kwp = base * M_day[d] * cl * np.exp(eps)
        out.append(max(0.0, kWp * per_kwp))
    return np.array(out, dtype=float)

# ---------- Matching ----------
def equal_level_fill(supply: float, demands: np.ndarray) -> (np.ndarray, float):
    n = len(demands)
    if n == 0 or supply <= 0:
        return np.zeros(n), supply
    d = demands.copy().astype(float)
    matched = np.zeros(n)
    active = [i for i in range(n) if d[i] > 1e-12]
    s = float(supply)
    while s > 1e-12 and len(active) > 0:
        a = s / len(active)
        used = 0.0
        new_active = []
        for idx in active:
            take = min(a, d[idx])
            matched[idx] += take
            d[idx] -= take
            used += take
            if d[idx] > 1e-12:
                new_active.append(idx)
        s -= used
        active = new_active
        if used < 1e-12: break
    return matched, s

# ---------- Fee params via setter ----------
price_params = {"f_pros": 0.0, "f_hh": 0.0, "f_shop": 0.0, "platform_fixed": 0.0}
def set_fee_params(f_pros, f_hh, f_shop, platform_fixed):
    price_params.update({"f_pros": float(f_pros), "f_hh": float(f_hh), "f_shop": float(f_shop), "platform_fixed": float(platform_fixed)})

def layer_fee(which: str) -> float:
    if which == "pros": return price_params["f_pros"]
    if which == "hh": return price_params["f_hh"]
    if which == "shop": return price_params["f_shop"]
    return 0.0

# ---------- Main MC ----------
def run_monte_carlo(
    hours: pd.DatetimeIndex,
    hh_fit: Dict, shop_fit: Dict, pv_fit: Dict,
    kwp_map: Dict[str, float], mapping: Dict[str, list],
    prosumer_ids: list, hh_ids: list, shop_ids: list,
    zonal: pd.DataFrame, pun_hourly_kwh: pd.DataFrame,
    price_layer, S: int, seed: int
) -> Dict[str, Any]:

    rng = np.random.default_rng(seed)
    # Align hourly price frames
    z = zonal.set_index("timestamp").reindex(hours)["zonal_price (EUR_per_MWh)"].ffill()
    p = pun_hourly_kwh.set_index("timestamp").reindex(hours)["PUN (EUR_per_KWh)".replace("KWh","kWh")].ffill()
    price_df = pd.DataFrame({"timestamp": hours, "zonal": z.values, "pun_kwh": p.values})

    nP = len(prosumer_ids); nH = len(hh_ids); nK = len(shop_ids)

    matched_hh = np.zeros(S); matched_shop = np.zeros(S)
    import_hh = np.zeros(S); import_shop = np.zeros(S)
    export = np.zeros(S)
    pros_rev = np.zeros(S); cons_cost = np.zeros(S)
    platform_gap = np.zeros(S); pv_gen = np.zeros(S); cons_total = np.zeros(S)
    baseline_cost = np.zeros(S)

    fees = 12.0 * (nP * price_params["f_pros"] + nH * price_params["f_hh"] + nK * price_params["f_shop"])
    fixed = 12.0 * price_params["platform_fixed"]

    for sidx in range(S):
        r = np.random.default_rng(rng.integers(0, 2**32-1))

        # >>> Key change: store as pandas Series indexed by 'hours'
        pros_load = {pid: pd.Series(sample_hh_like(hours, hh_fit, r), index=hours) for pid in prosumer_ids}
        pros_gen  = {pid: pd.Series(sample_pv(hours, pv_fit, float(kwp_map.get(pid, 0.0)), r), index=hours) for pid in prosumer_ids}
        hh_load   = {hid: pd.Series(sample_hh_like(hours, hh_fit, r), index=hours) for hid in hh_ids}
        shop_load = {kid: pd.Series(sample_shop_like(hours, shop_fit, r), index=hours) for kid in shop_ids}

        m_hh = 0.0; m_shop = 0.0
        imp_hh = 0.0; imp_shop = 0.0
        exp_total = 0.0
        rev = 0.0; cost = 0.0; gap_eur = 0.0
        pv_sum = 0.0; cons_sum = 0.0; base_cost = 0.0

        for _, row in price_df.iterrows():
            ts = row["timestamp"]; zon = float(row["zonal"]); pun_kwh = float(row["pun_kwh"])
            layer = price_layer(zon, pun_kwh)

            # HH and SHOP demands at this hour (now timestamp-safe)
            U_hh = np.array([hh_load[h].loc[ts] for h in hh_ids], dtype=float)
            U_sh = np.array([shop_load[k].loc[ts] for k in shop_ids], dtype=float)

            # Baseline retail-only cost (for savings): uses PUN-based retail
            base_cost += (U_hh.sum() * layer["ret_HH"] + U_sh.sum() * layer["ret_SH"])

            # Prosumer surpluses and self-consumption
            residuals = []
            for pid in prosumer_ids:
                L = float(pros_load[pid].loc[ts])
                G = float(pros_gen[pid].loc[ts])
                pv_sum += G
                SC = min(L, G)  # self-consumption (not explicitly used in pricing layer above)
                surplus = max(G - L, 0.0)

                # Designated HH first (ELWF)
                lst = mapping.get(pid, [])
                if len(lst) > 0 and surplus > 0:
                    idxs = [hh_ids.index(h) for h in lst if h in hh_ids]
                    need = U_hh[idxs].copy()
                    taken = 0.0
                    if need.size:
                        alloc, resid = equal_level_fill(surplus, need)
                        taken = float(alloc.sum())
                        U_hh[idxs] -= alloc
                        surplus = resid
                    m_hh += taken
                    rev += taken * layer["Ppros_HH"]
                    cost += taken * layer["Pcons_HH"]
                    gap_eur += (taken * (1 - layer["loss_factor"])) * layer["gap_HH"]

                residuals.append(surplus)

            # Pool residual to SHOPS
            pool = float(np.sum(residuals))
            if pool > 0 and len(U_sh) > 0:
                alloc_s, resid_pool = equal_level_fill(pool, U_sh.copy())
                taken_s = float(alloc_s.sum())
                U_sh -= alloc_s
                m_shop += taken_s
                rev += taken_s * layer["Ppros_SH"]
                cost += taken_s * layer["Pcons_SH"]
                gap_eur += (taken_s * (1 - layer["loss_factor"])) * layer["gap_SH"]
                pool = resid_pool

            # Imports for HH/SHOP at PUN-based retail
            imp_hh += U_hh.sum()
            imp_shop += U_sh.sum()
            cost += U_hh.sum() * layer["ret_HH"] + U_sh.sum() * layer["ret_SH"]

            # Export of residual pool
            exp_total += pool
            rev += pool * layer["P_unm"]

            cons_sum += (U_hh.sum() + m_hh) + (U_sh.sum() + m_shop)  # running aggregate (approx.)

        matched_hh[sidx] = m_hh; matched_shop[sidx] = m_shop
        import_hh[sidx] = imp_hh; import_shop[sidx] = imp_shop
        export[sidx] = exp_total
        pros_rev[sidx] = rev
        cons_cost[sidx] = cost
        platform_gap[sidx] = gap_eur
        pv_gen[sidx] = pv_sum
        cons_total[sidx] = cons_sum
        baseline_cost[sidx] = base_cost

    platform_margin = platform_gap + fees - fixed

    return dict(
        matched_hh=matched_hh, matched_shop=matched_shop,
        import_hh=import_hh, import_shop=import_shop,
        export=export,
        pros_rev=pros_rev, cons_cost=cons_cost,
        platform_gap=platform_gap, platform_fees=np.full_like(platform_gap, fees),
        platform_fixed=np.full_like(platform_gap, fixed),
        platform_margin=platform_margin,
        pv_gen=pv_gen, cons_total=cons_total, cons_baseline=baseline_cost,
    )
