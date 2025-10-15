from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .clustering import hh_shop_cluster, season_of


def _prepare_hour_cache(hours: pd.DatetimeIndex) -> Dict[str, Any]:
    hour_idx = hours.hour.to_numpy(dtype=int)
    day_codes, unique_days = pd.factorize(hours.normalize(), sort=False)
    clusters = np.array([hh_shop_cluster(ts) for ts in hours], dtype=object)
    day_clusters = np.array([hh_shop_cluster(ts) for ts in unique_days], dtype=object)
    seasons = np.array([season_of(ts) for ts in hours], dtype=object)
    day_seasons = np.array([season_of(ts) for ts in unique_days], dtype=object)
    return dict(
        hours=hours,
        hour_idx=hour_idx,
        day_codes=day_codes,
        days=unique_days,
        clusters=clusters,
        day_clusters=day_clusters,
        seasons=seasons,
        day_seasons=day_seasons,
        n_hours=len(hours),
        n_days=len(unique_days),
    )


def _build_cluster_index(fit: Mapping[str, Any], hour_cache: Dict[str, Any]) -> pd.Index:
    sources: List[Iterable[Any]] = [
        hour_cache["clusters"],
        hour_cache["day_clusters"],
    ]
    for key in ("mu", "sigma_resid", "p_zero"):
        val = fit.get(key)
        if isinstance(val, (pd.Series, pd.DataFrame)):
            sources.append(getattr(val, "index", []))
    sigma_lnD = fit.get("sigma_lnD", {})
    if isinstance(sigma_lnD, (pd.Series, pd.DataFrame)):
        sources.append(getattr(sigma_lnD, "index", []))
    else:
        if hasattr(sigma_lnD, "keys"):
            sources.append(list(sigma_lnD.keys()))
        else:
            sources.append(list(sigma_lnD))
    flat = set()
    for src in sources:
        idx = pd.Index(src, dtype=object)
        for item in idx:
            flat.add(str(item) if isinstance(item, pd.Interval) else item)
    return pd.Index(sorted(flat), dtype=object)


def _reindex_hour_matrix(df: Any, index: pd.Index, default: float) -> np.ndarray:
    if df is None:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(df)
    cols = pd.Index(range(24), dtype=int)
    return df.reindex(index=index, columns=cols).fillna(default).to_numpy(dtype=float)


def _prep_hh_shop_template(fit: Mapping[str, Any], hour_cache: Dict[str, Any]) -> Dict[str, Any]:
    cluster_index = _build_cluster_index(fit, hour_cache)
    cluster_codes = pd.Categorical(hour_cache["clusters"], categories=cluster_index).codes
    day_cluster_codes = pd.Categorical(hour_cache["day_clusters"], categories=cluster_index).codes

    mu_matrix = _reindex_hour_matrix(fit.get("mu"), cluster_index, 0.0)
    sigma_res_matrix = _reindex_hour_matrix(fit.get("sigma_resid"), cluster_index, 0.25)

    sigma_lnD = fit.get("sigma_lnD", {})
    if isinstance(sigma_lnD, pd.DataFrame):
        sigma_lnD_series = sigma_lnD.iloc[:, 0]
    elif isinstance(sigma_lnD, pd.Series):
        sigma_lnD_series = sigma_lnD
    else:
        sigma_lnD_series = pd.Series(sigma_lnD)
    sigma_lnD_vals = (
        sigma_lnD_series.reindex(cluster_index)
        .fillna(0.25)
        .to_numpy(dtype=float)
    )

    mu_hour = mu_matrix[cluster_codes, hour_cache["hour_idx"]]
    sigma_res_hour = sigma_res_matrix[cluster_codes, hour_cache["hour_idx"]]
    sigma_lnD_day = sigma_lnD_vals[day_cluster_codes]

    return dict(
        mu_hour=mu_hour,
        sigma_res_hour=sigma_res_hour,
        sigma_lnD_day=sigma_lnD_day,
        cluster_index=cluster_index,
    )


def _prep_shop_template(fit: Mapping[str, Any], hour_cache: Dict[str, Any]) -> Dict[str, Any]:
    base = _prep_hh_shop_template(fit, hour_cache)
    cluster_index = base["cluster_index"]
    p_zero_matrix = _reindex_hour_matrix(fit.get("p_zero"), cluster_index, 0.0)
    cluster_codes = pd.Categorical(hour_cache["clusters"], categories=cluster_index).codes
    p_zero_hour = p_zero_matrix[cluster_codes, hour_cache["hour_idx"]]
    base["p_zero_hour"] = p_zero_hour
    return base


def _prep_pv_template(pv_fit: Mapping[str, Any], hour_cache: Dict[str, Any]) -> Dict[str, Any]:
    season_index = pd.Index(
        sorted(
            set(hour_cache["seasons"]) | set(getattr(pv_fit.get("S", pd.DataFrame()), "index", []))
        )
    )
    S = pv_fit.get("S", pd.DataFrame())
    S_matrix = _reindex_hour_matrix(S, season_index, 0.0)
    season_codes = pd.Categorical(hour_cache["seasons"], categories=season_index).codes
    day_season_codes = pd.Categorical(hour_cache["day_seasons"], categories=season_index).codes
    base_hour = S_matrix[season_codes, hour_cache["hour_idx"]]

    MK = pv_fit.get("markov", {})
    P_default = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)
    beta_default = {"alpha": 5.0, "beta": 5.0}
    P_mats = []
    alpha_vals = []
    beta_vals = []
    for s in season_index:
        mk = MK.get(s, {})
        P = np.array(mk.get("P", P_default), dtype=float)
        if P.shape != (2, 2):
            P = P_default
        P_mats.append(P)
        beta = mk.get("beta", beta_default)
        alpha_vals.append(max(float(beta.get("alpha", 5.0)), 0.5))
        beta_vals.append(max(float(beta.get("beta", 5.0)), 0.5))
    P_mats = np.stack(P_mats, axis=0) if P_mats else np.zeros((0, 2, 2))
    alpha_vals = np.array(alpha_vals, dtype=float)
    beta_vals = np.array(beta_vals, dtype=float)
    alpha_hour = alpha_vals[season_codes]
    beta_hour = beta_vals[season_codes]

    LL = pv_fit.get("loglogistic", {})
    c_vals = []
    scale_vals = []
    for s in season_index:
        pars = LL.get(s, {"c": 2.0, "scale": 1.0})
        c_vals.append(max(float(pars.get("c", 2.0)), 0.1))
        scale_vals.append(max(float(pars.get("scale", 1.0)), 1e-6))
    c_vals = np.array(c_vals, dtype=float)
    scale_vals = np.array(scale_vals, dtype=float)
    c_day = c_vals[day_season_codes]
    scale_day = scale_vals[day_season_codes]

    return dict(
        base_hour=base_hour,
        alpha_hour=alpha_hour,
        beta_hour=beta_hour,
        P_mats=P_mats,
        day_season_codes=day_season_codes,
        c_day=c_day,
        scale_day=scale_day,
        season_index=season_index,
    )


def _sample_loglogistic(rng: np.random.Generator, shape: float, scale: float) -> float:
    shape = max(shape, 1e-6)
    scale = max(scale, 1e-12)
    u = rng.uniform()
    u = np.clip(u, 1e-9, 1 - 1e-9)
    return scale * np.power(u / (1.0 - u), 1.0 / shape)


def _sample_hh_like_cached(
    hour_cache: Dict[str, Any], template: Dict[str, np.ndarray], rng: np.random.Generator, size: int
) -> np.ndarray:
    if size <= 0:
        return np.zeros((0, hour_cache["n_hours"]), dtype=float)

    mu_hour = template["mu_hour"]
    sigma_res_hour = template["sigma_res_hour"]
    sigma_lnD_day = template["sigma_lnD_day"]

    lnD = rng.normal(loc=0.0, scale=sigma_lnD_day, size=(size, hour_cache["n_days"]))
    lnD_hour = lnD[:, hour_cache["day_codes"]]
    residual = rng.normal(scale=sigma_res_hour, size=(size, hour_cache["n_hours"]))
    draws = mu_hour * np.exp(lnD_hour + residual)
    return np.maximum(draws, 0.0)


def _sample_shop_like_cached(
    hour_cache: Dict[str, Any], template: Dict[str, np.ndarray], rng: np.random.Generator, size: int
) -> np.ndarray:
    draws = _sample_hh_like_cached(hour_cache, template, rng, size)
    if size <= 0:
        return draws
    p_zero = template.get("p_zero_hour")
    if p_zero is None:
        return draws
    mask = rng.uniform(size=(size, hour_cache["n_hours"])) < p_zero
    draws[mask] = 0.0
    return draws


def _sample_pv_cached(
    hour_cache: Dict[str, Any], template: Dict[str, Any], kwp: float, rng: np.random.Generator
) -> np.ndarray:
    if kwp <= 0:
        return np.zeros(hour_cache["n_hours"], dtype=float)

    n_hours = hour_cache["n_hours"]
    day_codes = hour_cache["day_codes"]
    season_codes = template["day_season_codes"]

    base_hour = template["base_hour"]
    alpha_hour = template["alpha_hour"]
    beta_hour = template["beta_hour"]
    c_day = template["c_day"]
    scale_day = template["scale_day"]

    P_mats = template.get("P_mats")
    prev_state = np.full(len(template["season_index"]), -1, dtype=int)
    M_day = np.empty(hour_cache["n_days"], dtype=float)
    default_P = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)
    for day_idx, season_code in enumerate(season_codes):
        idx = season_code if 0 <= season_code < len(P_mats) else -1
        P = P_mats[idx] if idx >= 0 else default_P
        state_slot = idx if idx >= 0 else 0
        if state_slot >= len(prev_state):
            prev_state = np.pad(prev_state, (0, state_slot - len(prev_state) + 1), constant_values=-1)
        if prev_state[state_slot] < 0:
            p01 = P[0, 1]
            p10 = P[1, 0]
            denom = p01 + p10
            pi_clear = p10 / denom if denom > 0 else 0.5
            stt = 1 if rng.random() < pi_clear else 0
        else:
            stt = 1 if rng.random() < P[prev_state[state_slot], 1] else 0
        prev_state[state_slot] = stt
        M_day[day_idx] = _sample_loglogistic(rng, c_day[day_idx], scale_day[day_idx])

    beta_draws = rng.beta(alpha_hour, beta_hour)
    ln_noise = rng.normal(loc=0.0, scale=0.1, size=n_hours)
    per_kwp = base_hour * M_day[day_codes] * beta_draws * np.exp(ln_noise)
    return np.maximum(per_kwp * kwp, 0.0)


def sample_hh_like(hours: pd.DatetimeIndex, fit: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
    hour_cache = _prepare_hour_cache(hours)
    template = _prep_hh_shop_template(fit or {}, hour_cache)
    return _sample_hh_like_cached(hour_cache, template, rng, 1)[0]


def sample_shop_like(hours: pd.DatetimeIndex, fit: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
    hour_cache = _prepare_hour_cache(hours)
    template = _prep_shop_template(fit or {}, hour_cache)
    return _sample_shop_like_cached(hour_cache, template, rng, 1)[0]


def sample_pv(hours: pd.DatetimeIndex, pv_fit: Mapping[str, Any], kWp: float, rng: np.random.Generator) -> np.ndarray:
    hour_cache = _prepare_hour_cache(hours)
    template = _prep_pv_template(pv_fit or {}, hour_cache)
    return _sample_pv_cached(hour_cache, template, kWp, rng)


def run_monte_carlo(
    *,
    hours: pd.DatetimeIndex,
    hh_fit: Mapping[str, Any],
    shop_fit: Mapping[str, Any],
    pv_fit: Mapping[str, Any],
    kwp_map: Mapping[str, float],
    mapping: Mapping[str, Sequence[str]],
    prosumer_ids: Sequence[str],
    hh_ids: Sequence[str],
    shop_ids: Sequence[str],
    zonal: pd.DataFrame,
    pun_hourly_kwh: pd.DataFrame,
    price_layer: Callable[[float, float], Mapping[str, float]],
    S: int,
    seed: int,
    price_params: Mapping[str, float],
    equal_level_fill: Callable[[float, np.ndarray], Tuple[np.ndarray, float]],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    hour_cache = _prepare_hour_cache(hours)

    zonal_series = zonal.set_index("timestamp").reindex(hour_cache["hours"])["zonal_price (EUR_per_MWh)"].ffill()
    pun_col = "PUN (EUR_per_KWh)".replace("KWh", "kWh")
    pun_series = pun_hourly_kwh.set_index("timestamp").reindex(hour_cache["hours"])[pun_col].ffill()
    zonal_prices = zonal_series.to_numpy(dtype=float)
    pun_prices = pun_series.to_numpy(dtype=float)

    n_hours = hour_cache["n_hours"]
    nP = len(prosumer_ids)
    nH = len(hh_ids)
    nK = len(shop_ids)

    hh_index = {hid: idx for idx, hid in enumerate(hh_ids)}
    mapping_idx = [
        np.array([hh_index[h] for h in mapping.get(pid, []) if h in hh_index], dtype=int)
        for pid in prosumer_ids
    ]

    hh_template = _prep_hh_shop_template(hh_fit or {}, hour_cache)
    shop_template = _prep_shop_template(shop_fit or {}, hour_cache) if nK else None
    pv_template = _prep_pv_template(pv_fit or {}, hour_cache)

    matched_hh = np.zeros(S)
    matched_shop = np.zeros(S)
    import_hh = np.zeros(S)
    import_shop = np.zeros(S)
    export = np.zeros(S)
    pros_rev = np.zeros(S)
    cons_cost = np.zeros(S)
    platform_gap = np.zeros(S)
    pv_gen = np.zeros(S)
    cons_total = np.zeros(S)
    baseline_cost = np.zeros(S)
    pros_demand = np.zeros(S)
    hh_demand = np.zeros(S)
    shop_demand = np.zeros(S)
    shared_surplus = np.zeros(S)

    fees = 12.0 * (
        nP * price_params.get("f_pros", 0.0)
        + nH * price_params.get("f_hh", 0.0)
        + nK * price_params.get("f_shop", 0.0)
    )
    fixed = 12.0 * price_params.get("platform_fixed", 0.0)

    for sidx in range(S):
        r = np.random.default_rng(rng.integers(0, 2**32 - 1))

        pros_load = (
            _sample_hh_like_cached(hour_cache, hh_template, r, nP)
            if nP
            else np.zeros((0, n_hours), dtype=float)
        )
        pros_gen = (
            np.vstack([
                _sample_pv_cached(hour_cache, pv_template, float(kwp_map.get(pid, 0.0)), r)
                for pid in prosumer_ids
            ])
            if nP
            else np.zeros((0, n_hours), dtype=float)
        )
        hh_load = (
            _sample_hh_like_cached(hour_cache, hh_template, r, nH)
            if nH
            else np.zeros((0, n_hours), dtype=float)
        )
        shop_load = (
            _sample_shop_like_cached(hour_cache, shop_template, r, nK)
            if nK and shop_template is not None
            else np.zeros((0, n_hours), dtype=float)
        )

        m_hh = 0.0
        m_shop = 0.0
        imp_hh = 0.0
        imp_shop = 0.0
        exp_total = 0.0
        rev = 0.0
        cost = 0.0
        gap_eur = 0.0
        pv_sum = 0.0
        cons_sum = 0.0
        base_cost = 0.0
        pros_sum = 0.0
        hh_sum = 0.0
        shop_sum = 0.0
        surplus_shared = 0.0

        for h_idx in range(n_hours):
            zon = zonal_prices[h_idx]
            pun_kwh = pun_prices[h_idx]
            layer = price_layer(zon, pun_kwh)

            hh_full = hh_load[:, h_idx] if nH else np.zeros(0, dtype=float)
            sh_full = shop_load[:, h_idx] if nK else np.zeros(0, dtype=float)
            U_hh = hh_full.copy()
            U_sh = sh_full.copy()

            base_cost += hh_full.sum() * layer["ret_HH"] + sh_full.sum() * layer["ret_SH"]

            hh_sum += hh_full.sum()
            shop_sum += sh_full.sum()

            residuals = []
            for p_idx, pid in enumerate(prosumer_ids):
                L = float(pros_load[p_idx, h_idx]) if nP else 0.0
                G = float(pros_gen[p_idx, h_idx]) if nP else 0.0
                pv_sum += G
                pros_sum += L
                surplus = max(G - L, 0.0)

                idxs = mapping_idx[p_idx] if p_idx < len(mapping_idx) else None
                if idxs is not None and idxs.size > 0 and surplus > 0 and nH:
                    need = U_hh[idxs].copy()
                    if need.size:
                        alloc, resid = equal_level_fill(surplus, need)
                        alloc = np.asarray(alloc, dtype=float)
                        taken = float(alloc.sum())
                        if taken > 0:
                            U_hh[idxs] -= alloc
                            surplus = resid
                            m_hh += taken
                            surplus_shared += taken
                            rev += taken * layer["Ppros_HH"]
                            cost += taken * layer["Pcons_HH"]
                            gap_eur += (taken * (1 - layer["loss_factor"])) * layer["gap_HH"]

                residuals.append(surplus)

            pool = float(np.sum(residuals))
            if pool > 0 and len(U_sh) > 0:
                alloc_s, resid_pool = equal_level_fill(pool, U_sh.copy())
                alloc_s = np.asarray(alloc_s, dtype=float)
                taken_s = float(alloc_s.sum())
                if taken_s > 0:
                    U_sh -= alloc_s
                    m_shop += taken_s
                    surplus_shared += taken_s
                    rev += taken_s * layer["Ppros_SH"]
                    cost += taken_s * layer["Pcons_SH"]
                    gap_eur += (taken_s * (1 - layer["loss_factor"])) * layer["gap_SH"]
                    pool = resid_pool

            imp_hh += U_hh.sum()
            imp_shop += U_sh.sum()
            cost += U_hh.sum() * layer["ret_HH"] + U_sh.sum() * layer["ret_SH"]

            exp_total += pool
            rev += pool * layer["P_unm"]

            cons_sum += hh_full.sum() + sh_full.sum()

        matched_hh[sidx] = m_hh
        matched_shop[sidx] = m_shop
        import_hh[sidx] = imp_hh
        import_shop[sidx] = imp_shop
        export[sidx] = exp_total
        pros_rev[sidx] = rev
        cons_cost[sidx] = cost
        platform_gap[sidx] = gap_eur
        pv_gen[sidx] = pv_sum
        cons_total[sidx] = cons_sum
        baseline_cost[sidx] = base_cost
        pros_demand[sidx] = pros_sum
        hh_demand[sidx] = hh_sum
        shop_demand[sidx] = shop_sum
        shared_surplus[sidx] = surplus_shared

    platform_margin = platform_gap + fees - fixed

    return dict(
        matched_hh=matched_hh,
        matched_shop=matched_shop,
        import_hh=import_hh,
        import_shop=import_shop,
        export=export,
        pros_rev=pros_rev,
        cons_cost=cons_cost,
        platform_gap=platform_gap,
        platform_fees=np.full_like(platform_gap, fees),
        platform_fixed=np.full_like(platform_gap, fixed),
        platform_margin=platform_margin,
        pv_gen=pv_gen,
        cons_total=cons_total,
        cons_baseline=baseline_cost,
        pros_demand=pros_demand,
        hh_demand=hh_demand,
        shop_demand=shop_demand,
        surplus_shared=shared_surplus,
    )
