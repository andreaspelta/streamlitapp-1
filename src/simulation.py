import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any

from .clustering import assign_clusters, CLUSTERS


@dataclass
class DeterministicResult:
    hours: pd.DatetimeIndex
    prices: pd.DataFrame
    community_hourly: pd.DataFrame
    prosumer_hourly: pd.DataFrame
    household_hourly: pd.DataFrame
    shop_hourly: pd.DataFrame
    prosumer_summary: pd.DataFrame
    household_summary: pd.DataFrame
    shop_summary: pd.DataFrame
    totals: Dict[str, float]
    parameters: Dict[str, Any]


def equal_level_fill(supply: float, demands: np.ndarray) -> (np.ndarray, float):
    """Water-filling allocation used for HH and shop matching."""

    n = len(demands)
    if n == 0 or supply <= 1e-12:
        return np.zeros(n), float(supply)
    remaining = demands.astype(float).copy()
    matched = np.zeros(n, dtype=float)
    active = [i for i, v in enumerate(remaining) if v > 1e-12]
    s = float(supply)
    while s > 1e-12 and active:
        share = s / len(active)
        used = 0.0
        new_active = []
        for idx in active:
            take = min(share, remaining[idx])
            matched[idx] += take
            remaining[idx] -= take
            used += take
            if remaining[idx] > 1e-12:
                new_active.append(idx)
        s -= used
        if used <= 1e-12:
            break
        active = new_active
    return matched, s


def build_hourly_profile(hours: pd.DatetimeIndex, medians: pd.DataFrame) -> pd.Series:
    """Expand cluster median table (24×12) to an hourly template."""

    medians = medians.copy()
    medians.index = medians.index.astype(int)
    medians = medians.reindex(range(24)).fillna(0.0)
    for cluster in CLUSTERS:
        if cluster not in medians.columns:
            medians[cluster] = 0.0
    medians = medians[CLUSTERS]
    clusters = assign_clusters(hours)
    values = [
        float(medians.at[ts.hour, cluster] if cluster in medians.columns else 0.0)
        for ts, cluster in zip(hours, clusters)
    ]
    return pd.Series(values, index=hours, name="kWh")


def _ensure_alignment(series: pd.Series, hours: pd.DatetimeIndex, label: str) -> pd.Series:
    aligned = series.reindex(hours)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index[:5]
        raise ValueError(
            f"[{label}] Missing values for hours: {list(missing)}"
        )
    return aligned.astype(float)


def run_deterministic(
    *,
    hours: pd.DatetimeIndex,
    pv_series: pd.Series,
    hh_series: pd.Series,
    shop_series: pd.Series,
    prosumers: pd.DataFrame,
    households: pd.DataFrame,
    shops: pd.DataFrame,
    mapping: Dict[str, List[str]],
    zonal: pd.DataFrame,
    pun_hourly: pd.DataFrame,
    price_layer,
    efficiency: float,
    hh_gift: bool,
    fees: Dict[str, float],
) -> DeterministicResult:
    """Run the deterministic energy/community model for a single year."""

    if hours is None or len(hours) == 0:
        raise ValueError("Calendar hours are missing; select a year and load prices.")

    hours = hours.sort_values()
    zonal_series = zonal.set_index("timestamp").reindex(hours)["zonal_price (EUR_per_MWh)"].ffill()
    if zonal_series.isna().any():
        raise ValueError("Zonal prices have missing hours after reindexing.")
    pun_series = pun_hourly.set_index("timestamp").reindex(hours)["PUN (EUR_per_kWh)"].ffill()
    if pun_series.isna().any():
        raise ValueError("Hourly PUN has missing values after reindexing.")

    pv_series = _ensure_alignment(pv_series, hours, "PV template")
    hh_series = _ensure_alignment(hh_series, hours, "Household template")
    shop_series = _ensure_alignment(shop_series, hours, "Shop template")

    n_hours = len(hours)
    prosumer_ids = prosumers["prosumer_id"].tolist()
    hh_ids = households["household_id"].tolist()
    shop_ids = shops["shop_id"].tolist()
    nP = len(prosumer_ids)
    nH = len(hh_ids)
    nS = len(shop_ids)

    kwp = prosumers["kWp"].astype(float).to_numpy() if nP else np.zeros(0)
    w_self = prosumers.get("w_self", pd.Series(0, index=prosumers.index)).astype(float).to_numpy() if nP else np.zeros(0)
    pros_province = prosumers.get("province", pd.Series("Province", index=prosumers.index)).astype(str).tolist()

    hh_weight = households.get("w_HH", pd.Series(0, index=households.index)).astype(float).to_numpy() if nH else np.zeros(0)
    shop_weight = shops.get("w_SHOP", pd.Series(0, index=shops.index)).astype(float).to_numpy() if nS else np.zeros(0)
    shop_province = shops.get("province", pd.Series("Province", index=shops.index)).astype(str).tolist()

    pv_vals = pv_series.to_numpy()
    hh_vals = hh_series.to_numpy()
    shop_vals = shop_series.to_numpy()

    pros_gen = (kwp[:, None] * pv_vals[None, :] * float(efficiency)) if nP else np.zeros((0, n_hours))
    pros_load = (w_self[:, None] * hh_vals[None, :]) if nP else np.zeros((0, n_hours))
    pros_self = np.minimum(pros_gen, pros_load)
    pros_import = np.maximum(pros_load - pros_gen, 0.0)
    pros_surplus = np.maximum(pros_gen - pros_load, 0.0)

    hh_load = (hh_weight[:, None] * hh_vals[None, :]) if nH else np.zeros((0, n_hours))
    shop_load = (shop_weight[:, None] * shop_vals[None, :]) if nS else np.zeros((0, n_hours))

    hh_matched = np.zeros_like(hh_load)
    hh_import = np.zeros_like(hh_load)
    shop_matched = np.zeros_like(shop_load)
    shop_import = np.zeros_like(shop_load)
    pros_matched_hh = np.zeros_like(pros_gen)
    pros_matched_shop = np.zeros_like(pros_gen)
    pros_exports = np.zeros_like(pros_gen)

    hh_index = {hid: idx for idx, hid in enumerate(hh_ids)}
    mapping_idx = [
        np.array([hh_index[h] for h in mapping.get(pid, []) if h in hh_index], dtype=int)
        for pid in prosumer_ids
    ]

    province_to_pros: Dict[str, np.ndarray] = {}
    for idx, prov in enumerate(pros_province):
        province_to_pros.setdefault(prov, []).append(idx)
    province_to_pros = {k: np.array(v, dtype=int) for k, v in province_to_pros.items()}

    province_to_shops: Dict[str, np.ndarray] = {}
    for idx, prov in enumerate(shop_province):
        province_to_shops.setdefault(prov, []).append(idx)
    province_to_shops = {k: np.array(v, dtype=int) for k, v in province_to_shops.items()}

    price_records = []
    for zonal_val, pun_val in zip(zonal_series.to_numpy(), pun_series.to_numpy()):
        price_records.append(price_layer(zonal_val, pun_val))

    ret_HH = np.array([rec["ret_HH"] for rec in price_records])
    ret_SH = np.array([rec["ret_SH"] for rec in price_records])
    Ppros_HH = np.array([rec["Ppros_HH"] for rec in price_records])
    Pcons_HH = np.array([rec["Pcons_HH"] for rec in price_records])
    gap_HH = np.array([rec["gap_HH"] for rec in price_records])
    Ppros_SH = np.array([rec["Ppros_SH"] for rec in price_records])
    Pcons_SH = np.array([rec["Pcons_SH"] for rec in price_records])
    gap_SH = np.array([rec["gap_SH"] for rec in price_records])
    P_unm = np.array([rec["P_unm"] for rec in price_records])
    loss_factor = float(price_records[0].get("loss_factor", 0.0)) if price_records else 0.0

    for h in range(n_hours):
        hh_need = hh_load[:, h].copy()
        residual = pros_surplus[:, h].copy()

        if nP and nH:
            for p_idx in range(nP):
                supply = residual[p_idx]
                idxs = mapping_idx[p_idx]
                if supply > 1e-12 and idxs.size > 0:
                    alloc, resid = equal_level_fill(supply, hh_need[idxs])
                    alloc = np.asarray(alloc, dtype=float)
                    if alloc.size:
                        hh_need[idxs] -= alloc
                        hh_matched[idxs, h] += alloc
                        pros_matched_hh[p_idx, h] += float(alloc.sum())
                    residual[p_idx] = resid

        hh_import[:, h] = hh_need

        shop_need = shop_load[:, h].copy()
        if residual.sum() > 1e-12 and nS:
            for prov, pros_idxs in province_to_pros.items():
                if not len(pros_idxs):
                    continue
                supply = residual[pros_idxs].sum()
                if supply <= 1e-12:
                    continue
                shop_idxs = province_to_shops.get(prov, np.array([], dtype=int))
                if shop_idxs.size == 0:
                    continue
                alloc, resid_pool = equal_level_fill(supply, shop_need[shop_idxs])
                alloc = np.asarray(alloc, dtype=float)
                taken = float(alloc.sum())
                if taken > 1e-12:
                    shop_need[shop_idxs] -= alloc
                    shop_matched[shop_idxs, h] += alloc
                    total_supply = residual[pros_idxs].sum()
                    if total_supply > 0:
                        shares = residual[pros_idxs] / total_supply
                        for idx, share in zip(pros_idxs, shares):
                            used = share * taken
                            pros_matched_shop[idx, h] += used
                            residual[idx] = max(residual[idx] - used, 0.0)
                # resid_pool implicitly carried in residual remaining
        shop_import[:, h] = shop_need
        pros_exports[:, h] = np.maximum(residual, 0.0)

    if nP:
        pros_rev_self = pros_self * ret_HH[None, :]
        pros_rev_matched = (
            pros_matched_hh * Ppros_HH[None, :]
            + pros_matched_shop * Ppros_SH[None, :]
        )
        pros_rev_export = pros_exports * P_unm[None, :]
        pros_rev_import_cost = pros_import * ret_HH[None, :]
        pros_rev = pros_rev_self + pros_rev_matched + pros_rev_export - pros_rev_import_cost
        pros_rev_no_share = (
            pros_rev_self
            + (pros_matched_hh + pros_matched_shop + pros_exports) * P_unm[None, :]
            - pros_rev_import_cost
        )
    else:
        pros_rev_self = np.zeros((0, n_hours))
        pros_rev_matched = np.zeros((0, n_hours))
        pros_rev_export = np.zeros((0, n_hours))
        pros_rev_import_cost = np.zeros((0, n_hours))
        pros_rev = np.zeros((0, n_hours))
        pros_rev_no_share = np.zeros((0, n_hours))

    hh_cost = (
        hh_matched * Pcons_HH[None, :]
        + hh_import * ret_HH[None, :]
    ) if nH else np.zeros((0, n_hours))
    hh_baseline = hh_load * ret_HH[None, :] if nH else np.zeros((0, n_hours))

    shop_cost = (
        shop_matched * Pcons_SH[None, :]
        + shop_import * ret_SH[None, :]
    ) if nS else np.zeros((0, n_hours))
    shop_baseline = shop_load * ret_SH[None, :] if nS else np.zeros((0, n_hours))

    gap_hh = hh_matched.sum(axis=0) * (1 - loss_factor) * gap_HH if nH else np.zeros(n_hours)
    gap_shop = shop_matched.sum(axis=0) * (1 - loss_factor) * gap_SH if nS else np.zeros(n_hours)
    platform_gap = gap_hh + gap_shop

    platform_fees = 12.0 * (
        nP * float(fees.get("f_pros", 0.0))
        + nH * float(fees.get("f_hh", 0.0))
        + nS * float(fees.get("f_shop", 0.0))
    )
    platform_fixed = 12.0 * float(fees.get("platform_fixed", 0.0))

    totals = dict(
        matched_hh=float(hh_matched.sum()),
        matched_shop=float(shop_matched.sum()),
        import_hh=float(hh_import.sum()),
        import_shop=float(shop_import.sum()),
        export=float(pros_exports.sum()),
        pros_rev=float(pros_rev.sum()),
        cons_cost=float(hh_cost.sum() + shop_cost.sum()),
        cons_cost_hh=float(hh_cost.sum()),
        cons_cost_shop=float(shop_cost.sum()),
        cons_baseline=float(hh_baseline.sum() + shop_baseline.sum()),
        cons_baseline_hh=float(hh_baseline.sum()),
        cons_baseline_shop=float(shop_baseline.sum()),
        cons_savings_hh=float(hh_baseline.sum() - hh_cost.sum()),
        cons_savings_shop=float(shop_baseline.sum() - shop_cost.sum()),
        platform_gap=float(platform_gap.sum()),
        platform_fees=float(platform_fees),
        platform_fixed=float(platform_fixed),
        platform_margin=float(platform_gap.sum() + platform_fees - platform_fixed),
        pv_gen=float(pros_gen.sum()),
        pros_demand=float(pros_load.sum()),
        hh_demand=float(hh_load.sum()),
        shop_demand=float(shop_load.sum()),
        surplus_shared=float(hh_matched.sum() + shop_matched.sum()),
        pros_import=float(pros_import.sum()),
        pros_rev_matched=float(pros_rev_matched.sum()),
        pros_rev_export=float(pros_rev_export.sum()),
        pros_rev_no_share=float(pros_rev_no_share.sum()),
    )

    prosumer_hourly = pd.DataFrame({
        "prosumer_id": np.repeat(prosumer_ids, n_hours),
        "timestamp": np.tile(hours, nP),
        "generation_kWh": pros_gen.reshape(-1),
        "load_kWh": pros_load.reshape(-1),
        "self_consumption_kWh": pros_self.reshape(-1),
        "imports_kWh": pros_import.reshape(-1),
        "matched_hh_kWh": pros_matched_hh.reshape(-1),
        "matched_shop_kWh": pros_matched_shop.reshape(-1),
        "exports_kWh": pros_exports.reshape(-1),
        "revenue_EUR": pros_rev.reshape(-1),
    }) if nP else pd.DataFrame(columns=[
        "prosumer_id", "timestamp", "generation_kWh", "load_kWh",
        "self_consumption_kWh", "imports_kWh", "matched_hh_kWh",
        "matched_shop_kWh", "exports_kWh", "revenue_EUR"
    ])

    household_hourly = pd.DataFrame({
        "household_id": np.repeat(hh_ids, n_hours),
        "timestamp": np.tile(hours, nH),
        "load_kWh": hh_load.reshape(-1),
        "matched_kWh": hh_matched.reshape(-1),
        "import_kWh": hh_import.reshape(-1),
        "cost_EUR": hh_cost.reshape(-1),
        "baseline_EUR": hh_baseline.reshape(-1),
    }) if nH else pd.DataFrame(columns=[
        "household_id", "timestamp", "load_kWh", "matched_kWh",
        "import_kWh", "cost_EUR", "baseline_EUR"
    ])

    shop_hourly = pd.DataFrame({
        "shop_id": np.repeat(shop_ids, n_hours),
        "timestamp": np.tile(hours, nS),
        "load_kWh": shop_load.reshape(-1),
        "matched_kWh": shop_matched.reshape(-1),
        "import_kWh": shop_import.reshape(-1),
        "cost_EUR": shop_cost.reshape(-1),
        "baseline_EUR": shop_baseline.reshape(-1),
    }) if nS else pd.DataFrame(columns=[
        "shop_id", "timestamp", "load_kWh", "matched_kWh",
        "import_kWh", "cost_EUR", "baseline_EUR"
    ])

    prosumer_summary = prosumer_hourly.groupby("prosumer_id").agg({
        "generation_kWh": "sum",
        "load_kWh": "sum",
        "self_consumption_kWh": "sum",
        "imports_kWh": "sum",
        "matched_hh_kWh": "sum",
        "matched_shop_kWh": "sum",
        "exports_kWh": "sum",
        "revenue_EUR": "sum",
    }).reset_index() if not prosumer_hourly.empty else pd.DataFrame(columns=[
        "prosumer_id", "generation_kWh", "load_kWh", "self_consumption_kWh",
        "imports_kWh", "matched_hh_kWh", "matched_shop_kWh",
        "exports_kWh", "revenue_EUR"
    ])

    household_summary = household_hourly.groupby("household_id").agg({
        "load_kWh": "sum",
        "matched_kWh": "sum",
        "import_kWh": "sum",
        "cost_EUR": "sum",
        "baseline_EUR": "sum",
    }).reset_index() if not household_hourly.empty else pd.DataFrame(columns=[
        "household_id", "load_kWh", "matched_kWh", "import_kWh",
        "cost_EUR", "baseline_EUR"
    ])
    if not household_summary.empty:
        household_summary["savings_EUR"] = household_summary["baseline_EUR"] - household_summary["cost_EUR"]

    shop_summary = shop_hourly.groupby("shop_id").agg({
        "load_kWh": "sum",
        "matched_kWh": "sum",
        "import_kWh": "sum",
        "cost_EUR": "sum",
        "baseline_EUR": "sum",
    }).reset_index() if not shop_hourly.empty else pd.DataFrame(columns=[
        "shop_id", "load_kWh", "matched_kWh", "import_kWh",
        "cost_EUR", "baseline_EUR"
    ])
    if not shop_summary.empty:
        shop_summary["savings_EUR"] = shop_summary["baseline_EUR"] - shop_summary["cost_EUR"]

    community_hourly = pd.DataFrame({
        "timestamp": hours,
        "pv_generation_kWh": pros_gen.sum(axis=0) if nP else np.zeros(n_hours),
        "prosumer_load_kWh": pros_load.sum(axis=0) if nP else np.zeros(n_hours),
        "hh_load_kWh": hh_load.sum(axis=0) if nH else np.zeros(n_hours),
        "shop_load_kWh": shop_load.sum(axis=0) if nS else np.zeros(n_hours),
        "matched_hh_kWh": hh_matched.sum(axis=0) if nH else np.zeros(n_hours),
        "matched_shop_kWh": shop_matched.sum(axis=0) if nS else np.zeros(n_hours),
        "exports_kWh": pros_exports.sum(axis=0) if nP else np.zeros(n_hours),
        "import_hh_kWh": hh_import.sum(axis=0) if nH else np.zeros(n_hours),
        "import_shop_kWh": shop_import.sum(axis=0) if nS else np.zeros(n_hours),
        "platform_gap_EUR": platform_gap,
    })

    prices_df = pd.DataFrame({
        "timestamp": hours,
        "zonal_price (EUR_per_MWh)": zonal_series.to_numpy(),
        "PUN (EUR_per_kWh)": pun_series.to_numpy(),
        "retail_HH (EUR_per_kWh)": ret_HH,
        "retail_SH (EUR_per_kWh)": ret_SH,
        "Ppros_HH (EUR_per_kWh)": Ppros_HH,
        "Pcons_HH (EUR_per_kWh)": Pcons_HH,
        "gap_HH (EUR_per_kWh)": gap_HH,
        "Ppros_SH (EUR_per_kWh)": Ppros_SH,
        "Pcons_SH (EUR_per_kWh)": Pcons_SH,
        "gap_SH (EUR_per_kWh)": gap_SH,
        "P_unm (EUR_per_kWh)": P_unm,
    })

    params = dict(
        efficiency=float(efficiency),
        hh_gift=bool(hh_gift),
        fees={k: float(v) for k, v in fees.items()},
    )

    return DeterministicResult(
        hours=hours,
        prices=prices_df,
        community_hourly=community_hourly,
        prosumer_hourly=prosumer_hourly,
        household_hourly=household_hourly,
        shop_hourly=shop_hourly,
        prosumer_summary=prosumer_summary,
        household_summary=household_summary,
        shop_summary=shop_summary,
        totals=totals,
        parameters=params,
    )
