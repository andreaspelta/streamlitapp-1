import numpy as np
import pandas as pd
from typing import Dict, Any

from . import _fast_simulation as _fast

# ---------- Sampling helpers (unchanged) ----------
def sample_hh_like(hours: pd.DatetimeIndex, fit: Dict, rng: np.random.Generator) -> np.ndarray:
    """Backwards-compatible wrapper delegating to the optimized sampler."""
    return _fast.sample_hh_like(hours, fit, rng)

def sample_shop_like(hours: pd.DatetimeIndex, fit: Dict, rng: np.random.Generator) -> np.ndarray:
    """Backwards-compatible wrapper delegating to the optimized sampler."""
    return _fast.sample_shop_like(hours, fit, rng)

def sample_pv(hours: pd.DatetimeIndex, pv_fit: Dict, kWp: float, rng: np.random.Generator) -> np.ndarray:
    """Backwards-compatible wrapper delegating to the optimized sampler."""
    return _fast.sample_pv(hours, pv_fit, kWp, rng)

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

    return _fast.run_monte_carlo(
        hours=hours,
        hh_fit=hh_fit,
        shop_fit=shop_fit,
        pv_fit=pv_fit,
        kwp_map=kwp_map,
        mapping=mapping,
        prosumer_ids=prosumer_ids,
        hh_ids=hh_ids,
        shop_ids=shop_ids,
        zonal=zonal,
        pun_hourly_kwh=pun_hourly_kwh,
        price_layer=price_layer,
        S=S,
        seed=seed,
        price_params=price_params,
        equal_level_fill=equal_level_fill,
    )
