import numpy as np

NEGATIVE_RULE = "If zonal < 0 → P_pros=0; P_cons=retail (PUN-monthly+spread); gap=0; export = max(zonal+δ_unm, 0)."

def build_price_layers(
    s_hh,
    s_sh,
    spread_split_hh,
    platform_gap_hh,
    spread_split_sh,
    platform_gap_sh,
    delta_unm,
    loss_factor,
    hh_gift,
):
    """
    Returns a callable that receives:
      - zonal_eur_per_mwh (float)   [hourly]
      - pun_kwh (float)             [monthly PUN expanded to hour; €/kWh]

    and produces per-hour prices dict for HH and SHOP segments:
      - ret_HH, ret_SH (€/kWh)          retail = PUN_monthly_kWh + spread
      - Ppros_HH, Pcons_HH, gap_HH
      - Ppros_SH, Pcons_SH, gap_SH
      - P_unm (€/kWh) for exports = max(zonal_kWh + δ_unm, 0)
      - loss_factor (float), hh_gift (bool)
    """
    def layer(zonal_eur_per_mwh, pun_kwh):
        z = float(zonal_eur_per_mwh) / 1000.0  # €/kWh
        pun = float(pun_kwh)                   # €/kWh (already in kWh units)

        # Retail now built from monthly PUN (€/kWh) + spreads
        ret_HH = pun + float(s_hh)
        ret_SH = pun + float(s_sh)

        if z < 0:
            # Negative-price override
            Ppros_HH = 0.0; Pcons_HH = ret_HH; gap_HH = 0.0
            Ppros_SH = 0.0; Pcons_SH = ret_SH; gap_SH = 0.0
        else:
            # Anchor the zonal price to the non-negative domain
            z_anchor = max(z, 0.0)

            # Segment spreads depend on the distance between retail and anchored zonal price
            S_HH = max(ret_HH - z_anchor, 0.0)
            S_SH = max(ret_SH - z_anchor, 0.0)

            # Prosumer price lift off anchored zonal (€/kWh), capped at retail
            Ppros_HH = min(
                z_anchor + (1 - platform_gap_hh) * spread_split_hh * S_HH,
                ret_HH,
            )
            Pcons_HH = ret_HH - (1 - platform_gap_hh) * (1 - spread_split_hh) * S_HH
            gap_HH = platform_gap_hh * S_HH

            Ppros_SH = min(
                z_anchor + (1 - platform_gap_sh) * spread_split_sh * S_SH,
                ret_SH,
            )
            Pcons_SH = ret_SH - (1 - platform_gap_sh) * (1 - spread_split_sh) * S_SH
            gap_SH = platform_gap_sh * S_SH

        if hh_gift and z >= 0:
            # Gift: HH matched at zero for both sides; no gap
            Ppros_HH = 0.0; Pcons_HH = 0.0; gap_HH = 0.0

        P_unm = max(z + float(delta_unm), 0.0)

        return {
            "ret_HH": ret_HH, "ret_SH": ret_SH,
            "Ppros_HH": Ppros_HH, "Pcons_HH": Pcons_HH, "gap_HH": gap_HH,
            "Ppros_SH": Ppros_SH, "Pcons_SH": Pcons_SH, "gap_SH": gap_SH,
            "P_unm": P_unm,
            "loss_factor": float(loss_factor),
            "hh_gift": bool(hh_gift),
        }
    return layer
