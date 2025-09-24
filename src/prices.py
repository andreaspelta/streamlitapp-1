import numpy as np
import pandas as pd

NEGATIVE_RULE = "If zonal < 0 → P_pros=0; P_cons=retail; gap=0; export price = max(zonal + δ_unm, 0)."

def build_price_layers(s_hh, s_sh, a_hh, phi_hh, a_sh, phi_sh, delta_unm, loss_factor, hh_gift):
    """
    Returns a callable:
      price_layer(ts) -> dict with fields for HH & SHOP:
        { 'ret_HH':..., 'ret_SH':..., 'Ppros_HH':..., 'Pcons_HH':..., 'gap_HH':...,
          'Ppros_SH':..., 'Pcons_SH':..., 'gap_SH':..., 'P_unm':..., 'loss_factor':..., 'hh_gift':... }
    """
    def layer(zonal_eur_per_mwh, pun_eur_per_mwh):
        z = float(zonal_eur_per_mwh)/1000.0
        ret_HH = z + float(s_hh)
        ret_SH = z + float(s_sh)

        if z < 0:
            Ppros_HH = 0.0; Pcons_HH = ret_HH; gap_HH = 0.0
            Ppros_SH = 0.0; Pcons_SH = ret_SH; gap_SH = 0.0
        else:
            rem_HH = (1 - phi_hh) * a_hh * s_hh
            gap_HH = phi_hh * s_hh
            Ppros_HH = min(z + rem_HH, ret_HH)
            Pcons_HH = ret_HH - (1 - phi_hh) * (1 - a_hh) * s_hh

            rem_SH = (1 - phi_sh) * a_sh * s_sh
            gap_SH = phi_sh * s_sh
            Ppros_SH = min(z + rem_SH, ret_SH)
            Pcons_SH = ret_SH - (1 - phi_sh) * (1 - a_sh) * s_sh

        if hh_gift and z >= 0:
            # HH matched at zero for both sides; no gap
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
