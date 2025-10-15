from dataclasses import dataclass, field
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Any

@dataclass
class AppState:
    # Uploads
    hh_df: Optional[pd.DataFrame] = None
    shop_df: Optional[pd.DataFrame] = None
    pv_perkwp: Optional[pd.DataFrame] = None
    zonal: Optional[pd.DataFrame] = None
    pun_m: Optional[pd.DataFrame] = None   # monthly PUN €/kWh
    pun_h: Optional[pd.DataFrame] = None   # hourly PUN €/kWh (expanded)
    hours: Optional[pd.DatetimeIndex] = None
    hh_file_digest: Optional[str] = None
    shop_file_digest: Optional[str] = None
    pv_file_digest: Optional[str] = None
    zonal_file_digest: Optional[str] = None
    pun_file_digest: Optional[str] = None

    # Fits
    hh_fit: Optional[dict] = None
    shop_fit: Optional[dict] = None
    pv_fit: Optional[dict] = None
    hh_diag: Optional[dict] = None
    shop_diag: Optional[dict] = None
    pv_diag: Optional[dict] = None

    # Scenario
    N_P: int = 3
    N_HH: int = 6
    N_SHOP: int = 3
    seed: int = 12345
    S: int = 5000
    hh_gift: bool = False

    s_HH: float = 0.10
    s_SH: float = 0.12
    alpha_HH: float = 0.5
    phi_HH: float = 0.3
    alpha_SH: float = 0.5
    phi_SH: float = 0.3
    delta_unm: float = 0.0
    loss_factor: float = 0.05

    f_pros: float = 2.0
    f_hh: float = 1.0
    f_shop: float = 1.5
    platform_fixed: float = 200.0

    prosumer_ids: List[str] = field(default_factory=list)
    hh_ids: List[str] = field(default_factory=list)
    shop_ids: List[str] = field(default_factory=list)
    kwp_map: Dict[str, float] = field(default_factory=dict)
    mapping: Dict[str, List[str]] = field(default_factory=dict)

    # Monte Carlo results
    mc: Optional[Dict[str, Any]] = None

def get_state() -> AppState:
    if "APP_STATE" not in st.session_state:
        st.session_state.APP_STATE = AppState()
    return st.session_state.APP_STATE

def reset_all_state():
    if "APP_STATE" in st.session_state:
        del st.session_state.APP_STATE
