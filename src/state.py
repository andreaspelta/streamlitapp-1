from dataclasses import dataclass, field
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Any


@dataclass
class AppState:
    year: int = 2024
    hours: Optional[pd.DatetimeIndex] = None

    # Uploaded deterministic templates
    pv_df: Optional[pd.DataFrame] = None
    pv_series: Optional[pd.Series] = None
    hh_medians: Optional[pd.DataFrame] = None
    hh_series: Optional[pd.Series] = None
    shop_medians: Optional[pd.DataFrame] = None
    shop_series: Optional[pd.Series] = None

    zonal: Optional[pd.DataFrame] = None
    pun_m: Optional[pd.DataFrame] = None   # monthly PUN €/kWh
    pun_h: Optional[pd.DataFrame] = None   # hourly PUN €/kWh (expanded)

    pv_file_digest: Optional[str] = None
    hh_file_digest: Optional[str] = None
    shop_file_digest: Optional[str] = None
    zonal_file_digest: Optional[str] = None
    pun_file_digest: Optional[str] = None

    efficiency: float = 1.0
    hh_gift: bool = False

    # Pricing parameters
    s_HH: float = 0.10
    s_SH: float = 0.12
    spread_split_HH: float = 0.5
    platform_gap_HH: float = 0.3
    spread_split_SH: float = 0.5
    platform_gap_SH: float = 0.3
    delta_unm: float = 0.0
    loss_factor: float = 0.05

    f_pros: float = 2.0
    f_hh: float = 1.0
    f_shop: float = 1.5
    platform_fixed: float = 200.0

    # Scenario definition
    N_P: int = 3
    N_HH: int = 6
    N_SHOP: int = 3

    prosumer_ids: List[str] = field(default_factory=list)
    hh_ids: List[str] = field(default_factory=list)
    shop_ids: List[str] = field(default_factory=list)

    kwp_map: Dict[str, float] = field(default_factory=dict)
    w_self_map: Dict[str, float] = field(default_factory=dict)
    prosumer_province: Dict[str, str] = field(default_factory=dict)
    hh_weights: Dict[str, float] = field(default_factory=dict)
    shop_weights: Dict[str, float] = field(default_factory=dict)
    shop_province: Dict[str, str] = field(default_factory=dict)
    mapping: Dict[str, List[str]] = field(default_factory=dict)

    result: Optional[Any] = None


def get_state() -> AppState:
    if "APP_STATE" not in st.session_state:
        st.session_state.APP_STATE = AppState()
    return st.session_state.APP_STATE

def reset_all_state():
    if "APP_STATE" in st.session_state:
        del st.session_state.APP_STATE
