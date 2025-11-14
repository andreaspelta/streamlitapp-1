import pandas as pd
import streamlit as st

from typing import Any, Dict, List


class spinner_block:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.ctx = st.spinner(self.msg)
        self.ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        self.ctx.__exit__(exc_type, exc, tb)


def info_box(msg):
    st.info(msg)


def warn_box(msg):
    st.warning(msg)


def error_box(msg):
    st.error(msg)


def build_prosumer_table(
    prosumer_ids: List[str],
    kwp_map: Dict[str, float],
    w_self_map: Dict[str, float],
    province_map: Dict[str, str],
) -> Dict[str, Dict[str, any]]:
    df = pd.DataFrame(
        {
            "prosumer_id": prosumer_ids,
            "kWp": [float(kwp_map.get(pid, 0.0)) for pid in prosumer_ids],
            "w_self": [float(w_self_map.get(pid, 1.0)) for pid in prosumer_ids],
            "province": [province_map.get(pid, "Province-1") for pid in prosumer_ids],
        }
    )
    edited = st.data_editor(
        df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config={
            "kWp": st.column_config.NumberColumn("kWp", min_value=0.0, step=0.1),
            "w_self": st.column_config.NumberColumn("w_self", min_value=0.0, step=0.1),
        },
        key="prosumer_table",
    )
    result = {}
    for _, row in edited.iterrows():
        result[row["prosumer_id"]] = {
            "kWp": float(row["kWp"]),
            "w_self": float(row["w_self"]),
            "province": str(row["province"]).strip() or "Province-1",
        }
    return result


def build_household_table(
    hh_ids: List[str],
    hh_weights: Dict[str, float],
    hh_price_mode: Dict[str, str],
    hh_price_value: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    mode_options = ["PUN-INDEX", "Fixed"]
    df = pd.DataFrame(
        {
            "household_id": hh_ids,
            "w_HH": [float(hh_weights.get(hid, 1.0)) for hid in hh_ids],
            "price_mode": [hh_price_mode.get(hid, "PUN-INDEX") for hid in hh_ids],
            "fixed_price": [float(hh_price_value.get(hid, 0.25)) for hid in hh_ids],
        }
    )
    edited = st.data_editor(
        df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config={
            "w_HH": st.column_config.NumberColumn("w_HH", min_value=0.0, step=0.1),
            "price_mode": st.column_config.SelectboxColumn(
                "Base price mode",
                options=mode_options,
            ),
            "fixed_price": st.column_config.NumberColumn(
                "Fixed price (€/kWh)",
                min_value=0.0,
                step=0.005,
                format="%.4f",
            ),
        },
        key="household_table",
    )
    result: Dict[str, Dict[str, Any]] = {}
    for _, row in edited.iterrows():
        mode_raw = str(row.get("price_mode", "PUN-INDEX")).strip().upper()
        mode = "FIXED" if mode_raw == "FIXED" else "PUN-INDEX"
        raw_fixed = row.get("fixed_price", 0.25)
        fixed_val = float(raw_fixed) if pd.notna(raw_fixed) else 0.0
        raw_weight = row.get("w_HH", 1.0)
        weight = float(raw_weight) if pd.notna(raw_weight) else 1.0
        result[str(row["household_id"])] = {
            "w_HH": weight,
            "price_mode": mode,
            "fixed_price": fixed_val,
        }
    return result


def build_shop_table(
    shop_ids: List[str],
    shop_weights: Dict[str, float],
    shop_province: Dict[str, str],
    shop_price_mode: Dict[str, str],
    shop_price_value: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    mode_options = ["PUN-INDEX", "Fixed"]
    df = pd.DataFrame(
        {
            "shop_id": shop_ids,
            "w_SHOP": [float(shop_weights.get(sid, 1.0)) for sid in shop_ids],
            "province": [shop_province.get(sid, "Province-1") for sid in shop_ids],
            "price_mode": [shop_price_mode.get(sid, "PUN-INDEX") for sid in shop_ids],
            "fixed_price": [float(shop_price_value.get(sid, 0.25)) for sid in shop_ids],
        }
    )
    edited = st.data_editor(
        df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config={
            "w_SHOP": st.column_config.NumberColumn("w_SHOP", min_value=0.0, step=0.1),
            "province": st.column_config.TextColumn("Province"),
            "price_mode": st.column_config.SelectboxColumn(
                "Base price mode",
                options=mode_options,
            ),
            "fixed_price": st.column_config.NumberColumn(
                "Fixed price (€/kWh)",
                min_value=0.0,
                step=0.005,
                format="%.4f",
            ),
        },
        key="shop_table",
    )
    result: Dict[str, Dict[str, Any]] = {}
    for _, row in edited.iterrows():
        mode_raw = str(row.get("price_mode", "PUN-INDEX")).strip().upper()
        mode = "FIXED" if mode_raw == "FIXED" else "PUN-INDEX"
        raw_fixed = row.get("fixed_price", 0.25)
        fixed_val = float(raw_fixed) if pd.notna(raw_fixed) else 0.0
        raw_weight = row.get("w_SHOP", 1.0)
        weight = float(raw_weight) if pd.notna(raw_weight) else 1.0
        province_val = str(row.get("province", "Province-1")).strip() or "Province-1"
        result[str(row["shop_id"])] = {
            "w_SHOP": weight,
            "province": province_val,
            "price_mode": mode,
            "fixed_price": fixed_val,
        }
    return result


def build_mapping_editor(
    prosumer_ids: List[str],
    hh_ids: List[str],
    existing: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    st.caption("Assign households to each prosumer for priority matching.")
    out = {}
    cols = st.columns(min(3, len(prosumer_ids)) or 1)
    for i, pid in enumerate(prosumer_ids):
        with cols[i % len(cols)]:
            sel = st.multiselect(
                f"{pid} → Households",
                hh_ids,
                default=existing.get(pid, []),
                key=f"map_{pid}",
            )
            out[pid] = sel
    return out
