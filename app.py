import hashlib
from io import BytesIO

import pandas as pd
import streamlit as st

from src.state import get_state, reset_all_state
from src.io_utils import (
    read_pv_excel,
    read_household_template,
    read_shop_template,
    read_zonal_csv,
    read_pun_monthly_csv,
    ensure_price_hours,
    expand_monthly_pun_to_hours,
)
from src.exporters import (
    build_pv_excel_template,
    build_household_template,
    build_shop_template,
    build_zonal_price_template,
    build_pun_monthly_template,
    export_all_in_one_xlsx,
    export_hourly_facts,
)
from src.simulation import build_hourly_profile, run_deterministic
from src.prices import build_price_layers, NEGATIVE_RULE
from src.ui_components import (
    spinner_block,
    info_box,
    warn_box,
    error_box,
    build_prosumer_table,
    build_household_table,
    build_shop_table,
    build_mapping_editor,
)
from src.kpi import render_kpi_dashboard
from src.sankey import build_energy_sankey_chart, build_economic_sankey_chart

TZ = "Europe/Rome"

st.set_page_config(page_title="Virtual Energy Community", layout="wide")

S = get_state()

st.sidebar.title("VEC App")
page = st.sidebar.radio(
    "Navigation",
    [
        "1) Templates & Inputs",
        "2) Scenario Builder",
        "3) Run Deterministic",
        "4) KPI Dashboard",
        "5) Sankey Diagram",
        "6) Exports",
        "About",
    ],
)


def _set_year(year: int):
    if S.year == year and S.hours is not None:
        return
    S.year = int(year)
    start = pd.Timestamp(f"{S.year}-01-01 00:00", tz=TZ)
    end = pd.Timestamp(f"{S.year + 1}-01-01 00:00", tz=TZ)
    S.hours = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    # Reset PV because the template depends on the calendar year
    S.pv_df = None
    S.pv_series = None
    S.pv_file_digest = None
    if S.hh_medians is not None:
        S.hh_series = build_hourly_profile(S.hours, S.hh_medians)
    if S.shop_medians is not None:
        S.shop_series = build_hourly_profile(S.hours, S.shop_medians)
    if S.pun_m is not None:
        try:
            S.pun_h = expand_monthly_pun_to_hours(S.pun_m, S.hours)
        except Exception:
            S.pun_h = None


# ---- Page 1: Templates & Inputs
if page == "1) Templates & Inputs":
    st.header("Deterministic Templates & Pricing Inputs")

    year_options = list(range(2023, 2031))
    col_year_select, col_year_button = st.columns([3, 1])
    with col_year_select:
        default_idx = year_options.index(S.year) if S.year in year_options else 0
        selected_year = st.selectbox("Simulation year", year_options, index=default_idx)
    with col_year_button:
        if st.button("Set calendar year", use_container_width=True):
            _set_year(selected_year)
            info_box(f"Calendar set to {selected_year}. Templates will use this year.")
    if S.hours is None:
        _set_year(selected_year)

    st.subheader("Prosumer PV template")
    st.download_button(
        f"Download PV template ({S.year})",
        build_pv_excel_template(S.year),
        file_name=f"pv_template_{S.year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    pv_file = st.file_uploader("Upload PV template (timestamp, kWh_per_kWp)", type=["xlsx"], key="pv")
    if pv_file:
        if S.hours is None:
            error_box("Set the simulation year before uploading PV data.")
        else:
            data = pv_file.getvalue()
            digest = hashlib.md5(data).hexdigest()
            if digest != S.pv_file_digest:
                try:
                    df = read_pv_excel(BytesIO(data))
                    series = df.set_index("timestamp")["kWh_per_kWp"].reindex(S.hours)
                    if series.isna().any():
                        raise ValueError("Uploaded PV file is missing some hours for the selected year.")
                    S.pv_df = df
                    S.pv_series = series
                    S.pv_file_digest = digest
                    info_box(f"PV template loaded for {len(series)} hours.")
                except Exception as exc:
                    error_box(f"Failed to read PV template: {exc}")
                    S.pv_df = None
                    S.pv_series = None
                    S.pv_file_digest = None

    st.subheader("Household median profile (12 clusters)")
    st.download_button(
        "Download HH template",
        build_household_template(S.year),
        file_name=f"household_medians_{S.year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    hh_file = st.file_uploader("Upload household template", type=["xlsx"], key="hh")
    if hh_file:
        data = hh_file.getvalue()
        digest = hashlib.md5(data).hexdigest()
        if digest != S.hh_file_digest:
            try:
                medians = read_household_template(BytesIO(data))
                S.hh_medians = medians
                if S.hours is not None:
                    S.hh_series = build_hourly_profile(S.hours, medians)
                S.hh_file_digest = digest
                info_box("Household template loaded.")
            except Exception as exc:
                error_box(f"Failed to read household template: {exc}")
                S.hh_medians = None
                S.hh_series = None
                S.hh_file_digest = None

    st.subheader("Shop median profile (12 clusters)")
    st.download_button(
        "Download SHOP template",
        build_shop_template(S.year),
        file_name=f"shop_medians_{S.year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    shop_file = st.file_uploader("Upload shop template", type=["xlsx"], key="shop")
    if shop_file:
        data = shop_file.getvalue()
        digest = hashlib.md5(data).hexdigest()
        if digest != S.shop_file_digest:
            try:
                medians = read_shop_template(BytesIO(data))
                S.shop_medians = medians
                if S.hours is not None:
                    S.shop_series = build_hourly_profile(S.hours, medians)
                S.shop_file_digest = digest
                info_box("Shop template loaded.")
            except Exception as exc:
                error_box(f"Failed to read shop template: {exc}")
                S.shop_medians = None
                S.shop_series = None
                S.shop_file_digest = None

    st.subheader("Price inputs")
    col_price1, col_price2 = st.columns(2)
    with col_price1:
        st.download_button(
            f"Download zonal template ({S.year})",
            build_zonal_price_template(S.year),
            file_name=f"zonal_template_{S.year}.csv",
            mime="text/csv",
        )
        zonal_file = st.file_uploader("Upload zonal price CSV", type=["csv"], key="zonal")
        if zonal_file:
            data = zonal_file.getvalue()
            digest = hashlib.md5(data).hexdigest()
            if digest != S.zonal_file_digest:
                try:
                    zonal_df = read_zonal_csv(BytesIO(data))
                    expected_hours = ensure_price_hours(zonal_df)
                    if S.hours is not None and not expected_hours.equals(S.hours):
                        raise ValueError("Zonal calendar does not match the selected year template.")
                    S.zonal = zonal_df
                    S.zonal_file_digest = digest
                    info_box(f"Zonal prices loaded: {len(zonal_df)} rows.")
                except Exception as exc:
                    error_box(f"Failed to read zonal CSV: {exc}")
                    S.zonal = None
                    S.zonal_file_digest = None
    with col_price2:
        st.download_button(
            f"Download monthly PUN template ({S.year})",
            build_pun_monthly_template(S.year),
            file_name=f"pun_monthly_{S.year}.csv",
            mime="text/csv",
        )
        pun_file = st.file_uploader("Upload monthly PUN CSV", type=["csv"], key="pun")
        if pun_file:
            data = pun_file.getvalue()
            digest = hashlib.md5(data).hexdigest()
            if digest != S.pun_file_digest:
                try:
                    pun_df = read_pun_monthly_csv(BytesIO(data))
                    S.pun_m = pun_df
                    if S.hours is not None:
                        S.pun_h = expand_monthly_pun_to_hours(pun_df, S.hours)
                    S.pun_file_digest = digest
                    info_box("Monthly PUN loaded and expanded to hours.")
                except Exception as exc:
                    error_box(f"Failed to read monthly PUN: {exc}")
                    S.pun_m = None
                    S.pun_h = None
                    S.pun_file_digest = None

    st.markdown("---")
    if st.button("Reset all inputs"):
        reset_all_state()
        st.experimental_rerun()

# ---- Page 2: Scenario Builder
elif page == "2) Scenario Builder":
    st.header("Scenario Builder")

    col_counts, col_flags = st.columns([3, 2])
    with col_counts:
        S.N_P = int(st.number_input("Number of prosumers", min_value=1, value=S.N_P))
        S.N_HH = int(st.number_input("Number of households", min_value=1, value=S.N_HH))
        S.N_SHOP = int(st.number_input("Number of shops", min_value=1, value=S.N_SHOP))
        S.efficiency = st.number_input(
            "PV efficiency multiplier",
            min_value=0.0,
            value=S.efficiency,
            step=0.01,
            format="%.3f",
            help="Applied to all PV profiles (1.00 = 100%).",
        )
    with col_flags:
        S.hh_gift = st.checkbox("HH-Gift (matched HH energy at zero €)", value=S.hh_gift)
        S.loss_factor = st.slider("Loss factor (platform gap loss)", 0.0, 1.0, value=S.loss_factor, step=0.01)

    st.subheader("Pricing parameters")
    col_price_a, col_price_b, col_price_c = st.columns(3)
    with col_price_a:
        S.s_HH = st.number_input("HH retail spread (€/kWh)", min_value=0.0, value=S.s_HH, step=0.01, format="%.4f")
        S.spread_split_HH = st.slider("spread_split_HH", 0.0, 1.0, value=S.spread_split_HH, step=0.01)
        S.platform_gap_HH = st.slider("platform_gap_HH", 0.0, 1.0, value=S.platform_gap_HH, step=0.01)
    with col_price_b:
        S.s_SH = st.number_input("SHOP retail spread (€/kWh)", min_value=0.0, value=S.s_SH, step=0.01, format="%.4f")
        S.spread_split_SH = st.slider("spread_split_SH", 0.0, 1.0, value=S.spread_split_SH, step=0.01)
        S.platform_gap_SH = st.slider("platform_gap_SH", 0.0, 1.0, value=S.platform_gap_SH, step=0.01)
    with col_price_c:
        S.delta_unm = st.number_input("δ_unm export uplift (€/kWh)", value=S.delta_unm, step=0.01, format="%.4f")

    st.subheader("Platform fees")
    col_fees_a, col_fees_b = st.columns(2)
    with col_fees_a:
        S.f_pros = st.number_input("Prosumer monthly fee (€/month)", min_value=0.0, value=S.f_pros, step=0.5)
        S.f_hh = st.number_input("Household monthly fee (€/month)", min_value=0.0, value=S.f_hh, step=0.5)
        S.f_shop = st.number_input("Shop monthly fee (€/month)", min_value=0.0, value=S.f_shop, step=0.5)
    with col_fees_b:
        S.platform_fixed = st.number_input("Platform fixed cost (€/month)", min_value=0.0, value=S.platform_fixed, step=10.0)

    S.prosumer_ids = [f"P{i:03d}" for i in range(1, S.N_P + 1)]
    S.hh_ids = [f"H{i:03d}" for i in range(1, S.N_HH + 1)]
    S.shop_ids = [f"K{i:03d}" for i in range(1, S.N_SHOP + 1)]

    st.subheader("Prosumer parameters")
    pros_data = build_prosumer_table(S.prosumer_ids, S.kwp_map, S.w_self_map, S.prosumer_province)
    S.kwp_map = {pid: data["kWp"] for pid, data in pros_data.items()}
    S.w_self_map = {pid: data["w_self"] for pid, data in pros_data.items()}
    S.prosumer_province = {pid: data["province"] for pid, data in pros_data.items()}

    st.subheader("Household weights (per template unit)")
    S.hh_weights = build_household_table(S.hh_ids, S.hh_weights)

    st.subheader("Shop weights and provinces")
    shop_data = build_shop_table(S.shop_ids, S.shop_weights, S.shop_province)
    S.shop_weights = {sid: data["w_SHOP"] for sid, data in shop_data.items()}
    S.shop_province = {sid: data["province"] for sid, data in shop_data.items()}

    st.subheader("Prosumer → Household mapping")
    S.mapping = build_mapping_editor(S.prosumer_ids, S.hh_ids, S.mapping)

    if st.button("Validate mapping"):
        missing = set(S.hh_ids)
        for lst in S.mapping.values():
            for hid in lst:
                missing.discard(hid)
        if missing:
            error_box(f"Households not assigned: {sorted(missing)}")
        else:
            info_box("All households assigned.")

# ---- Page 3: Run Deterministic
elif page == "3) Run Deterministic":
    st.header("Run deterministic scenario")

    required = {
        "calendar": S.hours,
        "PV template": S.pv_series,
        "HH template": S.hh_series,
        "SHOP template": S.shop_series,
        "Zonal prices": S.zonal,
        "Monthly PUN": S.pun_m,
        "Hourly PUN": S.pun_h,
    }
    missing = [name for name, val in required.items() if val is None]
    if missing:
        error_box(f"Missing inputs: {', '.join(missing)}. Complete the previous steps first.")
    else:
        st.write(f"Hours in calendar: {len(S.hours)}")
        st.write(f"Negative-price rule: {NEGATIVE_RULE}")
        if st.button("Run deterministic engine", use_container_width=True):
            prosumers_df = pd.DataFrame({
                "prosumer_id": S.prosumer_ids,
                "kWp": [S.kwp_map.get(pid, 0.0) for pid in S.prosumer_ids],
                "w_self": [S.w_self_map.get(pid, 0.0) for pid in S.prosumer_ids],
                "province": [S.prosumer_province.get(pid, "Province-1") for pid in S.prosumer_ids],
            })
            households_df = pd.DataFrame({
                "household_id": S.hh_ids,
                "w_HH": [S.hh_weights.get(hid, 1.0) for hid in S.hh_ids],
            })
            shops_df = pd.DataFrame({
                "shop_id": S.shop_ids,
                "w_SHOP": [S.shop_weights.get(sid, 1.0) for sid in S.shop_ids],
                "province": [S.shop_province.get(sid, "Province-1") for sid in S.shop_ids],
            })

            price_layer = build_price_layers(
                s_hh=S.s_HH,
                s_sh=S.s_SH,
                spread_split_hh=S.spread_split_HH,
                platform_gap_hh=S.platform_gap_HH,
                spread_split_sh=S.spread_split_SH,
                platform_gap_sh=S.platform_gap_SH,
                delta_unm=S.delta_unm,
                loss_factor=S.loss_factor,
                hh_gift=S.hh_gift,
            )
            fees = {
                "f_pros": S.f_pros,
                "f_hh": S.f_hh,
                "f_shop": S.f_shop,
                "platform_fixed": S.platform_fixed,
            }

            with spinner_block("Computing deterministic flows..."):
                try:
                    S.result = run_deterministic(
                        hours=S.hours,
                        pv_series=S.pv_series,
                        hh_series=S.hh_series,
                        shop_series=S.shop_series,
                        prosumers=prosumers_df,
                        households=households_df,
                        shops=shops_df,
                        mapping=S.mapping,
                        zonal=S.zonal,
                        pun_hourly=S.pun_h,
                        price_layer=price_layer,
                        efficiency=S.efficiency,
                        hh_gift=S.hh_gift,
                        fees=fees,
                    )
                    info_box("Deterministic scenario computed.")
                except Exception as exc:
                    S.result = None
                    error_box(f"Simulation failed: {exc}")

        if S.result is not None:
            st.success("Last run completed. Proceed to KPI Dashboard for results.")

# ---- Page 4: KPI Dashboard
elif page == "4) KPI Dashboard":
    st.header("KPI Dashboard")
    render_kpi_dashboard(S)

# ---- Page 5: Sankey Diagram
elif page == "5) Sankey Diagram":
    st.header("Sankey Diagram")
    energy_fig, energy_note, energy_placeholder = build_energy_sankey_chart(S)
    st.subheader("Energy flows")
    if energy_placeholder:
        st.info(energy_note)
    else:
        st.caption(energy_note)
    st.plotly_chart(energy_fig, use_container_width=True)

    economic_fig, economic_note, economic_placeholder = build_economic_sankey_chart(S)
    st.subheader("Economic flows")
    if economic_placeholder:
        st.info(economic_note)
    else:
        st.caption(economic_note)
    st.plotly_chart(economic_fig, use_container_width=True)

# ---- Page 6: Exports
elif page == "6) Exports":
    st.header("Exports")
    if S.result is None:
        warn_box("Run the deterministic engine before exporting.")
    else:
        st.download_button(
            "Download COMMUNITY_ALL_IN_ONE.xlsx",
            export_all_in_one_xlsx(S),
            file_name="COMMUNITY_ALL_IN_ONE.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download hourly price facts (CSV)",
            export_hourly_facts(S, fmt="csv"),
            file_name=f"hourly_prices_{S.year}.csv",
            mime="text/csv",
        )

# ---- About page
else:
    st.header("About this deterministic VEC app")
    st.markdown(
        """
        This version replaces the Monte Carlo engine with a single deterministic year.
        Users provide per-kWp PV generation and median load templates for twelve
        season/day-type clusters (Autumn/Spring/Summer/Winter × Sunday/Saturday/Weekday).
        Actor-level weights scale these templates to actual demand and supply.

        Matching, pricing, and cash-flow rules follow the original VEC specification: the
        water-filling algorithm first serves designated households, then allocates provincial
        residuals to shops, and applies the locked pricing formulas (including the negative-price
        override and HH-Gift toggle).

        All KPIs, balances, and exports are computed on hourly data for the selected year.
        """
    )
