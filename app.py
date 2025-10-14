import streamlit as st
from src.state import get_state, reset_all_state
from src.io_utils import (
    read_households_excel, read_shops_excel, read_pv_excel,
    read_zonal_csv, read_pun_monthly_csv, ensure_price_hours, expand_monthly_pun_to_hours
)
from src.fitting import fit_households, fit_shops
from src.pv_model import fit_pv_optionB_v3
from src.ui_components import (
    show_calibration_tables, show_histograms_and_qq,
    build_mapping_editor, build_kwp_table, spinner_block,
    info_box, warn_box, error_box
)
from src.prices import build_price_layers, NEGATIVE_RULE
from src.simulation import run_monte_carlo, set_fee_params
from src.kpi import render_kpi_dashboard, compute_kpi_distributions
from src.exporters import (
    build_calibration_workbook_hh, build_calibration_workbook_shop,
    build_calibration_workbook_pv, export_kpi_quantiles, export_kpi_samples,
    export_all_in_one_xlsx, export_hourly_facts, build_household_template,
    build_shop_template, build_pv_excel_template, build_zonal_price_template,
    build_pun_monthly_template,
)

st.set_page_config(page_title="Virtual Energy Community", layout="wide")

# ---- Sidebar navigation
st.sidebar.title("VEC App")
page = st.sidebar.radio(
    "Navigation",
    ["1) Upload & Fit", "2) Scenario Builder", "3) Run Monte Carlo", "4) KPI Dashboard", "5) Exports", "About"],
)

# Persistent state (stored in st.session_state)
S = get_state()

# ---- Page 1: Upload & Fit
if page == "1) Upload & Fit":
    st.header("Data Upload & Calibration (Fitting)")
    st.markdown("""
Upload the **Households**, **Small Shops**, **PV per-kWp Excel**, and **Prices**.
**Prices rule (NEW):** Retail = **Monthly PUN (€/kWh)** + spread. Zonal remains hourly (€/MWh) for export & split base.
    """)

    with st.expander("Upload — Households (Excel: 1 sheet per household; 15-min kW)", expanded=True):
        st.download_button(
            "Download 2022 HH template (15-min)",
            build_household_template(),
            file_name="households_template_2022.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Pre-filled timestamp column from 1 Jan 2022 00:30 to 1 Jan 2023 00:00 (Europe/Rome).",
        )
        st.markdown("**Modello di carico per le famiglie (HH)**")
        st.latex(r"E_{c,h,d} = \mu_{c,h} \cdot D_{c,d} \cdot e^{\eta_{c,h,d}}")
        with st.popover("ℹ️"):
            st.markdown(
                r"""
                * **\(c\)** is the seasonal/weekly cluster defined by the hour; for each cluster we estimate \(\mu_{c,h}\) as the median hourly profile of the representative day.
                * **\(D_{c,d}\)** is the lognormal daily scaler: \(D_{c,d} = \exp(\sigma_{\ln D}(c)\,Z_d)\) with \(Z_d \sim \mathcal{N}(0,1)\).
                * **\(\eta_{c,h,d}\)** is the hourly lognormal perturbation: \(\eta_{c,h,d} \sim \mathcal{N}(0, \sigma_{\eta}(c,h))\).
                * The best fitting derives the cluster baseline, daily log-scaler spread, and hourly residual variance directly from the uploaded measurements.
                """
            )
        hh_file = st.file_uploader("Households.xlsx", type=["xlsx"])
        if hh_file:
            with spinner_block("Reading Households Excel..."):
                S.hh_df = read_households_excel(hh_file)  # hourly kWh per household_id
            info_box(f"Households loaded: {S.hh_df['household_id'].nunique()} meters, hours = {S.hh_df['timestamp'].nunique()}.")

    with st.expander("Upload — Small Shops (Excel: 1 sheet per shop; 15-min kWh in 'ActiveEnergy_Generale')", expanded=True):
        st.download_button(
            "Download Shops template (15-min, 2022-09-15 to 2023-09-15)",
            build_shop_template(),
            file_name="small_shops_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="20 sheets pre-filled with timestamps every 15 minutes from 15 Sep 2022 to 15 Sep 2023.",
        )
        st.markdown("**Modello di carico per i piccoli negozi (SHOP)**")
        st.latex(r"K_{c,h,d} = I_{c,h} \cdot \mu_{c,h} \cdot D_{c,d} \cdot e^{\eta_{c,h,d}}")
        with st.popover("ℹ️"):
            st.markdown(
                r"""
                * **\(I_{c,h}\)** is a Bernoulli variable with probability \(1 - p_{0}(c,h)\) of being 1, capturing hours with zero demand.
                * **\(\mu_{c,h}\)**, **\(D_{c,d}\)**, and **\(\eta_{c,h,d}\)** follow the same lognormal structure as households, with parameters estimated cluster by cluster from the uploaded data.
                * The fitting recovers the median profile \(\mu_{c,h}\), daily log-scaler dispersion \(\sigma_{\ln D}(c)\), hourly residual spread \(\sigma_{\eta}(c,h)\), and off probability \(p_{0}(c,h)\) for every hour.
                """
            )
        shop_file = st.file_uploader("SmallShops.xlsx", type=["xlsx"])
        if shop_file:
            with spinner_block("Reading Small Shops Excel..."):
                S.shop_df = read_shops_excel(shop_file)  # hourly kWh per shop_id
            info_box(f"Shops loaded: {S.shop_df['shop_id'].nunique()} meters, hours = {S.shop_df['timestamp'].nunique()}.")

    with st.expander("Upload — Prosumer PV (Excel: per-kWp hourly kWh)", expanded=True):
        st.download_button(
            "Download PV Excel template (hourly, 2018-2023)",
            build_pv_excel_template(),
            file_name="pv_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Hourly timestamps (Europe/Rome) from 1 Jan 2018 00:30 to 31 Dec 2023 23:30.",
        )
        st.markdown("**Modello di produzione FV per kWp**")
        st.latex(r"G_{s,h,d} = kWp \cdot S_{s,h} \cdot M_{s,d} \cdot \rho_{s,h,d} \cdot e^{\epsilon_{s,h,d}}")
        with st.popover("ℹ️"):
            st.markdown(
                r"""
                * **\(S_{s,h}\)** is the normalized seasonal envelope (average profile for season \(s\) and hour \(h\)).
                * **\(M_{s,d}\)** is the daily energy drawn from a log-logistic distribution (parameters \(c_s\), \(scale_s\)) whose daily state follows a Markov chain ("clear" or "cloudy" days).
                * **\(\rho_{s,h,d}\)** is a Beta factor (\(\alpha_s, \beta_s\)) that modulates intra-day clearness.
                * **\(\epsilon_{s,h,d}\)** is a small Gaussian disturbance to capture residual variability.
                * The best fitting retrieves the seasonal envelope, fits the log-logistic distribution to daily totals, estimates the clear/cloud Markov chain, and calibrates the Beta clearness factors from the uploaded history.
                """
            )
        pv_file = st.file_uploader("PV.xlsx", type=["xlsx"])
        if pv_file:
            with spinner_block("Reading PV Excel..."):
                S.pv_perkwp = read_pv_excel(pv_file)
            info_box(f"PV per-kWp hours: {len(S.pv_perkwp)}.")

    with st.expander("Upload — Prices", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            zonal_year = st.selectbox("Zonal template year", list(range(2024, 2031)), key="zonal_template_year")
            st.download_button(
                f"Download zonal template {zonal_year}",
                build_zonal_price_template(zonal_year),
                file_name=f"zonal_template_{zonal_year}.csv",
                mime="text/csv",
                help="Hourly timestamps for the selected year with placeholder zonal prices.",
            )
            zonal_file = st.file_uploader("zonal.csv (timestamp, zonal_price (EUR_per_MWh))", type=["csv"], key="zonal")
            if zonal_file:
                S.zonal = read_zonal_csv(zonal_file)
                info_box(f"Zonal rows: {len(S.zonal)} (hourly).")
        with col2:
            pun_year = st.selectbox("Monthly PUN template year", list(range(2024, 2031)), key="pun_template_year")
            st.download_button(
                f"Download monthly PUN template {pun_year}",
                build_pun_monthly_template(pun_year),
                file_name=f"pun_monthly_template_{pun_year}.csv",
                mime="text/csv",
                help="Monthly timestamps (first day of month) for the selected year.",
            )
            pun_file = st.file_uploader("PUN_monthly.csv (timestamp(any day in month), PUN (EUR_per_kWh))", type=["csv"], key="pun")
            if pun_file:
                S.pun_m = read_pun_monthly_csv(pun_file)
                info_box(f"Monthly PUN rows: {len(S.pun_m)} (€/kWh).")

        # Build calendar from ZONAL hours, then expand monthly PUN to hours
        if S.zonal is not None:
            try:
                S.hours = ensure_price_hours(S.zonal)
                info_box(f"Calendar ok from zonal: {len(S.hours)} hours.")
                if S.pun_m is not None:
                    S.pun_h = expand_monthly_pun_to_hours(S.pun_m, S.hours)
                    info_box("Expanded monthly PUN to hourly (€/kWh).")
            except Exception as e:
                error_box(str(e))

    st.divider()
    st.subheader("Fit Distributions")
    if st.button("Run fitting (HH, SHOP, PV)"):
        if S.hh_df is None or S.shop_df is None or S.pv_perkwp is None or S.hours is None:
            error_box("Please upload Households, Shops, PV Excel, and Zonal (for calendar) first. Monthly PUN is also required for simulation.")
        else:
            with spinner_block("Fitting Households..."):
                S.hh_fit, S.hh_diag = fit_households(S.hh_df)
            with spinner_block("Fitting Shops..."):
                S.shop_fit, S.shop_diag = fit_shops(S.shop_df)
            with spinner_block("Fitting PV Option-B v3..."):
                S.pv_fit, S.pv_diag = fit_pv_optionB_v3(S.pv_perkwp)

            info_box("Fitting complete. See parameter tables and diagnostics below.")

    if S.hh_fit is not None or S.shop_fit is not None or S.pv_fit is not None:
        st.subheader("Calibration Results (parameters)")
        show_calibration_tables(S)

        st.subheader("Diagnostics (histograms + QQ)")
        show_histograms_and_qq(S)

# ---- Page 2: Scenario Builder
elif page == "2) Scenario Builder":
    st.header("Scenario Builder — Community sizing, mapping, economics")

    colA, colB = st.columns([2, 1])
    with colA:
        S.N_P = st.number_input("Number of Prosumers (N_P)", min_value=1, value=S.N_P or 3, step=1)
        S.N_HH = st.number_input("Number of Households (N_HH)", min_value=1, value=S.N_HH or 6, step=1)
        S.N_SHOP = st.number_input("Number of Small Shops (N_SHOP)", min_value=1, value=S.N_SHOP or 3, step=1)
        S.seed = st.number_input("Random Seed (integer)", value=S.seed or 12345, step=1)
        S.S = st.number_input("Monte Carlo Scenarios (S)", min_value=100, max_value=10000, value=S.S or 5000, step=100)
    with colB:
        st.write("Special toggles")
        S.hh_gift = st.checkbox("Enable HH-Gift (HH matched kWh billed at 0 to both sides)", value=S.hh_gift or False)
        st.caption("When HH-Gift is ON: P_pros,HH=0; P_cons,HH=0; gap_HH=0.")

    st.subheader("Economic Parameters (year-constant spreads; single zone)")
    col1, col2, col3 = st.columns(3)
    with col1:
        S.s_HH = st.number_input("Retail spread s_HH (€/kWh)", min_value=0.0, value=S.s_HH or 0.10, step=0.01, format="%.5f")
        S.alpha_HH = st.slider("alpha_HH (split)", 0.0, 1.0, S.alpha_HH or 0.5, 0.01)
        S.phi_HH = st.slider("phi_HH (gap share)", 0.0, 1.0, S.phi_HH or 0.3, 0.01)
    with col2:
        S.s_SH = st.number_input("Retail spread s_SHOP (€/kWh)", min_value=0.0, value=S.s_SH or 0.12, step=0.01, format="%.5f")
        S.alpha_SH = st.slider("alpha_SHOP (split)", 0.0, 1.0, S.alpha_SH or 0.5, 0.01)
        S.phi_SH = st.slider("phi_SHOP (gap share)", 0.0, 1.0, S.phi_SH or 0.3, 0.01)
    with col3:
        S.delta_unm = st.number_input("δ_unm (€/kWh) export uplift", value=S.delta_unm or 0.0, step=0.01, format="%.5f")
        S.loss_factor = st.slider("Loss factor ℓ (platform gap on delivered)", 0.0, 1.0, S.loss_factor or 0.05, 0.01)

    st.subheader("Platform Fees and Fixed Cost (can be zero)")
    colf1, colf2 = st.columns(2)
    with colf1:
        S.f_pros = st.number_input("Prosumer monthly fee (€/month)", min_value=0.0, value=S.f_pros or 2.0, step=0.5)
        S.f_hh = st.number_input("Household monthly fee (€/month)", min_value=0.0, value=S.f_hh or 1.0, step=0.5)
        S.f_shop = st.number_input("Shop monthly fee (€/month)", min_value=0.0, value=S.f_shop or 1.5, step=0.5)
    with colf2:
        S.platform_fixed = st.number_input("Platform fixed monthly cost (€/month)", min_value=0.0, value=S.platform_fixed or 200.0, step=10.0)

    # Synthetic IDs
    S.prosumer_ids = [f"P{i:03d}" for i in range(1, int(S.N_P) + 1)]
    S.hh_ids = [f"H{i:03d}" for i in range(1, int(S.N_HH) + 1)]
    S.shop_ids = [f"K{i:03d}" for i in range(1, int(S.N_SHOP) + 1)]

    st.subheader("Prosumer Capacities (kWp)")
    S.kwp_map = build_kwp_table(S.prosumer_ids, S.kwp_map)

    st.subheader("Manual Mapping: Prosumer → Designated Households")
    S.mapping = build_mapping_editor(S.prosumer_ids, S.hh_ids, S.mapping)

    if st.button("Validate Scenario"):
        # Hard validations
        missing = set(S.hh_ids)
        for pid, lst in S.mapping.items():
            for hh in lst:
                if hh in missing:
                    missing.remove(hh)
        if len(missing) > 0:
            error_box(f"Unassigned Households: {sorted(missing)}. Assign every HH to at least one prosumer.")
        elif any(S.kwp_map.get(pid, 0) < 0 for pid in S.prosumer_ids):
            error_box("kWp must be >= 0.")
        else:
            info_box("Scenario validation passed.")

# ---- Page 3: Run Monte Carlo
elif page == "3) Run Monte Carlo":
    st.header("Run Monte Carlo")
    required = [S.hh_fit, S.shop_fit, S.pv_fit, S.hours, S.zonal, S.pun_m]
    if any(x is None for x in required):
        error_box("Missing inputs or calibration. Upload Zonal & Monthly PUN, run fitting, then return here.")
    else:
        # Build hourly PUN (€/kWh) from monthly table if not done
        if S.pun_h is None:
            try:
                S.pun_h = expand_monthly_pun_to_hours(S.pun_m, S.hours)
                info_box("Expanded monthly PUN to hourly (€/kWh).")
            except Exception as e:
                error_box(f"Could not expand monthly PUN to hours: {e}")

        st.write("Calendar hours detected:", len(S.hours))
        st.write("Negative-price rule:", NEGATIVE_RULE)
        if st.button("Run Monte Carlo now"):
            # Build price layers callable (NOTE: now retail uses PUN hourly kWh)
            price_layer = build_price_layers(
                s_hh=S.s_HH, s_sh=S.s_SH,
                a_hh=S.alpha_HH, phi_hh=S.phi_HH,
                a_sh=S.alpha_SH, phi_sh=S.phi_SH,
                delta_unm=S.delta_unm, loss_factor=S.loss_factor,
                hh_gift=S.hh_gift
            )

            # Pass platform fees/fixed to simulation module
            set_fee_params(S.f_pros, S.f_hh, S.f_shop, S.platform_fixed)

            with st.spinner("Simulating ..."):
                S.mc = run_monte_carlo(
                    hours=S.hours,
                    hh_fit=S.hh_fit,
                    shop_fit=S.shop_fit,
                    pv_fit=S.pv_fit,
                    kwp_map=S.kwp_map,
                    mapping=S.mapping,
                    prosumer_ids=S.prosumer_ids,
                    hh_ids=S.hh_ids,
                    shop_ids=S.shop_ids,
                    zonal=S.zonal,
                    pun_hourly_kwh=S.pun_h,   # <— hourly PUN in €/kWh
                    price_layer=price_layer,
                    S=int(S.S),
                    seed=int(S.seed),
                )
            info_box("Monte Carlo completed and stored in session.")

# ---- Page 4: KPI Dashboard
elif page == "4) KPI Dashboard":
    st.header("KPI Dashboard")
    if S.mc is None:
        warn_box("No Monte Carlo results yet. Go to **Run Monte Carlo**.")
    else:
        render_kpi_dashboard(S)

# ---- Page 5: Exports
elif page == "5) Exports":
    st.header("Exports")
    if any(x is None for x in [S.hh_fit, S.shop_fit, S.pv_fit]):
        warn_box("Run fitting first to export calibration workbooks.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Download HH calibration workbook", use_container_width=True):
                st.download_button(
                    "Save HH_params.xlsx",
                    data=build_calibration_workbook_hh(S),
                    file_name="HH_params.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        with c2:
            if st.button("Download SHOP calibration workbook", use_container_width=True):
                st.download_button(
                    "Save SHOP_params.xlsx",
                    data=build_calibration_workbook_shop(S),
                    file_name="SHOP_params.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        with c3:
            if st.button("Download PV calibration workbook", use_container_width=True):
                st.download_button(
                    "Save PV_params.xlsx",
                    data=build_calibration_workbook_pv(S),
                    file_name="PV_params.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    st.divider()
    if S.mc is None:
        warn_box("Run Monte Carlo before KPI exports.")
    else:
        # KPI distributions
        kpi_dists, kpi_summary = compute_kpi_distributions(S)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "Download KPI summary (CSV)",
                data=kpi_summary.to_csv(index=False).encode("utf-8"),
                file_name="kpi_summary.csv", mime="text/csv"
            )
        with c2:
            st.download_button(
                "Download KPI quantiles (Parquet)",
                data=export_kpi_quantiles(kpi_summary),
                file_name="kpi_quantiles.parquet", mime="application/octet-stream"
            )
        with c3:
            st.download_button(
                "Download KPI samples (CSV)",
                data=export_kpi_samples(kpi_dists).encode("utf-8"),
                file_name="kpi_samples.csv", mime="text/csv"
            )

        st.divider()
        st.subheader("Optional large exports (hourly facts)")
        colx, coly = st.columns(2)
        with colx:
            st.download_button(
                "Download hourly facts (CSV, may be large)",
                data=export_hourly_facts(S, fmt="csv"),
                file_name="hourly_facts.csv", mime="text/csv"
            )
        with coly:
            st.download_button(
                "Download hourly facts (Parquet, recommended)",
                data=export_hourly_facts(S, fmt="parquet"),
                file_name="hourly_facts.parquet", mime="application/octet-stream"
            )

# ---- About
else:
    st.header("About this app")
    st.markdown("""
**Virtual Energy Community (VEC)** app implements the full pipeline:

1. Upload & fit distributions (Households, Shops, PV Option-B v3).  
2. Build scenarios: choose counts (N_P, N_HH, N_SHOP), enter kWp, map Prosumer→Households.  
3. Monte Carlo simulation aligned to **zonal hours**; **Retail = Monthly PUN (€/kWh) + spread**.  
4. KPI dashboard with **distributions** (mean, sd, p05, p10, p50, p90, p95).  
5. Exports (Excel + CSV/Parquet), including **calibration workbooks** and **hourly facts**.
""")
    if st.button("Reset all session data"):
        reset_all_state()
        st.success("Session state cleared.")
