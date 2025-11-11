"""Streamlit page for the household profile reshaping workflow."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .profile_reshaping import (
    ReshapingParameters,
    build_excel_payload,
    reshape_household_profile,
)
from .state import get_state
from .ui_components import error_box, info_box, warn_box


MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def _default_targets(value: float) -> dict[int, float]:
    return {i + 1: value for i in range(12)}


def render_profile_reshaping_page() -> None:
    st.header("Profile reshaping")
    st.write(
        "Adapt the household load profile to better align with PV surplus while"
        " respecting monthly coverage targets. The reshaped template can be exported"
        " and used like any other HH median profile."
    )

    S = get_state()

    if S.hours is None:
        warn_box("Set the simulation year in 'Templates & Inputs' before continuing.")
        return

    if S.hh_medians is None:
        warn_box("Upload a household median template in 'Templates & Inputs' first.")
        return

    if S.pv_series is None:
        warn_box("Upload a PV template in 'Templates & Inputs' to enable reshaping.")
        return

    previous = S.hh_reshaping_params or {}

    default_target = float(previous.get("base_target", 0.6))
    pv_capacity_kwp = float(previous.get("pv_capacity_kwp", 1.0))
    min_fraction = float(previous.get("min_fraction", 0.3))
    max_fraction = float(previous.get("max_fraction", 1.8))
    absolute_min = float(previous.get("absolute_min", 0.05))
    absolute_max = float(previous.get("absolute_max", 4.0))
    monthly_limit = float(previous.get("monthly_limit", 0.1))
    monthly_targets = previous.get("monthly_targets") or _default_targets(default_target)

    with st.expander("Coverage target", expanded=True):
        col_target, col_toggle = st.columns([3, 2])
        with col_target:
            base_target_pct = st.slider(
                "Monthly coverage target (%)",
                min_value=0,
                max_value=100,
                value=int(round(default_target * 100)),
                help="Desired share of household demand served by PV each month.",
            )
        with col_toggle:
            customise = st.checkbox(
                "Customise targets per month",
                value=bool(previous.get("customise", False)),
            )

        if customise:
            target_df = pd.DataFrame(
                {
                    "Month": MONTH_LABELS,
                    "Target coverage (%)": [
                        float(monthly_targets.get(i + 1, base_target_pct / 100)) * 100
                        for i in range(12)
                    ],
                }
            )
            edited_df = st.data_editor(
                target_df,
                num_rows="fixed",
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Target coverage (%)": st.column_config.NumberColumn(
                        min_value=0.0,
                        max_value=100.0,
                        format="%.1f",
                    )
                },
                key="reshaping_targets_editor",
            )
            monthly_targets = {
                idx + 1: float(row["Target coverage (%)"]) / 100.0
                for idx, row in edited_df.iterrows()
            }
        else:
            monthly_targets = _default_targets(base_target_pct / 100.0)

    with st.expander("Operational constraints", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            pv_capacity_kwp = st.number_input(
                "PV capacity considered (kWp)",
                min_value=0.0,
                value=pv_capacity_kwp,
                step=0.5,
                help="Scales the uploaded PV per-kWp profile to represent the available generation.",
            )
            min_fraction = st.slider(
                "Minimum hourly fraction of original load",
                min_value=0.0,
                max_value=1.0,
                value=float(min_fraction),
                step=0.05,
                help="Protect standby loads by keeping at least this share of the original hour.",
            )
            absolute_min = st.number_input(
                "Minimum standby load (kWh)",
                min_value=0.0,
                value=float(absolute_min),
                step=0.05,
                help="Absolute floor per hour; set to 0 to rely solely on the relative limit.",
            )
        with col2:
            max_fraction = st.slider(
                "Maximum hourly multiplier",
                min_value=1.0,
                max_value=3.0,
                value=float(max_fraction),
                step=0.1,
                help="Cap the amplification of each hour relative to the original profile.",
            )
            absolute_max = st.number_input(
                "Absolute hourly cap (kWh)",
                min_value=0.0,
                value=float(absolute_max),
                step=0.5,
                help="Hard ceiling per hour; set to 0 to disable the cap.",
            )
            monthly_limit = st.slider(
                "Monthly energy tolerance (±%)",
                min_value=0,
                max_value=30,
                value=int(round(monthly_limit * 100)),
                help="Allowable deviation of monthly energy from the original profile.",
            ) / 100.0

    params = ReshapingParameters(
        monthly_targets=monthly_targets,
        pv_capacity_kwp=pv_capacity_kwp,
        min_hourly_fraction=min_fraction,
        max_hourly_fraction=max_fraction,
        absolute_min_kwh=absolute_min,
        absolute_max_kwh=absolute_max,
        monthly_scale_limit=monthly_limit,
    )

    if st.button("Run profile reshaping", use_container_width=True):
        try:
            with st.spinner("Computing reshaped profile..."):
                result = reshape_household_profile(
                    hours=S.hours,
                    medians=S.hh_medians,
                    pv_series=S.pv_series,
                    params=params,
                )
        except Exception as exc:
            error_box(f"Reshaping failed: {exc}")
            return

        S.hh_reshaped_medians = result.reshaped_medians
        S.hh_reshaping_summary = result.monthly_summary
        S.hh_reshaping_params = {
            "base_target": base_target_pct / 100.0,
            "customise": customise,
            "monthly_targets": monthly_targets,
            "pv_capacity_kwp": pv_capacity_kwp,
            "min_fraction": min_fraction,
            "max_fraction": max_fraction,
            "absolute_min": absolute_min,
            "absolute_max": absolute_max,
            "monthly_limit": monthly_limit,
        }

        info_box("Profile reshaping completed. Review the summary below and export the template.")

        if result.warnings:
            for message in result.warnings:
                warn_box(message)

        summary_df = result.monthly_summary.copy()
        if not summary_df.empty:
            summary_df["Month"] = summary_df["month"].apply(lambda m: MONTH_LABELS[m - 1])
            summary_df["Target coverage (%)"] = summary_df["target_coverage"] * 100.0
            summary_df["Achieved coverage (%)"] = summary_df["achieved_coverage"] * 100.0
            summary_df = summary_df[
                [
                    "Month",
                    "Target coverage (%)",
                    "Achieved coverage (%)",
                    "original_load_kWh",
                    "reshaped_load_kWh",
                    "pv_energy_kWh",
                    "matched_energy_kWh",
                    "feasible",
                    "note",
                ]
            ]
            st.subheader("Monthly performance")
            st.dataframe(
                summary_df,
                use_container_width=True,
                column_config={
                    "original_load_kWh": st.column_config.NumberColumn(format="%.1f"),
                    "reshaped_load_kWh": st.column_config.NumberColumn(format="%.1f"),
                    "pv_energy_kWh": st.column_config.NumberColumn(format="%.1f"),
                    "matched_energy_kWh": st.column_config.NumberColumn(format="%.1f"),
                    "Target coverage (%)": st.column_config.NumberColumn(format="%.1f"),
                    "Achieved coverage (%)": st.column_config.NumberColumn(format="%.1f"),
                },
            )

            total_load = summary_df["reshaped_load_kWh"].sum()
            total_matched = summary_df["matched_energy_kWh"].sum()
            overall_cov = total_matched / total_load if total_load > 1e-9 else 0.0

            st.subheader("Aggregate indicators")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total load (kWh)", f"{total_load:,.1f}")
            with col_b:
                st.metric("Shared energy (kWh)", f"{total_matched:,.1f}")
            with col_c:
                st.metric("Overall coverage (%)", f"{overall_cov * 100:.1f}")

        st.subheader("Reshaped medians preview")
        st.dataframe(
            result.reshaped_medians,
            use_container_width=True,
        )

        download_name = f"hh_profile_reshaped_{S.year}.xlsx"
        st.download_button(
            "Download reshaped HH template",
            data=build_excel_payload(result.reshaped_medians, year=S.year),
            file_name=download_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

