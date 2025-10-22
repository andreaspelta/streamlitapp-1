import pandas as pd
import streamlit as st

from .state import AppState


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def compute_kpi_summary(S: AppState) -> pd.DataFrame:
    if S.result is None:
        return pd.DataFrame(columns=["KPI", "unit", "value"])

    totals = S.result.totals

    matched_hh = totals.get("matched_hh", 0.0)
    matched_shop = totals.get("matched_shop", 0.0)
    import_hh = totals.get("import_hh", 0.0)
    import_shop = totals.get("import_shop", 0.0)
    pv_gen = totals.get("pv_gen", 0.0)
    hh_demand = totals.get("hh_demand", 0.0)
    shop_demand = totals.get("shop_demand", 0.0)
    export = totals.get("export", 0.0)
    consumer_cost = totals.get("cons_cost", 0.0)
    consumer_base = totals.get("cons_baseline", 0.0)
    savings = consumer_base - consumer_cost

    coverage_total = _safe_ratio(matched_hh + matched_shop, hh_demand + shop_demand) * 100
    coverage_hh = _safe_ratio(matched_hh, matched_hh + import_hh) * 100
    coverage_shop = _safe_ratio(matched_shop, matched_shop + import_shop) * 100

    pv_util = _safe_ratio(matched_hh + matched_shop, pv_gen) * 100
    export_ratio = _safe_ratio(export, pv_gen) * 100

    autarky_total = (1.0 - _safe_ratio(import_hh + import_shop, hh_demand + shop_demand)) * 100
    autarky_hh = (1.0 - _safe_ratio(import_hh, matched_hh + import_hh)) * 100
    autarky_shop = (1.0 - _safe_ratio(import_shop, matched_shop + import_shop)) * 100

    savings_pct = _safe_ratio(savings, consumer_base) * 100

    rows = [
        ("PV generation", "kWh", pv_gen),
        ("Prosumer demand", "kWh", totals.get("pros_demand", 0.0)),
        ("HH demand", "kWh", hh_demand),
        ("Shop demand", "kWh", shop_demand),
        ("Matched to HH", "kWh", matched_hh),
        ("Matched to SHOP", "kWh", matched_shop),
        ("Exports", "kWh", export),
        ("Imports HH", "kWh", import_hh),
        ("Imports SHOP", "kWh", import_shop),
        ("Community demand", "kWh", hh_demand + shop_demand),
        ("Coverage — total", "%", coverage_total),
        ("Coverage — HH", "%", coverage_hh),
        ("Coverage — SHOP", "%", coverage_shop),
        ("PV utilization", "%", pv_util),
        ("Export ratio", "%", export_ratio),
        ("Autarky — total", "%", autarky_total),
        ("Autarky — HH", "%", autarky_hh),
        ("Autarky — SHOP", "%", autarky_shop),
        ("Prosumer revenue", "€", totals.get("pros_rev", 0.0)),
        ("Consumer baseline", "€", consumer_base),
        ("Consumer cost — total", "€", consumer_cost),
        ("Consumer savings", "€", savings),
        ("Consumer savings", "%", savings_pct),
        ("Platform gap revenue", "€", totals.get("platform_gap", 0.0)),
        ("Platform fees", "€", totals.get("platform_fees", 0.0)),
        ("Platform fixed cost", "€", totals.get("platform_fixed", 0.0)),
        ("Platform margin", "€", totals.get("platform_margin", 0.0)),
    ]

    df = pd.DataFrame(rows, columns=["KPI", "unit", "value"])
    return df


def render_kpi_dashboard(S: AppState):
    summary = compute_kpi_summary(S)
    if summary.empty:
        st.warning("Run the deterministic scenario to view KPIs.")
        return

    st.subheader("Headline rates")
    highlight = (
        summary[summary["unit"] == "%"].set_index("KPI").reindex(
            [
                "Coverage — total",
                "PV utilization",
                "Autarky — total",
                "Consumer savings",
            ]
        )
    ).dropna()
    if not highlight.empty:
        cols = st.columns(len(highlight))
        for col, (kpi, row) in zip(cols, highlight.iterrows()):
            with col:
                st.metric(kpi, f"{row['value']:.2f}%")

    st.subheader("Energy balances (kWh)")
    st.dataframe(summary[summary["unit"] == "kWh"].set_index("KPI").round(3))

    st.subheader("Rates (%)")
    st.dataframe(summary[summary["unit"] == "%"].set_index("KPI").round(2))

    st.subheader("Economics (€)")
    st.dataframe(summary[summary["unit"] == "€"].set_index("KPI").round(2))
