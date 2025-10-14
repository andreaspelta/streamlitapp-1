import numpy as np
import pandas as pd
import streamlit as st

from typing import Dict, Any

from .state import AppState


def _px():
    import plotly.express as px

    return px

def _stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=np.nan, sd=np.nan, p05=np.nan, p10=np.nan, p50=np.nan, p90=np.nan, p95=np.nan)
    return dict(
        mean=float(np.mean(x)),
        sd=float(np.std(x, ddof=1) if x.size>1 else 0.0),
        p05=float(np.percentile(x,5)), p10=float(np.percentile(x,10)),
        p50=float(np.percentile(x,50)), p90=float(np.percentile(x,90)),
        p95=float(np.percentile(x,95))
    )

def compute_kpi_distributions(S: AppState):
    mc = S.mc
    m_hh = mc["matched_hh"]; m_sh = mc["matched_shop"]
    imp_hh = mc["import_hh"]; imp_sh = mc["import_shop"]
    exp = mc["export"]; rev = mc["pros_rev"]; cost = mc["cons_cost"]
    pv = mc["pv_gen"]; cons = mc["cons_total"]
    base = mc["cons_baseline"]
    plat_gap = mc["platform_gap"]; fees = mc["platform_fees"]; fixed = mc["platform_fixed"]
    plat_margin = mc["platform_margin"]

    # Rates
    coverage_total = np.divide(m_hh+m_sh, cons, out=np.zeros_like(cons), where=cons>0)*100
    coverage_hh = np.divide(m_hh, m_hh+imp_hh, out=np.zeros_like(m_hh), where=(m_hh+imp_hh)>0)*100
    coverage_sh = np.divide(m_sh, m_sh+imp_sh, out=np.zeros_like(m_sh), where=(m_sh+imp_sh)>0)*100
    autarky_total = (1.0 - np.divide(imp_hh+imp_sh, cons, out=np.zeros_like(cons), where=cons>0))*100
    autarky_hh = (1.0 - np.divide(imp_hh, m_hh+imp_hh, out=np.zeros_like(imp_hh), where=(m_hh+imp_hh)>0))*100
    autarky_sh = (1.0 - np.divide(imp_sh, m_sh+imp_sh, out=np.zeros_like(imp_sh), where=(m_sh+imp_sh)>0))*100
    pv_util = np.divide(m_hh+m_sh, pv, out=np.zeros_like(pv), where=pv>0)*100
    export_ratio = np.divide(exp, pv, out=np.zeros_like(exp), where=pv>0)*100

    savings = base - cost
    savings_pct = np.divide(savings, base, out=np.zeros_like(savings), where=base>0)*100

    dists = {
        "PV generation (kWh)": pv,
        "Matched to HH (kWh)": m_hh,
        "Matched to SHOP (kWh)": m_sh,
        "Exports (kWh)": exp,
        "Imports HH (kWh)": imp_hh,
        "Imports SHOP (kWh)": imp_sh,
        "Community demand (kWh)": cons,

        "Coverage — total (%)": coverage_total,
        "Coverage — HH (%)": coverage_hh,
        "Coverage — SHOP (%)": coverage_sh,
        "PV utilization (%)": pv_util,
        "Export ratio (%)": export_ratio,
        "Autarky — total (%)": autarky_total,
        "Autarky — HH (%)": autarky_hh,
        "Autarky — SHOP (%)": autarky_sh,

        "Prosumer revenue (€)": rev,
        "Consumer baseline (€)": base,
        "Consumer cost — total (€)": cost,
        "Consumer savings (€)": savings,
        "Consumer savings (%)": savings_pct,
        "Platform gap revenue (€)": plat_gap,
        "Platform fees (€)": fees,
        "Platform fixed cost (€)": fixed,
        "Platform margin (€)": plat_margin,
    }

    rows = []
    for k, v in dists.items():
        unit = "%" if "(%)" in k else ("€" if "€" in k else "kWh")
        stt = _stats(np.asarray(v))
        stt.update({"KPI": k.replace(" (%)","").replace(" (€)",""), "unit": unit})
        rows.append(stt)
    summary = pd.DataFrame(rows)[["KPI","unit","mean","sd","p05","p10","p50","p90","p95"]]
    return dists, summary

def render_kpi_dashboard(S: AppState):
    dists, summary = compute_kpi_distributions(S)

    st.subheader("Headline medians")
    med = summary.set_index("KPI")["p50"]
    keep = [x for x in ["Coverage — total","PV utilization","Autarky — total","Consumer savings (%)"] if x in med.index]
    if keep:
        px = _px()
        fig = px.bar(med[keep], title="Headline medians")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Energy (kWh)")
    st.dataframe(summary[summary["unit"]=="kWh"].set_index("KPI").round(3))

    st.subheader("Energy Rates (%)")
    st.dataframe(summary[summary["unit"]=="%"].set_index("KPI").round(2))

    st.subheader("Economics (€)")
    st.dataframe(summary[summary["unit"]=="€"].set_index("KPI").round(2))

    st.caption("All statistics are Monte-Carlo distributions: mean, sd, p05, p10, p50, p90, p95.")
