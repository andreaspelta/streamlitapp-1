import streamlit as st
import pandas as pd

from typing import Dict, List

from .state import AppState


def _px():
    import plotly.express as px

    return px

class spinner_block:
    def __init__(self, msg): self.msg=msg
    def __enter__(self): self.ctx = st.spinner(self.msg); self.ctx.__enter__()
    def __exit__(self, exc_type, exc, tb): self.ctx.__exit__(exc_type, exc, tb)

def info_box(msg): st.info(msg)
def warn_box(msg): st.warning(msg)
def error_box(msg): st.error(msg)

def show_calibration_tables(S: AppState):
    col1, col2, col3 = st.columns(3)
    if S.hh_fit:
        with col1:
            st.markdown("**Households — μ_{c,h}**")
            st.dataframe(S.hh_fit["mu"].round(4))
    if S.shop_fit:
        with col2:
            st.markdown("**Shops — μ_{c,h}**")
            st.dataframe(S.shop_fit["mu"].round(4))
    if S.pv_fit:
        with col3:
            st.markdown("**PV — S_{s,h} envelope**")
            st.dataframe(S.pv_fit["S"].round(4))

    with st.expander("More parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        if S.hh_fit:
            with c1:
                st.write("HH σ_lnD (per cluster)")
                st.dataframe(S.hh_fit["sigma_lnD"].round(4))
                st.write("HH σ_resid (per cluster/hour)")
                st.dataframe(S.hh_fit["sigma_resid"].round(4))
        if S.shop_fit:
            with c2:
                st.write("SHOP σ_lnD")
                st.dataframe(S.shop_fit["sigma_lnD"].round(4))
                st.write("SHOP σ_resid")
                st.dataframe(S.shop_fit["sigma_resid"].round(4))
                st.write("SHOP p_zero")
                st.dataframe(S.shop_fit["p_zero"].round(3))
        if S.pv_fit:
            with c3:
                st.write("PV Log-logistic (per season)")
                st.json(S.pv_fit["loglogistic"])
                st.write("PV Markov & Beta (per season)")
                st.json(S.pv_fit["markov"])

def show_histograms_and_qq(S: AppState):
    # HH lnD & resid
    if S.hh_diag:
        px = _px()
        st.subheader("Households Diagnostics")
        lnD = S.hh_diag["lnD"]
        fig = px.histogram(lnD, x="lnD", color="cluster", nbins=50, title="HH ln D")
        st.plotly_chart(fig, use_container_width=True)

        resid = S.hh_diag["resid"]
        fig2 = px.histogram(resid, x="resid", color="cluster", nbins=50, title="HH residuals (log-scale)")
        st.plotly_chart(fig2, use_container_width=True)

    if S.shop_diag:
        px = _px()
        st.subheader("Shops Diagnostics")
        lnD = S.shop_diag["lnD"]
        fig = px.histogram(lnD, x="lnD", color="cluster", nbins=50, title="SHOP ln D")
        st.plotly_chart(fig, use_container_width=True)

        resid = S.shop_diag["resid"]
        fig2 = px.histogram(resid, x="resid", color="cluster", nbins=50, title="SHOP residuals (log-scale)")
        st.plotly_chart(fig2, use_container_width=True)

    if S.pv_diag:
        px = _px()
        st.subheader("PV Diagnostics")
        env = S.pv_diag["envelope"]
        fig = px.line(env.melt(id_vars="season", var_name="hour", value_name="S"),
                      x="hour", y="S", color="season", title="PV Envelope S_{s,h}")
        st.plotly_chart(fig, use_container_width=True)

def build_kwp_table(prosumer_ids: List[str], existing: Dict[str, float]) -> Dict[str, float]:
    df = pd.DataFrame({"prosumer_id": prosumer_ids,
                       "kWp": [existing.get(pid, 5.0) for pid in prosumer_ids]})
    edited = st.data_editor(df, num_rows="fixed", use_container_width=True, key="kwp_editor")
    return {row["prosumer_id"]: float(row["kWp"]) for _, row in edited.iterrows()}

def build_mapping_editor(prosumer_ids: List[str], hh_ids: List[str], existing: Dict[str, list]) -> Dict[str, list]:
    st.caption("Select designated Households for each prosumer (manual mapping). Every HH must be assigned at least once.")
    out = {}
    cols = st.columns(min(3, len(prosumer_ids)) or 1)
    for i, pid in enumerate(prosumer_ids):
        with cols[i % len(cols)]:
            sel = st.multiselect(f"{pid} → HH", hh_ids, default=existing.get(pid, []), key=f"map_{pid}")
            out[pid] = sel
    return out
