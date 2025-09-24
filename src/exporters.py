import io
import pandas as pd
import numpy as np
from typing import Dict
from .state import AppState

def build_calibration_workbook_hh(S: AppState) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xl:
        S.hh_fit["mu"].to_excel(xl, sheet_name="mu_c_h")
        S.hh_fit["sigma_lnD"].to_excel(xl, sheet_name="sigma_lnD")
        S.hh_fit["sigma_resid"].to_excel(xl, sheet_name="sigma_resid")
        # Diagnostics summaries
        pd.DataFrame(S.hh_diag["medday"]).to_excel(xl, sheet_name="medday")
    return out.getvalue()

def build_calibration_workbook_shop(S: AppState) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xl:
        S.shop_fit["mu"].to_excel(xl, sheet_name="mu_c_h")
        S.shop_fit["sigma_lnD"].to_excel(xl, sheet_name="sigma_lnD")
        S.shop_fit["sigma_resid"].to_excel(xl, sheet_name="sigma_resid")
        S.shop_fit["p_zero"].to_excel(xl, sheet_name="p_zero")
    return out.getvalue()

def build_calibration_workbook_pv(S: AppState) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xl:
        S.pv_fit["S"].to_excel(xl, sheet_name="envelope_S")
        # JSON-ish dicts to sheets
        pd.DataFrame.from_records(
            [{"season":k, **v} for k,v in S.pv_fit["loglogistic"].items()]
        ).to_excel(xl, sheet_name="loglogistic", index=False)
        # markov / beta
        recs = []
        for k, v in S.pv_fit["markov"].items():
            recs.append({"season": k, "P_00": v["P"][0][0], "P_01": v["P"][0][1], "P_10": v["P"][1][0], "P_11": v["P"][1][1],
                         "beta_alpha": v["beta"]["alpha"], "beta_beta": v["beta"]["beta"]})
        pd.DataFrame(recs).to_excel(xl, sheet_name="markov_beta", index=False)
    return out.getvalue()

def export_kpi_quantiles(summary_df: pd.DataFrame) -> bytes:
    # Parquet
    out = io.BytesIO()
    summary_df.to_parquet(out, index=False)
    return out.getvalue()

def export_kpi_samples(dists: Dict[str, any]) -> str:
    # CSV of samples, columns=KPI, rows=scenarios
    df = pd.DataFrame({k: pd.Series(v).astype(float) for k, v in dists.items()})
    return df.to_csv(index=False)

def export_all_in_one_xlsx(S: AppState) -> bytes:
    from .kpi import compute_kpi_distributions
    dists, summary = compute_kpi_distributions(S)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xl:
        summary.to_excel(xl, sheet_name="kpi_summary", index=False)
        pd.DataFrame({k: pd.Series(v) for k, v in dists.items()}).to_excel(xl, sheet_name="kpi_samples", index=False)
        md = {
            "S": [S.S], "seed":[S.seed], "hh_gift":[S.hh_gift],
            "N_P":[len(S.prosumer_ids)], "N_HH":[len(S.hh_ids)], "N_SHOP":[len(S.shop_ids)]
        }
        pd.DataFrame(md).to_excel(xl, sheet_name="metadata", index=False)
    return out.getvalue()

def export_hourly_facts(S: AppState, fmt="csv") -> bytes:
    # Minimal hourly facts example: only prices calendar for now (extend with flows if needed)
    df = pd.DataFrame({"timestamp": S.hours})
    df = df.merge(S.zonal, on="timestamp", how="left")
    df = df.merge(S.pun, on="timestamp", how="left")
    if fmt=="csv":
        return df.to_csv(index=False).encode("utf-8")
    else:
        out = io.BytesIO(); df.to_parquet(out, index=False); return out.getvalue()
