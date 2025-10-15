import io
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .state import AppState
from .io_utils import TZ
from .clustering import season_of


def _format_timestamp_index(idx: pd.DatetimeIndex) -> pd.Series:
    """Return timestamps formatted with timezone offset for CSV/Excel templates."""

    return idx.strftime("%Y-%m-%d %H:%M:%S%z")

def _normalize_ts_arg(value) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


@lru_cache(maxsize=16)
def _build_household_template_cached(start: str, end: str) -> bytes:
    rng = pd.date_range(start=start, end=end, freq="15min", tz=TZ)
    df = pd.DataFrame(
        {
            "timestamp": rng.strftime("%Y-%m-%d %H:%M:%S%z"),
            "Power_kW": np.nan,
        }
    )
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xl:
        df.to_excel(xl, sheet_name="HH01", index=False)
    return out.getvalue()


def build_household_template(start="2022-01-01 00:30", end="2023-01-01 00:00") -> bytes:
    """Create a 15-minute household template workbook for 2022."""

    return _build_household_template_cached(
        _normalize_ts_arg(start), _normalize_ts_arg(end)
    )


@lru_cache(maxsize=8)
def _build_shop_template_cached(start: str, end: str, sheets: int) -> bytes:
    rng = pd.date_range(start=start, end=end, freq="15min", tz=TZ, inclusive="left")
    df = pd.DataFrame(
        {
            "timestamp": _format_timestamp_index(rng),
            "ActiveEnergy_Generale": np.nan,
        }
    )
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xl:
        for sheet_idx in range(1, sheets + 1):
            sheet_name = f"Shop{sheet_idx:02d}"
            df.to_excel(xl, sheet_name=sheet_name, index=False)
    return out.getvalue()


def build_shop_template(
    start="2022-09-15 00:00", end="2023-09-15 00:00", sheets: int = 20
) -> bytes:
    """Create a 15-minute small shops template spanning one year across multiple sheets."""

    return _build_shop_template_cached(
        _normalize_ts_arg(start), _normalize_ts_arg(end), int(sheets)
    )


@lru_cache(maxsize=16)
def _build_pv_excel_template_cached(start: str, end: str) -> bytes:
    rng = pd.date_range(start=start, end=end, freq="h", tz=TZ)
    df = pd.DataFrame(
        {
            "timestamp": _format_timestamp_index(rng),
            "energy_kWh_per_kWp": np.nan,
        }
    )

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xl:
        df.to_excel(xl, sheet_name="PV_per_kWp", index=False)
    return out.getvalue()


def build_pv_excel_template(
    start="2018-01-01 00:30", end="2023-12-31 23:30"
) -> bytes:
    """Create an Excel template for PV per-kWp hourly data."""

    return _build_pv_excel_template_cached(
        _normalize_ts_arg(start), _normalize_ts_arg(end)
    )


@lru_cache(maxsize=16)
def _build_zonal_price_template_cached(year: int) -> bytes:
    start = pd.Timestamp(f"{year}-01-01 00:00", tz=TZ)
    end = pd.Timestamp(f"{year + 1}-01-01 00:00", tz=TZ)
    rng = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    df = pd.DataFrame(
        {
            "timestamp": _format_timestamp_index(rng),
            "zonal_price (EUR_per_MWh)": np.nan,
        }
    )
    return df.to_csv(index=False).encode("utf-8")


def build_zonal_price_template(year: int) -> bytes:
    """Create an hourly zonal price CSV template for the selected year."""

    return _build_zonal_price_template_cached(int(year))


@lru_cache(maxsize=16)
def _build_pun_monthly_template_cached(year: int) -> bytes:
    rng = pd.date_range(
        start=pd.Timestamp(f"{year}-01-01 00:00", tz=TZ),
        end=pd.Timestamp(f"{year}-12-01 00:00", tz=TZ),
        freq="MS",
    )
    df = pd.DataFrame(
        {
            "timestamp": _format_timestamp_index(rng),
            "PUN (EUR_per_kWh)": np.nan,
        }
    )
    return df.to_csv(index=False).encode("utf-8")


def build_pun_monthly_template(year: int) -> bytes:
    """Create a monthly PUN CSV template for the selected year."""

    return _build_pun_monthly_template_cached(int(year))
    
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

def _extract_pv_daily_totals(S: AppState) -> Optional[pd.DataFrame]:
    """Return historical PV daily totals grouped by season, if available."""

    daily_totals = None
    if getattr(S, "pv_diag", None):
        daily_totals = S.pv_diag.get("daily_totals")
        if daily_totals is not None and not daily_totals.empty:
            return daily_totals.copy()

    pv_perkwp = getattr(S, "pv_perkwp", None)
    if pv_perkwp is None or pv_perkwp.empty:
        return None

    df = pv_perkwp.copy()
    if "timestamp" not in df.columns or "kWh_per_kWp" not in df.columns:
        return None

    df = df[df["kWh_per_kWp"].notna()]
    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["season"] = df["timestamp"].apply(season_of)
    df["date"] = df["timestamp"].dt.date
    daily = (
        df.groupby(["season", "date"])["kWh_per_kWp"].sum().reset_index()
    )
    return daily if not daily.empty else None


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

        # Daily medians sandbox
        daily_totals = _extract_pv_daily_totals(S)

        if daily_totals is not None and not daily_totals.empty:
            daily_df = daily_totals.copy()
            daily_df["date"] = pd.to_datetime(daily_df["date"])
            medians = (
                daily_df.groupby("season")["kWh_per_kWp"].median()
                .rename("observed_median_kWh_per_kWp")
                .reset_index()
            )
            medians["test_median_kWh_per_kWp"] = medians["observed_median_kWh_per_kWp"]

            instructions = pd.DataFrame(
                {
                    "Note": [
                        "Adjust the test median column to experiment with alternative seasonal medians.",
                        "Use the daily totals table below to compute additional statistics if needed.",
                        "These medians are available right after running the PV fitting step (Monte Carlo not required).",
                    ]
                }
            )

            instructions.to_excel(xl, sheet_name="daily_medians", index=False)
            start_row = len(instructions) + 2
            medians.to_excel(
                xl,
                sheet_name="daily_medians",
                index=False,
                startrow=start_row,
            )

            daily_export = daily_df.sort_values(["season", "date"])
            daily_export["date"] = daily_export["date"].dt.strftime("%Y-%m-%d")
            daily_start = start_row + len(medians) + 2
            daily_export.to_excel(
                xl,
                sheet_name="daily_medians",
                index=False,
                startrow=daily_start,
            )
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

def export_hourly_facts(S, fmt="csv"):
    """
    Build a lightweight hourly facts table aligned to S.hours, including:
      - zonal_price (EUR_per_MWh) [hourly]
      - PUN (EUR_per_kWh)         [hourly, expanded from monthly PUN]
    Returns bytes suitable for Streamlit download_button.
    """
    import pandas as pd
    import io

    # Defensive checks (note: we now use S.pun_h, not S.pun)
    if S.hours is None:
        raise ValueError("Hourly calendar (S.hours) is missing.")
    if S.zonal is None:
        raise ValueError("Zonal prices are missing. Upload zonal.csv on the Upload & Fit page.")
    if S.pun_h is None:
        raise ValueError("Hourly PUN (S.pun_h) is missing. Upload monthly PUN and run the expansion step.")

    # Prepare frames with correct dtypes
    hours_df = pd.DataFrame({"timestamp": S.hours})
    zonal_df = S.zonal.copy()
    punh_df = S.pun_h.copy()

    # Ensure timestamp columns are datetime with tz
    zonal_df["timestamp"] = pd.to_datetime(zonal_df["timestamp"]).dt.tz_convert(hours_df["timestamp"].dt.tz)
    punh_df["timestamp"] = pd.to_datetime(punh_df["timestamp"]).dt.tz_convert(hours_df["timestamp"].dt.tz)

    # Merge hourly zonal and hourly PUN (EUR_per_kWh)
    out = hours_df.merge(zonal_df, on="timestamp", how="left")
    out = out.merge(punh_df, on="timestamp", how="left")

    # Optional: ensure expected column names exist
    if "zonal_price (EUR_per_MWh)" not in out.columns:
        raise KeyError("Column 'zonal_price (EUR_per_MWh)' not found after merge.")
    if "PUN (EUR_per_kWh)" not in out.columns:
        # Some CSV editors might alter capitalization; try common variants once
        for cand in out.columns:
            if cand.strip().lower() == "pun (eur_per_kwh)":
                out.rename(columns={cand: "PUN (EUR_per_kWh)"}, inplace=True)
                break
        if "PUN (EUR_per_kWh)" not in out.columns:
            raise KeyError("Column 'PUN (EUR_per_kWh)' not found after merge.")

    # Emit CSV or Parquet as bytes for Streamlit
    if fmt == "csv":
        return out.to_csv(index=False).encode("utf-8")
    elif fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        sink = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(out), sink)
        return sink.getvalue()
    else:
        raise ValueError("fmt must be 'csv' or 'parquet'")

