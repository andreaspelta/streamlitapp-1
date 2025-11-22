import io
from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64tz_dtype

from .state import AppState
from .io_utils import TZ
from .clustering import CLUSTERS
from .kpi import compute_kpi_summary


def _transpose_with_baseline(
    df: pd.DataFrame,
    *,
    id_col: str,
    drop_metrics: set[str] | None = None,
) -> pd.DataFrame:
    """Return a transposed summary with community/baseline columns per actor."""

    if df is None or df.empty:
        return df

    drop_metrics = drop_metrics or set()
    df = df.copy()
    baseline_lookup = {
        col[: -len("_baseline")]: col for col in df.columns if col.endswith("_baseline")
    }
    metrics = [
        col
        for col in df.columns
        if col not in {id_col}
        and not col.endswith("_baseline")
        and col not in drop_metrics
    ]

    frames = []
    for actor_id, row in df.set_index(id_col).iterrows():
        community_values = [row.get(metric, np.nan) for metric in metrics]
        baseline_values = [
            row.get(baseline_lookup[metric], np.nan) if metric in baseline_lookup else np.nan
            for metric in metrics
        ]
        actor_frame = pd.DataFrame(
            {
                f"{actor_id}_community": community_values,
                f"{actor_id}_baseline": baseline_values,
            },
            index=metrics,
        )
        frames.append(actor_frame)

    if not frames:
        return df

    combined = pd.concat(frames, axis=1)
    combined = combined.reset_index().rename(columns={"index": id_col})
    return combined


def _format_timestamp_index(idx: pd.DatetimeIndex) -> pd.Series:
    """Return timestamps formatted with timezone offset for CSV/Excel templates."""

    return idx.strftime("%Y-%m-%d %H:%M:%S%z")


def _timezone_naive(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return a copy of *df* with any timezone-aware datetimes made naive.

    Excel writers reject timezone-aware timestamps. This helper is used before
    exporting DataFrames to Excel so that Streamlit downloads succeed regardless
    of the upstream timezone handling.
    """

    if df is None:
        return None

    out = df.copy()

    for col in out.columns:
        if is_datetime64tz_dtype(out[col]):
            out[col] = out[col].dt.tz_localize(None)

    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    return out

def _profile_template_dataframe() -> pd.DataFrame:
    df = pd.DataFrame({"hour": np.arange(24, dtype=int)})
    for cluster in CLUSTERS:
        df[cluster] = np.nan
    return df


@lru_cache(maxsize=32)
def _build_profile_template_cached(year: int, label: str) -> bytes:
    df = _profile_template_dataframe()
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xl:
        sheet = f"{label}_median"
        df.to_excel(xl, sheet_name=sheet, index=False, startrow=3)
        ws = xl.sheets[sheet]
        ws.write(0, 0, f"Deterministic {label} template — year {year}")
        ws.write(1, 0, "Paste the median hourly demand (kWh) for each cluster.")
        ws.write(2, 0, "Hour 0 corresponds to 00:00-01:00 local time.")
    return out.getvalue()


def build_household_template(year: int) -> bytes:
    """Excel template for household cluster medians (24×12 grid)."""

    return _build_profile_template_cached(int(year), "HH")


def build_shop_template(year: int) -> bytes:
    """Excel template for shop cluster medians (24×12 grid)."""

    return _build_profile_template_cached(int(year), "SHOP")


def build_prosumer_template(year: int) -> bytes:
    """Excel template for prosumer cluster medians (24×12 grid)."""

    return _build_profile_template_cached(int(year), "PROSUMER")


@lru_cache(maxsize=16)
def _build_pv_excel_template_cached(year: int) -> bytes:
    start = pd.Timestamp(f"{int(year)}-01-01 00:00", tz=TZ)
    end = pd.Timestamp(f"{int(year) + 1}-01-01 00:00", tz=TZ)
    rng = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    df = pd.DataFrame(
        {
            "timestamp": _format_timestamp_index(rng),
            "kWh_per_kWp": np.nan,
        }
    )
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xl:
        sheet = "PV_per_kWp"
        df.to_excel(xl, sheet_name=sheet, index=False, startrow=2)
        ws = xl.sheets[sheet]
        ws.write(0, 0, f"PV template — hourly per-kWp yield for {year}")
        ws.write(1, 0, "Fill kWh/kWp in the second column.")
    return out.getvalue()


def build_pv_excel_template(year: int) -> bytes:
    """Create an Excel template for PV per-kWp hourly data for the selected year."""

    return _build_pv_excel_template_cached(int(year))


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
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xl:
        sheet = "Zonal_prices"
        df.to_excel(xl, sheet_name=sheet, index=False, startrow=2)
        ws = xl.sheets[sheet]
        ws.write(0, 0, f"Zonal price template — hourly prices for {year}")
        ws.write(1, 0, "Keep the timestamp column unchanged and fill EUR/MWh values in column B.")
    return out.getvalue()


def build_zonal_price_template(year: int) -> bytes:
    """Create an hourly zonal price Excel template for the selected year."""

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
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xl:
        sheet = "Monthly_PUN"
        df.to_excel(xl, sheet_name=sheet, index=False, startrow=2)
        ws = xl.sheets[sheet]
        ws.write(0, 0, f"Monthly PUN template — €/kWh for {year}")
        ws.write(1, 0, "Provide one value per month; timestamps should remain as the first of each month at 00:00.")
    return out.getvalue()


def build_pun_monthly_template(year: int) -> bytes:
    """Create a monthly PUN Excel template for the selected year."""

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
        daily_totals_raw = None
        if getattr(S, "pv_diag", None):
            daily_totals_raw = S.pv_diag.get("daily_totals")

        medians_df = None
        fit_medians = None
        if getattr(S, "pv_fit", None):
            fit_medians = S.pv_fit.get("daily_median_M")

        if isinstance(fit_medians, dict) and fit_medians:
            medians_df = pd.DataFrame(
                [
                    {
                        "season": season,
                        "observed_median_kWh_per_kWp": float(value),
                    }
                    for season, value in fit_medians.items()
                ]
            )
        elif hasattr(fit_medians, "items") and fit_medians:
            medians_df = pd.DataFrame(
                [
                    {
                        "season": season,
                        "observed_median_kWh_per_kWp": float(value),
                    }
                    for season, value in dict(fit_medians).items()
                ]
            )

        if medians_df is None and getattr(S, "pv_diag", None):
            diag_medians = S.pv_diag.get("daily_median_M")
            if isinstance(diag_medians, pd.DataFrame) and not diag_medians.empty:
                medians_df = diag_medians.rename(
                    columns={"median_kWh_per_kWp": "observed_median_kWh_per_kWp"}
                ).copy()

        daily_df = None
        if isinstance(daily_totals_raw, pd.DataFrame) and not daily_totals_raw.empty:
            daily_df = daily_totals_raw.copy()
            daily_df["date"] = pd.to_datetime(daily_df["date"])

        if medians_df is None and daily_df is not None:
            medians_df = (
                daily_df.groupby("season")["kWh_per_kWp"].median()
                .rename("observed_median_kWh_per_kWp")
                .reset_index()
            )

        if medians_df is not None and not medians_df.empty:
            medians_df = medians_df.sort_values("season").reset_index(drop=True)
            medians_df["test_median_kWh_per_kWp"] = medians_df[
                "observed_median_kWh_per_kWp"
            ]

            instructions = pd.DataFrame(
                {
                    "Note": [
                        "Adjust the test median column to experiment with alternative seasonal medians.",
                        "Use the daily totals table below to compute additional statistics if needed.",
                    ]
                }
            )

            instructions.to_excel(xl, sheet_name="daily_medians", index=False)
            start_row = len(instructions) + 2
            medians_df.to_excel(
                xl,
                sheet_name="daily_medians",
                index=False,
                startrow=start_row,
            )

            if daily_df is not None:
                daily_export = daily_df.sort_values(["season", "date"])
                daily_export["date"] = daily_export["date"].dt.strftime("%Y-%m-%d")
                daily_start = start_row + len(medians_df) + 2
                daily_export.to_excel(
                    xl,
                    sheet_name="daily_medians",
                    index=False,
                    startrow=daily_start,
                )
    return out.getvalue()

def export_all_in_one_xlsx(S: AppState) -> bytes:
    if S.result is None:
        raise ValueError("Run the deterministic scenario before exporting.")

    summary = compute_kpi_summary(S)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xl:
        scenario_params = {
            "year": S.year,
            "N_P": len(S.prosumer_ids),
            "N_HH": len(S.hh_ids),
            "N_SHOP": len(S.shop_ids),
            "efficiency": S.efficiency,
            "hh_gift": S.hh_gift,
            "s_HH": S.s_HH,
            "spread_split_HH": S.spread_split_HH,
            "platform_gap_HH": S.platform_gap_HH,
            "s_SH": S.s_SH,
            "spread_split_SH": S.spread_split_SH,
            "platform_gap_SH": S.platform_gap_SH,
            "delta_unm": S.delta_unm,
            "f_pros": S.f_pros,
            "f_hh": S.f_hh,
            "f_shop": S.f_shop,
            "platform_fixed": S.platform_fixed,
        }
        meta = pd.DataFrame(
            {
                "parameter": list(scenario_params.keys()),
                "value": list(scenario_params.values()),
            }
        )

        prosumer_summary = _transpose_with_baseline(
            S.result.prosumer_summary,
            id_col="prosumer_id",
        )
        household_summary = _transpose_with_baseline(
            S.result.household_summary,
            id_col="household_id",
        )
        shop_summary = _transpose_with_baseline(
            S.result.shop_summary,
            id_col="shop_id",
        )

        tables = [
            ("kpi_summary", summary),
            ("prosumer_summary", prosumer_summary),
            ("household_summary", household_summary),
            ("shop_summary", shop_summary),
            ("community_hourly", S.result.community_hourly),
            ("prices", S.result.prices),
            ("metadata", meta),
        ]

        for sheet_name, df in tables:
            sanitised = _timezone_naive(df)
            if sanitised is not None:
                sanitised.to_excel(xl, sheet_name=sheet_name, index=False)
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

