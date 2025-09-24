# Virtual Energy Community (Streamlit)

This repository hosts a complete Streamlit app that models a **Virtual Energy Community** (VEC):
1) Upload & fit distributions for **Households**, **Small Shops**, and **Prosumer PV (Option-B v3)**  
2) Build a scenario with explicit **community sizing** and **manual Prosumer→Households mapping**  
3) Run a **Monte Carlo** simulation aligned to your **price calendar**  
4) Explore **KPI distributions** and **download** calibration workbooks, summary tables, and raw samples

> The app enforces the rules we agreed: negative-price override; HH-Gift; loss-factor on gap; spreads are year-constant scalars.

## One-time setup (beginner friendly)

1. Create a new GitHub repo and copy the file tree of this project (preserve folders).  
2. Go to https://share.streamlit.io/ and **Connect your GitHub repo**.  
3. Select `app.py` as the entry point.  
4. Confirm the Python version (Streamlit Cloud defaults work) and **requirements.txt** is detected.  
5. Deploy. That’s it.

## How to use the app

**Page 1 — Upload & Fit**  
- Upload:
  - Households.xlsx (one sheet per household, 15-min kW → app converts to hourly kWh)
  - SmallShops.xlsx (one sheet per shop, 15-min kWh column `ActiveEnergy_Generale`)
  - PV.json (per-kWp hourly kWh)
  - zonal.csv (timestamp, `zonal_price (EUR_per_MWh)`), PUN.csv (timestamp, `PUN (EUR_per_MWh)`)
- Click **Run fitting**. You’ll see parameter tables and diagnostics (histograms + QQ).

**Page 2 — Scenario Builder**  
- Enter N_P, N_HH, N_SHOP; **seed** and **S** (number of scenarios).  
- Enter economic scalars: spreads, α, φ, δ_unm, ℓ; fees and platform fixed cost.  
- Enter **kWp per prosumer** and **manually map** each prosumer to a set of HH.  
- **Validate scenario** (every HH must be assigned at least once; kWp ≥ 0).

- ## Price inputs (NEW)

- **Zonal**: `zonal.csv` with columns `timestamp`, `zonal_price (EUR_per_MWh)` (hourly). Used for:
  - Prosumer matched-price lift (before the retail cap)
  - Export price \(P^{unm}_t = \max(Z^{kWh}_t + \delta_{unm}, 0)\)
  - Negative-price override trigger (if \(Z^{kWh}_t < 0\))

- **PUN (monthly in €/kWh)**: `PUN_monthly.csv` with columns:
  - `timestamp`: any date within each month (e.g., 2024-01-01, 2024-02-01, …)
  - `PUN (EUR_per_kWh)`: monthly value (€/kWh)
  
  The app expands monthly PUN to **hourly** and builds retail:
  - \(R_{HH,t} = \text{PUN}^{(m)}_{kWh} + s_{HH}\)
  - \(R_{SHOP,t} = \text{PUN}^{(m)}_{kWh} + s_{SHOP}\)

All **spreads, lift shares (\(\alpha\)), gap shares (\(\phi\)), uplift \(\delta_{unm}\), loss factor \(\ell\), fees, and fixed costs** can be set to **zero** in the Scenario Builder.

**Page 3 — Run Monte Carlo**  
- The simulation uses the **price calendar** (hard fail if gaps).  
- It samples loads/generation from the fitted distributions, applies matching/pricing rules, and stores the results.

**Page 4 — KPI Dashboard**  
- See energy and economic KPI distributions (mean, sd, p05, p10, p50, p90, p95).

**Page 5 — Exports**  
- Download calibration workbooks, KPI summary (CSV/Parquet), KPI samples (CSV), and optional hourly facts (CSV/Parquet).

## Notes
- If you change inputs or parameters, re-run **fitting** and **simulation**.
- For large S (e.g., 10,000), consider using Parquet exports and avoid hourly facts on Streamlit Cloud.

Enjoy!
