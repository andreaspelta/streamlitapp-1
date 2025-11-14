"""Pricing utilities and documentation constants for the deterministic model."""

NEGATIVE_RULE = (
    "If zonal < 0 → P_pros=0; P_cons=retail (PUN-monthly+spread); "
    "gap=0; export = max(zonal+δ_unm, 0)."
)
