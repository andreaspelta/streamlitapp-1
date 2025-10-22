from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from .state import AppState
from .simulation import DeterministicResult

_EPS = 1e-6


@dataclass
class _SankeyLink:
    source: str
    target: str
    value: float
    label: str = ""
    color: Optional[str] = None


@dataclass
class _SankeySpec:
    nodes: List[str]
    links: List[_SankeyLink]
    unit: str
    note: str


_NODE_BASE_COLORS: Dict[str, str] = {
    "Grid Import": "#636363",
    "Grid Export": "#d62728",
    "Self Consumption": "#2ca02c",
    "Community Matched": "#1f77b4",
    "Plenitude Platform": "#17becf",
    "Platform Gap": "#bcbd22",
    "Actor Fees": "#ff9896",
    "Platform Fixed Costs": "#8c564b",
    "Platform Margin": "#bcbd22",
    "Residual Balance": "#c49c94",
    "Balance Adjustment": "#9edae5",
}


def _hex_to_rgba(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return color
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _node_color(label: str) -> str:
    if label in _NODE_BASE_COLORS:
        return _NODE_BASE_COLORS[label]
    if label.startswith("Prosumer"):
        return "#2ca02c"
    if label.startswith("Household"):
        return "#ff7f0e"
    if label.startswith("Shop"):
        return "#9467bd"
    if "Import" in label:
        return "#636363"
    if "Export" in label:
        return "#d62728"
    if "Community" in label:
        return "#1f77b4"
    return "#7f7f7f"


def _link_color_from_source(source_label: str) -> str:
    return _hex_to_rgba(_node_color(source_label), 0.5)


def _build_sankey_figure(spec: _SankeySpec, title: str) -> go.Figure:
    node_indices: Dict[str, int] = {}
    labels: List[str] = []
    for label in spec.nodes:
        if label not in node_indices:
            node_indices[label] = len(labels)
            labels.append(label)
    for link in spec.links:
        if link.source not in node_indices:
            node_indices[link.source] = len(labels)
            labels.append(link.source)
        if link.target not in node_indices:
            node_indices[link.target] = len(labels)
            labels.append(link.target)

    node_colors = [_node_color(label) for label in labels]

    sources: List[int] = []
    targets: List[int] = []
    values: List[float] = []
    link_colors: List[str] = []
    link_labels: List[str] = []
    hovertemplates: List[str] = []

    for link in spec.links:
        if link.value <= _EPS:
            continue
        source_idx = node_indices[link.source]
        target_idx = node_indices[link.target]
        sources.append(source_idx)
        targets.append(target_idx)
        values.append(float(link.value))
        link_labels.append(link.label)
        color = link.color or _link_color_from_source(link.source)
        link_colors.append(color)
        description = link.label
        src_label = link.source
        tgt_label = link.target
        if description:
            hover_text = (
                f"{src_label} → {tgt_label}<br>{description}<br><b>{link.value:,.2f} {spec.unit}</b><extra></extra>"
            )
        else:
            hover_text = f"{src_label} → {tgt_label}<br><b>{link.value:,.2f} {spec.unit}</b><extra></extra>"
        hovertemplates.append(hover_text)

    sankey = go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="rgba(0,0,0,0)", width=0),
            label=labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            label=link_labels,
            hovertemplate=hovertemplates,
        ),
    )

    fig = go.Figure(sankey)
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left"),
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=12),
    )
    return fig


def build_energy_sankey_chart(state: AppState) -> Tuple[go.Figure, str, bool]:
    result = state.result if isinstance(state.result, DeterministicResult) else None
    if result is None or result.prosumer_summary.empty:
        spec = _dummy_energy_spec()
        fig = _build_sankey_figure(spec, "Energy flows across the community")
        return fig, spec.note, True
    spec = _energy_spec_from_result(result)
    fig = _build_sankey_figure(spec, "Energy flows across the community")
    return fig, spec.note, False


def build_economic_sankey_chart(state: AppState) -> Tuple[go.Figure, str, bool]:
    result = state.result if isinstance(state.result, DeterministicResult) else None
    if result is None or result.prosumer_summary.empty:
        spec = _dummy_economic_spec()
        fig = _build_sankey_figure(spec, "Economic flows between actors and platform")
        return fig, spec.note, True
    spec = _economic_spec_from_result(state, result)
    fig = _build_sankey_figure(spec, "Economic flows between actors and platform")
    return fig, spec.note, False


def _dummy_energy_spec() -> _SankeySpec:
    nodes: List[str] = []
    links: List[_SankeyLink] = []

    def add_link(source: str, target: str, value: float, label: str) -> None:
        if value <= _EPS:
            return
        if source not in nodes:
            nodes.append(source)
        if target not in nodes:
            nodes.append(target)
        links.append(_SankeyLink(source, target, value, label))

    add_link("Grid Import", "Prosumer P1", 50.0, "Imports for own use")
    add_link("Grid Import", "Prosumer P2", 80.0, "Imports for own use")
    add_link("Grid Import", "Household H1", 110.0, "Residual demand")
    add_link("Grid Import", "Household H2", 95.0, "Residual demand")
    add_link("Grid Import", "Household H3", 90.0, "Residual demand")
    add_link("Grid Import", "Shop S1", 120.0, "Residual demand")

    add_link("Prosumer P1", "Self Consumption", 220.0, "Self-consumed PV")
    add_link("Prosumer P2", "Self Consumption", 160.0, "Self-consumed PV")

    add_link("Prosumer P1", "Community Matched", 520.0, "Shared with community")
    add_link("Prosumer P2", "Community Matched", 430.0, "Shared with community")

    add_link("Prosumer P1", "Grid Export", 90.0, "Residual export")
    add_link("Prosumer P2", "Grid Export", 70.0, "Residual export")

    add_link("Community Matched", "Household H1", 240.0, "Community supply")
    add_link("Community Matched", "Household H2", 210.0, "Community supply")
    add_link("Community Matched", "Household H3", 200.0, "Community supply")
    add_link("Community Matched", "Shop S1", 300.0, "Community supply")

    note = "Placeholder Sankey built with illustrative values. Run the deterministic engine to replace it with actual data."
    return _SankeySpec(nodes, links, "kWh", note)


def _dummy_economic_spec() -> _SankeySpec:
    nodes: List[str] = []
    links: List[_SankeyLink] = []

    def add_link(source: str, target: str, value: float, label: str) -> None:
        if value <= _EPS:
            return
        if source not in nodes:
            nodes.append(source)
        if target not in nodes:
            nodes.append(target)
        links.append(_SankeyLink(source, target, value, label))

    plenitude = "Plenitude Platform"

    add_link("Household H1", plenitude, 420.0, "Energy + fee")
    add_link("Household H2", plenitude, 390.0, "Energy + fee")
    add_link("Household H3", plenitude, 360.0, "Energy + fee")
    add_link("Shop S1", plenitude, 520.0, "Energy + fee")

    add_link("Prosumer P1", plenitude, 120.0, "Imports + fee")
    add_link("Prosumer P2", plenitude, 140.0, "Imports + fee")

    add_link(plenitude, "Prosumer P1", 360.0, "Community + export revenue")
    add_link(plenitude, "Prosumer P2", 310.0, "Community + export revenue")

    add_link(plenitude, "Platform Gap", 180.0, "Spread retained")
    add_link(plenitude, "Actor Fees", 160.0, "Annual fees")
    add_link(plenitude, "Platform Fixed Costs", 200.0, "O&M and services")

    incoming = 420.0 + 390.0 + 360.0 + 520.0 + 120.0 + 140.0
    outgoing = 360.0 + 310.0 + 180.0 + 160.0 + 200.0
    residual = incoming - outgoing
    if residual > _EPS:
        add_link(plenitude, "Platform Margin", residual, "Net platform margin")

    note = "Placeholder monetary flows. Run the deterministic engine to populate values from the latest scenario."
    return _SankeySpec(nodes, links, "€", note)


def _energy_spec_from_result(result: DeterministicResult) -> _SankeySpec:
    nodes: List[str] = []
    links: List[_SankeyLink] = []

    def add_link(source: str, target: str, value: float, label: str) -> None:
        if value <= _EPS:
            return
        if source not in nodes:
            nodes.append(source)
        if target not in nodes:
            nodes.append(target)
        links.append(_SankeyLink(source, target, float(value), label))

    for row in result.prosumer_summary.itertuples():
        pros_label = f"Prosumer {row.prosumer_id}"
        add_link("Grid Import", pros_label, float(row.imports_kWh), "Imports from grid")
        add_link(pros_label, "Self Consumption", float(row.self_consumption_kWh), "Self-consumed energy")
        matched_total = float(row.matched_hh_kWh + row.matched_shop_kWh)
        add_link(pros_label, "Community Matched", matched_total, "Shared with community")
        add_link(pros_label, "Grid Export", float(row.exports_kWh), "Residual export")

    for row in result.household_summary.itertuples():
        hh_label = f"Household {row.household_id}"
        add_link("Community Matched", hh_label, float(row.matched_kWh), "Community supply")
        add_link("Grid Import", hh_label, float(row.import_kWh), "Grid supply")

    for row in result.shop_summary.itertuples():
        shop_label = f"Shop {row.shop_id}"
        add_link("Community Matched", shop_label, float(row.matched_kWh), "Community supply")
        add_link("Grid Import", shop_label, float(row.import_kWh), "Grid supply")

    note = "Energy flows computed from the most recent deterministic calculation (annual totals in kWh)."
    return _SankeySpec(nodes, links, "kWh", note)


def _economic_spec_from_result(state: AppState, result: DeterministicResult) -> _SankeySpec:
    nodes: List[str] = []
    links: List[_SankeyLink] = []
    plenitude = "Plenitude Platform"

    def add_link(source: str, target: str, value: float, label: str) -> None:
        if value <= _EPS:
            return
        if source not in nodes:
            nodes.append(source)
        if target not in nodes:
            nodes.append(target)
        links.append(_SankeyLink(source, target, float(value), label))

    prices = result.prices.copy()
    prices = prices.rename(
        columns={
            "retail_HH (EUR_per_kWh)": "retail_HH",
            "retail_SH (EUR_per_kWh)": "retail_SH",
            "Pcons_HH (EUR_per_kWh)": "Pcons_HH",
            "Pcons_SH (EUR_per_kWh)": "Pcons_SH",
            "Ppros_HH (EUR_per_kWh)": "Ppros_HH",
            "Ppros_SH (EUR_per_kWh)": "Ppros_SH",
            "P_unm (EUR_per_kWh)": "P_unm",
        }
    )

    hh_hourly = result.household_hourly
    if not hh_hourly.empty:
        hh_merge = hh_hourly.merge(
            prices[["timestamp", "retail_HH", "Pcons_HH"]], on="timestamp", how="left"
        )
        hh_merge[["retail_HH", "Pcons_HH"]] = hh_merge[["retail_HH", "Pcons_HH"]].fillna(0.0)
        hh_merge["matched_cost"] = hh_merge["matched_kWh"] * hh_merge["Pcons_HH"]
        hh_merge["import_cost"] = hh_merge["import_kWh"] * hh_merge["retail_HH"]
        hh_costs = hh_merge.groupby("household_id")[["matched_cost", "import_cost"]].sum().reset_index()
    else:
        hh_costs = pd.DataFrame(columns=["household_id", "matched_cost", "import_cost"])

    shop_hourly = result.shop_hourly
    if not shop_hourly.empty:
        shop_merge = shop_hourly.merge(
            prices[["timestamp", "retail_SH", "Pcons_SH"]], on="timestamp", how="left"
        )
        shop_merge[["retail_SH", "Pcons_SH"]] = shop_merge[["retail_SH", "Pcons_SH"]].fillna(0.0)
        shop_merge["matched_cost"] = shop_merge["matched_kWh"] * shop_merge["Pcons_SH"]
        shop_merge["import_cost"] = shop_merge["import_kWh"] * shop_merge["retail_SH"]
        shop_costs = shop_merge.groupby("shop_id")[["matched_cost", "import_cost"]].sum().reset_index()
    else:
        shop_costs = pd.DataFrame(columns=["shop_id", "matched_cost", "import_cost"])

    pros_hourly = result.prosumer_hourly
    if not pros_hourly.empty:
        pros_merge = pros_hourly.merge(
            prices[["timestamp", "retail_HH", "Ppros_HH", "Ppros_SH", "P_unm"]],
            on="timestamp",
            how="left",
        )
        pros_merge[["retail_HH", "Ppros_HH", "Ppros_SH", "P_unm"]] = pros_merge[
            ["retail_HH", "Ppros_HH", "Ppros_SH", "P_unm"]
        ].fillna(0.0)
        pros_merge["import_cost"] = pros_merge["imports_kWh"] * pros_merge["retail_HH"]
        pros_merge["matched_hh_rev"] = pros_merge["matched_hh_kWh"] * pros_merge["Ppros_HH"]
        pros_merge["matched_shop_rev"] = pros_merge["matched_shop_kWh"] * pros_merge["Ppros_SH"]
        pros_merge["export_rev"] = pros_merge["exports_kWh"] * pros_merge["P_unm"]
        pros_cash = (
            pros_merge.groupby("prosumer_id")[
                ["import_cost", "matched_hh_rev", "matched_shop_rev", "export_rev"]
            ]
            .sum()
            .reset_index()
        )
    else:
        pros_cash = pd.DataFrame(
            columns=["prosumer_id", "import_cost", "matched_hh_rev", "matched_shop_rev", "export_rev"]
        )

    hh_fee = 12.0 * float(state.f_hh)
    shop_fee = 12.0 * float(state.f_shop)
    pros_fee = 12.0 * float(state.f_pros)

    for _, row in hh_costs.iterrows():
        hh_label = f"Household {row['household_id']}"
        energy_payment = float(row["matched_cost"] + row["import_cost"])
        total_payment = energy_payment + hh_fee
        label = f"Energy: {energy_payment:,.2f} €"
        if hh_fee > _EPS:
            label += f" | Fees: {hh_fee:,.2f} €"
        add_link(hh_label, plenitude, total_payment, label)

    for _, row in shop_costs.iterrows():
        shop_label = f"Shop {row['shop_id']}"
        energy_payment = float(row["matched_cost"] + row["import_cost"])
        total_payment = energy_payment + shop_fee
        label = f"Energy: {energy_payment:,.2f} €"
        if shop_fee > _EPS:
            label += f" | Fees: {shop_fee:,.2f} €"
        add_link(shop_label, plenitude, total_payment, label)

    for _, row in pros_cash.iterrows():
        pros_label = f"Prosumer {row['prosumer_id']}"
        import_cost = float(row["import_cost"])
        receipts = float(row["matched_hh_rev"] + row["matched_shop_rev"] + row["export_rev"])
        payment_label = ""
        payment_total = import_cost
        if import_cost > _EPS:
            payment_label = f"Imports: {import_cost:,.2f} €"
        if pros_fee > _EPS:
            payment_total += pros_fee
            payment_label = (
                (payment_label + " | " if payment_label else "") + f"Fees: {pros_fee:,.2f} €"
            )
        if payment_total > _EPS:
            add_link(pros_label, plenitude, payment_total, payment_label)
        receipt_label = []
        if row["matched_hh_rev"] > _EPS:
            receipt_label.append(f"HH matched: {row['matched_hh_rev']:,.2f} €")
        if row["matched_shop_rev"] > _EPS:
            receipt_label.append(f"Shop matched: {row['matched_shop_rev']:,.2f} €")
        if row["export_rev"] > _EPS:
            receipt_label.append(f"Exports: {row['export_rev']:,.2f} €")
        if receipts > _EPS:
            add_link(plenitude, pros_label, receipts, " | ".join(receipt_label))

    total_fees = (
        hh_fee * len(hh_costs)
        + shop_fee * len(shop_costs)
        + pros_fee * len(pros_cash)
    )

    gap_value = float(result.totals.get("platform_gap", 0.0))
    fees_value = total_fees
    fixed_value = float(result.totals.get("platform_fixed", 0.0))

    if gap_value > _EPS:
        add_link(plenitude, "Platform Gap", gap_value, "Spread retained")
    if fees_value > _EPS:
        add_link(plenitude, "Actor Fees", fees_value, "Subscription fees")
    if fixed_value > _EPS:
        add_link(plenitude, "Platform Fixed Costs", fixed_value, "O&M and services")

    incoming = sum(link.value for link in links if link.target == plenitude)
    outgoing = sum(link.value for link in links if link.source == plenitude)
    residual = incoming - outgoing
    if residual > _EPS:
        add_link(plenitude, "Platform Margin", residual, "Net margin")
    elif residual < -_EPS:
        add_link("Balance Adjustment", plenitude, abs(residual), "Balance correction")

    note = "Monetary flows derived from deterministic results (annual totals in €)."
    return _SankeySpec(nodes, links, "€", note)
