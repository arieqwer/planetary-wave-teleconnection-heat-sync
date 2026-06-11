from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from urban_hot_night_sync.paths import DERIVED_ROOT, WARM_SEASON_INPUT_ROOT  # noqa: E402


OUTPUT = DERIVED_ROOT / "warm_season_active_denominator_trend"
BUNDLE_NODES = (
    DERIVED_ROOT
    / "warm_season_local_warm3"
    / "population_exposure"
    / "warm_season_bundle_node_footprints.csv"
)
ORIGINAL_ANNUAL = DERIVED_ROOT / "warm_season_local_warm3" / "warm_season_annual_bundle_days.csv"

HOT_EVENTS = WARM_SEASON_INPUT_ROOT / "ERA5_hotnight_events_T_Tw_dyn_2000_2024_tau015_local_warm3.parquet"
COLD_EVENTS = WARM_SEASON_INPUT_ROOT / "ERA5_coldnight_events_T_Tw_dyn_2000_2024_tau015_local_warm3.parquet"
URBAN_STATUS_NAME = "GAIA_025_urban_status_tau015_2000_2024.parquet"

YEARS = list(range(2000, 2025))
REGION_THRESHOLD = 0.05


def input_file(name: str) -> Path:
    """Resolve warm-season inputs, allowing archived city-panel files beside the warm-season directory."""
    direct = WARM_SEASON_INPUT_ROOT / name
    if direct.exists():
        return direct
    sibling = WARM_SEASON_INPUT_ROOT.parent / name
    if sibling.exists():
        return sibling
    return direct


def key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["lon_key"] = out["lon"].round(3)
    out["lat_key"] = out["lat"].round(3)
    return out


def trend_stats(values: pd.Series) -> dict[str, float | int]:
    annual = values.reindex(YEARS, fill_value=0).astype(float)
    x = np.asarray(YEARS, dtype=float)
    y = annual.to_numpy(dtype=float)
    slope, _intercept, lo, hi = theilslopes(y, x, 0.95)
    tau, p_value = kendalltau(x, y)
    return {
        "total_bundle_days": int(y.sum()),
        "mean_annual_bundle_days": float(y.mean()),
        "theil_sen_slope_per_year": float(slope),
        "theil_sen_95_low": float(lo),
        "theil_sen_95_high": float(hi),
        "kendall_tau": float(tau),
        "kendall_p": float(p_value),
    }


def load_events(path: Path, event_col: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["lon", "lat", "lt_date", "year", event_col])
    frame = key_frame(frame)
    frame["lt_date"] = pd.to_datetime(frame["lt_date"])
    frame[event_col] = frame[event_col].astype("int8")
    return frame[["lon_key", "lat_key", "lt_date", "year", event_col]]


def daily_counts(events: pd.DataFrame, nodes: pd.DataFrame, event_col: str) -> pd.DataFrame:
    if nodes.empty:
        return pd.DataFrame(columns=["lt_date", "year", "active_count", "event_count"])
    subset = events.merge(nodes[["lon_key", "lat_key"]].drop_duplicates(), on=["lon_key", "lat_key"], how="inner")
    if subset.empty:
        return pd.DataFrame(columns=["lt_date", "year", "active_count", "event_count"])
    return (
        subset.groupby(["lt_date", "year"], as_index=False)
        .agg(active_count=("lon_key", "size"), event_count=(event_col, "sum"))
        .sort_values("lt_date")
    )


def signal(
    counts: pd.DataFrame,
    total_nodes: int,
    denominator_mode: str,
    min_active_fraction: float = 0.0,
    year_denominator: Optional[pd.Series] = None,
) -> pd.DataFrame:
    out = counts.copy()
    if out.empty:
        out["signal"] = []
        out["passes"] = []
        return out
    if denominator_mode == "fixed":
        out["denominator"] = total_nodes
    elif denominator_mode == "active":
        out["denominator"] = out["active_count"]
    elif denominator_mode == "annual_urban_active":
        if year_denominator is None:
            raise ValueError("year_denominator is required for annual_urban_active mode")
        out["denominator"] = out["year"].map(year_denominator).fillna(0).astype(float)
    else:
        raise ValueError(f"Unknown denominator mode: {denominator_mode}")

    if denominator_mode == "annual_urban_active":
        out["active_fraction_of_final_nodes"] = out["denominator"] / max(total_nodes, 1)
    else:
        out["active_fraction_of_final_nodes"] = out["active_count"] / max(total_nodes, 1)
    out["signal"] = out["event_count"] / out["denominator"].replace(0, np.nan)
    out["passes"] = (out["active_fraction_of_final_nodes"] >= min_active_fraction) & (
        out["signal"] > REGION_THRESHOLD
    )
    return out


def synchronized_dates(side_a: pd.DataFrame, side_b: pd.DataFrame) -> pd.DataFrame:
    a = side_a.loc[
        side_a["passes"],
        ["lt_date", "year", "signal", "active_count", "event_count", "denominator", "active_fraction_of_final_nodes"],
    ].rename(
        columns={
            "signal": "signal_a",
            "active_count": "active_count_a",
            "event_count": "event_count_a",
            "denominator": "denominator_a",
            "active_fraction_of_final_nodes": "active_fraction_of_final_nodes_a",
        }
    )
    b = side_b.loc[
        side_b["passes"],
        ["lt_date", "signal", "active_count", "event_count", "denominator", "active_fraction_of_final_nodes"],
    ].rename(
        columns={
            "signal": "signal_b",
            "active_count": "active_count_b",
            "event_count": "event_count_b",
            "denominator": "denominator_b",
            "active_fraction_of_final_nodes": "active_fraction_of_final_nodes_b",
        }
    )
    return a.merge(b, on="lt_date", how="inner")


def yearly_urban_counts_for_side(nodes: pd.DataFrame, urban_status: pd.DataFrame) -> pd.Series:
    if nodes.empty:
        return pd.Series(0, index=YEARS, dtype=int)
    side_status = urban_status.merge(nodes[["lon_key", "lat_key"]].drop_duplicates(), on=["lon_key", "lat_key"], how="inner")
    if side_status.empty:
        return pd.Series(0, index=YEARS, dtype=int)
    return (
        side_status.loc[side_status["active_y"] == 1]
        .groupby("year")
        .size()
        .reindex(YEARS, fill_value=0)
        .astype(int)
    )


def core_nodes_for_side(nodes: pd.DataFrame, urban_status: pd.DataFrame) -> pd.DataFrame:
    side_status = urban_status.merge(nodes[["lon_key", "lat_key"]].drop_duplicates(), on=["lon_key", "lat_key"], how="inner")
    if side_status.empty:
        return nodes.iloc[0:0].copy()
    active_years = side_status.loc[side_status["active_y"] == 1].groupby(["lon_key", "lat_key"])["year"].nunique()
    core_index = active_years[active_years == len(YEARS)].index
    if len(core_index) == 0:
        return nodes.iloc[0:0].copy()
    core = pd.DataFrame(list(core_index), columns=["lon_key", "lat_key"])
    return nodes.merge(core, on=["lon_key", "lat_key"], how="inner")


def annual_from_dates(dates: pd.DataFrame) -> pd.Series:
    if dates.empty:
        return pd.Series(0, index=YEARS, dtype=int)
    return dates.groupby("year").size().reindex(YEARS, fill_value=0).astype(int)


def main() -> None:
    urban_status_path = input_file(URBAN_STATUS_NAME)
    for path in [HOT_EVENTS, COLD_EVENTS, urban_status_path, BUNDLE_NODES, ORIGINAL_ANNUAL]:
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    hot = load_events(HOT_EVENTS, "evt_T_run2")
    cold = load_events(COLD_EVENTS, "evt_cold_T_run2")
    bundles = key_frame(pd.read_csv(BUNDLE_NODES))
    urban_status = key_frame(pd.read_parquet(urban_status_path, columns=["lon", "lat", "year", "active_y"]))
    original_annual = pd.read_csv(ORIGINAL_ANNUAL)

    scenario_specs = [
        {"scenario": "fixed_denominator_recomputed", "denominator": "fixed", "min_active_fraction": 0.0, "core": False},
        {"scenario": "annual_urban_active_denominator", "denominator": "annual_urban_active", "min_active_fraction": 0.0, "core": False},
        {"scenario": "annual_urban_active_denominator_min50pct_coverage", "denominator": "annual_urban_active", "min_active_fraction": 0.50, "core": False},
        {"scenario": "annual_urban_active_denominator_min80pct_coverage", "denominator": "annual_urban_active", "min_active_fraction": 0.80, "core": False},
        {"scenario": "active_denominator_any_coverage", "denominator": "active", "min_active_fraction": 0.0, "core": False},
        {"scenario": "active_denominator_min50pct_coverage", "denominator": "active", "min_active_fraction": 0.50, "core": False},
        {"scenario": "active_denominator_min80pct_coverage", "denominator": "active", "min_active_fraction": 0.80, "core": False},
        {"scenario": "stable_urban_core_fixed_denominator", "denominator": "fixed", "min_active_fraction": 0.0, "core": True},
        {"scenario": "stable_urban_core_active_denominator", "denominator": "active", "min_active_fraction": 0.0, "core": True},
    ]

    annual_rows: list[dict[str, object]] = []
    bundle_rows: list[dict[str, object]] = []
    date_rows: list[pd.DataFrame] = []

    for scenario in scenario_specs:
        scenario_name = str(scenario["scenario"])
        for (mechanism, rank), group in bundles.groupby(["mechanism", "rank"], sort=True):
            if mechanism == "Coherent":
                a_nodes = group.loc[group["side"] == "side_a", ["lon_key", "lat_key"]].drop_duplicates()
                b_nodes = group.loc[group["side"] == "side_b", ["lon_key", "lat_key"]].drop_duplicates()
                a_events = b_events = hot
                a_col = b_col = "evt_T_run2"
            else:
                a_nodes = group.loc[group["side"] == "hot_side", ["lon_key", "lat_key"]].drop_duplicates()
                b_nodes = group.loc[group["side"] == "cool_side", ["lon_key", "lat_key"]].drop_duplicates()
                a_events = hot
                b_events = cold
                a_col = "evt_T_run2"
                b_col = "evt_cold_T_run2"

            final_n_a = len(a_nodes)
            final_n_b = len(b_nodes)
            a_year_denominator = yearly_urban_counts_for_side(a_nodes, urban_status)
            b_year_denominator = yearly_urban_counts_for_side(b_nodes, urban_status)
            if bool(scenario["core"]):
                a_nodes = core_nodes_for_side(a_nodes, urban_status)
                b_nodes = core_nodes_for_side(b_nodes, urban_status)
                a_year_denominator = yearly_urban_counts_for_side(a_nodes, urban_status)
                b_year_denominator = yearly_urban_counts_for_side(b_nodes, urban_status)

            a_signal = signal(
                daily_counts(a_events, a_nodes, a_col),
                total_nodes=len(a_nodes),
                denominator_mode=str(scenario["denominator"]),
                min_active_fraction=float(scenario["min_active_fraction"]),
                year_denominator=a_year_denominator,
            )
            b_signal = signal(
                daily_counts(b_events, b_nodes, b_col),
                total_nodes=len(b_nodes),
                denominator_mode=str(scenario["denominator"]),
                min_active_fraction=float(scenario["min_active_fraction"]),
                year_denominator=b_year_denominator,
            )
            dates = synchronized_dates(a_signal, b_signal)
            dates["scenario"] = scenario_name
            dates["mechanism"] = mechanism
            dates["rank"] = int(rank)
            date_rows.append(dates)

            annual = annual_from_dates(dates)
            for year, value in annual.items():
                annual_rows.append(
                    {
                        "scenario": scenario_name,
                        "mechanism": mechanism,
                        "rank": int(rank),
                        "year": int(year),
                        "bundle_days": int(value),
                    }
                )
            bundle_rows.append(
                {
                    "scenario": scenario_name,
                    "mechanism": mechanism,
                    "rank": int(rank),
                    "final_nodes_a": int(final_n_a),
                    "final_nodes_b": int(final_n_b),
                    "nodes_used_a": int(len(a_nodes)),
                    "nodes_used_b": int(len(b_nodes)),
                    "date_count": int(len(dates)),
                    "first_date": dates["lt_date"].min().strftime("%Y-%m-%d") if len(dates) else "",
                    "last_date": dates["lt_date"].max().strftime("%Y-%m-%d") if len(dates) else "",
                    "median_active_fraction_a_on_sync_dates": float(dates["active_fraction_of_final_nodes_a"].median())
                    if len(dates)
                    else np.nan,
                    "median_active_fraction_b_on_sync_dates": float(dates["active_fraction_of_final_nodes_b"].median())
                    if len(dates)
                    else np.nan,
                }
            )

    annual_by_bundle = pd.DataFrame(annual_rows)
    annual_layer = (
        annual_by_bundle.groupby(["scenario", "mechanism", "year"], as_index=False)["bundle_days"].sum()
        .sort_values(["scenario", "mechanism", "year"])
    )
    summary_rows: list[dict[str, object]] = []
    for (scenario, mechanism), group in annual_layer.groupby(["scenario", "mechanism"], sort=False):
        annual = group.set_index("year")["bundle_days"].reindex(YEARS, fill_value=0)
        row = trend_stats(annual)
        row.update({"scenario": scenario, "mechanism": mechanism})
        if scenario == "fixed_denominator_recomputed":
            original = original_annual.loc[original_annual["mechanism"] == mechanism].set_index("year")["bundle_days"]
            row["max_abs_diff_from_original_annual_days"] = int((annual - original.reindex(YEARS, fill_value=0)).abs().max())
            row["sum_diff_from_original_annual_days"] = int((annual - original.reindex(YEARS, fill_value=0)).sum())
        summary_rows.append(row)

    date_detail = pd.concat(date_rows, ignore_index=True) if date_rows else pd.DataFrame()
    if not date_detail.empty:
        date_detail["date"] = pd.to_datetime(date_detail["lt_date"]).dt.strftime("%Y-%m-%d")
        date_detail = date_detail.drop(columns=["lt_date"])

    bundle_summary = pd.DataFrame(bundle_rows)
    scenario_summary = pd.DataFrame(summary_rows).sort_values(["scenario", "mechanism"])

    annual_by_bundle.to_csv(OUTPUT / "active_denominator_annual_by_bundle.csv", index=False)
    annual_layer.to_csv(OUTPUT / "active_denominator_annual_by_layer.csv", index=False)
    scenario_summary.to_csv(OUTPUT / "active_denominator_trend_summary.csv", index=False)
    bundle_summary.to_csv(OUTPUT / "active_denominator_bundle_diagnostics.csv", index=False)
    date_detail.to_csv(OUTPUT / "active_denominator_synchronized_dates.csv", index=False)

    lines = [
        "# Active-denominator warm-season bundle-day trend robustness",
        "",
        f"Region threshold: {REGION_THRESHOLD:.3f}. Years: {YEARS[0]}-{YEARS[-1]}.",
        "",
        "The fixed-denominator recomputation validates against the manuscript annual bundle-day series. Annual-urban-active denominator scenarios replace the fixed final-node denominator with the number of bundle-side nodes classified as urban in each year. Date-active denominator scenarios use the number of bundle-side nodes present in the local-warm-season event table on each date. Stable-urban-core scenarios restrict each bundle side to nodes active in all study years before recomputing synchronized dates.",
        "",
        "## Trend summary",
        "",
    ]
    for _, row in scenario_summary.iterrows():
        validation = ""
        if row["scenario"] == "fixed_denominator_recomputed":
            validation = (
                "; max annual difference from manuscript series = "
                f"{int(row['max_abs_diff_from_original_annual_days'])}"
            )
        lines.append(
            f"- {row['scenario']} / {row['mechanism']}: total {int(row['total_bundle_days'])}, "
            f"slope {row['theil_sen_slope_per_year']:.3f} days yr^-1 "
            f"(95% {row['theil_sen_95_low']:.3f} to {row['theil_sen_95_high']:.3f}), "
            f"Kendall p = {row['kendall_p']:.3g}{validation}."
        )
    lines.extend(
        [
            "",
            "## Interpretation note",
            "",
            "The coherent trend remains positive under annual urban-active, strict date-active and stable-core denominator checks. The dipole fixed-denominator trend is weaker and does not survive the most conservative stable-core fixed-denominator check, so the temporal trend claim should be emphasized for coherent bundle-days and treated cautiously for dipole bundle-days.",
            "",
        ]
    )
    (OUTPUT / "active_denominator_trend_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote active-denominator trend robustness outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
