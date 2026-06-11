from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, poisson, theilslopes
from sklearn.cluster import KMeans


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from urban_hot_night_sync.paths import WARM_SEASON_INPUT_ROOT

SCRIPT04 = REPO_ROOT / "scripts/04_build_primary_warm_season_outputs.py"
SCRIPT09 = REPO_ROOT / "scripts/09_warm_season_network_parameter_sensitivity.py"
DEFAULT_INPUT_ROOT = WARM_SEASON_INPUT_ROOT
DEFAULT_DERIVED_ROOT = REPO_ROOT / "data/derived/warm_season_final_statistical_robustness"
DEFAULT_PRIMARY_TREND_SUMMARY = REPO_ROOT / "data/derived/warm_season_local_warm3/warm_season_trend_summary.csv"
TAG = "tau015_local_warm3"
YEARS = list(range(2000, 2025))


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


primary_outputs = import_module(SCRIPT04, "warm_season_primary_outputs")
parameter_sensitivity = import_module(SCRIPT09, "warm_season_parameter_sensitivity")


SETTINGS = [
    {"setting": "D=1", "duration_days": 1, "k_days": 3, "alpha": 0.005},
    {"setting": "K=1", "duration_days": 2, "k_days": 1, "alpha": 0.005},
    {"setting": "p<0.001", "duration_days": 2, "k_days": 3, "alpha": 0.001},
    {"setting": "Primary", "duration_days": 2, "k_days": 3, "alpha": 0.005},
    {"setting": "p<0.01", "duration_days": 2, "k_days": 3, "alpha": 0.01},
    {"setting": "K=5", "duration_days": 2, "k_days": 5, "alpha": 0.005},
    {"setting": "D=3", "duration_days": 3, "k_days": 3, "alpha": 0.005},
]


def safe_setting(setting: str) -> str:
    return setting.replace("<", "lt").replace("=", "").replace(".", "p")


def edge_paths(edge_root: Path, setting: str) -> tuple[Path, Path]:
    safe = safe_setting(setting)
    return edge_root / f"coherent_edges_{safe}.parquet", edge_root / f"dipole_edges_{safe}.parquet"


def coherent_edges_for_setting(
    nodes: pd.DataFrame,
    active: np.ndarray,
    events: np.ndarray,
    event_lists,
    k_days: int,
    alpha: float,
) -> pd.DataFrame:
    active_i = active.astype(np.int16)
    event_i = events.astype(np.int16)
    n_eff = active_i @ active_i.T
    n_a = event_i @ active_i.T
    n_b = n_a.T
    obs = parameter_sensitivity.observed_hot_hot(event_lists, active, k_days).astype(np.int32)
    tri_i, tri_j = np.triu_indices(len(nodes), k=1)
    n_eff_tri = n_eff[tri_i, tri_j].astype(np.int32)
    lam = (n_a[tri_i, tri_j].astype(float) * n_b[tri_i, tri_j].astype(float) * (2 * k_days + 1)) / np.maximum(
        n_eff_tri, 1
    )
    p_values = np.ones_like(lam, dtype=float)
    valid = n_eff_tri >= 200
    p_values[valid] = poisson.sf(obs[valid] - 1, lam[valid])
    keep = p_values < alpha
    ii = tri_i[keep]
    jj = tri_j[keep]
    dist = parameter_sensitivity.geodetic_distance_km(
        nodes.loc[ii, "lon"].to_numpy(),
        nodes.loc[ii, "lat"].to_numpy(),
        nodes.loc[jj, "lon"].to_numpy(),
        nodes.loc[jj, "lat"].to_numpy(),
    )
    return pd.DataFrame(
        {
            "lon1": nodes.loc[ii, "lon"].to_numpy(),
            "lat1": nodes.loc[ii, "lat"].to_numpy(),
            "lon2": nodes.loc[jj, "lon"].to_numpy(),
            "lat2": nodes.loc[jj, "lat"].to_numpy(),
            "dist_km": dist,
            "p": p_values[keep],
            "obs": obs[keep],
            "lam": lam[keep],
            "N_eff": n_eff_tri[keep],
        }
    )


def dipole_edges_for_setting(
    hot_nodes: pd.DataFrame,
    cold_nodes: pd.DataFrame,
    active_hot: np.ndarray,
    hot_events: np.ndarray,
    hot_lists,
    active_cold: np.ndarray,
    cold_events: np.ndarray,
    cold_lists,
    k_days: int,
    alpha: float,
) -> pd.DataFrame:
    active_hot_i = active_hot.astype(np.int16)
    active_cold_i = active_cold.astype(np.int16)
    hot_i = hot_events.astype(np.int16)
    cold_i = cold_events.astype(np.int16)
    n_eff = active_hot_i @ active_cold_i.T
    n_hot = hot_i @ active_cold_i.T
    n_cold = active_hot_i @ cold_i.T
    obs = parameter_sensitivity.observed_hot_cold(hot_lists, cold_lists, active_hot, active_cold, k_days).astype(np.int32)
    same_node = hot_nodes[["lon", "lat"]].merge(
        cold_nodes[["lon", "lat"]].reset_index(names="cold_idx"),
        on=["lon", "lat"],
        how="left",
    )["cold_idx"]
    same_node = same_node.dropna().astype(int)
    n_eff[same_node.index.to_numpy(), same_node.to_numpy()] = 0
    lam = (n_hot.astype(float) * n_cold.astype(float) * (2 * k_days + 1)) / np.maximum(n_eff, 1)
    p_values = np.ones_like(lam, dtype=float)
    valid = n_eff >= 200
    p_values[valid] = poisson.sf(obs[valid] - 1, lam[valid])
    keep = p_values < alpha
    ii, jj = np.where(keep)
    dist = parameter_sensitivity.geodetic_distance_km(
        hot_nodes.loc[ii, "lon"].to_numpy(),
        hot_nodes.loc[ii, "lat"].to_numpy(),
        cold_nodes.loc[jj, "lon"].to_numpy(),
        cold_nodes.loc[jj, "lat"].to_numpy(),
    )
    return pd.DataFrame(
        {
            "hot_lon": hot_nodes.loc[ii, "lon"].to_numpy(),
            "hot_lat": hot_nodes.loc[ii, "lat"].to_numpy(),
            "cold_lon": cold_nodes.loc[jj, "lon"].to_numpy(),
            "cold_lat": cold_nodes.loc[jj, "lat"].to_numpy(),
            "dist_km": dist,
            "p": p_values[keep],
            "obs": obs[keep],
            "lam": lam[keep],
            "N_eff": n_eff[keep],
        }
    )


def build_edges_for_settings(hot: pd.DataFrame, cold: pd.DataFrame, edge_root: Path, date_start, date_end) -> None:
    edge_root.mkdir(parents=True, exist_ok=True)
    for setting in SETTINGS:
        coherent_path, dipole_path = edge_paths(edge_root, setting["setting"])
        if coherent_path.exists() and dipole_path.exists():
            print(f"Edges exist, skipping {setting['setting']}")
            continue
        duration_days = int(setting["duration_days"])
        k_days = int(setting["k_days"])
        alpha = float(setting["alpha"])
        print(f"Building edges for {setting['setting']} (D={duration_days}, K={k_days}, p<{alpha:g})")
        hot_col = parameter_sensitivity.build_event_column(hot, "evt_T", duration_days)
        cold_col = parameter_sensitivity.build_event_column(cold, "evt_cold_T", duration_days)
        hot_nodes, active_hot, hot_events, hot_lists, _, _ = parameter_sensitivity.build_matrices(
            hot, hot_col, min_events=5, date_start=date_start, date_end=date_end
        )
        cold_nodes, active_cold, cold_events, cold_lists, _, _ = parameter_sensitivity.build_matrices(
            cold, cold_col, min_events=5, date_start=date_start, date_end=date_end
        )
        coherent = coherent_edges_for_setting(hot_nodes, active_hot, hot_events, hot_lists, k_days, alpha)
        dipole = dipole_edges_for_setting(
            hot_nodes,
            cold_nodes,
            active_hot,
            hot_events,
            hot_lists,
            active_cold,
            cold_events,
            cold_lists,
            k_days,
            alpha,
        )
        coherent.to_parquet(coherent_path, index=False)
        dipole.to_parquet(dipole_path, index=False)


def run_matching_sensitivity(
    edge_root: Path,
    input_root: Path,
    output_root: Path,
    n_boot: int,
) -> pd.DataFrame:
    rows = []
    hot_events = input_root / f"ERA5_hotnight_events_T_Tw_dyn_2000_2024_{TAG}.parquet"
    cold_events = input_root / f"ERA5_coldnight_events_T_Tw_dyn_2000_2024_{TAG}.parquet"
    output_root.mkdir(parents=True, exist_ok=True)
    for setting in SETTINGS:
        coherent_path, dipole_path = edge_paths(edge_root, setting["setting"])
        setting_dir = output_root / safe_setting(setting["setting"])
        setting_dir.mkdir(parents=True, exist_ok=True)
        print(f"Matched amplification for {setting['setting']}")
        summary, raw = primary_outputs.run_matched_amplification(
            coherent_edges_path=coherent_path,
            dipole_edges_path=dipole_path,
            hot_events_path=hot_events,
            cold_events_path=cold_events,
            output_dir=setting_dir,
            min_dist_km=2500.0,
            k_days=int(setting["k_days"]),
            min_duration=2,
            n_boot=n_boot,
        )
        raw.to_csv(setting_dir / "raw_lock_summary.csv", index=False)
        summary.insert(0, "setting", setting["setting"])
        summary.insert(1, "edge_duration_days", int(setting["duration_days"]))
        summary.insert(2, "edge_alpha", float(setting["alpha"]))
        rows.append(summary)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(output_root / "warm_season_matched_amplification_parameter_sensitivity.csv", index=False)
    return result


def daily_signal(events: pd.DataFrame, nodes: pd.DataFrame) -> pd.Series:
    if nodes.empty:
        return pd.Series(dtype=float)
    active = events.merge(nodes, on=["lon", "lat"])
    if active.empty:
        return pd.Series(dtype=float)
    return active.groupby("lt_date").size() / len(nodes)


def synchronized_dates(signal_a: pd.Series, signal_b: pd.Series, threshold: float) -> pd.DatetimeIndex:
    common = signal_a.index.intersection(signal_b.index)
    if common.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(common[(signal_a.loc[common] > threshold) & (signal_b.loc[common] > threshold)])


def trend_stats(mechanism: str, setting: dict[str, object], annual: pd.Series) -> dict[str, object]:
    x = np.asarray(YEARS, dtype=float)
    y = annual.reindex(YEARS, fill_value=0).to_numpy(dtype=float)
    slope, intercept, lo, hi = theilslopes(y, x, 0.95)
    tau, p_value = kendalltau(x, y)
    return {
        "setting": setting["setting"],
        "network_layer": mechanism,
        "duration_days": int(setting["duration_days"]),
        "k_days": int(setting["k_days"]),
        "alpha": float(setting["alpha"]),
        "total_bundle_days": int(y.sum()),
        "mean_annual_bundle_days": float(y.mean()),
        "theil_sen_slope_per_year": float(slope),
        "theil_sen_95_low": float(lo),
        "theil_sen_95_high": float(hi),
        "kendall_tau": float(tau) if not np.isnan(tau) else np.nan,
        "kendall_p": float(p_value) if not np.isnan(p_value) else np.nan,
    }


def top_bundle_annual_counts(
    edges: pd.DataFrame,
    event_a: pd.DataFrame,
    event_b: pd.DataFrame,
    layer: str,
    threshold: float,
    top_n: int,
    n_clusters: int,
) -> tuple[pd.Series, int, list[str]]:
    long_edges = edges.loc[edges["dist_km"] > 2500].copy()
    if len(long_edges) < n_clusters:
        return pd.Series(0, index=YEARS, dtype=float), len(long_edges), []
    if layer == "Coherent":
        features = ["lon1", "lat1", "lon2", "lat2"]
        cols_a = {"lon1": "lon", "lat1": "lat"}
        cols_b = {"lon2": "lon", "lat2": "lat"}
    else:
        features = ["hot_lon", "hot_lat", "cold_lon", "cold_lat"]
        cols_a = {"hot_lon": "lon", "hot_lat": "lat"}
        cols_b = {"cold_lon": "lon", "cold_lat": "lat"}
    long_edges["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(long_edges[features])
    annual = pd.Series(0, index=YEARS, dtype=float)
    labels = []
    for cluster_id in long_edges["cluster"].value_counts().index[:top_n]:
        subset = long_edges.loc[long_edges["cluster"] == cluster_id].copy()
        nodes_a = subset[list(cols_a)].rename(columns=cols_a).drop_duplicates()
        nodes_b = subset[list(cols_b)].rename(columns=cols_b).drop_duplicates()
        dates = synchronized_dates(daily_signal(event_a, nodes_a), daily_signal(event_b, nodes_b), threshold=threshold)
        if len(dates):
            counts = pd.Series(dates.year).value_counts().sort_index()
            annual = annual.add(counts.reindex(YEARS, fill_value=0), fill_value=0)
        labels.append(f"{float(nodes_a['lon'].mean()):.0f},{float(nodes_a['lat'].mean()):.0f}->{float(nodes_b['lon'].mean()):.0f},{float(nodes_b['lat'].mean()):.0f}")
    return annual, len(long_edges), labels


def event_days_for_column(frame: pd.DataFrame, event_col: str) -> pd.DataFrame:
    out = frame.loc[frame[event_col] == 1, ["lon", "lat", "lt_date"]].copy()
    out["lt_date"] = pd.to_datetime(out["lt_date"])
    return out


def run_trend_sensitivity(
    hot: pd.DataFrame,
    cold: pd.DataFrame,
    edge_root: Path,
    output_root: Path,
    region_threshold: float,
    top_n: int,
    n_clusters: int,
    primary_trend_summary: Path | None = None,
) -> pd.DataFrame:
    rows = []
    for setting in SETTINGS:
        print(f"Trend sensitivity for {setting['setting']}")
        duration_days = int(setting["duration_days"])
        hot_col = parameter_sensitivity.build_event_column(hot, "evt_T", duration_days)
        cold_col = parameter_sensitivity.build_event_column(cold, "evt_cold_T", duration_days)
        hot_events = event_days_for_column(hot, hot_col)
        cold_events = event_days_for_column(cold, cold_col)
        coherent_path, dipole_path = edge_paths(edge_root, setting["setting"])
        coherent_edges = pd.read_parquet(coherent_path)
        dipole_edges = pd.read_parquet(dipole_path)
        annual, n_long, labels = top_bundle_annual_counts(
            coherent_edges,
            hot_events,
            hot_events,
            "Coherent",
            region_threshold,
            top_n,
            n_clusters,
        )
        row = trend_stats("Coherent", setting, annual)
        row["long_range_edges"] = int(n_long)
        row["top_bundle_labels"] = "; ".join(labels)
        rows.append(row)
        annual, n_long, labels = top_bundle_annual_counts(
            dipole_edges,
            hot_events,
            cold_events,
            "Dipole",
            region_threshold,
            top_n,
            n_clusters,
        )
        row = trend_stats("Dipole", setting, annual)
        row["long_range_edges"] = int(n_long)
        row["top_bundle_labels"] = "; ".join(labels)
        rows.append(row)
    result = pd.DataFrame(rows)
    if primary_trend_summary is not None and primary_trend_summary.exists():
        primary = pd.read_csv(primary_trend_summary)
        for _, source in primary.iterrows():
            mask = (result["setting"] == "Primary") & (result["network_layer"] == source["mechanism"])
            if not mask.any():
                continue
            for target_col, source_col in [
                ("total_bundle_days", "total_bundle_days"),
                ("mean_annual_bundle_days", "mean_annual_bundle_days"),
                ("theil_sen_slope_per_year", "theil_sen_slope_per_year"),
                ("theil_sen_95_low", "theil_sen_95_low"),
                ("theil_sen_95_high", "theil_sen_95_high"),
                ("kendall_tau", "kendall_tau"),
                ("kendall_p", "kendall_p"),
            ]:
                result.loc[mask, target_col] = source[source_col]
            result.loc[mask, "top_bundle_labels"] = "fixed main-text top-three bundles"
    output_root.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_root / "warm_season_bundle_trend_parameter_sensitivity.csv", index=False)
    return result


def summarize_final_outputs(matched: pd.DataFrame, trends: pd.DataFrame, output_root: Path) -> None:
    matched_summary = (
        matched.groupby(["setting", "mechanism", "edge_set"], as_index=False)
        .agg(
            n_pairs=("n_pairs", "first"),
            mean_delta=("mean_delta", "first"),
            boot95_mean_lo=("boot95_mean_lo", "first"),
            boot95_mean_hi=("boot95_mean_hi", "first"),
            wilcoxon_p_greater=("wilcoxon_p_greater", "first"),
        )
        .sort_values(["setting", "mechanism", "edge_set"])
    )
    trend_summary = trends.sort_values(["setting", "network_layer"]).copy()
    matched_summary.to_csv(output_root / "warm_season_matched_amplification_parameter_sensitivity_compact.csv", index=False)
    trend_summary.to_csv(output_root / "warm_season_bundle_trend_parameter_sensitivity_compact.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final statistical robustness checks for matched amplification and trends.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--n-boot", type=int, default=800)
    parser.add_argument("--region-threshold", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--primary-trend-summary", type=Path, default=DEFAULT_PRIMARY_TREND_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hot_path = args.input_root / f"ERA5_hotnight_events_T_Tw_dyn_2000_2024_{TAG}.parquet"
    cold_path = args.input_root / f"ERA5_coldnight_events_T_Tw_dyn_2000_2024_{TAG}.parquet"
    missing = [path for path in [hot_path, cold_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing corrected warm-season event tables:\n" + "\n".join(str(path) for path in missing))

    hot = pd.read_parquet(hot_path, columns=["lon", "lat", "lt_date", "Tmin_C", "Tmin95", "evt_T", "evt_T_run2"])
    cold = pd.read_parquet(
        cold_path,
        columns=["lon", "lat", "lt_date", "Tmin_C", "Tmin05", "evt_cold_T", "evt_cold_T_run2"],
    )
    hot["lt_date"] = pd.to_datetime(hot["lt_date"])
    cold["lt_date"] = pd.to_datetime(cold["lt_date"])
    date_start = min(hot["lt_date"].min(), cold["lt_date"].min())
    date_end = max(hot["lt_date"].max(), cold["lt_date"].max())
    edge_root = args.derived_root / "edges"
    build_edges_for_settings(hot, cold, edge_root, date_start, date_end)
    matched = run_matching_sensitivity(edge_root, args.input_root, args.derived_root / "matched", args.n_boot)
    trends = run_trend_sensitivity(
        hot,
        cold,
        edge_root,
        args.derived_root / "trends",
        args.region_threshold,
        args.top_n,
        args.n_clusters,
        args.primary_trend_summary,
    )
    summarize_final_outputs(matched, trends, args.derived_root)
    print(f"Wrote final statistical robustness outputs to {args.derived_root}")


if __name__ == "__main__":
    main()
