from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.cluster import KMeans


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from urban_hot_night_sync.paths import WARM_SEASON_INPUT_ROOT

SCRIPT09 = REPO_ROOT / "scripts/09_warm_season_network_parameter_sensitivity.py"
DEFAULT_INPUT_ROOT = WARM_SEASON_INPUT_ROOT
DEFAULT_DERIVED_ROOT = REPO_ROOT / "data/derived/warm_season_bundle_geography_sensitivity"
TAG = "tau015_local_warm3"

spec = importlib.util.spec_from_file_location("parameter_sensitivity", SCRIPT09)
parameter_sensitivity = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(parameter_sensitivity)


def region_name(lon: float, lat: float, compact: bool = False) -> str:
    if lat > 20 and -130 < lon < -60:
        return "N.America" if compact else "NorthAmerica"
    if lat > 20 and 60 < lon < 150:
        return "E.Asia" if compact else "EastAsia"
    if lat > 30 and -10 < lon < 40:
        return "Europe"
    return f"({lon:.0f},{lat:.0f})"


def setting_label(duration_days: int, k_days: int, alpha: float) -> str:
    if duration_days == 2 and k_days == 3 and abs(alpha - 0.005) < 1e-12:
        return "Primary"
    if duration_days != 2:
        return f"D={duration_days}"
    if k_days != 3:
        return f"K={k_days}"
    return f"p<{alpha:g}"


def mean_centroid_distance(
    a_lon: float,
    a_lat: float,
    b_lon: float,
    b_lat: float,
    ref_a_lon: float,
    ref_a_lat: float,
    ref_b_lon: float,
    ref_b_lat: float,
    allow_swap: bool,
) -> float:
    direct = parameter_sensitivity.geodetic_distance_km(
        np.asarray([a_lon, b_lon]),
        np.asarray([a_lat, b_lat]),
        np.asarray([ref_a_lon, ref_b_lon]),
        np.asarray([ref_a_lat, ref_b_lat]),
    ).mean()
    if not allow_swap:
        return float(direct)
    swapped = parameter_sensitivity.geodetic_distance_km(
        np.asarray([a_lon, b_lon]),
        np.asarray([a_lat, b_lat]),
        np.asarray([ref_b_lon, ref_a_lon]),
        np.asarray([ref_b_lat, ref_a_lat]),
    ).mean()
    return float(min(direct, swapped))


def bundle_rows(
    edges: pd.DataFrame,
    layer: str,
    duration_days: int,
    k_days: int,
    alpha: float,
    total_long_edges: int,
    top_n: int,
    n_clusters: int,
) -> list[dict[str, object]]:
    if len(edges) < n_clusters:
        return []
    if layer == "Coherent":
        features = ["lon1", "lat1", "lon2", "lat2"]
        a_lon, a_lat, b_lon, b_lat = "lon1", "lat1", "lon2", "lat2"
    else:
        features = ["hot_lon", "hot_lat", "cold_lon", "cold_lat"]
        a_lon, a_lat, b_lon, b_lat = "hot_lon", "hot_lat", "cold_lon", "cold_lat"
    clustered = edges.copy()
    clustered["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(clustered[features])
    rows: list[dict[str, object]] = []
    for rank, cluster_id in enumerate(clustered["cluster"].value_counts().index[:top_n], start=1):
        subset = clustered.loc[clustered["cluster"] == cluster_id].copy()
        centroid_a_lon = float(subset[a_lon].mean())
        centroid_a_lat = float(subset[a_lat].mean())
        centroid_b_lon = float(subset[b_lon].mean())
        centroid_b_lat = float(subset[b_lat].mean())
        if layer == "Coherent":
            corridor = (
                f"{region_name(centroid_a_lon, centroid_a_lat, compact=True)}-"
                f"{region_name(centroid_b_lon, centroid_b_lat, compact=True)}"
            )
        else:
            corridor = f"{region_name(centroid_a_lon, centroid_a_lat)} to {region_name(centroid_b_lon, centroid_b_lat)}"
        rows.append(
            {
                "setting": setting_label(duration_days, k_days, alpha),
                "network_layer": layer,
                "duration_days": duration_days,
                "k_days": k_days,
                "alpha": alpha,
                "rank": rank,
                "cluster_id": int(cluster_id),
                "edge_count": int(len(subset)),
                "edge_share_of_long_range": float(len(subset) / total_long_edges) if total_long_edges else np.nan,
                "total_long_range_edges": int(total_long_edges),
                "centroid_a_lon": centroid_a_lon,
                "centroid_a_lat": centroid_a_lat,
                "centroid_b_lon": centroid_b_lon,
                "centroid_b_lat": centroid_b_lat,
                "corridor_label": corridor,
            }
        )
    return rows


def coherent_long_edges(
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
    dist = parameter_sensitivity.geodetic_distance_km(
        nodes.loc[tri_i, "lon"].to_numpy(),
        nodes.loc[tri_i, "lat"].to_numpy(),
        nodes.loc[tri_j, "lon"].to_numpy(),
        nodes.loc[tri_j, "lat"].to_numpy(),
    )
    keep = (p_values < alpha) & (dist > 2500)
    ii = tri_i[keep]
    jj = tri_j[keep]
    return pd.DataFrame(
        {
            "lon1": nodes.loc[ii, "lon"].to_numpy(),
            "lat1": nodes.loc[ii, "lat"].to_numpy(),
            "lon2": nodes.loc[jj, "lon"].to_numpy(),
            "lat2": nodes.loc[jj, "lat"].to_numpy(),
            "dist_km": dist[keep],
            "p": p_values[keep],
            "obs": obs[keep],
            "lam": lam[keep],
            "N_eff": n_eff_tri[keep],
        }
    )


def dipole_long_edges(
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
    lon1 = np.repeat(hot_nodes["lon"].to_numpy(), len(cold_nodes))
    lat1 = np.repeat(hot_nodes["lat"].to_numpy(), len(cold_nodes))
    lon2 = np.tile(cold_nodes["lon"].to_numpy(), len(hot_nodes))
    lat2 = np.tile(cold_nodes["lat"].to_numpy(), len(hot_nodes))
    dist = parameter_sensitivity.geodetic_distance_km(lon1, lat1, lon2, lat2).reshape(p_values.shape)
    keep = (p_values < alpha) & (dist > 2500)
    ii, jj = np.where(keep)
    return pd.DataFrame(
        {
            "hot_lon": hot_nodes.loc[ii, "lon"].to_numpy(),
            "hot_lat": hot_nodes.loc[ii, "lat"].to_numpy(),
            "cold_lon": cold_nodes.loc[jj, "lon"].to_numpy(),
            "cold_lat": cold_nodes.loc[jj, "lat"].to_numpy(),
            "dist_km": dist[keep],
            "p": p_values[keep],
            "obs": obs[keep],
            "lam": lam[keep],
            "N_eff": n_eff[keep],
        }
    )


def compare_to_primary(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = all_rows.loc[all_rows["setting"] == "Primary"].copy()
    for _, row in all_rows.iterrows():
        refs = primary.loc[primary["network_layer"] == row["network_layer"]]
        if refs.empty:
            rows.append({**row.to_dict(), "closest_primary_rank": np.nan, "closest_primary_distance_km": np.nan})
            continue
        distances = []
        for _, ref in refs.iterrows():
            distances.append(
                mean_centroid_distance(
                    row["centroid_a_lon"],
                    row["centroid_a_lat"],
                    row["centroid_b_lon"],
                    row["centroid_b_lat"],
                    ref["centroid_a_lon"],
                    ref["centroid_a_lat"],
                    ref["centroid_b_lon"],
                    ref["centroid_b_lat"],
                    allow_swap=row["network_layer"] == "Coherent",
                )
            )
        best = int(np.argmin(distances))
        ref_row = refs.iloc[best]
        rows.append(
            {
                **row.to_dict(),
                "closest_primary_rank": int(ref_row["rank"]),
                "closest_primary_distance_km": float(distances[best]),
                "closest_primary_corridor": ref_row["corridor_label"],
                "matches_primary_within_1500km": bool(distances[best] <= 1500),
                "matches_primary_within_2500km": bool(distances[best] <= 2500),
            }
        )
    return pd.DataFrame(rows)


def build_summary(compared: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, layer), group in compared.groupby(["setting", "network_layer"], sort=False):
        rows.append(
            {
                "setting": setting,
                "network_layer": layer,
                "duration_days": int(group["duration_days"].iloc[0]),
                "k_days": int(group["k_days"].iloc[0]),
                "alpha": float(group["alpha"].iloc[0]),
                "total_long_range_edges": int(group["total_long_range_edges"].iloc[0]),
                "top3_edge_share": float(group["edge_count"].sum() / group["total_long_range_edges"].iloc[0]),
                "top3_matches_within_1500km": int(group["matches_primary_within_1500km"].sum()),
                "top3_matches_within_2500km": int(group["matches_primary_within_2500km"].sum()),
                "median_closest_primary_distance_km": float(group["closest_primary_distance_km"].median()),
                "top_corridors": "; ".join(group.sort_values("rank")["corridor_label"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test dominant-bundle geography stability under alternate warm-season ECA settings.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--n-clusters", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hot_path = args.input_root / f"ERA5_hotnight_events_T_Tw_dyn_2000_2024_{TAG}.parquet"
    cold_path = args.input_root / f"ERA5_coldnight_events_T_Tw_dyn_2000_2024_{TAG}.parquet"
    missing = [path for path in [hot_path, cold_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing corrected warm-season event tables:\n" + "\n".join(str(path) for path in missing))

    hot = pd.read_parquet(hot_path, columns=["lon", "lat", "lt_date", "evt_T", "evt_T_run2"])
    cold = pd.read_parquet(cold_path, columns=["lon", "lat", "lt_date", "evt_cold_T", "evt_cold_T_run2"])
    hot["lt_date"] = pd.to_datetime(hot["lt_date"])
    cold["lt_date"] = pd.to_datetime(cold["lt_date"])
    date_start = min(hot["lt_date"].min(), cold["lt_date"].min())
    date_end = max(hot["lt_date"].max(), cold["lt_date"].max())

    settings = [
        (1, 3, [0.005]),
        (2, 1, [0.005]),
        (2, 3, [0.001, 0.005, 0.01]),
        (2, 5, [0.005]),
        (3, 3, [0.005]),
    ]
    all_rows: list[dict[str, object]] = []
    for duration_days, k_days, alpha_values in settings:
        print(f"Preparing D={duration_days}, K={k_days}, alpha={alpha_values}")
        hot_col = parameter_sensitivity.build_event_column(hot, "evt_T", duration_days)
        cold_col = parameter_sensitivity.build_event_column(cold, "evt_cold_T", duration_days)
        hot_nodes, active_hot, hot_events, hot_lists, _, _ = parameter_sensitivity.build_matrices(
            hot, hot_col, min_events=5, date_start=date_start, date_end=date_end
        )
        cold_nodes, active_cold, cold_events, cold_lists, _, _ = parameter_sensitivity.build_matrices(
            cold, cold_col, min_events=5, date_start=date_start, date_end=date_end
        )
        for alpha in alpha_values:
            coherent_edges = coherent_long_edges(hot_nodes, active_hot, hot_events, hot_lists, k_days, alpha)
            dipole_edges = dipole_long_edges(
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
            all_rows.extend(
                bundle_rows(
                    coherent_edges,
                    "Coherent",
                    duration_days,
                    k_days,
                    alpha,
                    len(coherent_edges),
                    args.top_n,
                    args.n_clusters,
                )
            )
            all_rows.extend(
                bundle_rows(
                    dipole_edges,
                    "Dipole",
                    duration_days,
                    k_days,
                    alpha,
                    len(dipole_edges),
                    args.top_n,
                    args.n_clusters,
                )
            )

    detailed = compare_to_primary(pd.DataFrame(all_rows))
    summary = build_summary(detailed)
    args.derived_root.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.derived_root / "warm_season_bundle_geography_sensitivity_detailed.csv", index=False)
    summary.to_csv(args.derived_root / "warm_season_bundle_geography_sensitivity_summary.csv", index=False)
    print(f"Wrote bundle geography sensitivity outputs to {args.derived_root}")


if __name__ == "__main__":
    main()
