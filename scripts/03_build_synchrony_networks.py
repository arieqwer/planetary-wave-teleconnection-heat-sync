from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.stats import poisson

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from urban_hot_night_sync.paths import ANALYSIS_ROOT


def tau_tag(tau: float) -> str:
    return f"tau{int(round(tau * 100)):03d}"


def event_column(metric: str, min_run_length: int, cold: bool = False) -> str:
    prefix = "evt_cold_" if cold else "evt_"
    suffix = f"_run{min_run_length}" if min_run_length > 1 else ""
    return f"{prefix}{metric}{suffix}"


def load_event_series(path: Path, column: str, min_events: int) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    frame = pd.read_parquet(path, columns=["lon", "lat", "lt_date", column]).rename(columns={column: "evt"})
    frame["lt_date"] = pd.to_datetime(frame["lt_date"])
    frame = frame.drop_duplicates(subset=["lon", "lat", "lt_date"])

    counts = frame.groupby(["lon", "lat"])["evt"].sum()
    valid_nodes = counts[counts >= min_events].index
    frame = frame.set_index(["lon", "lat"]).loc[valid_nodes].reset_index()

    series: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    for node, group in frame.groupby(["lon", "lat"], sort=False):
        active = group["lt_date"].sort_values().to_numpy(dtype="datetime64[D]").astype(np.int64)
        events = (
            group.loc[group["evt"] == 1, "lt_date"]
            .sort_values()
            .to_numpy(dtype="datetime64[D]")
            .astype(np.int64)
        )
        series[node] = (active, events)
    return series


def eca_p_value(
    active_a: np.ndarray,
    events_a: np.ndarray,
    active_b: np.ndarray,
    events_b: np.ndarray,
    k_days: int,
    min_overlap_days: int,
) -> tuple[float, int, float, int]:
    overlap = np.intersect1d(active_a, active_b, assume_unique=True)
    n_eff = len(overlap)
    if n_eff < min_overlap_days:
        return 1.0, 0, 0.0, n_eff

    events_a = np.intersect1d(events_a, overlap, assume_unique=True)
    events_b = np.intersect1d(events_b, overlap, assume_unique=True)
    if len(events_a) == 0 or len(events_b) == 0:
        return 1.0, 0, 0.0, n_eff

    observed = 0
    for value in events_a:
        left = np.searchsorted(events_b, value - k_days, side="left")
        right = np.searchsorted(events_b, value + k_days, side="right")
        if left != right:
            observed += 1

    lam = (len(events_a) * len(events_b) * (2 * k_days + 1)) / n_eff
    p_value = 1.0 if lam <= 0 else 1 - poisson.cdf(observed - 1, lam)
    return float(p_value), int(observed), float(lam), int(n_eff)


def build_hot_hot_network(
    events_path: Path,
    output_edges: Path,
    output_nodes: Path,
    metric: str,
    min_run_length: int,
    alpha: float,
    k_days: int,
    min_events: int,
    min_overlap_days: int,
) -> None:
    series = load_event_series(events_path, event_column(metric, min_run_length), min_events=min_events)
    nodes = list(series.keys())
    geod = Geod(ellps="WGS84")
    edges: list[tuple[float, float, float, float, float, float, int, float, int]] = []

    start = time.time()
    total_pairs = (len(nodes) * (len(nodes) - 1)) // 2
    for idx, (node_a, node_b) in enumerate(combinations(nodes, 2), start=1):
        if idx % 200000 == 0:
            rate = idx / max(time.time() - start, 1e-9)
            print(f"Processed {idx:,}/{total_pairs:,} pairs at {rate:.0f} pairs/sec")

        p_value, observed, lam, n_eff = eca_p_value(
            *series[node_a],
            *series[node_b],
            k_days=k_days,
            min_overlap_days=min_overlap_days,
        )
        if p_value >= alpha:
            continue

        _, _, distance_m = geod.inv(node_a[0], node_a[1], node_b[0], node_b[1])
        edges.append((node_a[0], node_a[1], node_b[0], node_b[1], distance_m / 1000.0, p_value, observed, lam, n_eff))

    pd.DataFrame(
        edges,
        columns=["lon1", "lat1", "lon2", "lat2", "dist_km", "p", "obs", "lam", "N_eff"],
    ).to_parquet(output_edges, index=False)
    pd.DataFrame(nodes, columns=["lon", "lat"]).to_parquet(output_nodes, index=False)
    print(f"Wrote Hot-Hot edges: {output_edges}")
    print(f"Wrote Hot-Hot nodes: {output_nodes}")


def build_seesaw_network(
    hot_events_path: Path,
    cold_events_path: Path,
    output_edges: Path,
    metric: str,
    min_run_length: int,
    alpha: float,
    k_days: int,
    min_events: int,
    min_overlap_days: int,
) -> None:
    hot_series = load_event_series(hot_events_path, event_column(metric, min_run_length), min_events=min_events)
    cold_series = load_event_series(cold_events_path, event_column(metric, min_run_length, cold=True), min_events=min_events)

    all_nodes = list(set(hot_series).union(cold_series))
    geod = Geod(ellps="WGS84")
    edges: list[tuple[float, float, float, float, float, float, int, float, int]] = []

    start = time.time()
    total_pairs = (len(all_nodes) * (len(all_nodes) - 1)) // 2
    for idx, (node_a, node_b) in enumerate(combinations(all_nodes, 2), start=1):
        if idx % 200000 == 0:
            rate = idx / max(time.time() - start, 1e-9)
            print(f"Processed {idx:,}/{total_pairs:,} pairs at {rate:.0f} pairs/sec")

        if node_a in hot_series and node_b in cold_series:
            p_value, observed, lam, n_eff = eca_p_value(
                *hot_series[node_a],
                *cold_series[node_b],
                k_days=k_days,
                min_overlap_days=min_overlap_days,
            )
            if p_value < alpha:
                _, _, distance_m = geod.inv(node_a[0], node_a[1], node_b[0], node_b[1])
                edges.append((node_a[0], node_a[1], node_b[0], node_b[1], distance_m / 1000.0, p_value, observed, lam, n_eff))

        if node_b in hot_series and node_a in cold_series:
            p_value, observed, lam, n_eff = eca_p_value(
                *hot_series[node_b],
                *cold_series[node_a],
                k_days=k_days,
                min_overlap_days=min_overlap_days,
            )
            if p_value < alpha:
                _, _, distance_m = geod.inv(node_b[0], node_b[1], node_a[0], node_a[1])
                edges.append((node_b[0], node_b[1], node_a[0], node_a[1], distance_m / 1000.0, p_value, observed, lam, n_eff))

    pd.DataFrame(
        edges,
        columns=["hot_lon", "hot_lat", "cold_lon", "cold_lat", "dist_km", "p", "obs", "lam", "N_eff"],
    ).to_parquet(output_edges, index=False)
    print(f"Wrote seesaw edges: {output_edges}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Hot-Hot and seesaw synchronization networks.")
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--metric", choices=["T", "Tw"], default="T")
    parser.add_argument("--mode", choices=["hothot", "seesaw", "both"], default="both")
    parser.add_argument("--min-run-length", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--k-days", type=int, default=3)
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--min-overlap-days", type=int, default=200)
    parser.add_argument("--input-root", type=Path, default=ANALYSIS_ROOT)
    parser.add_argument("--output-root", type=Path, default=ANALYSIS_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = tau_tag(args.tau)
    run_label = str(args.min_run_length)
    alpha_label = f"{args.alpha:g}"

    hot_events = args.input_root / f"ERA5_hotnight_events_T_Tw_dyn_2000_2024_{tag}.parquet"
    cold_events = args.input_root / f"ERA5_coldnight_events_T_Tw_dyn_2000_2024_{tag}.parquet"

    if args.mode in {"hothot", "both"}:
        build_hot_hot_network(
            events_path=hot_events,
            output_edges=args.output_root / f"ECA_edges_dyn_{args.metric}{run_label}_a{alpha_label}_K{args.k_days}_{tag}.parquet",
            output_nodes=args.output_root / f"ECA_nodes_used_{args.metric}{run_label}_{tag}.parquet",
            metric=args.metric,
            min_run_length=args.min_run_length,
            alpha=args.alpha,
            k_days=args.k_days,
            min_events=args.min_events,
            min_overlap_days=args.min_overlap_days,
        )

    if args.mode in {"seesaw", "both"}:
        build_seesaw_network(
            hot_events_path=hot_events,
            cold_events_path=cold_events,
            output_edges=args.output_root / f"ECA_edges_SEESAW_{args.metric}{run_label}_a{alpha_label}_K{args.k_days}_{tag}.parquet",
            metric=args.metric,
            min_run_length=args.min_run_length,
            alpha=args.alpha,
            k_days=args.k_days,
            min_events=args.min_events,
            min_overlap_days=args.min_overlap_days,
        )


if __name__ == "__main__":
    main()
