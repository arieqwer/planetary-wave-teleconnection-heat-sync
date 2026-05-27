from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from urban_hot_night_sync.paths import WARM_SEASON_INPUT_ROOT

DERIVED = REPO_ROOT / "data/derived/warm_season_local_warm3"
INPUT_ROOT = Path(os.environ.get("WARM_SEASON_INPUT_ROOT", WARM_SEASON_INPUT_ROOT))
TAG = "tau015_local_warm3"


def valid_nodes(path: Path, event_col: str, min_events: int = 5) -> set[tuple[float, float]]:
    frame = pd.read_parquet(path, columns=["lon", "lat", event_col])
    counts = frame.groupby(["lon", "lat"])[event_col].sum()
    return set(counts[counts >= min_events].index)


def bh_keep_count(p_values: np.ndarray, total_tests: int, q: float) -> tuple[int, float]:
    sorted_p = np.sort(np.asarray(p_values, dtype=float))
    thresholds = (np.arange(1, len(sorted_p) + 1) / total_tests) * q
    keep = sorted_p <= thresholds
    n_keep = int(keep.sum())
    cutoff = float(sorted_p[n_keep - 1]) if n_keep else np.nan
    return n_keep, cutoff


def summarize_edges(label: str, path: Path, total_tests: int, q: float) -> dict[str, object]:
    edges = pd.read_parquet(path)
    n_keep, cutoff = bh_keep_count(edges["p"].to_numpy(), total_tests, q)
    kept = edges.loc[edges["p"] <= cutoff].copy() if n_keep else edges.iloc[0:0].copy()
    return {
        "network_layer": label,
        "candidate_edges_p_lt_0_005": int(len(edges)),
        "total_pairwise_tests": int(total_tests),
        "bh_q": q,
        "bh_retained_edges": n_keep,
        "bh_p_cutoff": cutoff,
        "bh_retained_long_range_edges_gt2500km": int((kept["dist_km"] > 2500).sum()) if n_keep else 0,
        "bh_retained_long_range_percent": float((kept["dist_km"] > 2500).mean() * 100) if n_keep else np.nan,
        "bh_retained_median_distance_km": float(kept["dist_km"].median()) if n_keep else np.nan,
    }


def main() -> None:
    hot_events = INPUT_ROOT / f"ERA5_hotnight_events_T_Tw_dyn_2000_2024_{TAG}.parquet"
    cold_events = INPUT_ROOT / f"ERA5_coldnight_events_T_Tw_dyn_2000_2024_{TAG}.parquet"
    coherent_edges = INPUT_ROOT / f"ECA_edges_dyn_T2_a0.005_K3_{TAG}.parquet"
    dipole_edges = INPUT_ROOT / f"ECA_edges_SEESAW_T2_a0.005_K3_{TAG}.parquet"

    missing = [p for p in [hot_events, cold_events, coherent_edges, dipole_edges] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing warm-season inputs:\n" + "\n".join(str(p) for p in missing))

    hot_nodes = valid_nodes(hot_events, "evt_T_run2")
    cold_nodes = valid_nodes(cold_events, "evt_cold_T_run2")
    all_nodes = hot_nodes | cold_nodes

    coherent_tests = len(hot_nodes) * (len(hot_nodes) - 1) // 2
    # The dipole builder tests both hot->cool and cool->hot directions over unordered node pairs.
    dipole_tests = 2 * (len(all_nodes) * (len(all_nodes) - 1) // 2)

    rows = [
        summarize_edges("Coherent", coherent_edges, coherent_tests, q=0.05),
        summarize_edges("Dipole", dipole_edges, dipole_tests, q=0.05),
    ]
    output = DERIVED / "warm_season_edge_screen_fdr_diagnostic.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
