from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange
from numba.typed import List
from pyproj import Geod
from scipy.stats import poisson


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from urban_hot_night_sync.paths import WARM_SEASON_INPUT_ROOT

DEFAULT_INPUT_ROOT = WARM_SEASON_INPUT_ROOT
DEFAULT_DERIVED_ROOT = REPO_ROOT / "data/derived/warm_season_parameter_sensitivity"
TAG = "tau015_local_warm3"


def min_run_indicator(series: pd.Series, length: int) -> pd.Series:
    values = series.to_numpy()
    flagged = np.zeros_like(values, dtype="int8")
    run = 0
    for idx, value in enumerate(values):
        if value == 1:
            run += 1
            if run >= length:
                flagged[idx] = 1
        else:
            run = 0
    return pd.Series(flagged, index=series.index)


@njit
def window_exists(events: np.ndarray, left: int, right: int, active_source: np.ndarray) -> bool:
    pos = np.searchsorted(events, left)
    while pos < len(events) and events[pos] <= right:
        if active_source[events[pos]]:
            return True
        pos += 1
    return False


@njit(parallel=True)
def observed_hot_hot(event_lists, active: np.ndarray, k_days: int) -> np.ndarray:
    n = len(event_lists)
    out = np.zeros((n * (n - 1)) // 2, dtype=np.int16)
    for i in prange(n - 1):
        offset = i * n - (i * (i + 1)) // 2
        events_i = event_lists[i]
        active_i = active[i]
        for j in range(i + 1, n):
            events_j = event_lists[j]
            active_j = active[j]
            observed = 0
            for event_day in events_i:
                if active_j[event_day] and window_exists(events_j, event_day - k_days, event_day + k_days, active_i):
                    observed += 1
            out[offset + (j - i - 1)] = observed
    return out


@njit(parallel=True)
def observed_hot_cold(
    hot_event_lists,
    cold_event_lists,
    active_hot: np.ndarray,
    active_cold: np.ndarray,
    k_days: int,
) -> np.ndarray:
    n_hot = len(hot_event_lists)
    n_cold = len(cold_event_lists)
    out = np.zeros(n_hot * n_cold, dtype=np.int16)
    for i in prange(n_hot):
        events_i = hot_event_lists[i]
        active_i = active_hot[i]
        for j in range(n_cold):
            events_j = cold_event_lists[j]
            active_j = active_cold[j]
            observed = 0
            for event_day in events_i:
                if active_j[event_day] and window_exists(events_j, event_day - k_days, event_day + k_days, active_i):
                    observed += 1
            out[i * n_cold + j] = observed
    return out.reshape((n_hot, n_cold))


def bh_keep_count(p_values: np.ndarray, total_tests: int, q: float = 0.05) -> tuple[int, float]:
    if len(p_values) == 0:
        return 0, np.nan
    sorted_p = np.sort(np.asarray(p_values, dtype=float))
    thresholds = (np.arange(1, len(sorted_p) + 1) / total_tests) * q
    keep = sorted_p <= thresholds
    n_keep = int(keep.sum())
    cutoff = float(sorted_p[n_keep - 1]) if n_keep else np.nan
    return n_keep, cutoff


def build_event_column(frame: pd.DataFrame, base_col: str, duration_days: int) -> str:
    if duration_days == 1:
        return base_col
    existing = f"{base_col}_run{duration_days}"
    if existing in frame.columns:
        return existing
    out_col = existing
    frame[out_col] = (
        frame.sort_values(["lon", "lat", "lt_date"])
        .groupby(["lon", "lat"], sort=False)[base_col]
        .transform(lambda series: min_run_indicator(series, duration_days))
        .astype("int8")
    )
    return out_col


def build_matrices(
    frame: pd.DataFrame,
    event_col: str,
    min_events: int,
    date_start: pd.Timestamp | None = None,
    date_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, List, pd.Timestamp, pd.Timestamp]:
    use = frame[["lon", "lat", "lt_date", event_col]].drop_duplicates(subset=["lon", "lat", "lt_date"]).copy()
    event_counts = use.groupby(["lon", "lat"])[event_col].sum()
    valid_index = event_counts[event_counts >= min_events].index
    use = use.set_index(["lon", "lat"]).loc[valid_index].reset_index()
    nodes = use[["lon", "lat"]].drop_duplicates().sort_values(["lon", "lat"]).reset_index(drop=True)
    node_codes = {(row.lon, row.lat): idx for idx, row in nodes.iterrows()}
    use["node_idx"] = [node_codes[(lon, lat)] for lon, lat in zip(use["lon"], use["lat"])]
    start = use["lt_date"].min() if date_start is None else pd.Timestamp(date_start)
    end = use["lt_date"].max() if date_end is None else pd.Timestamp(date_end)
    use = use.loc[(use["lt_date"] >= start) & (use["lt_date"] <= end)].copy()
    use["date_idx"] = (use["lt_date"] - start).dt.days.astype("int32")

    n_nodes = len(nodes)
    n_days = int((end - start).days + 1)
    active = np.zeros((n_nodes, n_days), dtype=np.bool_)
    events = np.zeros((n_nodes, n_days), dtype=np.bool_)
    active[use["node_idx"].to_numpy(), use["date_idx"].to_numpy()] = True
    event_frame = use.loc[use[event_col] == 1, ["node_idx", "date_idx"]]
    events[event_frame["node_idx"].to_numpy(), event_frame["date_idx"].to_numpy()] = True

    typed_events = List()
    for idx in range(n_nodes):
        typed_events.append(np.flatnonzero(events[idx]).astype(np.int64))
    return nodes, active, events, typed_events, start, end


def geodetic_distance_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    geod = Geod(ellps="WGS84")
    _, _, distance_m = geod.inv(lon1, lat1, lon2, lat2)
    return np.asarray(distance_m, dtype=float) / 1000.0


def summarize_thresholds(
    p_values: np.ndarray,
    dist_km: np.ndarray,
    total_tests: int,
    alpha_values: list[float],
    base_row: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    for alpha in alpha_values:
        keep = p_values < alpha
        kept_p = p_values[keep]
        kept_dist = dist_km[keep]
        bh_n, bh_cutoff = bh_keep_count(kept_p, total_tests=total_tests, q=0.05)
        bh_keep = kept_p <= bh_cutoff if bh_n else np.zeros_like(kept_p, dtype=bool)
        rows.append(
            {
                **base_row,
                "alpha": alpha,
                "candidate_edges_p_lt_alpha": int(keep.sum()),
                "long_range_edges_gt2500km": int((kept_dist > 2500).sum()),
                "long_range_percent": float((kept_dist > 2500).mean() * 100) if len(kept_dist) else np.nan,
                "median_distance_km": float(np.median(kept_dist)) if len(kept_dist) else np.nan,
                "bh_retained_edges_q05": int(bh_n),
                "bh_p_cutoff": float(bh_cutoff) if bh_n else np.nan,
                "bh_retained_long_range_edges_gt2500km": int((kept_dist[bh_keep] > 2500).sum()) if bh_n else 0,
            }
        )
    return rows


def exact_eca_from_matrices(
    active_a: np.ndarray,
    events_a: np.ndarray,
    active_b: np.ndarray,
    events_b: np.ndarray,
    k_days: int,
    min_overlap_days: int,
) -> tuple[float, int, float, int]:
    overlap = active_a & active_b
    n_eff = int(overlap.sum())
    if n_eff < min_overlap_days:
        return 1.0, 0, 0.0, n_eff
    a_idx = np.flatnonzero(events_a & overlap)
    b_idx = np.flatnonzero(events_b & overlap)
    if len(a_idx) == 0 or len(b_idx) == 0:
        return 1.0, 0, 0.0, n_eff
    observed = 0
    for value in a_idx:
        left = np.searchsorted(b_idx, value - k_days, side="left")
        right = np.searchsorted(b_idx, value + k_days, side="right")
        if left != right:
            observed += 1
    lam = (len(a_idx) * len(b_idx) * (2 * k_days + 1)) / n_eff
    return float(poisson.sf(observed - 1, lam)), int(observed), float(lam), n_eff


def validate_coherent(
    rng: np.random.Generator,
    nodes: pd.DataFrame,
    active: np.ndarray,
    events: np.ndarray,
    tri_i: np.ndarray,
    tri_j: np.ndarray,
    p_values: np.ndarray,
    obs: np.ndarray,
    lam: np.ndarray,
    n_eff: np.ndarray,
    k_days: int,
    sample_size: int,
) -> dict[str, object]:
    candidate = np.flatnonzero(p_values < 0.005)
    if len(candidate) == 0:
        candidate = np.arange(len(p_values))
    sample = rng.choice(candidate, size=min(sample_size, len(candidate)), replace=False)
    obs_bad = 0
    n_eff_bad = 0
    max_lam = 0.0
    max_p = 0.0
    for idx in sample:
        i = tri_i[idx]
        j = tri_j[idx]
        p0, obs0, lam0, n_eff0 = exact_eca_from_matrices(active[i], events[i], active[j], events[j], k_days, 200)
        obs_bad += int(obs0 != int(obs[idx]))
        n_eff_bad += int(n_eff0 != int(n_eff[idx]))
        max_lam = max(max_lam, abs(lam0 - float(lam[idx])))
        max_p = max(max_p, abs(p0 - float(p_values[idx])))
    return {
        "network_layer": "Coherent",
        "sampled_edges": int(len(sample)),
        "obs_mismatches": obs_bad,
        "n_eff_mismatches": n_eff_bad,
        "max_abs_lam_diff": max_lam,
        "max_abs_p_diff": max_p,
    }


def validate_dipole(
    rng: np.random.Generator,
    active_hot: np.ndarray,
    hot_events: np.ndarray,
    active_cold: np.ndarray,
    cold_events: np.ndarray,
    p_values: np.ndarray,
    obs: np.ndarray,
    lam: np.ndarray,
    n_eff: np.ndarray,
    k_days: int,
    sample_size: int,
) -> dict[str, object]:
    candidate = np.flatnonzero(p_values.ravel() < 0.005)
    if len(candidate) == 0:
        candidate = np.arange(p_values.size)
    sample = rng.choice(candidate, size=min(sample_size, len(candidate)), replace=False)
    n_cold = p_values.shape[1]
    obs_bad = 0
    n_eff_bad = 0
    max_lam = 0.0
    max_p = 0.0
    for flat_idx in sample:
        i = int(flat_idx // n_cold)
        j = int(flat_idx % n_cold)
        p0, obs0, lam0, n_eff0 = exact_eca_from_matrices(
            active_hot[i],
            hot_events[i],
            active_cold[j],
            cold_events[j],
            k_days,
            200,
        )
        obs_bad += int(obs0 != int(obs[i, j]))
        n_eff_bad += int(n_eff0 != int(n_eff[i, j]))
        max_lam = max(max_lam, abs(lam0 - float(lam[i, j])))
        max_p = max(max_p, abs(p0 - float(p_values[i, j])))
    return {
        "network_layer": "Dipole",
        "sampled_edges": int(len(sample)),
        "obs_mismatches": obs_bad,
        "n_eff_mismatches": n_eff_bad,
        "max_abs_lam_diff": max_lam,
        "max_abs_p_diff": max_p,
    }


def summarize_coherent(
    hot_nodes: pd.DataFrame,
    active: np.ndarray,
    events: np.ndarray,
    event_lists,
    duration_days: int,
    k_days: int,
    alpha_values: list[float],
    rng: np.random.Generator,
    sample_size: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    active_i = active.astype(np.int16)
    event_i = events.astype(np.int16)
    n_eff = active_i @ active_i.T
    n_a = event_i @ active_i.T
    n_b = n_a.T
    obs_flat = observed_hot_hot(event_lists, active, k_days).astype(np.int32)
    tri_i, tri_j = np.triu_indices(len(hot_nodes), k=1)
    n_eff_tri = n_eff[tri_i, tri_j].astype(np.int32)
    lam = (n_a[tri_i, tri_j].astype(float) * n_b[tri_i, tri_j].astype(float) * (2 * k_days + 1)) / np.maximum(
        n_eff_tri, 1
    )
    p_values = np.ones_like(lam, dtype=float)
    valid = n_eff_tri >= 200
    p_values[valid] = poisson.sf(obs_flat[valid] - 1, lam[valid])
    dist = geodetic_distance_km(
        hot_nodes.loc[tri_i, "lon"].to_numpy(),
        hot_nodes.loc[tri_i, "lat"].to_numpy(),
        hot_nodes.loc[tri_j, "lon"].to_numpy(),
        hot_nodes.loc[tri_j, "lat"].to_numpy(),
    )
    base = {
        "network_layer": "Coherent",
        "duration_days": duration_days,
        "k_days": k_days,
        "hot_valid_nodes": int(len(hot_nodes)),
        "cold_valid_nodes": np.nan,
        "total_pairwise_tests": int(len(tri_i)),
    }
    rows = summarize_thresholds(p_values, dist, len(tri_i), alpha_values, base)
    validation = validate_coherent(
        rng, hot_nodes, active, events, tri_i, tri_j, p_values, obs_flat, lam, n_eff_tri, k_days, sample_size
    )
    validation.update({"duration_days": duration_days, "k_days": k_days})
    return rows, validation


def summarize_dipole(
    hot_nodes: pd.DataFrame,
    cold_nodes: pd.DataFrame,
    active_hot: np.ndarray,
    hot_events: np.ndarray,
    hot_lists,
    active_cold: np.ndarray,
    cold_events: np.ndarray,
    cold_lists,
    duration_days: int,
    k_days: int,
    alpha_values: list[float],
    rng: np.random.Generator,
    sample_size: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    active_hot_i = active_hot.astype(np.int16)
    active_cold_i = active_cold.astype(np.int16)
    hot_i = hot_events.astype(np.int16)
    cold_i = cold_events.astype(np.int16)
    n_eff = active_hot_i @ active_cold_i.T
    n_hot = hot_i @ active_cold_i.T
    n_cold = active_hot_i @ cold_i.T
    obs = observed_hot_cold(hot_lists, cold_lists, active_hot, active_cold, k_days).astype(np.int32)
    same_node = hot_nodes[["lon", "lat"]].merge(
        cold_nodes[["lon", "lat"]].reset_index(names="cold_idx"),
        on=["lon", "lat"],
        how="left",
    )["cold_idx"]
    same_node = same_node.dropna().astype(int)
    same_hot = same_node.index.to_numpy()
    same_cold = same_node.to_numpy()
    n_eff[same_hot, same_cold] = 0
    lam = (n_hot.astype(float) * n_cold.astype(float) * (2 * k_days + 1)) / np.maximum(n_eff, 1)
    p_values = np.ones_like(lam, dtype=float)
    valid = n_eff >= 200
    p_values[valid] = poisson.sf(obs[valid] - 1, lam[valid])
    lon1 = np.repeat(hot_nodes["lon"].to_numpy(), len(cold_nodes))
    lat1 = np.repeat(hot_nodes["lat"].to_numpy(), len(cold_nodes))
    lon2 = np.tile(cold_nodes["lon"].to_numpy(), len(hot_nodes))
    lat2 = np.tile(cold_nodes["lat"].to_numpy(), len(hot_nodes))
    dist = geodetic_distance_km(lon1, lat1, lon2, lat2).reshape(p_values.shape)
    total_tests = int(len(hot_nodes) * len(cold_nodes) - len(same_hot))
    base = {
        "network_layer": "Dipole",
        "duration_days": duration_days,
        "k_days": k_days,
        "hot_valid_nodes": int(len(hot_nodes)),
        "cold_valid_nodes": int(len(cold_nodes)),
        "total_pairwise_tests": total_tests,
    }
    rows = summarize_thresholds(p_values.ravel(), dist.ravel(), total_tests, alpha_values, base)
    validation = validate_dipole(
        rng, active_hot, hot_events, active_cold, cold_events, p_values, obs, lam, n_eff, k_days, sample_size
    )
    validation.update({"duration_days": duration_days, "k_days": k_days})
    return rows, validation


def event_count_summary(frame: pd.DataFrame, event_col: str, label: str, duration_days: int) -> dict[str, object]:
    values = frame.loc[frame[event_col] == 1, "Tmin_C"].astype(float)
    return {
        "event_set": label,
        "duration_days": duration_days,
        "event_days": int(len(values)),
        "median_tmin_c": float(values.median()) if len(values) else np.nan,
        "pct_ge_20c": float((values >= 20).mean() * 100) if len(values) else np.nan,
        "pct_ge_25c": float((values >= 25).mean() * 100) if len(values) else np.nan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm-season ECA parameter sensitivity for corrected local-warm3 events.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--durations", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sample-size", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hot_path = args.input_root / f"ERA5_hotnight_events_T_Tw_dyn_2000_2024_{TAG}.parquet"
    cold_path = args.input_root / f"ERA5_coldnight_events_T_Tw_dyn_2000_2024_{TAG}.parquet"
    missing = [path for path in [hot_path, cold_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing corrected warm-season event tables:\n" + "\n".join(str(path) for path in missing))

    hot = pd.read_parquet(hot_path, columns=["lon", "lat", "lt_date", "Tmin_C", "evt_T", "evt_T_run2"])
    cold = pd.read_parquet(cold_path, columns=["lon", "lat", "lt_date", "Tmin_C", "evt_cold_T", "evt_cold_T_run2"])
    hot["lt_date"] = pd.to_datetime(hot["lt_date"])
    cold["lt_date"] = pd.to_datetime(cold["lt_date"])
    date_start = min(hot["lt_date"].min(), cold["lt_date"].min())
    date_end = max(hot["lt_date"].max(), cold["lt_date"].max())

    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20240527)

    for duration_days in args.durations:
        print(f"Preparing duration D={duration_days}")
        hot_col = build_event_column(hot, "evt_T", duration_days)
        cold_col = build_event_column(cold, "evt_cold_T", duration_days)
        event_rows.append(event_count_summary(hot, hot_col, "hot-night event days", duration_days))
        event_rows.append(event_count_summary(cold, cold_col, "cool-anomaly event days", duration_days))

        hot_nodes, active_hot, hot_events, hot_lists, _, _ = build_matrices(
            hot, hot_col, min_events=5, date_start=date_start, date_end=date_end
        )
        cold_nodes, active_cold, cold_events, cold_lists, _, _ = build_matrices(
            cold, cold_col, min_events=5, date_start=date_start, date_end=date_end
        )

        k_values = [3] if duration_days != 2 else [1, 3, 5]
        for k_days in k_values:
            alpha_values = [0.005]
            if duration_days == 2 and k_days == 3:
                alpha_values = [0.001, 0.005, 0.01]
            print(f"Computing D={duration_days}, K={k_days}, alpha={alpha_values}")
            coherent_rows, coherent_validation = summarize_coherent(
                hot_nodes,
                active_hot,
                hot_events,
                hot_lists,
                duration_days,
                k_days,
                alpha_values,
                rng,
                args.sample_size,
            )
            dipole_rows, dipole_validation = summarize_dipole(
                hot_nodes,
                cold_nodes,
                active_hot,
                hot_events,
                hot_lists,
                active_cold,
                cold_events,
                cold_lists,
                duration_days,
                k_days,
                alpha_values,
                rng,
                args.sample_size,
            )
            rows.extend(coherent_rows)
            rows.extend(dipole_rows)
            validation_rows.extend([coherent_validation, dipole_validation])

    summary = pd.DataFrame(rows)
    primary = summary.loc[
        (summary["duration_days"] == 2) & (summary["k_days"] == 3) & (summary["alpha"] == 0.005)
    ][["network_layer", "candidate_edges_p_lt_alpha"]].rename(
        columns={"candidate_edges_p_lt_alpha": "primary_candidate_edges"}
    )
    summary = summary.merge(primary, on="network_layer", how="left")
    summary["edge_count_relative_to_primary"] = (
        summary["candidate_edges_p_lt_alpha"] / summary["primary_candidate_edges"]
    )
    summary = summary.drop(columns=["primary_candidate_edges"])
    args.derived_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.derived_root / "warm_season_network_parameter_sensitivity.csv", index=False)
    pd.DataFrame(event_rows).to_csv(args.derived_root / "warm_season_duration_event_count_sensitivity.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(
        args.derived_root / "warm_season_parameter_sensitivity_validation.csv", index=False
    )
    print(f"Wrote parameter sensitivity outputs to {args.derived_root}")


if __name__ == "__main__":
    main()
