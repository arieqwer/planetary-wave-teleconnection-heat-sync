from __future__ import annotations

import argparse
import sys
from bisect import bisect_left
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes, wilcoxon
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from urban_hot_night_sync.paths import DERIVED_ROOT, WARM_SEASON_INPUT_ROOT


ROUND_DECIMALS = 2
YEARS = list(range(2000, 2025))


def local_warm_suffix(season_mode: str) -> str:
    return "" if season_mode == "all" else f"_{season_mode.replace('-', '_')}"


def safe_name(value: str) -> str:
    return value.replace(" ", "").replace("->", "_to_").replace("<->", "_and_")


def region_name(lon: float, lat: float, compact: bool = False) -> str:
    if lat > 20 and -130 < lon < -60:
        return "N.America" if compact else "NorthAmerica"
    if lat > 20 and 60 < lon < 150:
        return "E.Asia" if compact else "EastAsia"
    if lat > 30 and -10 < lon < 40:
        return "Europe"
    return f"({lon:.0f},{lat:.0f})"


def load_event_days(path: Path, column: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["lon", "lat", "lt_date", column])
    frame = frame.loc[frame[column] == 1, ["lon", "lat", "lt_date"]].copy()
    frame["lt_date"] = pd.to_datetime(frame["lt_date"])
    return frame


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


def write_dates(path: Path, dates: Iterable[pd.Timestamp]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d")}).to_csv(path, index=False)


def bundle_summary_row(
    mechanism: str,
    rank: int,
    cluster_id: int,
    edge_count: int,
    total_edges: int,
    centroid_a_lon: float,
    centroid_a_lat: float,
    centroid_b_lon: float,
    centroid_b_lat: float,
    node_count_a: int,
    node_count_b: int,
    dates: pd.DatetimeIndex,
    filename: str,
) -> dict[str, object]:
    return {
        "mechanism": mechanism,
        "rank": rank,
        "cluster_id": int(cluster_id),
        "edge_count": int(edge_count),
        "edge_share": float(edge_count / total_edges) if total_edges else np.nan,
        "centroid_a_lon": float(centroid_a_lon),
        "centroid_a_lat": float(centroid_a_lat),
        "centroid_b_lon": float(centroid_b_lon),
        "centroid_b_lat": float(centroid_b_lat),
        "node_count_a": int(node_count_a),
        "node_count_b": int(node_count_b),
        "date_count": int(len(dates)),
        "first_date": dates.min().strftime("%Y-%m-%d") if len(dates) else "",
        "last_date": dates.max().strftime("%Y-%m-%d") if len(dates) else "",
        "date_file": filename,
    }


def extract_coherent_bundles(
    edges_path: Path,
    hot_events_path: Path,
    output_dir: Path,
    top_n: int,
    n_clusters: int,
    min_dist_km: float,
    region_threshold: float,
) -> list[dict[str, object]]:
    edges = pd.read_parquet(edges_path)
    long_edges = edges.loc[edges["dist_km"] > min_dist_km].copy()
    if len(long_edges) < n_clusters:
        raise ValueError(f"Only {len(long_edges)} coherent edges exceed {min_dist_km:g} km; cannot cluster {n_clusters}.")

    long_edges["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(
        long_edges[["lon1", "lat1", "lon2", "lat2"]]
    )
    hot_events = load_event_days(hot_events_path, "evt_T_run2")
    rows: list[dict[str, object]] = []

    for rank, cluster_id in enumerate(long_edges["cluster"].value_counts().index[:top_n], start=1):
        subset = long_edges.loc[long_edges["cluster"] == cluster_id].copy()
        nodes_a = subset[["lon1", "lat1"]].rename(columns={"lon1": "lon", "lat1": "lat"}).drop_duplicates()
        nodes_b = subset[["lon2", "lat2"]].rename(columns={"lon2": "lon", "lat2": "lat"}).drop_duplicates()

        dates = synchronized_dates(
            daily_signal(hot_events, nodes_a),
            daily_signal(hot_events, nodes_b),
            threshold=region_threshold,
        )
        name = (
            f"{region_name(subset['lon1'].mean(), subset['lat1'].mean(), compact=True)}"
            f"_and_{region_name(subset['lon2'].mean(), subset['lat2'].mean(), compact=True)}"
        )
        filename = f"dates_coherent_local_warm3_bundle_{rank}_{safe_name(name)}.csv"
        write_dates(output_dir / filename, dates)
        rows.append(
            bundle_summary_row(
                mechanism="Coherent",
                rank=rank,
                cluster_id=cluster_id,
                edge_count=len(subset),
                total_edges=len(long_edges),
                centroid_a_lon=subset["lon1"].mean(),
                centroid_a_lat=subset["lat1"].mean(),
                centroid_b_lon=subset["lon2"].mean(),
                centroid_b_lat=subset["lat2"].mean(),
                node_count_a=len(nodes_a),
                node_count_b=len(nodes_b),
                dates=dates,
                filename=filename,
            )
        )
    return rows


def extract_dipole_bundles(
    edges_path: Path,
    hot_events_path: Path,
    cold_events_path: Path,
    output_dir: Path,
    top_n: int,
    n_clusters: int,
    min_dist_km: float,
    region_threshold: float,
) -> list[dict[str, object]]:
    edges = pd.read_parquet(edges_path)
    long_edges = edges.loc[edges["dist_km"] > min_dist_km].copy()
    if len(long_edges) < n_clusters:
        raise ValueError(f"Only {len(long_edges)} dipole edges exceed {min_dist_km:g} km; cannot cluster {n_clusters}.")

    long_edges["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(
        long_edges[["hot_lon", "hot_lat", "cold_lon", "cold_lat"]]
    )
    hot_events = load_event_days(hot_events_path, "evt_T_run2")
    cold_events = load_event_days(cold_events_path, "evt_cold_T_run2")
    rows: list[dict[str, object]] = []

    for rank, cluster_id in enumerate(long_edges["cluster"].value_counts().index[:top_n], start=1):
        subset = long_edges.loc[long_edges["cluster"] == cluster_id].copy()
        hot_nodes = subset[["hot_lon", "hot_lat"]].rename(columns={"hot_lon": "lon", "hot_lat": "lat"}).drop_duplicates()
        cold_nodes = subset[["cold_lon", "cold_lat"]].rename(columns={"cold_lon": "lon", "cold_lat": "lat"}).drop_duplicates()

        dates = synchronized_dates(
            daily_signal(hot_events, hot_nodes),
            daily_signal(cold_events, cold_nodes),
            threshold=region_threshold,
        )
        name = (
            f"{region_name(subset['hot_lon'].mean(), subset['hot_lat'].mean())}"
            f"_to_{region_name(subset['cold_lon'].mean(), subset['cold_lat'].mean())}"
        )
        filename = f"dates_dipole_local_warm3_bundle_{rank}_{safe_name(name)}.csv"
        write_dates(output_dir / filename, dates)
        rows.append(
            bundle_summary_row(
                mechanism="Dipole",
                rank=rank,
                cluster_id=cluster_id,
                edge_count=len(subset),
                total_edges=len(long_edges),
                centroid_a_lon=subset["hot_lon"].mean(),
                centroid_a_lat=subset["hot_lat"].mean(),
                centroid_b_lon=subset["cold_lon"].mean(),
                centroid_b_lat=subset["cold_lat"].mean(),
                node_count_a=len(hot_nodes),
                node_count_b=len(cold_nodes),
                dates=dates,
                filename=filename,
            )
        )
    return rows


def annual_counts_for_files(bundle_dir: Path, filenames: list[str]) -> pd.Series:
    annual = pd.Series(0, index=YEARS, dtype=float)
    for filename in filenames:
        frame = pd.read_csv(bundle_dir / filename)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        counts = frame.dropna(subset=["date"]).groupby(frame["date"].dt.year).size()
        annual = annual.add(counts.reindex(YEARS, fill_value=0), fill_value=0)
    return annual


def trend_row(mechanism: str, annual: pd.Series, filenames: list[str]) -> dict[str, object]:
    x = np.asarray(YEARS, dtype=float)
    y = annual.to_numpy(dtype=float)
    slope, intercept, lo, hi = theilslopes(y, x, 0.95)
    tau, p_value = kendalltau(x, y)
    return {
        "mechanism": mechanism,
        "years": f"{YEARS[0]}-{YEARS[-1]}",
        "top_bundle_files": "; ".join(filenames),
        "total_bundle_days": int(y.sum()),
        "mean_annual_bundle_days": float(y.mean()),
        "theil_sen_slope_per_year": float(slope),
        "theil_sen_intercept": float(intercept),
        "theil_sen_95_low": float(lo),
        "theil_sen_95_high": float(hi),
        "kendall_tau": float(tau) if not np.isnan(tau) else np.nan,
        "kendall_p": float(p_value) if not np.isnan(p_value) else np.nan,
    }


def build_runs_from_daily_events(
    events_path: Path,
    flag_col: str,
    value_col: str,
    threshold_col: str,
    min_duration: int,
) -> pd.DataFrame:
    frame = pd.read_parquet(
        events_path,
        columns=["lon", "lat", "lt_date", flag_col, value_col, threshold_col],
    )
    frame["lt_date"] = pd.to_datetime(frame["lt_date"])
    frame = frame.loc[frame[flag_col] == 1, ["lon", "lat", "lt_date", value_col, threshold_col]].copy()
    frame["lat_r"] = frame["lat"].round(ROUND_DECIMALS).astype(float)
    frame["lon_r"] = frame["lon"].round(ROUND_DECIMALS).astype(float)
    frame["node"] = list(zip(frame["lat_r"], frame["lon_r"]))
    frame["excess"] = frame[value_col].astype("float32") - frame[threshold_col].astype("float32")
    frame = frame.sort_values(["node", "lt_date"]).reset_index(drop=True)

    date_gap = frame.groupby("node")["lt_date"].diff()
    frame["run_id"] = ((date_gap.isna()) | (date_gap != pd.Timedelta(days=1))).groupby(frame["node"]).cumsum()
    runs = (
        frame.groupby(["node", "run_id"], as_index=False)
        .agg(
            start=("lt_date", "min"),
            end=("lt_date", "max"),
            duration=("lt_date", "count"),
            intensity_mean=("excess", "mean"),
            intensity_max=("excess", "max"),
        )
    )
    runs = runs.loc[runs["duration"] >= min_duration].copy()
    runs["month"] = runs["start"].dt.month.astype("int16")
    runs["dur_bin"] = pd.cut(
        runs["duration"],
        bins=[1, 2, 4, 10_000],
        labels=["2", "3-4", "5+"],
        right=True,
        include_lowest=False,
    )
    runs["run_uid"] = np.arange(len(runs), dtype=np.int64)
    return runs


def coherent_pairs(edges: pd.DataFrame, min_dist_km: float | None = None) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if min_dist_km is not None:
        edges = edges.loc[edges["dist_km"] > min_dist_km].copy()
    nodes_a = list(zip(edges["lat1"].round(ROUND_DECIMALS).astype(float), edges["lon1"].round(ROUND_DECIMALS).astype(float)))
    nodes_b = list(zip(edges["lat2"].round(ROUND_DECIMALS).astype(float), edges["lon2"].round(ROUND_DECIMALS).astype(float)))
    pairs = [(a, b) if a <= b else (b, a) for a, b in zip(nodes_a, nodes_b)]
    return list(dict.fromkeys(pairs))


def dipole_pairs(edges: pd.DataFrame, min_dist_km: float | None = None) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if min_dist_km is not None:
        edges = edges.loc[edges["dist_km"] > min_dist_km].copy()
    hot_nodes = list(
        zip(edges["hot_lat"].round(ROUND_DECIMALS).astype(float), edges["hot_lon"].round(ROUND_DECIMALS).astype(float))
    )
    cold_nodes = list(
        zip(edges["cold_lat"].round(ROUND_DECIMALS).astype(float), edges["cold_lon"].round(ROUND_DECIMALS).astype(float))
    )
    return list(dict.fromkeys(zip(hot_nodes, cold_nodes)))


def overlaps(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp, k_days: int) -> bool:
    k = pd.Timedelta(days=k_days)
    return (a_start - k) <= (b_end + k) and (b_start - k) <= (a_end + k)


def run_groups(runs: pd.DataFrame) -> dict[tuple[float, float], np.ndarray]:
    return {
        node: group[["run_uid", "start", "end"]].sort_values("start").to_numpy()
        for node, group in runs.groupby("node", sort=False)
    }


def mark_coherent_locked(
    hot_runs: pd.DataFrame,
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
    k_days: int,
) -> pd.DataFrame:
    by_node = run_groups(hot_runs)
    locked: set[int] = set()
    for n1, n2 in pairs:
        if n1 not in by_node or n2 not in by_node:
            continue
        a_runs = by_node[n1]
        b_runs = by_node[n2]
        i = j = 0
        while i < len(a_runs) and j < len(b_runs):
            uid_a, a_start, a_end = a_runs[i]
            uid_b, b_start, b_end = b_runs[j]
            if overlaps(a_start, a_end, b_start, b_end, k_days):
                locked.add(int(uid_a))
                locked.add(int(uid_b))
                if a_end <= b_end:
                    i += 1
                else:
                    j += 1
            elif a_start < b_start:
                i += 1
            else:
                j += 1
    out = hot_runs.copy()
    out["is_locked"] = out["run_uid"].isin(locked)
    return out


def mark_dipole_hot_side_locked(
    hot_runs: pd.DataFrame,
    cold_runs: pd.DataFrame,
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
    k_days: int,
) -> pd.DataFrame:
    hot_by_node = run_groups(hot_runs)
    cold_by_node = run_groups(cold_runs)
    locked: set[int] = set()
    for hot_node, cold_node in pairs:
        if hot_node not in hot_by_node or cold_node not in cold_by_node:
            continue
        hot_arr = hot_by_node[hot_node]
        cold_arr = cold_by_node[cold_node]
        i = j = 0
        while i < len(hot_arr) and j < len(cold_arr):
            uid_hot, hot_start, hot_end = hot_arr[i]
            _, cold_start, cold_end = cold_arr[j]
            if overlaps(hot_start, hot_end, cold_start, cold_end, k_days):
                locked.add(int(uid_hot))
                if hot_end <= cold_end:
                    i += 1
                else:
                    j += 1
            elif hot_start < cold_start:
                i += 1
            else:
                j += 1
    out = hot_runs.copy()
    out["is_locked"] = out["run_uid"].isin(locked)
    return out


def nearest_unused_control(locked_start: pd.Timestamp, controls: pd.DataFrame, used: set[int]) -> int | None:
    available = controls.loc[~controls["run_uid"].isin(used)].sort_values("start")
    if available.empty:
        return None
    starts = available["start"].to_numpy(dtype="datetime64[ns]")
    target = np.datetime64(locked_start.to_datetime64(), "ns")
    pos = bisect_left(starts, target)
    candidate_positions = [p for p in (pos - 1, pos, pos + 1) if 0 <= p < len(available)]
    if not candidate_positions:
        candidate_positions = [0]
    best_pos = min(candidate_positions, key=lambda p: abs((starts[p] - target).astype("timedelta64[s]").astype(int)))
    return int(available.iloc[best_pos]["run_uid"])


def deterministic_match_pairs(runs: pd.DataFrame) -> pd.DataFrame:
    columns = ["run_uid", "node", "month", "dur_bin", "start", "is_locked", "intensity_mean"]
    frame = runs[columns].dropna().copy()
    frame["key"] = list(zip(frame["node"], frame["month"], frame["dur_bin"].astype(str)))
    controls_by_key = {key: group.copy() for key, group in frame.loc[~frame["is_locked"]].groupby("key", sort=False)}

    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for key, locked_group in frame.loc[frame["is_locked"]].sort_values("start").groupby("key", sort=False):
        controls = controls_by_key.get(key)
        if controls is None:
            continue
        for _, locked_row in locked_group.iterrows():
            control_uid = nearest_unused_control(locked_row["start"], controls, used)
            if control_uid is None:
                continue
            pairs.append((int(locked_row["run_uid"]), control_uid))
            used.add(control_uid)
    return pd.DataFrame(pairs, columns=["locked_uid", "control_uid"])


def bootstrap_mean_ci(deltas: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    index = np.arange(len(deltas))
    for boot_idx in range(n_boot):
        means[boot_idx] = np.mean(deltas[rng.choice(index, size=len(index), replace=True)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_summary(runs: pd.DataFrame, pairs: pd.DataFrame, n_boot: int, seed: int) -> tuple[dict[str, object], pd.DataFrame]:
    if pairs.empty:
        return {
            "n_pairs": 0,
            "mean_delta": np.nan,
            "median_delta": np.nan,
            "boot95_mean_lo": np.nan,
            "boot95_mean_hi": np.nan,
            "wilcoxon_p_greater": np.nan,
        }, pd.DataFrame()

    indexed = runs.set_index("run_uid")
    locked = indexed.loc[pairs["locked_uid"], "intensity_mean"].to_numpy(dtype=float)
    control = indexed.loc[pairs["control_uid"], "intensity_mean"].to_numpy(dtype=float)
    deltas = locked - control
    lo, hi = bootstrap_mean_ci(deltas, n_boot=n_boot, seed=seed)
    p_value = float(wilcoxon(deltas, alternative="greater").pvalue) if np.any(deltas != 0) else np.nan
    details = pairs.copy()
    details["locked_intensity_mean"] = locked
    details["control_intensity_mean"] = control
    details["delta"] = deltas
    return {
        "n_pairs": int(len(deltas)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "boot95_mean_lo": lo,
        "boot95_mean_hi": hi,
        "wilcoxon_p_greater": p_value,
    }, details


def stratified_summary(runs: pd.DataFrame, pairs: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    indexed = runs.set_index("run_uid")
    for dur_bin in ["2", "3-4", "5+"]:
        mask = indexed.loc[pairs["locked_uid"], "dur_bin"].astype(str).to_numpy() == dur_bin
        if mask.sum() < 30:
            continue
        stats, _ = paired_summary(runs, pairs.loc[mask].copy(), n_boot=n_boot, seed=seed)
        stats["dur_bin"] = dur_bin
        rows.append(stats)
    return pd.DataFrame(rows)


def raw_lock_summary(mechanism: str, run_set: str, runs: pd.DataFrame, pair_count: int) -> dict[str, object]:
    locked = runs.loc[runs["is_locked"], "intensity_mean"].dropna()
    free = runs.loc[~runs["is_locked"], "intensity_mean"].dropna()
    return {
        "mechanism": mechanism,
        "edge_set": run_set,
        "edge_pairs_used": int(pair_count),
        "n_runs": int(len(runs)),
        "n_locked_runs": int(runs["is_locked"].sum()),
        "locked_run_fraction": float(runs["is_locked"].mean()),
        "raw_locked_mean": float(locked.mean()) if len(locked) else np.nan,
        "raw_nonlocked_mean": float(free.mean()) if len(free) else np.nan,
        "raw_mean_delta": float(locked.mean() - free.mean()) if len(locked) and len(free) else np.nan,
    }


def run_matched_amplification(
    coherent_edges_path: Path,
    dipole_edges_path: Path,
    hot_events_path: Path,
    cold_events_path: Path,
    output_dir: Path,
    min_dist_km: float,
    k_days: int,
    min_duration: int,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hot_runs = build_runs_from_daily_events(
        hot_events_path,
        flag_col="evt_T_run2",
        value_col="Tmin_C",
        threshold_col="Tmin95",
        min_duration=min_duration,
    )
    cold_runs = build_runs_from_daily_events(
        cold_events_path,
        flag_col="evt_cold_T_run2",
        value_col="Tmin_C",
        threshold_col="Tmin05",
        min_duration=min_duration,
    )

    coherent_edges = pd.read_parquet(coherent_edges_path)
    dipole_edges = pd.read_parquet(dipole_edges_path)
    tasks = [
        ("Coherent", "all_edges", coherent_pairs(coherent_edges, None), None),
        ("Coherent", "long_range_gt2500km", coherent_pairs(coherent_edges, min_dist_km), None),
        ("Dipole", "all_edges", dipole_pairs(dipole_edges, None), "hot_cold"),
        ("Dipole", "long_range_gt2500km", dipole_pairs(dipole_edges, min_dist_km), "hot_cold"),
    ]

    summary_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for idx, (mechanism, edge_set, pairs, mode) in enumerate(tasks, start=1):
        if mechanism == "Coherent":
            locked_runs = mark_coherent_locked(hot_runs, pairs, k_days=k_days)
        else:
            locked_runs = mark_dipole_hot_side_locked(hot_runs, cold_runs, pairs, k_days=k_days)

        matched_pairs = deterministic_match_pairs(locked_runs)
        stats, _ = paired_summary(locked_runs, matched_pairs, n_boot=n_boot, seed=100 + idx)
        stats.update({"mechanism": mechanism, "edge_set": edge_set, "k_days": k_days, "min_duration": min_duration})
        summary_rows.append(stats)
        raw_rows.append(raw_lock_summary(mechanism, edge_set, locked_runs, pair_count=len(pairs)))

        prefix = f"matched_{mechanism.lower()}_{edge_set}_local_warm3"
        matched_pairs.to_csv(output_dir / f"{prefix}_pairs.csv", index=False)
        stratified_summary(locked_runs, matched_pairs, n_boot=max(500, n_boot // 2), seed=200 + idx).to_csv(
            output_dir / f"{prefix}_stratified.csv",
            index=False,
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(raw_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build warm-season bundle dates, trends, and matched amplification outputs for reviewer revision."
    )
    parser.add_argument("--input-root", type=Path, default=WARM_SEASON_INPUT_ROOT)
    parser.add_argument("--repo-output-root", type=Path, default=DERIVED_ROOT / "warm_season_local_warm3")
    parser.add_argument("--season-mode", default="local-warm3")
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--min-dist-km", type=float, default=2500.0)
    parser.add_argument("--region-threshold", type=float, default=0.05)
    parser.add_argument("--k-days", type=int, default=3)
    parser.add_argument("--min-duration", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--skip-matched", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = local_warm_suffix(args.season_mode)
    tag = f"tau{int(round(args.tau * 100)):03d}"
    args.repo_output_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = args.repo_output_root / "bundle_dates"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    coherent_edges = args.input_root / f"ECA_edges_dyn_T2_a0.005_K3_{tag}{suffix}.parquet"
    dipole_edges = args.input_root / f"ECA_edges_SEESAW_T2_a0.005_K3_{tag}{suffix}.parquet"
    hot_events = args.input_root / f"ERA5_hotnight_events_T_Tw_dyn_2000_2024_{tag}{suffix}.parquet"
    cold_events = args.input_root / f"ERA5_coldnight_events_T_Tw_dyn_2000_2024_{tag}{suffix}.parquet"

    missing = [path for path in [coherent_edges, dipole_edges, hot_events, cold_events] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required warm-season input files:\n" + "\n".join(str(path) for path in missing))

    rows = []
    rows.extend(
        extract_coherent_bundles(
            edges_path=coherent_edges,
            hot_events_path=hot_events,
            output_dir=bundle_dir,
            top_n=args.top_n,
            n_clusters=args.n_clusters,
            min_dist_km=args.min_dist_km,
            region_threshold=args.region_threshold,
        )
    )
    rows.extend(
        extract_dipole_bundles(
            edges_path=dipole_edges,
            hot_events_path=hot_events,
            cold_events_path=cold_events,
            output_dir=bundle_dir,
            top_n=args.top_n,
            n_clusters=args.n_clusters,
            min_dist_km=args.min_dist_km,
            region_threshold=args.region_threshold,
        )
    )
    bundle_summary = pd.DataFrame(rows)
    bundle_summary.to_csv(args.repo_output_root / "warm_season_bundle_summary.csv", index=False)

    trend_rows = []
    annual_rows = []
    for mechanism in ["Coherent", "Dipole"]:
        files = bundle_summary.loc[bundle_summary["mechanism"] == mechanism, "date_file"].head(args.top_n).tolist()
        annual = annual_counts_for_files(bundle_dir, files)
        trend_rows.append(trend_row(mechanism, annual, files))
        annual_rows.extend({"mechanism": mechanism, "year": int(year), "bundle_days": int(value)} for year, value in annual.items())
    pd.DataFrame(trend_rows).to_csv(args.repo_output_root / "warm_season_trend_summary.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(args.repo_output_root / "warm_season_annual_bundle_days.csv", index=False)

    if not args.skip_matched:
        matched_summary, raw_summary = run_matched_amplification(
            coherent_edges_path=coherent_edges,
            dipole_edges_path=dipole_edges,
            hot_events_path=hot_events,
            cold_events_path=cold_events,
            output_dir=args.repo_output_root,
            min_dist_km=args.min_dist_km,
            k_days=args.k_days,
            min_duration=args.min_duration,
            n_boot=args.n_boot,
        )
        matched_summary.to_csv(args.repo_output_root / "warm_season_matched_amplification_summary.csv", index=False)
        raw_summary.to_csv(args.repo_output_root / "warm_season_raw_lock_summary.csv", index=False)

    print(f"Wrote warm-season reviewer outputs to: {args.repo_output_root}")


if __name__ == "__main__":
    main()
