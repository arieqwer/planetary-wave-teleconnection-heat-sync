from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange
from numba.typed import List
from scipy.stats import poisson


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from urban_hot_night_sync.paths import WARM_SEASON_INPUT_ROOT, WARM_SEASON_WORK_ROOT

SCRIPT_DIR = REPO_ROOT / "scripts"
SOURCE_EVENT_ROOT = Path(os.environ.get("UHN_SYNC_SOURCE_EVENT_ROOT", WARM_SEASON_INPUT_ROOT.parent))
DEFAULT_WORK_ROOT = WARM_SEASON_WORK_ROOT / "definition_sensitivity"
DEFAULT_DERIVED_ROOT = REPO_ROOT / "data/derived/warm_season_definition_sensitivity"


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


def smooth_thresholds(frame: pd.DataFrame, spatial_cols: list[str], value_col: str, out_col: str) -> pd.DataFrame:
    frame = frame.sort_values(spatial_cols + ["doy"]).copy()
    frame[out_col] = (
        frame.groupby(spatial_cols)[value_col]
        .transform(lambda series: series.rolling(window=15, center=True, min_periods=1).mean())
    )
    return frame


def warm_month_table(frame: pd.DataFrame, spatial_cols: list[str], n_months: int) -> pd.DataFrame:
    baseline = frame.loc[(frame["year"] >= 2001) & (frame["year"] <= 2020)].copy()
    monthly = (
        baseline.groupby(spatial_cols + ["month"], as_index=False)["Tmin_C"]
        .mean()
        .rename(columns={"Tmin_C": "mean_Tmin_C"})
    )
    monthly = monthly.sort_values(spatial_cols + ["mean_Tmin_C"], ascending=[True] * len(spatial_cols) + [False])
    monthly["warm_rank"] = monthly.groupby(spatial_cols).cumcount() + 1
    return monthly.loc[monthly["warm_rank"] <= n_months, spatial_cols + ["month"]]


def build_event_table(
    source_path: Path,
    output_path: Path,
    n_months: int,
    kind: str,
    min_run_length: int,
) -> None:
    if output_path.exists():
        print(f"Exists, skipping event table: {output_path}")
        return

    value_cols = ["lon", "lat", "lt_date", "year", "doy", "Tmin_C", "Twmin_C"]
    frame = pd.read_parquet(source_path, columns=value_cols)
    frame["lt_date"] = pd.to_datetime(frame["lt_date"])
    frame["month"] = frame["lt_date"].dt.month.astype("int8")
    spatial_cols = ["lon", "lat"]

    warm_months = warm_month_table(frame, spatial_cols, n_months=n_months)
    frame = frame.merge(warm_months.assign(_warm=1), on=spatial_cols + ["month"], how="left")
    frame = frame.loc[frame["_warm"] == 1].drop(columns=["_warm"]).copy()
    baseline = frame.loc[(frame["year"] >= 2001) & (frame["year"] <= 2020)].copy()

    if kind == "hot":
        percentile = 95
        comparator = np.greater_equal
        prefix = "evt_"
        t_raw, tw_raw = "Tmin95_raw", "Twmin95_raw"
        t_thr, tw_thr = "Tmin95", "Twmin95"
    elif kind == "cold":
        percentile = 5
        comparator = np.less_equal
        prefix = "evt_cold_"
        t_raw, tw_raw = "Tmin05_raw", "Twmin05_raw"
        t_thr, tw_thr = "Tmin05", "Twmin05"
    else:
        raise ValueError(kind)

    thr_t = (
        baseline.groupby(spatial_cols + ["doy"], as_index=False)["Tmin_C"]
        .agg(lambda values: np.nanpercentile(values, percentile))
        .rename(columns={"Tmin_C": t_raw})
    )
    thr_tw = (
        baseline.groupby(spatial_cols + ["doy"], as_index=False)["Twmin_C"]
        .agg(lambda values: np.nanpercentile(values, percentile))
        .rename(columns={"Twmin_C": tw_raw})
    )
    thr_t = smooth_thresholds(thr_t, spatial_cols, t_raw, t_thr)
    thr_tw = smooth_thresholds(thr_tw, spatial_cols, tw_raw, tw_thr)

    out = (
        frame.merge(thr_t, on=spatial_cols + ["doy"], how="left")
        .merge(thr_tw, on=spatial_cols + ["doy"], how="left")
        .sort_values(spatial_cols + ["lt_date"])
        .copy()
    )
    out[f"{prefix}T"] = comparator(out["Tmin_C"].to_numpy(), out[t_thr].to_numpy()).astype("int8")
    out[f"{prefix}Tw"] = comparator(out["Twmin_C"].to_numpy(), out[tw_thr].to_numpy()).astype("int8")
    out[f"{prefix}T_run{min_run_length}"] = (
        out.groupby(spatial_cols, sort=False)[f"{prefix}T"]
        .transform(lambda series: min_run_indicator(series, min_run_length))
        .astype("int8")
    )
    out[f"{prefix}Tw_run{min_run_length}"] = (
        out.groupby(spatial_cols, sort=False)[f"{prefix}Tw"]
        .transform(lambda series: min_run_indicator(series, min_run_length))
        .astype("int8")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["month"]).to_parquet(output_path, index=False)
    print(f"Wrote {kind} local-warm{n_months} event table: {output_path}")


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


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    radius = 6371.0088
    lon1r = np.deg2rad(lon1)
    lon2r = np.deg2rad(lon2)
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def build_matrices(
    path: Path,
    event_col: str,
    min_events: int = 5,
    date_start: pd.Timestamp | None = None,
    date_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, List, pd.Timestamp, pd.Timestamp]:
    frame = pd.read_parquet(path, columns=["lon", "lat", "lt_date", event_col])
    frame["lt_date"] = pd.to_datetime(frame["lt_date"])
    frame = frame.drop_duplicates(subset=["lon", "lat", "lt_date"]).copy()
    event_counts = frame.groupby(["lon", "lat"])[event_col].sum()
    valid_index = event_counts[event_counts >= min_events].index
    frame = frame.set_index(["lon", "lat"]).loc[valid_index].reset_index()
    nodes = frame[["lon", "lat"]].drop_duplicates().sort_values(["lon", "lat"]).reset_index(drop=True)
    node_codes = {(row.lon, row.lat): idx for idx, row in nodes.iterrows()}
    frame["node_idx"] = [node_codes[(lon, lat)] for lon, lat in zip(frame["lon"], frame["lat"])]
    start = frame["lt_date"].min() if date_start is None else pd.Timestamp(date_start)
    end = frame["lt_date"].max() if date_end is None else pd.Timestamp(date_end)
    frame = frame.loc[(frame["lt_date"] >= start) & (frame["lt_date"] <= end)].copy()
    frame["date_idx"] = (frame["lt_date"] - start).dt.days.astype("int32")
    n_nodes = len(nodes)
    n_days = int((end - start).days + 1)
    active = np.zeros((n_nodes, n_days), dtype=np.bool_)
    events = np.zeros((n_nodes, n_days), dtype=np.bool_)
    active[frame["node_idx"].to_numpy(), frame["date_idx"].to_numpy()] = True
    event_frame = frame.loc[frame[event_col] == 1, ["node_idx", "date_idx"]]
    events[event_frame["node_idx"].to_numpy(), event_frame["date_idx"].to_numpy()] = True

    typed_events = List()
    for idx in range(n_nodes):
        typed_events.append(np.flatnonzero(events[idx]).astype(np.int64))
    return nodes, active, events, typed_events, start, end


def write_hot_hot_network(hot_path: Path, output_path: Path, k_days: int, alpha: float, min_overlap_days: int) -> None:
    if output_path.exists():
        print(f"Exists, skipping coherent network: {output_path}")
        return
    nodes, active, events, event_lists, _, _ = build_matrices(hot_path, "evt_T_run2")
    active_i = active.astype(np.int16)
    event_i = events.astype(np.int16)
    n_eff = active_i @ active_i.T
    n_a = event_i @ active_i.T
    n_b = n_a.T
    obs_flat = observed_hot_hot(event_lists, active, k_days)
    tri_i, tri_j = np.triu_indices(len(nodes), k=1)
    obs = obs_flat.astype(np.int32)
    n_eff_tri = n_eff[tri_i, tri_j].astype(np.int32)
    lam = (n_a[tri_i, tri_j].astype(float) * n_b[tri_i, tri_j].astype(float) * (2 * k_days + 1)) / np.maximum(n_eff_tri, 1)
    p_values = np.ones_like(lam, dtype=float)
    valid = n_eff_tri >= min_overlap_days
    p_values[valid] = poisson.sf(obs[valid] - 1, lam[valid])
    keep = valid & (p_values < alpha)
    ii = tri_i[keep]
    jj = tri_j[keep]
    dist = haversine_km(
        nodes.loc[ii, "lon"].to_numpy(),
        nodes.loc[ii, "lat"].to_numpy(),
        nodes.loc[jj, "lon"].to_numpy(),
        nodes.loc[jj, "lat"].to_numpy(),
    )
    out = pd.DataFrame(
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
    out.to_parquet(output_path, index=False)
    print(f"Wrote coherent sensitivity network: {output_path}")


def write_dipole_network(
    hot_path: Path,
    cold_path: Path,
    output_path: Path,
    k_days: int,
    alpha: float,
    min_overlap_days: int,
) -> None:
    if output_path.exists():
        print(f"Exists, skipping dipole network: {output_path}")
        return
    hot_minmax = pd.read_parquet(hot_path, columns=["lt_date"])
    cold_minmax = pd.read_parquet(cold_path, columns=["lt_date"])
    date_start = min(pd.to_datetime(hot_minmax["lt_date"]).min(), pd.to_datetime(cold_minmax["lt_date"]).min())
    date_end = max(pd.to_datetime(hot_minmax["lt_date"]).max(), pd.to_datetime(cold_minmax["lt_date"]).max())
    hot_nodes, active_hot, hot_events, hot_lists, _, _ = build_matrices(
        hot_path,
        "evt_T_run2",
        date_start=date_start,
        date_end=date_end,
    )
    cold_nodes, active_cold, cold_events, cold_lists, _, _ = build_matrices(
        cold_path,
        "evt_cold_T_run2",
        date_start=date_start,
        date_end=date_end,
    )
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
    valid = n_eff >= min_overlap_days
    p_values[valid] = poisson.sf(obs[valid] - 1, lam[valid])
    keep = valid & (p_values < alpha)
    ii, jj = np.where(keep)
    dist = haversine_km(
        hot_nodes.loc[ii, "lon"].to_numpy(),
        hot_nodes.loc[ii, "lat"].to_numpy(),
        cold_nodes.loc[jj, "lon"].to_numpy(),
        cold_nodes.loc[jj, "lat"].to_numpy(),
    )
    out = pd.DataFrame(
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
    out.to_parquet(output_path, index=False)
    print(f"Wrote dipole sensitivity network: {output_path}")


def run_networks(work_root: Path, n_months: int) -> None:
    suffix = f"local_warm{n_months}"
    hot = work_root / f"ERA5_hotnight_events_T_Tw_dyn_2000_2024_tau015_{suffix}.parquet"
    cold = work_root / f"ERA5_coldnight_events_T_Tw_dyn_2000_2024_tau015_{suffix}.parquet"
    write_hot_hot_network(
        hot,
        work_root / f"ECA_edges_dyn_T2_a0.005_K3_tau015_{suffix}.parquet",
        k_days=3,
        alpha=0.005,
        min_overlap_days=200,
    )
    write_dipole_network(
        hot,
        cold,
        work_root / f"ECA_edges_SEESAW_T2_a0.005_K3_tau015_{suffix}.parquet",
        k_days=3,
        alpha=0.005,
        min_overlap_days=200,
    )


def bh_keep_count(p_values: np.ndarray, total_tests: int, q: float) -> tuple[int, float]:
    sorted_p = np.sort(np.asarray(p_values, dtype=float))
    thresholds = (np.arange(1, len(sorted_p) + 1) / total_tests) * q
    keep = sorted_p <= thresholds
    n_keep = int(keep.sum())
    cutoff = float(sorted_p[n_keep - 1]) if n_keep else np.nan
    return n_keep, cutoff


def event_temperature_summary(path: Path, flag_col: str, label: str) -> dict[str, object]:
    frame = pd.read_parquet(path, columns=["Tmin_C", flag_col])
    values = frame.loc[frame[flag_col] == 1, "Tmin_C"].astype(float)
    return {
        "event_set": label,
        "event_days": int(len(values)),
        "median_tmin_c": float(values.median()),
        "pct_ge_20c": float((values >= 20).mean() * 100),
        "pct_ge_25c": float((values >= 25).mean() * 100),
    }


def valid_nodes(path: Path, event_col: str, min_events: int = 5) -> set[tuple[float, float]]:
    frame = pd.read_parquet(path, columns=["lon", "lat", event_col])
    counts = frame.groupby(["lon", "lat"])[event_col].sum()
    return set(counts[counts >= min_events].index)


def network_summary(path: Path, label: str, total_tests: int) -> dict[str, object]:
    edges = pd.read_parquet(path, columns=["dist_km", "p"])
    bh_n, bh_cutoff = bh_keep_count(edges["p"].to_numpy(), total_tests, q=0.05)
    kept = edges.loc[edges["p"] <= bh_cutoff] if bh_n else edges.iloc[0:0]
    return {
        "network_layer": label,
        "edges": int(len(edges)),
        "long_range_edges_gt2500km": int((edges["dist_km"] > 2500).sum()),
        "long_range_percent": float((edges["dist_km"] > 2500).mean() * 100),
        "median_distance_km": float(edges["dist_km"].median()),
        "bh_retained_edges_q05": bh_n,
        "bh_retained_long_range_edges_gt2500km": int((kept["dist_km"] > 2500).sum()) if bh_n else 0,
    }


def summarize(work_root: Path, derived_root: Path, n_months_values: list[int]) -> None:
    rows = []
    temp_rows = []
    for n_months in n_months_values:
        if n_months == 3:
            input_root = WARM_SEASON_INPUT_ROOT
        else:
            input_root = work_root
        suffix = f"local_warm{n_months}"
        hot = input_root / f"ERA5_hotnight_events_T_Tw_dyn_2000_2024_tau015_{suffix}.parquet"
        cold = input_root / f"ERA5_coldnight_events_T_Tw_dyn_2000_2024_tau015_{suffix}.parquet"
        coherent = input_root / f"ECA_edges_dyn_T2_a0.005_K3_tau015_{suffix}.parquet"
        dipole = input_root / f"ECA_edges_SEESAW_T2_a0.005_K3_tau015_{suffix}.parquet"
        for item in [
            event_temperature_summary(hot, "evt_T_run2", "hot-night event days"),
            event_temperature_summary(cold, "evt_cold_T_run2", "cool-anomaly event days"),
        ]:
            item["warm_season_definition"] = f"local-warm{n_months}"
            temp_rows.append(item)
        hot_nodes = valid_nodes(hot, "evt_T_run2")
        cold_nodes = valid_nodes(cold, "evt_cold_T_run2")
        coherent_tests = len(hot_nodes) * (len(hot_nodes) - 1) // 2
        dipole_tests = len(hot_nodes) * len(cold_nodes) - len(hot_nodes & cold_nodes)
        for item in [
            network_summary(coherent, "Coherent", coherent_tests),
            network_summary(dipole, "Dipole", dipole_tests),
        ]:
            item["warm_season_definition"] = f"local-warm{n_months}"
            rows.append(item)

    derived_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(derived_root / "warm_season_definition_network_sensitivity.csv", index=False)
    pd.DataFrame(temp_rows).to_csv(derived_root / "warm_season_definition_event_temperature_sensitivity.csv", index=False)
    print(f"Wrote sensitivity summaries to {derived_root}")


def run_contiguity_audit(derived_root: Path) -> None:
    path = WARM_SEASON_INPUT_ROOT / "ERA5_hotnight_events_T_Tw_dyn_2000_2024_tau015_local_warm3.parquet"
    frame = pd.read_parquet(path, columns=["lon", "lat", "lt_date", "evt_T", "evt_T_run2"])
    frame["lt_date"] = pd.to_datetime(frame["lt_date"])
    frame = frame.sort_values(["lon", "lat", "lt_date"]).copy()

    def run2_gapaware(group: pd.DataFrame) -> pd.Series:
        out = []
        run = 0
        prev = None
        for date, event in zip(group["lt_date"], group["evt_T"]):
            if event == 1 and prev is not None and (date - prev).days == 1:
                run += 1
            elif event == 1:
                run = 1
            else:
                run = 0
            out.append(1 if run >= 2 else 0)
            prev = date
        return pd.Series(out, index=group.index, dtype="int8")

    frame["evt_T_run2_gapaware"] = frame.groupby(["lon", "lat"], group_keys=False)[["lt_date", "evt_T"]].apply(
        run2_gapaware
    )
    changed = frame["evt_T_run2"] != frame["evt_T_run2_gapaware"]
    audit = pd.DataFrame(
        [
            {
                "warm_season_definition": "local-warm3",
                "rows": int(len(frame)),
                "original_run2_event_days": int(frame["evt_T_run2"].sum()),
                "calendar_gapaware_run2_event_days": int(frame["evt_T_run2_gapaware"].sum()),
                "changed_event_day_flags": int(changed.sum()),
                "changed_flag_percent_of_rows": float(changed.mean() * 100),
                "changed_flag_percent_of_original_run2_event_days": float(changed.sum() / max(frame["evt_T_run2"].sum(), 1) * 100),
            }
        ]
    )
    derived_root.mkdir(parents=True, exist_ok=True)
    audit.to_csv(derived_root / "warm_season_run_contiguity_audit.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local warm-season definition sensitivity checks.")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--n-months", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--skip-networks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_hot = SOURCE_EVENT_ROOT / "ERA5_hotnight_events_T_Tw_dyn_2000_2024.parquet"
    source_cold = SOURCE_EVENT_ROOT / "ERA5_coldnight_events_T_Tw_dyn_2000_2024.parquet"
    args.work_root.mkdir(parents=True, exist_ok=True)
    for n_months in args.n_months:
        suffix = f"local_warm{n_months}"
        build_event_table(
            source_hot,
            args.work_root / f"ERA5_hotnight_events_T_Tw_dyn_2000_2024_tau015_{suffix}.parquet",
            n_months,
            "hot",
            min_run_length=2,
        )
        build_event_table(
            source_cold,
            args.work_root / f"ERA5_coldnight_events_T_Tw_dyn_2000_2024_tau015_{suffix}.parquet",
            n_months,
            "cold",
            min_run_length=2,
        )
        if not args.skip_networks:
            run_networks(args.work_root, n_months)
    summarize(args.work_root, args.derived_root, [3, *args.n_months])
    run_contiguity_audit(args.derived_root)


if __name__ == "__main__":
    main()
