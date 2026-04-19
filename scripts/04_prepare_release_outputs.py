from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from urban_hot_night_sync.paths import ANALYSIS_ROOT, BUNDLE_DATES_ROOT


HOTHOT_RELEASE_FILES = [
    "dates_hothot_bundle_1_E.Asia_and_E.Asia.csv",
    "dates_hothot_bundle_2_(-85,19)_and_E.Asia.csv",
    "dates_hothot_bundle_3_(10,20)_and_(104,8).csv",
]
SEESAW_RELEASE_FILES = [
    "dates_bundle_1_NorthAmerica_to_NorthAmerica.csv",
    "dates_bundle_2_Europe_to_Europe.csv",
    "dates_bundle_3_(52,31)_to_EastAsia.csv",
]


def hothot_region_name(lon: float, lat: float) -> str:
    if lat > 20 and -130 < lon < -60:
        return "N.America"
    if lat > 20 and 60 < lon < 150:
        return "E.Asia"
    if lat > 30 and -10 < lon < 40:
        return "Europe"
    return f"({lon:.0f},{lat:.0f})"


def seesaw_region_name(lon: float, lat: float) -> str:
    if lat > 20 and -130 < lon < -60:
        return "NorthAmerica"
    if lat > 20 and 60 < lon < 150:
        return "EastAsia"
    if lat > 30 and -10 < lon < 40:
        return "Europe"
    return f"({lon:.0f},{lat:.0f})"


def load_hot_event_days(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["lon", "lat", "lt_date", "evt_T_run2"])
    frame = frame.loc[frame["evt_T_run2"] == 1, ["lon", "lat", "lt_date"]].copy()
    frame["lt_date"] = pd.to_datetime(frame["lt_date"])
    return frame


def load_cold_event_days(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["lon", "lat", "lt_date", "evt_cold_T_run2"])
    frame = frame.loc[frame["evt_cold_T_run2"] == 1, ["lon", "lat", "lt_date"]].copy()
    frame["lt_date"] = pd.to_datetime(frame["lt_date"])
    return frame


def bundle_signal(events: pd.DataFrame, nodes: pd.DataFrame) -> pd.Series:
    if nodes.empty:
        return pd.Series(dtype=float)
    active = events.merge(nodes, on=["lon", "lat"])
    if active.empty:
        return pd.Series(dtype=float)
    return active.groupby("lt_date").size() / len(nodes)


def write_hothot_bundle_dates(
    edges_path: Path,
    events_path: Path,
    output_dir: Path,
    top_n: int,
    min_dist_km: float,
    region_threshold: float,
) -> None:
    from sklearn.cluster import KMeans

    edges = pd.read_parquet(edges_path)
    edges = edges.loc[edges["dist_km"] > min_dist_km].copy()
    if len(edges) < top_n:
        raise ValueError("Not enough long-range Hot-Hot edges to extract the requested number of bundles.")

    labels = KMeans(n_clusters=top_n, random_state=42, n_init=10).fit_predict(edges[["lon1", "lat1", "lon2", "lat2"]])
    edges["cluster"] = labels
    events = load_hot_event_days(events_path)

    for rank, cluster_id in enumerate(edges["cluster"].value_counts().index[:top_n], start=1):
        subset = edges.loc[edges["cluster"] == cluster_id]
        nodes_a = subset[["lon1", "lat1"]].rename(columns={"lon1": "lon", "lat1": "lat"}).drop_duplicates()
        nodes_b = subset[["lon2", "lat2"]].rename(columns={"lon2": "lon", "lat2": "lat"}).drop_duplicates()

        signal_a = bundle_signal(events, nodes_a)
        signal_b = bundle_signal(events, nodes_b)
        common_dates = signal_a.index.intersection(signal_b.index)
        synced = common_dates[(signal_a.loc[common_dates] > region_threshold) & (signal_b.loc[common_dates] > region_threshold)]

        name = f"{hothot_region_name(subset['lon1'].mean(), subset['lat1'].mean())}_and_{hothot_region_name(subset['lon2'].mean(), subset['lat2'].mean())}"
        filename = f"dates_hothot_bundle_{rank}_{name}.csv"
        pd.DataFrame({"date": synced.strftime("%Y-%m-%d")}).to_csv(output_dir / filename, index=False)


def write_seesaw_bundle_dates(
    edges_path: Path,
    hot_events_path: Path,
    cold_events_path: Path,
    output_dir: Path,
    top_n: int,
    region_threshold: float,
) -> None:
    from sklearn.cluster import KMeans

    edges = pd.read_parquet(edges_path)
    if len(edges) < top_n:
        raise ValueError("Not enough seesaw edges to extract the requested number of bundles.")

    labels = KMeans(n_clusters=top_n, random_state=42, n_init=10).fit_predict(edges[["hot_lon", "hot_lat", "cold_lon", "cold_lat"]])
    edges["cluster"] = labels
    hot_events = load_hot_event_days(hot_events_path)
    cold_events = load_cold_event_days(cold_events_path)

    for rank, cluster_id in enumerate(edges["cluster"].value_counts().index[:top_n], start=1):
        subset = edges.loc[edges["cluster"] == cluster_id]
        hot_nodes = subset[["hot_lon", "hot_lat"]].rename(columns={"hot_lon": "lon", "hot_lat": "lat"}).drop_duplicates()
        cold_nodes = subset[["cold_lon", "cold_lat"]].rename(columns={"cold_lon": "lon", "cold_lat": "lat"}).drop_duplicates()

        signal_hot = bundle_signal(hot_events, hot_nodes)
        signal_cold = bundle_signal(cold_events, cold_nodes)
        common_dates = signal_hot.index.intersection(signal_cold.index)
        synced = common_dates[(signal_hot.loc[common_dates] > region_threshold) & (signal_cold.loc[common_dates] > region_threshold)]

        name = f"{seesaw_region_name(subset['hot_lon'].mean(), subset['hot_lat'].mean())}_to_{seesaw_region_name(subset['cold_lon'].mean(), subset['cold_lat'].mean())}"
        filename = f"dates_bundle_{rank}_{name}.csv"
        pd.DataFrame({"date": synced.strftime("%Y-%m-%d")}).to_csv(output_dir / filename, index=False)


def yearly_counts(path: Path, years: list[int]) -> pd.Series:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    counts = frame.groupby(frame["date"].dt.year).size()
    return counts.reindex(years, fill_value=0)


def build_trend_table(output_path: Path, years: list[int]) -> None:
    rows = []
    specs = [
        ("Hot-Hot (top 3 bundles)", HOTHOT_RELEASE_FILES),
        ("Seesaw (top 3 bundles)", SEESAW_RELEASE_FILES),
    ]

    for mechanism, names in specs:
        annual = pd.Series(0, index=years, dtype=float)
        for name in names:
            annual = annual.add(yearly_counts(BUNDLE_DATES_ROOT / name, years), fill_value=0)

        x = np.asarray(years, dtype=float)
        y = annual.to_numpy(dtype=float)
        slope, _, slope_lo, slope_hi = theilslopes(y, x, 0.95)
        tau, p_value = kendalltau(x, y)

        rows.append(
            {
                "Mechanism": mechanism,
                "Years": f"{years[0]}-{years[-1]}",
                "Bundle files": "; ".join(names),
                "Mean annual bundle-days": float(np.mean(y)),
                "Theil-Sen slope": float(slope),
                "Theil-Sen 95% low": float(slope_lo),
                "Theil-Sen 95% high": float(slope_hi),
                "Kendall tau": float(tau),
                "Kendall p": float(p_value),
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote trend table: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the lightweight bundle-date and trend-summary files kept on GitHub.")
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--region-threshold", type=float, default=0.05)
    parser.add_argument("--min-hot-hot-distance-km", type=float, default=2500.0)
    parser.add_argument("--trend-only", action="store_true", help="Skip bundle extraction and only rebuild the trend summary table.")
    parser.add_argument("--input-root", type=Path, default=ANALYSIS_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=BUNDLE_DATES_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=ANALYSIS_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = f"tau{int(round(args.tau * 100)):03d}"
    args.bundle_root.mkdir(parents=True, exist_ok=True)
    args.analysis_root.mkdir(parents=True, exist_ok=True)

    if not args.trend_only:
        hot_edges = args.input_root / f"ECA_edges_dyn_T2_a0.005_K3_{tag}.parquet"
        seesaw_edges = args.input_root / f"ECA_edges_SEESAW_T2_a0.005_K3_{tag}.parquet"
        hot_events = args.input_root / f"ERA5_hotnight_events_T_Tw_dyn_2000_2024_{tag}.parquet"
        cold_events = args.input_root / f"ERA5_coldnight_events_T_Tw_dyn_2000_2024_{tag}.parquet"

        if hot_edges.exists() and hot_events.exists():
            write_hothot_bundle_dates(
                edges_path=hot_edges,
                events_path=hot_events,
                output_dir=args.bundle_root,
                top_n=args.top_n,
                min_dist_km=args.min_hot_hot_distance_km,
                region_threshold=args.region_threshold,
            )
        else:
            print("Skipping Hot-Hot bundle extraction because the required large files are not present locally.")

        if seesaw_edges.exists() and hot_events.exists() and cold_events.exists():
            write_seesaw_bundle_dates(
                edges_path=seesaw_edges,
                hot_events_path=hot_events,
                cold_events_path=cold_events,
                output_dir=args.bundle_root,
                top_n=args.top_n,
                region_threshold=args.region_threshold,
            )
        else:
            print("Skipping seesaw bundle extraction because the required large files are not present locally.")

    build_trend_table(args.analysis_root / "Table5_Trends_Updated.csv", years=list(range(2000, 2025)))


if __name__ == "__main__":
    main()
