from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from urban_hot_night_sync.paths import WARM_SEASON_INPUT_ROOT

REPO_DERIVED = ROOT / "data/derived/warm_season_local_warm3"
INPUT = Path(os.environ.get("WARM_SEASON_INPUT_ROOT", WARM_SEASON_INPUT_ROOT))
OUTPUT = REPO_DERIVED / "population_exposure"
FIG_DIR = ROOT / "figures/warm_season_exposure"

COHERENT_EDGES = INPUT / "ECA_edges_dyn_T2_a0.005_K3_tau015_local_warm3.parquet"
DIPOLE_EDGES = INPUT / "ECA_edges_SEESAW_T2_a0.005_K3_tau015_local_warm3.parquet"
BUNDLE_SUMMARY = REPO_DERIVED / "warm_season_bundle_summary.csv"
DATE_DIR = REPO_DERIVED / "bundle_dates"

EE_PROJECT = os.environ.get("EE_PROJECT", "ee-zuruyuyu")
WORLDPOP_COLLECTION = "WorldPop/GP/100m/pop"
PRIMARY_POP_YEAR = 2020
GRID_HALF_DEG = 0.125
EE_SCALE_M = 1000
EE_TILE_SCALE = 4

COLORS = {"Coherent": "#D62728", "Dipole": "#1F77B4"}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )


def cluster_edges() -> tuple[pd.DataFrame, pd.DataFrame]:
    coherent = pd.read_parquet(COHERENT_EDGES)
    coherent = coherent.loc[coherent["dist_km"] > 2500].copy()
    coherent["mechanism"] = "Coherent"
    coherent["cluster"] = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(
        coherent[["lon1", "lat1", "lon2", "lat2"]]
    )

    dipole = pd.read_parquet(DIPOLE_EDGES)
    dipole = dipole.loc[dipole["dist_km"] > 2500].copy()
    dipole["mechanism"] = "Dipole"
    dipole["cluster"] = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(
        dipole[["hot_lon", "hot_lat", "cold_lon", "cold_lat"]]
    )
    return coherent, dipole


def bundle_label(row: pd.Series) -> str:
    if row["mechanism"] == "Coherent" and row["rank"] == 1:
        return "C1 North America-Europe"
    if row["mechanism"] == "Coherent" and row["rank"] == 2:
        return "C2 North America-East Asia"
    if row["mechanism"] == "Coherent" and row["rank"] == 3:
        return "C3 Europe-East Asia"
    if row["mechanism"] == "Dipole" and row["rank"] == 1:
        return "D1 East Asia-North America"
    if row["mechanism"] == "Dipole" and row["rank"] == 2:
        return "D2 West/Central Asia-East Asia"
    if row["mechanism"] == "Dipole" and row["rank"] == 3:
        return "D3 North Atlantic-North America"
    return f"{row['mechanism']} {int(row['rank'])}"


def extract_bundle_nodes(coherent: pd.DataFrame, dipole: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        mechanism = row["mechanism"]
        rank = int(row["rank"])
        cluster_id = int(row["cluster_id"])
        source = coherent if mechanism == "Coherent" else dipole
        subset = source.loc[source["cluster"] == cluster_id].copy()
        label = bundle_label(row)

        if mechanism == "Coherent":
            side_specs = [
                ("side_a", "Warm side A", "lon1", "lat1"),
                ("side_b", "Warm side B", "lon2", "lat2"),
            ]
        else:
            side_specs = [
                ("hot_side", "Hot side", "hot_lon", "hot_lat"),
                ("cool_side", "Cool side", "cold_lon", "cold_lat"),
            ]

        for side_code, side_label, lon_col, lat_col in side_specs:
            nodes = (
                subset[[lon_col, lat_col]]
                .rename(columns={lon_col: "lon", lat_col: "lat"})
                .drop_duplicates()
                .sort_values(["lat", "lon"])
            )
            for node in nodes.itertuples(index=False):
                rows.append(
                    {
                        "mechanism": mechanism,
                        "rank": rank,
                        "bundle": label,
                        "cluster_id": cluster_id,
                        "side": side_code,
                        "side_label": side_label,
                        "lon": round(float(node.lon), 3),
                        "lat": round(float(node.lat), 3),
                    }
                )

        side_counts = (
            pd.DataFrame(rows)
            .loc[lambda f: (f["mechanism"] == mechanism) & (f["rank"] == rank)]
            .groupby("side")
            .size()
            .to_dict()
        )
        validation_rows.append(
            {
                "mechanism": mechanism,
                "rank": rank,
                "bundle": label,
                "cluster_id": cluster_id,
                "edge_count_recomputed": int(len(subset)),
                "edge_count_expected": int(row["edge_count"]),
                "node_count_a_recomputed": int(side_counts.get("side_a", side_counts.get("hot_side", 0))),
                "node_count_a_expected": int(row["node_count_a"]),
                "node_count_b_recomputed": int(side_counts.get("side_b", side_counts.get("cool_side", 0))),
                "node_count_b_expected": int(row["node_count_b"]),
            }
        )

    nodes = pd.DataFrame(rows)
    validation = pd.DataFrame(validation_rows)
    validation["edge_count_match"] = validation["edge_count_recomputed"] == validation["edge_count_expected"]
    validation["node_count_a_match"] = validation["node_count_a_recomputed"] == validation["node_count_a_expected"]
    validation["node_count_b_match"] = validation["node_count_b_recomputed"] == validation["node_count_b_expected"]
    return nodes, validation


def init_earth_engine():
    try:
        import ee
    except ModuleNotFoundError as exc:
        raise RuntimeError("earthengine-api is required. Install with: python3 -m pip install --user earthengine-api") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ee.Initialize(project=EE_PROJECT)
    return ee


def available_worldpop_years(ee) -> list[int]:
    years = ee.ImageCollection(WORLDPOP_COLLECTION).aggregate_array("year").distinct().sort().getInfo()
    return sorted(int(year) for year in years if year is not None)


def nearest_year(year: int, years: list[int]) -> int:
    if year <= years[0]:
        return years[0]
    if year >= years[-1]:
        return years[-1]
    return min(years, key=lambda value: abs(value - year))


def population_image(ee, year: int):
    collection = ee.ImageCollection(WORLDPOP_COLLECTION).filter(ee.Filter.eq("year", int(year)))
    first = ee.Image(collection.first())
    native_projection = first.projection()
    return (
        ee.Image(collection.mosaic())
        .setDefaultProjection(native_projection)
        .reduceResolution(reducer=ee.Reducer.sum(), maxPixels=8192)
        .reproject(crs=native_projection, scale=EE_SCALE_M)
    )


def cell_polygon_coords(lon: float, lat: float) -> list[list[list[float]]]:
    west = max(-180.0, lon - GRID_HALF_DEG)
    east = min(180.0, lon + GRID_HALF_DEG)
    south = max(-90.0, lat - GRID_HALF_DEG)
    north = min(90.0, lat + GRID_HALF_DEG)
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return [ring]


def footprint_geometry(ee, nodes: pd.DataFrame):
    unique_nodes = nodes[["lon", "lat"]].drop_duplicates().sort_values(["lat", "lon"])
    polygons = [cell_polygon_coords(float(row.lon), float(row.lat)) for row in unique_nodes.itertuples(index=False)]
    return ee.Geometry.MultiPolygon(polygons, proj="EPSG:4326", geodesic=False)


def build_footprints(nodes: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, bundle_row in summary.iterrows():
        mechanism = bundle_row["mechanism"]
        rank = int(bundle_row["rank"])
        label = bundle_label(bundle_row)
        subset = nodes.loc[(nodes["mechanism"] == mechanism) & (nodes["rank"] == rank)].copy()
        unique_bundle = subset.drop_duplicates(["lon", "lat"])
        rows.append(
            {
                "mechanism": mechanism,
                "rank": rank,
                "bundle": label,
                "footprint": "bundle_union",
                "footprint_label": "Union of both paired sides",
                "node_count": int(len(unique_bundle)),
                "side_sum_node_count": int(len(subset.drop_duplicates(["side", "lon", "lat"]))),
            }
        )
        for side, side_df in subset.groupby("side", sort=False):
            unique_side = side_df.drop_duplicates(["lon", "lat"])
            rows.append(
                {
                    "mechanism": mechanism,
                    "rank": rank,
                    "bundle": label,
                    "footprint": side,
                    "footprint_label": unique_side["side_label"].iloc[0],
                    "node_count": int(len(unique_side)),
                    "side_sum_node_count": int(len(unique_side)),
                }
            )
    return pd.DataFrame(rows)


def nodes_for_footprint(nodes: pd.DataFrame, footprint: pd.Series) -> pd.DataFrame:
    subset = nodes.loc[(nodes["mechanism"] == footprint["mechanism"]) & (nodes["rank"] == footprint["rank"])].copy()
    if footprint["footprint"] != "bundle_union":
        subset = subset.loc[subset["side"] == footprint["footprint"]].copy()
    return subset.drop_duplicates(["lon", "lat"])


def query_footprint_population(nodes: pd.DataFrame, footprints: pd.DataFrame) -> pd.DataFrame:
    cache = OUTPUT / f"worldpop_footprint_population_{PRIMARY_POP_YEAR}_{EE_SCALE_M}m.csv"
    if cache.exists():
        return pd.read_csv(cache)

    ee = init_earth_engine()
    years = available_worldpop_years(ee)
    if PRIMARY_POP_YEAR not in years:
        raise RuntimeError(f"WorldPop {PRIMARY_POP_YEAR} is unavailable; available years are {years[0]}-{years[-1]}.")
    img = population_image(ee, PRIMARY_POP_YEAR)
    all_rows = []
    for fp in footprints.itertuples(index=False):
        fp_series = pd.Series(fp._asdict())
        fp_nodes = nodes_for_footprint(nodes, fp_series)
        geom = footprint_geometry(ee, fp_nodes)
        print(
            f"Querying WorldPop {PRIMARY_POP_YEAR}: {fp.bundle} / {fp.footprint_label} "
            f"({len(fp_nodes)} grid cells)...",
            flush=True,
        )
        result = None
        for attempt in range(1, 6):
            try:
                result = img.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=geom,
                    scale=EE_SCALE_M,
                    tileScale=EE_TILE_SCALE,
                    maxPixels=1_000_000_000_000,
                ).getInfo()
                break
            except Exception as exc:
                if attempt == 5:
                    raise
                sleep_s = 1.5 * attempt
                print(f"  retry {attempt} after {type(exc).__name__}: {exc}", flush=True)
                time.sleep(sleep_s)
        population = float(next(iter(result.values()))) if result else 0.0
        all_rows.append(
                {
                    "mechanism": fp.mechanism,
                    "rank": int(fp.rank),
                    "bundle": fp.bundle,
                    "footprint": fp.footprint,
                    "footprint_label": fp.footprint_label,
                    "node_count": int(fp.node_count),
                    "side_sum_node_count": int(fp.side_sum_node_count),
                    "population_year": PRIMARY_POP_YEAR,
                    "population": population,
                }
        )
        time.sleep(0.5)

    frame = pd.DataFrame(all_rows)
    frame.to_csv(cache, index=False)
    return frame


def event_year_counts(row: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(pd.read_csv(DATE_DIR / row["date_file"])["date"], errors="coerce").dropna()
    counts = dates.dt.year.value_counts().sort_index()
    return pd.DataFrame({"event_year": counts.index.astype(int), "event_nights": counts.values.astype(int)})


def summarize_exposure(footprint_pop: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    side_rows: list[dict[str, object]] = []
    bundle_rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []

    for _, bundle_row in summary.iterrows():
        mechanism = bundle_row["mechanism"]
        rank = int(bundle_row["rank"])
        label = bundle_label(bundle_row)
        dates_by_year = event_year_counts(bundle_row)
        total_nights = int(dates_by_year["event_nights"].sum())

        sub_pop = footprint_pop.loc[(footprint_pop["mechanism"] == mechanism) & (footprint_pop["rank"] == rank)].copy()
        for _, side in sub_pop.loc[sub_pop["footprint"] != "bundle_union"].iterrows():
            side_pop = float(side["population"])
            side_rows.append(
                {
                    "mechanism": mechanism,
                    "rank": rank,
                    "bundle": label,
                    "side": side["footprint"],
                    "side_label": side["footprint_label"],
                    "node_count": int(side["node_count"]),
                    "population_2020": side_pop,
                    "event_nights": total_nights,
                    "static_2020_person_nights": side_pop * total_nights,
                }
            )

        bundle_pop = sub_pop.loc[sub_pop["footprint"] == "bundle_union"].iloc[0]
        bundle_pop_2020 = float(bundle_pop["population"])
        static_2020 = bundle_pop_2020 * total_nights
        for item in dates_by_year.itertuples(index=False):
            exposure = bundle_pop_2020 * int(item.event_nights)
            annual_rows.append(
                {
                    "mechanism": mechanism,
                    "rank": rank,
                    "bundle": label,
                    "event_year": int(item.event_year),
                    "worldpop_year": PRIMARY_POP_YEAR,
                    "event_nights": int(item.event_nights),
                    "population_for_year": bundle_pop_2020,
                    "person_nights": exposure,
                }
            )

        bundle_rows.append(
            {
                "mechanism": mechanism,
                "rank": rank,
                "bundle": label,
                "node_count_unique": int(bundle_pop["node_count"]),
                "node_count_side_sum": int(bundle_pop["side_sum_node_count"]),
                "population_2020": bundle_pop_2020,
                "event_nights": total_nights,
                "static_2020_person_nights": static_2020,
                "timevarying_person_nights": static_2020,
                "timevarying_minus_static_pct": 0.0,
            }
        )

    side_summary = pd.DataFrame(side_rows)
    bundle_summary = pd.DataFrame(bundle_rows)
    annual = pd.DataFrame(annual_rows)
    return side_summary, bundle_summary, annual


def make_exposure_figure(bundle_summary: pd.DataFrame, side_summary: pd.DataFrame) -> tuple[Path, Path]:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bundle_summary = bundle_summary.copy()
    side_summary = side_summary.copy()
    bundle_summary["label"] = bundle_summary.apply(lambda row: f"{row['mechanism'][0]}{int(row['rank'])}", axis=1)
    side_summary["label"] = side_summary["bundle"].str.replace(" ", "\n", n=1)

    fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.05]})
    order = bundle_summary.sort_values(["mechanism", "rank"]).index
    ordered = bundle_summary.loc[order].reset_index(drop=True)
    colors = ordered["mechanism"].map(COLORS).tolist()
    x = np.arange(len(ordered))
    axes[0].bar(x, ordered["population_2020"] / 1e6, color=colors, alpha=0.86)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ordered["label"])
    axes[0].set_ylabel("Population in bundle footprint (millions)")
    axes[0].set_title("a  Static 2020 footprint", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="0.88", lw=0.6)
    for idx, value in enumerate(ordered["population_2020"] / 1e6):
        axes[0].text(idx, value * 1.02, f"{value:.1f}", ha="center", va="bottom", fontsize=6.6)

    axes[1].bar(x, ordered["timevarying_person_nights"] / 1e9, color=colors, alpha=0.86)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ordered["label"])
    axes[1].set_ylabel("Exposure (billion person-nights)")
    axes[1].set_title("b  Static 2020 event exposure", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="0.88", lw=0.6)
    for idx, value in enumerate(ordered["timevarying_person_nights"] / 1e9):
        axes[1].text(idx, value * 1.02, f"{value:.2f}", ha="center", va="bottom", fontsize=6.6)
    handles = [
        mpl.patches.Patch(facecolor=COLORS["Coherent"], label="Coherent"),
        mpl.patches.Patch(facecolor=COLORS["Dipole"], label="Dipole"),
    ]
    axes[1].legend(handles=handles, frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.14, wspace=0.34)
    main_path = FIG_DIR / "fig5c_warm_season_population_exposure.png"
    fig.set_size_inches(2200 / 300, 1100 / 300)
    fig.savefig(main_path, dpi=300, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots()
    pivot = side_summary.copy()
    pivot["side_display"] = np.where(
        pivot["mechanism"].eq("Coherent"),
        pivot["side"].map({"side_a": "Warm side A", "side_b": "Warm side B"}),
        pivot["side"].map({"hot_side": "Hot side", "cool_side": "Cool side"}),
    )
    pivot["side_label_short"] = pivot["bundle"] + "\n" + pivot["side_display"]
    y = np.arange(len(pivot))[::-1]
    colors = pivot["mechanism"].map(COLORS).tolist()
    ax.barh(y, pivot["population_2020"] / 1e6, color=colors, alpha=0.86)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot["side_label_short"], fontsize=6.5)
    ax.set_xlabel("Side-specific population (millions, 2020)")
    ax.set_title("Bundle-side population footprints", loc="left", fontweight="bold", fontsize=8)
    ax.grid(axis="x", color="0.88", lw=0.6)
    for yi, value in zip(y, pivot["population_2020"] / 1e6):
        ax.text(value + max(pivot["population_2020"] / 1e6) * 0.012, yi, f"{value:.1f}", va="center", fontsize=6.2)
    fig.subplots_adjust(left=0.39, right=0.96, top=0.88, bottom=0.14)
    extended_path = FIG_DIR / "si_fig_s8_population_exposure_footprints.png"
    fig.set_size_inches(1814 / 300, 1400 / 300)
    fig.savefig(extended_path, dpi=300, facecolor="white")
    plt.close(fig)
    return main_path, extended_path


def write_outputs(nodes: pd.DataFrame, validation: pd.DataFrame, footprints: pd.DataFrame, footprint_pop: pd.DataFrame, side_summary: pd.DataFrame, bundle_summary: pd.DataFrame, annual: pd.DataFrame) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(OUTPUT / "warm_season_bundle_node_footprints.csv", index=False)
    validation.to_csv(OUTPUT / "warm_season_bundle_reconstruction_validation.csv", index=False)
    footprints.to_csv(OUTPUT / "warm_season_bundle_footprints.csv", index=False)
    footprint_pop.to_csv(OUTPUT / f"worldpop_footprint_population_{PRIMARY_POP_YEAR}_{EE_SCALE_M}m.csv", index=False)
    side_summary.to_csv(OUTPUT / "warm_season_population_exposure_by_bundle_side.csv", index=False)
    bundle_summary.to_csv(OUTPUT / "warm_season_population_exposure_by_bundle.csv", index=False)
    annual.to_csv(OUTPUT / "warm_season_population_exposure_by_bundle_year.csv", index=False)
    metadata = {
        "bundle_source": str(BUNDLE_SUMMARY),
        "coherent_edges": str(COHERENT_EDGES),
        "dipole_edges": str(DIPOLE_EDGES),
        "worldpop_collection": WORLDPOP_COLLECTION,
        "primary_population_year": PRIMARY_POP_YEAR,
        "grid_cell_half_width_degrees": GRID_HALF_DEG,
        "earth_engine_project": EE_PROJECT,
        "earth_engine_scale_m": EE_SCALE_M,
        "notes": (
            "Population is summed inside exact 0.25 degree grid-cell footprints for unique nodes in each corrected warm-season bundle. "
            "Exposure is reported as static 2020 WorldPop population multiplied by corrected warm-season event nights."
        ),
    }
    (OUTPUT / "warm_season_population_exposure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(BUNDLE_SUMMARY)
    coherent, dipole = cluster_edges()
    nodes, validation = extract_bundle_nodes(coherent, dipole, summary)
    if not validation[["edge_count_match", "node_count_a_match", "node_count_b_match"]].all().all():
        print(validation.to_string(index=False))
        raise RuntimeError("Bundle reconstruction failed validation.")

    footprints = build_footprints(nodes, summary)
    footprint_pop = query_footprint_population(nodes, footprints)
    side_summary, bundle_exposure, annual = summarize_exposure(footprint_pop, summary)
    write_outputs(nodes, validation, footprints, footprint_pop, side_summary, bundle_exposure, annual)
    main_fig, extended_fig = make_exposure_figure(bundle_exposure, side_summary)

    display = bundle_exposure.copy()
    display["population_2020_millions"] = display["population_2020"] / 1e6
    display["timevarying_billion_person_nights"] = display["timevarying_person_nights"] / 1e9
    cols = [
        "mechanism",
        "rank",
        "bundle",
        "node_count_unique",
        "population_2020_millions",
        "event_nights",
        "timevarying_billion_person_nights",
        "timevarying_minus_static_pct",
    ]
    print(display[cols].to_string(index=False, formatters={
        "population_2020_millions": "{:.2f}".format,
        "timevarying_billion_person_nights": "{:.2f}".format,
        "timevarying_minus_static_pct": "{:.2f}".format,
    }))
    print(f"Wrote exposure outputs to {OUTPUT}")
    print(f"Figures: {main_fig}; {extended_fig}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
