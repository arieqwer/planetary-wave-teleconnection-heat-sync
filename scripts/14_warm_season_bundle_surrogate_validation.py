from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from urban_hot_night_sync.paths import WARM_SEASON_INPUT_ROOT

DEFAULT_INPUT_ROOT = WARM_SEASON_INPUT_ROOT
DEFAULT_DERIVED = REPO_ROOT / "data/derived/warm_season_local_warm3"
OUTPUT_DIR = REPO_ROOT / "data/derived/warm_season_surrogate_validation"
YEARS = np.arange(2000, 2025)
DATE_INDEX = pd.date_range("2000-01-01", "2024-12-31", freq="D")
DATE_TO_POS = pd.Series(np.arange(len(DATE_INDEX), dtype=np.int32), index=DATE_INDEX)


@dataclass(frozen=True)
class BundleSide:
    mechanism: str
    rank: int
    side: str
    event_kind: str
    nodes: tuple[tuple[float, float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dependence-preserving surrogate validation for warm-season top-bundle synchronized dates. "
            "Each node-year event sequence is circularly shifted independently, preserving local annual event "
            "counts and run structure while destroying cross-node phase alignment."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-surrogates", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def normalize_node_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["lon"] = out["lon"].round(3).astype(float)
    out["lat"] = out["lat"].round(3).astype(float)
    return out


def load_event_table(path: Path, column: str, nodes: set[tuple[float, float]]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["lon", "lat", "lt_date", column])
    frame = normalize_node_frame(frame)
    node_index = pd.MultiIndex.from_tuples(nodes, names=["lon", "lat"])
    frame = frame.set_index(["lon", "lat"]).loc[lambda x: x.index.isin(node_index)].reset_index()
    frame["lt_date"] = pd.to_datetime(frame["lt_date"])
    frame["year"] = frame["lt_date"].dt.year.astype(np.int16)
    frame["date_pos"] = DATE_TO_POS.loc[frame["lt_date"]].to_numpy(dtype=np.int32)
    frame["evt"] = frame[column].astype(np.int8)
    return frame[["lon", "lat", "lt_date", "year", "date_pos", "evt"]]


def load_bundle_sides(footprints_path: Path) -> list[BundleSide]:
    footprints = normalize_node_frame(pd.read_csv(footprints_path))
    sides: list[BundleSide] = []
    for (mechanism, rank, side), group in footprints.groupby(["mechanism", "rank", "side"], sort=True):
        event_kind = "hot"
        if mechanism == "Dipole" and side == "cool_side":
            event_kind = "cold"
        nodes = tuple(sorted({(float(row.lon), float(row.lat)) for row in group.itertuples(index=False)}))
        sides.append(BundleSide(str(mechanism), int(rank), str(side), event_kind, nodes))
    return sides


def side_observed_counts(side: BundleSide, event_table: pd.DataFrame) -> np.ndarray:
    nodes = pd.MultiIndex.from_tuples(side.nodes, names=["lon", "lat"])
    frame = event_table.set_index(["lon", "lat"]).loc[lambda x: x.index.isin(nodes)]
    counts = np.zeros(len(DATE_INDEX), dtype=np.int16)
    event_positions = frame.loc[frame["evt"] == 1, "date_pos"].to_numpy(dtype=np.int32)
    if len(event_positions):
        np.add.at(counts, event_positions, 1)
    return counts


def node_year_arrays(event_table: pd.DataFrame, nodes: tuple[tuple[float, float], ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    node_set = set(nodes)
    frame = event_table.loc[[node in node_set for node in zip(event_table["lon"], event_table["lat"])]].copy()
    arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for _, group in frame.groupby(["lon", "lat", "year"], sort=False):
        ordered = group.sort_values("lt_date")
        positions = ordered["date_pos"].to_numpy(dtype=np.int32)
        flags = ordered["evt"].to_numpy(dtype=np.int8)
        if len(positions) and flags.sum() > 0:
            arrays.append((positions, flags))
    return arrays


def side_surrogate_counts(arrays: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator) -> np.ndarray:
    counts = np.zeros(len(DATE_INDEX), dtype=np.int16)
    for positions, flags in arrays:
        n = len(flags)
        if n <= 1:
            shifted = flags
        else:
            shifted = np.roll(flags, int(rng.integers(0, n)))
        event_positions = positions[shifted == 1]
        if len(event_positions):
            np.add.at(counts, event_positions, 1)
    return counts


def synchronized_mask(count_a: np.ndarray, n_a: int, count_b: np.ndarray, n_b: int, threshold: float) -> np.ndarray:
    return (count_a / n_a > threshold) & (count_b / n_b > threshold)


def annual_counts(mask: np.ndarray) -> np.ndarray:
    dates = DATE_INDEX[mask]
    if len(dates) == 0:
        return np.zeros(len(YEARS), dtype=np.int16)
    counts = pd.Series(1, index=dates).groupby(dates.year).sum()
    return counts.reindex(YEARS, fill_value=0).to_numpy(dtype=np.int16)


def trend_stats(values: np.ndarray) -> tuple[float, float]:
    slope, _, _, _ = theilslopes(values.astype(float), YEARS.astype(float), 0.95)
    _, p_value = kendalltau(YEARS.astype(float), values.astype(float))
    return float(slope), float(p_value)


def empirical_p_greater(observed: float, surrogates: np.ndarray) -> float:
    return float((np.sum(surrogates >= observed) + 1) / (len(surrogates) + 1))


def summarize_surrogate(observed: float, surrogate_values: np.ndarray) -> dict[str, float]:
    return {
        "observed": float(observed),
        "surrogate_mean": float(np.mean(surrogate_values)),
        "surrogate_p95": float(np.percentile(surrogate_values, 95)),
        "surrogate_p99": float(np.percentile(surrogate_values, 99)),
        "empirical_p_greater": empirical_p_greater(float(observed), surrogate_values),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    footprints_path = args.derived_root / "population_exposure/warm_season_bundle_node_footprints.csv"
    bundle_summary_path = args.derived_root / "warm_season_bundle_summary.csv"
    hot_events_path = args.input_root / "ERA5_hotnight_events_T_Tw_dyn_2000_2024_tau015_local_warm3.parquet"
    cold_events_path = args.input_root / "ERA5_coldnight_events_T_Tw_dyn_2000_2024_tau015_local_warm3.parquet"

    missing = [p for p in [footprints_path, bundle_summary_path, hot_events_path, cold_events_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(str(p) for p in missing))

    sides = load_bundle_sides(footprints_path)
    hot_nodes = {node for side in sides if side.event_kind == "hot" for node in side.nodes}
    cold_nodes = {node for side in sides if side.event_kind == "cold" for node in side.nodes}
    hot_events = load_event_table(hot_events_path, "evt_T_run2", hot_nodes)
    cold_events = load_event_table(cold_events_path, "evt_cold_T_run2", cold_nodes) if cold_nodes else pd.DataFrame()

    observed_side_counts: dict[tuple[str, int, str], np.ndarray] = {}
    surrogate_inputs: dict[tuple[str, int, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    node_counts: dict[tuple[str, int, str], int] = {}
    for side in sides:
        key = (side.mechanism, side.rank, side.side)
        table = hot_events if side.event_kind == "hot" else cold_events
        observed_side_counts[key] = side_observed_counts(side, table)
        surrogate_inputs[key] = node_year_arrays(table, side.nodes)
        node_counts[key] = len(side.nodes)

    bundle_summary = pd.read_csv(bundle_summary_path)
    observed_bundle_rows: list[dict[str, object]] = []
    for row in bundle_summary.itertuples(index=False):
        if row.mechanism == "Coherent":
            key_a = ("Coherent", int(row.rank), "side_a")
            key_b = ("Coherent", int(row.rank), "side_b")
        else:
            key_a = ("Dipole", int(row.rank), "hot_side")
            key_b = ("Dipole", int(row.rank), "cool_side")
        mask = synchronized_mask(
            observed_side_counts[key_a],
            node_counts[key_a],
            observed_side_counts[key_b],
            node_counts[key_b],
            args.threshold,
        )
        annual = annual_counts(mask)
        slope, p_value = trend_stats(annual)
        observed_bundle_rows.append(
            {
                "mechanism": row.mechanism,
                "rank": int(row.rank),
                "observed_date_count_recomputed": int(mask.sum()),
                "manuscript_date_count": int(row.date_count),
                "observed_slope": slope,
                "observed_kendall_p": p_value,
            }
        )

    rng = np.random.default_rng(args.seed)
    bundle_surrogate_counts = {(row["mechanism"], row["rank"]): np.zeros(args.n_surrogates, dtype=np.int16) for row in observed_bundle_rows}
    layer_surrogate_annual = {
        "Coherent": np.zeros((args.n_surrogates, len(YEARS)), dtype=np.int16),
        "Dipole": np.zeros((args.n_surrogates, len(YEARS)), dtype=np.int16),
    }

    side_keys = list(surrogate_inputs.keys())
    for surrogate_idx in range(args.n_surrogates):
        side_counts = {
            key: side_surrogate_counts(surrogate_inputs[key], rng)
            for key in side_keys
        }
        for row in bundle_summary.itertuples(index=False):
            if row.mechanism == "Coherent":
                key_a = ("Coherent", int(row.rank), "side_a")
                key_b = ("Coherent", int(row.rank), "side_b")
            else:
                key_a = ("Dipole", int(row.rank), "hot_side")
                key_b = ("Dipole", int(row.rank), "cool_side")
            mask = synchronized_mask(side_counts[key_a], node_counts[key_a], side_counts[key_b], node_counts[key_b], args.threshold)
            bundle_surrogate_counts[(row.mechanism, int(row.rank))][surrogate_idx] = int(mask.sum())
            layer_surrogate_annual[row.mechanism][surrogate_idx, :] += annual_counts(mask)

    bundle_rows: list[dict[str, object]] = []
    observed_lookup = {(r["mechanism"], r["rank"]): r for r in observed_bundle_rows}
    for key, values in bundle_surrogate_counts.items():
        obs = observed_lookup[key]
        row = {
            "mechanism": key[0],
            "rank": key[1],
            "manuscript_date_count": obs["manuscript_date_count"],
            "observed_date_count_recomputed": obs["observed_date_count_recomputed"],
            **summarize_surrogate(obs["observed_date_count_recomputed"], values.astype(float)),
        }
        bundle_rows.append(row)

    layer_rows: list[dict[str, object]] = []
    observed_annual_by_layer = {"Coherent": np.zeros(len(YEARS), dtype=np.int16), "Dipole": np.zeros(len(YEARS), dtype=np.int16)}
    for row in bundle_summary.itertuples(index=False):
        if row.mechanism == "Coherent":
            key_a = ("Coherent", int(row.rank), "side_a")
            key_b = ("Coherent", int(row.rank), "side_b")
        else:
            key_a = ("Dipole", int(row.rank), "hot_side")
            key_b = ("Dipole", int(row.rank), "cool_side")
        mask = synchronized_mask(
            observed_side_counts[key_a],
            node_counts[key_a],
            observed_side_counts[key_b],
            node_counts[key_b],
            args.threshold,
        )
        observed_annual_by_layer[row.mechanism] += annual_counts(mask)

    for mechanism in ["Coherent", "Dipole"]:
        observed_annual = observed_annual_by_layer[mechanism]
        observed_total = float(observed_annual.sum())
        observed_slope, observed_p = trend_stats(observed_annual)
        surrogate_totals = layer_surrogate_annual[mechanism].sum(axis=1).astype(float)
        surrogate_slopes = np.array([trend_stats(layer_surrogate_annual[mechanism][idx, :])[0] for idx in range(args.n_surrogates)])
        layer_rows.append(
            {
                "mechanism": mechanism,
                "observed_total_bundle_days_recomputed": observed_total,
                **{f"total_{k}": v for k, v in summarize_surrogate(observed_total, surrogate_totals).items()},
                "observed_slope": observed_slope,
                "observed_kendall_p": observed_p,
                "surrogate_slope_mean": float(np.mean(surrogate_slopes)),
                "surrogate_slope_p95": float(np.percentile(surrogate_slopes, 95)),
                "surrogate_slope_p99": float(np.percentile(surrogate_slopes, 99)),
                "slope_empirical_p_greater": empirical_p_greater(observed_slope, surrogate_slopes),
            }
        )

    pd.DataFrame(bundle_rows).to_csv(args.output_dir / "warm_season_bundle_surrogate_counts.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(args.output_dir / "warm_season_layer_surrogate_trends.csv", index=False)
    pd.DataFrame(
        {
            "surrogate_index": np.arange(args.n_surrogates, dtype=int),
            "coherent_total_bundle_days": layer_surrogate_annual["Coherent"].sum(axis=1),
            "dipole_total_bundle_days": layer_surrogate_annual["Dipole"].sum(axis=1),
        }
    ).to_csv(args.output_dir / "warm_season_layer_surrogate_draws.csv", index=False)
    print(f"Wrote surrogate validation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
