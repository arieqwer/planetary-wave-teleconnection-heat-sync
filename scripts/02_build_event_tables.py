from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from urban_hot_night_sync.paths import ANALYSIS_ROOT, ERA5_YEARLY_ROOT


def tau_tag(tau: float) -> str:
    return f"tau{int(round(tau * 100)):03d}"


def kelvin_to_c(values: np.ndarray) -> np.ndarray:
    return values - 273.15


def rh_from_t_tdew(temp_c: np.ndarray, dew_c: np.ndarray) -> np.ndarray:
    a, b = 17.625, 243.04
    es = 6.1094 * np.exp((a * temp_c) / (b + temp_c))
    e = 6.1094 * np.exp((a * dew_c) / (b + dew_c))
    return np.clip(100.0 * (e / es), 0, 100)


def tw_stull(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    temp_c = np.asarray(temp_c, float)
    rh = np.asarray(rh, float)
    return (
        temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(temp_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


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


def nightly_minima_for_year(df: pd.DataFrame) -> pd.DataFrame:
    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"])
    elif "time_iso" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_iso"])
    else:
        raise KeyError("Expected either time_utc or time_iso in ERA5 hourly parquet input.")

    df["T_C"] = kelvin_to_c(df["temperature_2m"].to_numpy())
    df["Td_C"] = kelvin_to_c(df["dewpoint_temperature_2m"].to_numpy())
    df["lt"] = df["time_utc"] + pd.to_timedelta(df["lon"] / 15.0, unit="h")
    df["lt_hour"] = df["lt"].dt.hour

    night = df.loc[(df["lt_hour"] >= 0) & (df["lt_hour"] <= 6)].copy()
    if night.empty:
        return night

    night["lt_date"] = pd.to_datetime(night["lt"].dt.date)
    night["RH"] = rh_from_t_tdew(night["T_C"].to_numpy(), night["Td_C"].to_numpy())
    night["Tw_C"] = tw_stull(night["T_C"].to_numpy(), night["RH"].to_numpy())

    group_cols = ["lon", "lat", "lt_date"]
    if {"row_idx", "col_idx"}.issubset(night.columns):
        group_cols = ["lon", "lat", "row_idx", "col_idx", "lt_date"]

    return (
        night.groupby(group_cols, as_index=False)
        .agg(Tmin_C=("T_C", "min"), Twmin_C=("Tw_C", "min"))
    )


def smooth_thresholds(frame: pd.DataFrame, spatial_cols: list[str], value_col: str, out_col: str) -> pd.DataFrame:
    frame = frame.sort_values(spatial_cols + ["doy"]).copy()
    frame[out_col] = (
        frame.groupby(spatial_cols)[value_col]
        .transform(lambda series: series.rolling(window=15, center=True, min_periods=1).mean())
    )
    return frame


def build_event_table(
    kind: str,
    urban_panel_path: Path,
    era5_root: Path,
    output_path: Path,
    baseline_start: int,
    baseline_end: int,
    min_run_length: int,
) -> None:
    panel = pd.read_parquet(urban_panel_path)
    years = sorted(panel["year"].unique())

    yearly_frames: list[pd.DataFrame] = []
    missing: list[Path] = []

    for year in years:
        hourly_path = era5_root / f"ERA5_hourly_T_Td_UNION_{year}.parquet"
        if not hourly_path.exists():
            missing.append(hourly_path)
            continue

        hourly = pd.read_parquet(hourly_path)
        minima = nightly_minima_for_year(hourly)
        if minima.empty:
            continue

        minima["year"] = year
        minima["doy"] = minima["lt_date"].dt.dayofyear
        minima = minima.merge(panel[["lon", "lat", "year", "active_y"]], on=["lon", "lat", "year"], how="left")
        minima = minima.loc[minima["active_y"] == 1].copy()
        yearly_frames.append(minima.drop(columns=["active_y"]))

    if not yearly_frames:
        raise RuntimeError(f"No nightly minima could be built from {era5_root}")

    full = pd.concat(yearly_frames, ignore_index=True)
    spatial_cols = ["lon", "lat"]
    if {"row_idx", "col_idx"}.issubset(full.columns):
        spatial_cols.extend(["row_idx", "col_idx"])

    baseline = full.loc[(full["year"] >= baseline_start) & (full["year"] <= baseline_end)].copy()
    if baseline.empty:
        raise RuntimeError("Baseline period produced no rows; check the event years and baseline settings.")

    if kind == "hot":
        percentile = 95
        comparator = np.greater_equal
        prefix = "evt_"
        t_raw, tw_raw = "Tmin95_raw", "Twmin95_raw"
        t_thr, tw_thr = "Tmin95", "Twmin95"
    else:
        percentile = 5
        comparator = np.less_equal
        prefix = "evt_cold_"
        t_raw, tw_raw = "Tmin05_raw", "Twmin05_raw"
        t_thr, tw_thr = "Tmin05", "Twmin05"

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
        full.merge(thr_t, on=spatial_cols + ["doy"], how="left")
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
    out.to_parquet(output_path, index=False)
    print(f"Wrote {kind} event table: {output_path}")
    if missing:
        print("Missing yearly ERA5 inputs:")
        for path in missing:
            print(f"  - {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the hot-night and cold-night event tables from yearly ERA5 hourly parquet files."
    )
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--kind", choices=["hot", "cold", "both"], default="both")
    parser.add_argument("--baseline-start", type=int, default=2001)
    parser.add_argument("--baseline-end", type=int, default=2020)
    parser.add_argument("--min-run-length", type=int, default=2)
    parser.add_argument(
        "--urban-panel-path",
        type=Path,
        default=None,
        help="Override the default GAIA panel path. By default this is inferred from tau and the 2000-2024 study period.",
    )
    parser.add_argument("--era5-root", type=Path, default=ERA5_YEARLY_ROOT)
    parser.add_argument("--output-root", type=Path, default=ANALYSIS_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = tau_tag(args.tau)
    panel_path = args.urban_panel_path or (args.output_root / f"GAIA_025_urban_status_{tag}_2000_2024.parquet")

    jobs = []
    if args.kind in {"hot", "both"}:
        jobs.append(("hot", args.output_root / f"ERA5_hotnight_events_T_Tw_dyn_2000_2024_{tag}.parquet"))
    if args.kind in {"cold", "both"}:
        jobs.append(("cold", args.output_root / f"ERA5_coldnight_events_T_Tw_dyn_2000_2024_{tag}.parquet"))

    for kind, output_path in jobs:
        build_event_table(
            kind=kind,
            urban_panel_path=panel_path,
            era5_root=args.era5_root,
            output_path=output_path,
            baseline_start=args.baseline_start,
            baseline_end=args.baseline_end,
            min_run_length=args.min_run_length,
        )


if __name__ == "__main__":
    main()
