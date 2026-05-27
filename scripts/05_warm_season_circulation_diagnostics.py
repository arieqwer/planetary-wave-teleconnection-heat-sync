from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from urban_hot_night_sync.paths import DERIVED_ROOT, NCEP_ROOT


def lon_360(lon: float) -> float:
    return lon % 360


def longitudes(n_points: int) -> np.ndarray:
    return np.linspace(0, 360, n_points, endpoint=False)


def fourier_filter(values: np.ndarray, min_wavenumber: int = 4, max_wavenumber: int = 8) -> np.ndarray:
    fft = np.fft.rfft(values)
    mask = np.zeros_like(fft, dtype=bool)
    mask[min_wavenumber : max_wavenumber + 1] = True
    return np.fft.irfft(fft * mask, n=len(values))


def pattern_drift_deg_per_day(early: np.ndarray, late: np.ndarray, max_shift: int = 20) -> float:
    """Estimate longitudinal drift of the lag-correlation maximum, not wave phase speed."""
    shifts = range(-max_shift, max_shift + 1)
    correlations = []
    for shift in shifts:
        corr = np.corrcoef(early, np.roll(late, shift))[0, 1]
        correlations.append(corr)
    best_shift = shifts[int(np.nanargmax(correlations))]
    return float(-best_shift / 6.0)


def nearest_value(wave: np.ndarray, lons: np.ndarray, lon: float) -> float:
    idx = int(np.abs(lons - lon_360(lon)).argmin())
    return float(wave[idx])


def open_hgt(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path, decode_times=True)
    except Exception:
        coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        return xr.open_dataset(path, decode_times=coder)


def composite_waves(
    dates: pd.Series,
    target_lat: float,
    ncep_cache_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    t0_values: list[np.ndarray] = []
    early_values: list[np.ndarray] = []
    late_values: list[np.ndarray] = []

    dates = pd.to_datetime(dates, errors="coerce").dropna()
    for year in sorted(dates.dt.year.unique()):
        hgt_path = ncep_cache_root / f"hgt.{int(year)}.nc"
        if not hgt_path.exists():
            continue

        with open_hgt(hgt_path) as dataset:
            z500 = dataset["hgt"].sel(level=500, lat=target_lat, method="nearest")
            year_dates = dates.loc[dates.dt.year == year]
            for date in year_dates:
                try:
                    d0 = z500.sel(time=date, method="nearest").values
                    d_early = z500.sel(time=date - pd.Timedelta(days=3), method="nearest").values
                    d_late = z500.sel(time=date + pd.Timedelta(days=3), method="nearest").values
                except Exception:
                    continue
                t0_values.append(np.asarray(d0) - np.nanmean(d0))
                early_values.append(np.asarray(d_early) - np.nanmean(d_early))
                late_values.append(np.asarray(d_late) - np.nanmean(d_late))

    if not t0_values:
        raise RuntimeError(f"No valid Z500 windows found for target latitude {target_lat:g}.")

    return (
        fourier_filter(np.nanmean(t0_values, axis=0)),
        fourier_filter(np.nanmean(early_values, axis=0)),
        fourier_filter(np.nanmean(late_values, axis=0)),
        len(t0_values),
    )


def diagnostics_for_bundle(row: pd.Series, bundle_dir: Path, ncep_cache_root: Path) -> dict[str, object]:
    date_file = bundle_dir / row["date_file"]
    dates = pd.read_csv(date_file)["date"]
    target_lat = float(np.nanmean([row["centroid_a_lat"], row["centroid_b_lat"]]))
    wave_t0, wave_early, wave_late, n_events = composite_waves(dates, target_lat, ncep_cache_root)
    lons = longitudes(len(wave_t0))

    lon_a = lon_360(float(row["centroid_a_lon"]))
    lon_b = lon_360(float(row["centroid_b_lon"]))
    val_a = nearest_value(wave_t0, lons, lon_a)
    val_b = nearest_value(wave_t0, lons, lon_b)
    delta_z = abs(val_a - val_b)
    speed = pattern_drift_deg_per_day(wave_early, wave_late)

    return {
        "mechanism": row["mechanism"],
        "rank": int(row["rank"]),
        "date_file": row["date_file"],
        "n_event_windows": int(n_events),
        "target_lat": target_lat,
        "lon_a_0_360": lon_a,
        "lon_b_0_360": lon_b,
        "z500_wave_a_m": val_a,
        "z500_wave_b_m": val_b,
        "abs_delta_z_m": float(delta_z),
        "same_sign": bool(np.sign(val_a) == np.sign(val_b)) if val_a != 0 and val_b != 0 else False,
        "pattern_drift_deg_per_day": speed,
        "diagnostic_class_by_delta_z": "coherent-like" if delta_z < 45 else "dipole-like",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute warm-season Z500 wave diagnostics for extracted bundles.")
    parser.add_argument("--output-root", type=Path, default=DERIVED_ROOT / "warm_season_local_warm3")
    parser.add_argument("--ncep-cache-root", type=Path, default=NCEP_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_summary_path = args.output_root / "warm_season_bundle_summary.csv"
    bundle_dir = args.output_root / "bundle_dates"
    if not bundle_summary_path.exists():
        raise FileNotFoundError(f"Missing bundle summary: {bundle_summary_path}")
    if not args.ncep_cache_root.exists():
        raise FileNotFoundError(f"Missing NCEP cache directory: {args.ncep_cache_root}")

    summary = pd.read_csv(bundle_summary_path)
    rows = [diagnostics_for_bundle(row, bundle_dir, args.ncep_cache_root) for _, row in summary.iterrows()]
    output_path = args.output_root / "warm_season_z500_wave_diagnostics.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote warm-season Z500 diagnostics to: {output_path}")


if __name__ == "__main__":
    main()
