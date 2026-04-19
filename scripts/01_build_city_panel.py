from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from urban_hot_night_sync.paths import ANALYSIS_ROOT


def tau_tag(tau: float) -> str:
    return f"tau{int(round(tau * 100)):03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the yearly urban-activity panel and union-node table from yearly GAIA parquet files."
    )
    parser.add_argument("--tau", type=float, default=0.15, help="Urban-fraction threshold used to mark active urban cells.")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--gaia-root",
        type=Path,
        default=ANALYSIS_ROOT,
        help="Directory containing GAIA_frac_025_<year>.parquet files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ANALYSIS_ROOT,
        help="Directory for the panel and union-node parquet outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = range(args.start_year, args.end_year + 1)
    tag = tau_tag(args.tau)

    panel_path = args.output_root / f"GAIA_025_urban_status_{tag}_{args.start_year}_{args.end_year}.parquet"
    union_path = args.output_root / f"GAIA_025_union_nodes_{tag}_{args.start_year}_{args.end_year}.parquet"

    frames: list[pd.DataFrame] = []
    missing: list[Path] = []

    for year in years:
        path = args.gaia_root / f"GAIA_frac_025_{year}.parquet"
        if not path.exists():
            missing.append(path)
            continue

        frame = pd.read_parquet(path, columns=["lon", "lat", "urban_frac"])
        frame["year"] = year
        frame["active_y"] = (frame["urban_frac"] >= args.tau).astype("int8")
        frames.append(frame[["lon", "lat", "year", "urban_frac", "active_y"]])

    if not frames:
        raise FileNotFoundError(f"No GAIA yearly parquet files were found under {args.gaia_root}")

    panel = pd.concat(frames, ignore_index=True)
    union = (
        panel.groupby(["lon", "lat"], as_index=False)["active_y"]
        .max()
        .rename(columns={"active_y": "ever_active"})
    )
    union = union.loc[union["ever_active"] == 1, ["lon", "lat"]]

    args.output_root.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path, index=False)
    union.to_parquet(union_path, index=False)

    print(f"Wrote panel: {panel_path}")
    print(f"Wrote union nodes: {union_path}")
    if missing:
        print("Missing yearly GAIA inputs:")
        for path in missing:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
