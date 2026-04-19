from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_repo_root(start: Path | None = None) -> Path:
    start = start or Path.cwd()
    here = start.resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "src" / "urban_hot_night_sync").exists():
            return candidate
    raise RuntimeError("Could not locate repository root. Run from inside the repository checkout.")


REPO_ROOT = resolve_repo_root(Path(__file__).resolve())
DATA_ROOT = Path(os.environ.get("UHN_SYNC_DATA_ROOT", REPO_ROOT / "data"))
EXTERNAL_ROOT = Path(os.environ.get("UHN_SYNC_EXTERNAL_ROOT", DATA_ROOT / "external"))
DERIVED_ROOT = Path(os.environ.get("UHN_SYNC_DERIVED_ROOT", DATA_ROOT / "derived"))
ANALYSIS_ROOT = Path(os.environ.get("UHN_SYNC_ANALYSIS_ROOT", DERIVED_ROOT / "analysis"))
BUNDLE_DATES_ROOT = Path(os.environ.get("UHN_SYNC_BUNDLE_DATES_ROOT", DERIVED_ROOT / "bundle_dates"))
NCEP_ROOT = Path(os.environ.get("UHN_SYNC_NCEP_ROOT", EXTERNAL_ROOT / "ncep"))
ERA5_YEARLY_ROOT = Path(os.environ.get("UHN_SYNC_ERA5_YEARLY_ROOT", EXTERNAL_ROOT / "era5_hourly_yearly"))


def bootstrap_notebook() -> None:
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
