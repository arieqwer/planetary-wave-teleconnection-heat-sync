# Urban Hot Night Synchrony

This is the trimmed public code-and-data release for the manuscript on synchronized nocturnal urban heat extremes and planetary-wave locking. The repository now keeps only the core processing scripts plus the lightweight manuscript outputs that are appropriate to host on GitHub.


## Core workflow

1. `scripts/01_build_city_panel.py`
   Builds the yearly GAIA urban-activity panel and the union-node table from yearly `GAIA_frac_025_*.parquet` inputs.
2. `scripts/02_build_event_tables.py`
   Builds the hot-night and cold-night daily event tables from yearly ERA5 hourly parquet inputs.
3. `scripts/03_build_synchrony_networks.py`
   Builds the Hot-Hot and seesaw event-coincidence networks from the event tables.
4. `scripts/04_prepare_release_outputs.py`
   Rebuilds the top-bundle date CSV files and the trend summary table used in the manuscript release package.

## What stays on GitHub

- the four scripts above
- the repository-relative path helper in `src/urban_hot_night_sync/`
- lightweight manuscript result files in `data/derived/`

The exact GitHub data filenames are listed in `data/GITHUB_DATA_FILES.txt`.

## What does not stay on GitHub

Do not commit the heavy source or intermediate files directly to GitHub. Put them in Zenodo, Figshare, OSF, or another archive and link that archive from the manuscript.

The required large files are summarized in `data/external/README.md`.

## Repository layout

- `scripts/`: the minimal integrated workflow
- `src/urban_hot_night_sync/`: repository path helper
- `data/derived/analysis/`: lightweight summary tables kept on GitHub
- `data/derived/bundle_dates/`: top-bundle event-day CSV files kept on GitHub
- `data/external/`: placeholder for large archived inputs and outputs

## Minimal GitHub data package

This trimmed repository keeps only:

- top-3 Hot-Hot bundle date CSVs
- top-3 seesaw bundle date CSVs
- matched amplification summary outputs
- Supplementary Table 5 exposure outputs
- Table 5 trend summary

## Quick start

1. Create a Python environment from `requirements.txt`.
2. Place the large external inputs described in `data/external/README.md` in your archive or local external-data directory.
3. Run the workflow scripts from the repository root.

## Notes

- The GitHub release is intentionally narrower than the full private working directory.
- Figure rendering scripts were removed because the release now prefers lightweight data products over plot-specific code clutter.
