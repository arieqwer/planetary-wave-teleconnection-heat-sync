# Urban Hot Night Synchrony

This repository is the trimmed public code-and-data release for the manuscript on synchronized warm-season nocturnal urban heat extremes and planetary-wave locking. It keeps only the integrated workflow scripts, lightweight derived outputs, and manuscript-support files that are suitable for GitHub.

The previous notebook-heavy working directory was intentionally reduced. The release excludes figure-by-figure notebooks, local-versus-Colab duplicates, exploratory drafting scripts, and stale year-round outputs that no longer support the corrected warm-season manuscript.

## Manuscript framing

The corrected analysis uses each node's local warm season, defined as the warmest three calendar months during the 2001-2020 baseline. The network layers use the manuscript terminology:

- `Coherent`: hot-night event coincidence at both endpoints.
- `Dipole`: hot-night event coincidence at one endpoint with cool-anomaly coincidence at the other endpoint.

Some archived source filenames retain the historical token `SEESAW` for provenance. In the manuscript and release documentation, those files support the `Dipole` network layer.

## Core workflow

Run scripts from the repository root. Heavy inputs are expected outside GitHub; see `data/external/README.md`.

1. `scripts/01_build_city_panel.py`
   Builds the yearly GAIA urban-activity panel and union-node table from yearly `GAIA_frac_025_<year>.parquet` inputs.
2. `scripts/02_build_event_tables.py`
   Builds hot-night and cool-anomaly daily event tables from yearly ERA5 hourly parquet inputs. Use `--season-mode local-warm3` for the corrected manuscript analysis.
3. `scripts/03_build_synchrony_networks.py`
   Builds the Coherent and Dipole event-coincidence networks from the event tables.
4. `scripts/04_warm_season_reviewer_outputs.py`
   Rebuilds the corrected warm-season bundle dates, trend summaries, raw-lock summaries, and matched-amplification summaries.
5. `scripts/05_warm_season_circulation_diagnostics.py`
   Computes lightweight Z500 wave diagnostics for the corrected warm-season bundles from cached NCEP daily geopotential-height files.
6. `scripts/06_warm_season_population_exposure.py`
   Recomputes 2020 WorldPop footprint populations and static person-night exposure for corrected warm-season Coherent and Dipole bundles.
7. `scripts/07_warm_season_edge_screen_diagnostic.py`
   Recomputes the Benjamini-Hochberg FDR diagnostic for the pairwise ECA candidate-edge screen.
8. `scripts/08_warm_season_definition_sensitivity.py`
   Rebuilds local-warmest-4-month and local-warmest-6-month sensitivity summaries.
9. `scripts/09_warm_season_network_parameter_sensitivity.py`
   Recomputes network summaries under alternate ECA lag windows, p-value screens, and sustained-event durations.
10. `scripts/10_warm_season_bundle_geography_sensitivity.py`
    Reclusters long-range edges under alternate ECA settings and compares dominant bundle centroids with the primary corrected bundles.
11. `scripts/11_warm_season_final_statistical_robustness.py`
    Rebuilds alternate edge sets, matched-amplification diagnostics, and top-bundle trend diagnostics for the final robustness audit.

## Reproducibility checks

Run this lightweight validation before submission or repository release:

```bash
python3 scripts/12_validate_release_package.py
```

The check verifies that the manifest matches the public data files, stale year-round outputs are absent, key corrected manuscript numbers are reproducible from the CSV outputs, and release files do not expose local absolute paths.

## GitHub data package

The exact public data files are listed in `data/GITHUB_DATA_FILES.txt`. The package includes:

- corrected warm-season top-three Coherent and Dipole bundle-date CSVs
- corrected warm-season bundle, trend, raw-lock, Z500, matched-amplification, and FDR summaries
- corrected warm-season population-footprint and exposure summaries
- warm-season definition, network-parameter, bundle-geography, and statistical-robustness sensitivity outputs
- compact robustness summaries plus small edge-list Parquet files retained to support independent checking

The complete public data package is intentionally lightweight. At the time of packaging, `data/derived/` is about 16 MB and contains no single file larger than GitHub's recommended practical limits.

## Figure and table map

- Main network/event summaries: `data/derived/warm_season_local_warm3/warm_season_bundle_summary.csv`, `warm_season_raw_lock_summary.csv`, and `warm_season_trend_summary.csv`.
- Matched amplification results: `data/derived/warm_season_local_warm3/warm_season_matched_amplification_summary.csv`.
- Z500 diagnostics: `data/derived/warm_season_local_warm3/warm_season_z500_wave_diagnostics.csv`.
- Population/exposure results: `data/derived/warm_season_local_warm3/population_exposure/`.
- Robustness tables: `data/derived/warm_season_definition_sensitivity/`, `data/derived/warm_season_parameter_sensitivity/`, `data/derived/warm_season_bundle_geography_sensitivity/`, and `data/derived/warm_season_final_statistical_robustness/`.
- Generated exposure figure panels retained in this repository: `figures/warm_season_exposure/`.

## Repository layout

- `scripts/`: integrated workflow and validation scripts
- `src/urban_hot_night_sync/`: repository-relative path helpers
- `data/derived/`: lightweight manuscript outputs kept on GitHub
- `data/external/`: placeholder and provenance notes for large external inputs
- `figures/`: lightweight generated exposure panels retained for manuscript traceability

## Environment

Create a Python environment from `requirements.txt`, then point the workflow to archived inputs with environment variables as needed:

```bash
export UHN_SYNC_EXTERNAL_ROOT=/path/to/external/archive
export UHN_SYNC_WARM_SEASON_INPUT_ROOT=$UHN_SYNC_EXTERNAL_ROOT/warm_season_local_warm3
export UHN_SYNC_NCEP_ROOT=$UHN_SYNC_EXTERNAL_ROOT/ncep
```

For Earth Engine population exposure, also set `EE_PROJECT` if your project differs from the default used during analysis.

## Notes

- The GitHub release is narrower than the private working directory by design.
- Raw GAIA, ERA5, NCEP and WorldPop products should be cited and accessed from their public providers or from a separate data archive, not committed directly to GitHub.
- The repository supports reproducibility of the corrected warm-season manuscript outputs; it is not a full mirror of every exploratory step used during manuscript development.
