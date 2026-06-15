# Urban Hot Night Synchrony

Code and lightweight derived data for the manuscript **Atmospheric Circulation Synchronizes Urban Heat Exposure across Distant Regions**.

This repository is designed as a compact reproducibility package. It contains the integrated scripts and compact manuscript outputs needed to audit the warm-season event-network, circulation, intensity, trend, population-exposure, and robustness results. Large public source datasets are not committed to GitHub; their provenance and expected local layout are described in `data/external/README.md`.

## What Is Included

- Primary warm-season event-network summaries for the coherent shared-hot-night layer and the dipole hot-cool-contrast layer.
- Bundle dates, trend summaries, matched-intensity summaries, Z500 diagnostics, population-footprint exposure tables, and figure-support files.
- Sensitivity outputs for warm-season definition, event-duration threshold, ECA lag window, raw pairwise p-value threshold, urban-footprint denominators, stable urban core, timing-randomized surrogates, and adjacent Z500 wavenumber bands.
- A validation script that checks the public manifest, key manuscript numbers, robustness outputs, and absence of local absolute paths.

## Quick Validation

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/12_validate_release_package.py
```

The validation should end with:

```text
Release validation passed.
```

## Core Workflow

The complete workflow is ordered below. Scripts accept command-line arguments or environment variables for non-GitHub input locations.

1. `scripts/01_build_city_panel.py` builds yearly GAIA urban-activity panels and union-node tables.
2. `scripts/02_build_event_tables.py` builds daily hot-night and cool-anomaly event tables from ERA5-Land inputs.
3. `scripts/03_build_synchrony_networks.py` builds coherent and dipole event-coincidence networks.
4. `scripts/04_build_primary_warm_season_outputs.py` rebuilds primary bundle dates, trend summaries, raw-lock summaries, and matched-amplification summaries.
5. `scripts/05_warm_season_circulation_diagnostics.py` computes Z500 wave diagnostics and optional adjacent-band sensitivity outputs.
6. `scripts/06_warm_season_population_exposure.py` recomputes WorldPop footprint populations and static person-night exposure.
7. `scripts/07_warm_season_edge_screen_diagnostic.py` recomputes the Benjamini-Hochberg FDR diagnostic for the ECA edge screen.
8. `scripts/08_warm_season_definition_sensitivity.py` rebuilds local-warmest-4-month and local-warmest-6-month sensitivity summaries.
9. `scripts/09_warm_season_network_parameter_sensitivity.py` recomputes network summaries under alternative ECA settings.
10. `scripts/10_warm_season_bundle_geography_sensitivity.py` tests dominant-bundle geography under alternative ECA settings.
11. `scripts/11_warm_season_final_statistical_robustness.py` rebuilds compact matched-intensity and trend robustness outputs.
12. `scripts/13_warm_season_active_denominator_trend.py` evaluates annual active-denominator and stable-urban-core trend checks.
13. `scripts/14_warm_season_bundle_surrogate_validation.py` runs the node-year circular-shift surrogate diagnostic.

## Repository Layout

- `scripts/`: ordered processing, diagnostic, and validation scripts.
- `src/urban_hot_night_sync/`: repository-relative path helpers.
- `data/derived/`: lightweight CSV and Parquet outputs used by the manuscript and Supporting Information.
- `data/GITHUB_DATA_FILES.txt`: manifest of all public derived-data files.
- `data/external/README.md`: public source-data provenance and expected external-input layout.
- `figures/warm_season_exposure/`: lightweight exposure figure panels retained for traceability.
- `MANUSCRIPT_OUTPUT_MAP.md`: concise map from manuscript/SI items to repository files.

## Data Policy Note

For initial peer review, this GitHub repository provides code and lightweight derived outputs for auditability. Before publication, the repository will be archived in Zenodo with DOI.
