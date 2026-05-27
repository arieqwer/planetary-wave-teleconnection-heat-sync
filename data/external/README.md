# External Data Inputs

The public GitHub repository intentionally excludes heavy raw and intermediate files. Place those files in a separate archive or local external-data directory and point the scripts to them with environment variables.

## Public source products

- GAIA impervious/urban-fraction grids aggregated to the manuscript grid.
- ERA5 hourly 2 m air temperature and dew-point temperature at the urban-node union.
- NCEP/NCAR daily geopotential-height files used for the lightweight Z500 diagnostics.
- WorldPop 100 m population accessed through Google Earth Engine for 2020 footprint population.

## Expected local/archive layout

These names are the defaults assumed by the scripts. You may use another layout if you pass command-line arguments or set the environment variables below.

```text
external/
  gaia_yearly/
    GAIA_frac_025_2000.parquet
    ...
    GAIA_frac_025_2024.parquet
  era5_hourly_yearly/
    ERA5_hourly_T_Td_UNION_2000.parquet
    ...
    ERA5_hourly_T_Td_UNION_2024.parquet
  warm_season_local_warm3/
    GAIA_025_urban_status_tau015_2000_2024.parquet
    GAIA_025_union_nodes_tau015_2000_2024.parquet
    ERA5_hotnight_events_T_Tw_dyn_2000_2024_tau015_local_warm3.parquet
    ERA5_coldnight_events_T_Tw_dyn_2000_2024_tau015_local_warm3.parquet
    ECA_edges_dyn_T2_a0.005_K3_tau015_local_warm3.parquet
    ECA_edges_SEESAW_T2_a0.005_K3_tau015_local_warm3.parquet
    ECA_nodes_used_T2_tau015_local_warm3.parquet
  ncep/
    hgt.2000.nc
    ...
    hgt.2024.nc
```

The filename `ECA_edges_SEESAW...` is retained only for continuity with the original analysis pipeline. It is the edge file used for the manuscript's `Dipole` network layer.

## Environment variables

- `UHN_SYNC_EXTERNAL_ROOT`: parent directory for external inputs.
- `UHN_SYNC_GAIA_YEARLY_ROOT`: directory containing yearly GAIA parquet files.
- `UHN_SYNC_ERA5_YEARLY_ROOT`: directory containing yearly ERA5 hourly parquet files.
- `UHN_SYNC_WARM_SEASON_INPUT_ROOT`: directory containing corrected warm-season event tables and edge files.
- `UHN_SYNC_NCEP_ROOT`: directory containing `hgt.<year>.nc` files.
- `UHN_SYNC_WARM_SEASON_WORK_ROOT`: scratch/work directory for sensitivity outputs that are too large or too temporary for GitHub.
- `UHN_SYNC_SOURCE_EVENT_ROOT`: directory containing all-season source event tables if rebuilding local-warm4/local-warm6 sensitivity products.

## Archiving recommendation

For journal submission, archive non-GitHub files in Zenodo, Figshare, OSF, or an institutional repository, then cite that archive in the Data Availability and Code Availability statements. GitHub should contain only the scripts and lightweight derived outputs listed in `data/GITHUB_DATA_FILES.txt`.
