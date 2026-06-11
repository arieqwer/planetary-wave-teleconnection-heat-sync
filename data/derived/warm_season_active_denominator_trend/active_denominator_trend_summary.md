# Active-denominator warm-season bundle-day trend robustness

Region threshold: 0.050. Years: 2000-2024.

The fixed-denominator recomputation validates against the manuscript annual bundle-day series. Annual-urban-active denominator scenarios replace the fixed final-node denominator with the number of bundle-side nodes classified as urban in each year. Date-active denominator scenarios use the number of bundle-side nodes present in the local-warm-season event table on each date. Stable-urban-core scenarios restrict each bundle side to nodes active in all study years before recomputing synchronized dates.

## Trend summary

- active_denominator_any_coverage / Coherent: total 948, slope 3.031 days yr^-1 (95% 1.400 to 5.125), Kendall p = 9.29e-05.
- active_denominator_any_coverage / Dipole: total 492, slope 0.923 days yr^-1 (95% 0.100 to 1.474), Kendall p = 0.0263.
- active_denominator_min50pct_coverage / Coherent: total 318, slope 1.000 days yr^-1 (95% 0.400 to 1.750), Kendall p = 7.71e-05.
- active_denominator_min50pct_coverage / Dipole: total 121, slope 0.500 days yr^-1 (95% 0.167 to 0.778), Kendall p = 1.5e-05.
- active_denominator_min80pct_coverage / Coherent: total 168, slope 0.414 days yr^-1 (95% 0.000 to 1.000), Kendall p = 0.000119.
- active_denominator_min80pct_coverage / Dipole: total 47, slope 0.000 days yr^-1 (95% 0.000 to 0.200), Kendall p = 0.000646.
- annual_urban_active_denominator / Coherent: total 315, slope 0.710 days yr^-1 (95% 0.400 to 1.500), Kendall p = 0.00026.
- annual_urban_active_denominator / Dipole: total 193, slope 0.218 days yr^-1 (95% -0.143 to 0.545), Kendall p = 0.197.
- annual_urban_active_denominator_min50pct_coverage / Coherent: total 306, slope 0.750 days yr^-1 (95% 0.500 to 1.500), Kendall p = 6.03e-05.
- annual_urban_active_denominator_min50pct_coverage / Dipole: total 94, slope 0.372 days yr^-1 (95% 0.000 to 0.588), Kendall p = 8.74e-05.
- annual_urban_active_denominator_min80pct_coverage / Coherent: total 264, slope 0.786 days yr^-1 (95% 0.250 to 1.333), Kendall p = 4.19e-06.
- annual_urban_active_denominator_min80pct_coverage / Dipole: total 70, slope 0.000 days yr^-1 (95% 0.000 to 0.429), Kendall p = 0.000228.
- fixed_denominator_recomputed / Coherent: total 281, slope 0.800 days yr^-1 (95% 0.364 to 1.429), Kendall p = 4.17e-05; max annual difference from manuscript series = 0.
- fixed_denominator_recomputed / Dipole: total 106, slope 0.264 days yr^-1 (95% 0.100 to 0.588), Kendall p = 0.000522; max annual difference from manuscript series = 0.
- stable_urban_core_active_denominator / Coherent: total 691, slope 1.967 days yr^-1 (95% 0.667 to 3.267), Kendall p = 0.000435.
- stable_urban_core_active_denominator / Dipole: total 331, slope -0.200 days yr^-1 (95% -0.647 to 0.200), Kendall p = 0.261.
- stable_urban_core_fixed_denominator / Coherent: total 307, slope 0.667 days yr^-1 (95% 0.333 to 1.562), Kendall p = 0.000194.
- stable_urban_core_fixed_denominator / Dipole: total 147, slope 0.000 days yr^-1 (95% -0.368 to 0.200), Kendall p = 0.907.

## Interpretation note

The coherent trend remains positive under annual urban-active, strict date-active and stable-core denominator checks. The dipole fixed-denominator trend is weaker and does not survive the most conservative stable-core fixed-denominator check, so the temporal trend claim should be emphasized for coherent bundle-days and treated cautiously for dipole bundle-days.
