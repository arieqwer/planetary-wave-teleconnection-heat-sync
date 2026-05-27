from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "derived"
MANIFEST = REPO_ROOT / "data" / "GITHUB_DATA_FILES.txt"


def fail(message: str) -> None:
    raise AssertionError(message)


def assert_close(actual: float, expected: float, tol: float, label: str) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=tol):
        fail(f"{label}: expected {expected}, found {actual}")


def read_csv(relative_path: str) -> pd.DataFrame:
    path = REPO_ROOT / relative_path
    if not path.exists():
        fail(f"Missing required file: {relative_path}")
    return pd.read_csv(path)


def validate_manifest() -> None:
    if not MANIFEST.exists():
        fail("Missing data/GITHUB_DATA_FILES.txt")

    listed = [line.strip() for line in MANIFEST.read_text().splitlines() if line.strip()]
    missing = [path for path in listed if not (REPO_ROOT / path).exists()]
    if missing:
        fail("Manifest entries missing on disk:\n" + "\n".join(missing))

    actual = sorted(str(path.relative_to(REPO_ROOT)) for path in DATA_ROOT.rglob("*") if path.is_file())
    extra = sorted(set(actual) - set(listed))
    stale = sorted(set(listed) - set(actual))
    if extra or stale:
        message = []
        if extra:
            message.append("Data files absent from manifest:\n" + "\n".join(extra))
        if stale:
            message.append("Manifest entries not in data/derived:\n" + "\n".join(stale))
        fail("\n\n".join(message))


def validate_no_stale_outputs() -> None:
    stale_dirs = [
        DATA_ROOT / "analysis",
        DATA_ROOT / "bundle_dates",
    ]
    present = [str(path.relative_to(REPO_ROOT)) for path in stale_dirs if path.exists()]
    if present:
        fail("Stale year-round output directories are still present:\n" + "\n".join(present))

    forbidden_path_tokens = [
        "Hot" + "Hot",
        "hot" + "hot",
        "SEE" + "SAW",
        "see" + "saw",
    ]
    bad_paths = [
        str(path.relative_to(REPO_ROOT))
        for path in DATA_ROOT.rglob("*")
        if path.is_file() and any(token in path.name for token in forbidden_path_tokens)
    ]
    if bad_paths:
        fail("Stale terminology remains in public derived-data filenames:\n" + "\n".join(sorted(bad_paths)))


def validate_primary_numbers() -> None:
    bundle = read_csv("data/derived/warm_season_local_warm3/warm_season_bundle_summary.csv")
    if len(bundle) != 6:
        fail(f"Expected 6 primary bundle rows, found {len(bundle)}")
    if set(bundle["mechanism"]) != {"Coherent", "Dipole"}:
        fail(f"Unexpected bundle mechanisms: {sorted(bundle['mechanism'].unique())}")
    for mechanism in ["Coherent", "Dipole"]:
        ranks = sorted(bundle.loc[bundle["mechanism"] == mechanism, "rank"].tolist())
        if ranks != [1, 2, 3]:
            fail(f"{mechanism} bundle ranks should be [1, 2, 3], found {ranks}")

    trends = read_csv("data/derived/warm_season_local_warm3/warm_season_trend_summary.csv")
    trend_by_layer = trends.set_index("mechanism")
    assert_close(trend_by_layer.loc["Coherent", "theil_sen_slope_per_year"], 0.8, 1e-9, "Coherent trend slope")
    assert_close(trend_by_layer.loc["Dipole", "theil_sen_slope_per_year"], 0.2638888889, 1e-9, "Dipole trend slope")
    assert_close(trend_by_layer.loc["Coherent", "total_bundle_days"], 281, 1e-9, "Coherent total bundle days")
    assert_close(trend_by_layer.loc["Dipole", "total_bundle_days"], 106, 1e-9, "Dipole total bundle days")

    matched = read_csv("data/derived/warm_season_local_warm3/warm_season_matched_amplification_summary.csv")
    matched_by_key = matched.set_index(["mechanism", "edge_set"])
    expected_matched = {
        ("Coherent", "all_edges"): 0.232185,
        ("Coherent", "long_range_gt2500km"): 0.020369,
        ("Dipole", "all_edges"): 0.029356,
        ("Dipole", "long_range_gt2500km"): 0.042439,
    }
    for key, expected in expected_matched.items():
        assert_close(matched_by_key.loc[key, "mean_delta"], expected, 1e-6, f"{key} matched mean_delta")

    raw = read_csv("data/derived/warm_season_local_warm3/warm_season_raw_lock_summary.csv")
    raw_by_key = raw.set_index(["mechanism", "edge_set"])
    expected_edges = {
        ("Coherent", "all_edges"): 101348,
        ("Coherent", "long_range_gt2500km"): 2403,
        ("Dipole", "all_edges"): 13194,
        ("Dipole", "long_range_gt2500km"): 11439,
    }
    for key, expected in expected_edges.items():
        assert_close(raw_by_key.loc[key, "edge_pairs_used"], expected, 1e-9, f"{key} edge_pairs_used")

    fdr = read_csv("data/derived/warm_season_local_warm3/warm_season_edge_screen_fdr_diagnostic.csv")
    fdr_by_layer = fdr.set_index("network_layer")
    assert_close(fdr_by_layer.loc["Coherent", "candidate_edges_p_lt_0_005"], 101348, 1e-9, "Coherent candidate edges")
    assert_close(fdr_by_layer.loc["Dipole", "candidate_edges_p_lt_0_005"], 13194, 1e-9, "Dipole candidate edges")
    assert_close(fdr_by_layer.loc["Coherent", "bh_retained_edges"], 80769, 1e-9, "Coherent BH retained edges")
    assert_close(fdr_by_layer.loc["Dipole", "bh_retained_edges"], 81, 1e-9, "Dipole BH retained edges")


def validate_population_and_diagnostics() -> None:
    reconstruction = read_csv(
        "data/derived/warm_season_local_warm3/population_exposure/warm_season_bundle_reconstruction_validation.csv"
    )
    for column in ["edge_count_match", "node_count_a_match", "node_count_b_match"]:
        if not reconstruction[column].astype(bool).all():
            fail(f"Population footprint reconstruction failed in column {column}")

    exposure = read_csv(
        "data/derived/warm_season_local_warm3/population_exposure/warm_season_population_exposure_by_bundle.csv"
    )
    if len(exposure) != 6:
        fail(f"Expected 6 exposure bundle rows, found {len(exposure)}")
    if (exposure["static_2020_person_nights"] <= 0).any():
        fail("Exposure table contains non-positive static person-night values")

    z500 = read_csv("data/derived/warm_season_local_warm3/warm_season_z500_wave_diagnostics.csv")
    removed_column = "phase" + "_speed_deg_per_day"
    if removed_column in z500.columns:
        fail(f"Z500 diagnostics still contain the removed {removed_column} column")
    if len(z500) != 6:
        fail(f"Expected 6 Z500 diagnostic rows, found {len(z500)}")


def validate_robustness_files() -> None:
    required = [
        "data/derived/warm_season_definition_sensitivity/warm_season_definition_network_sensitivity.csv",
        "data/derived/warm_season_parameter_sensitivity/warm_season_network_parameter_sensitivity.csv",
        "data/derived/warm_season_bundle_geography_sensitivity/warm_season_bundle_geography_sensitivity_summary.csv",
        "data/derived/warm_season_final_statistical_robustness/warm_season_matched_amplification_parameter_sensitivity_compact.csv",
        "data/derived/warm_season_final_statistical_robustness/warm_season_bundle_trend_parameter_sensitivity_compact.csv",
    ]
    for relative_path in required:
        frame = read_csv(relative_path)
        if frame.empty:
            fail(f"Required robustness file is empty: {relative_path}")


def validate_no_local_paths() -> None:
    forbidden = [
        "/" + "Users" + "/" + "apple",
        "/" + "Volumes" + "/" + "duck",
        "/" + "content" + "/" + "drive",
        "My" + "Drive",
    ]
    text_suffixes = {".csv", ".json", ".md", ".py", ".txt", ".toml", ".yml", ".yaml"}
    bad: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file() or path.suffix not in text_suffixes:
            continue
        text = path.read_text(errors="ignore")
        hits = [token for token in forbidden if token in text]
        if hits:
            bad.append(f"{path.relative_to(REPO_ROOT)}: {', '.join(hits)}")
    if bad:
        fail("Local absolute paths remain in release text files:\n" + "\n".join(sorted(bad)))


def validate_file_sizes() -> None:
    large = [
        f"{path.relative_to(REPO_ROOT)} ({path.stat().st_size / 1024 / 1024:.1f} MB)"
        for path in REPO_ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and path.stat().st_size > 50 * 1024 * 1024
    ]
    if large:
        fail("Files larger than 50 MB should not be committed to GitHub:\n" + "\n".join(large))


def main() -> None:
    validate_manifest()
    validate_no_stale_outputs()
    validate_primary_numbers()
    validate_population_and_diagnostics()
    validate_robustness_files()
    validate_no_local_paths()
    validate_file_sizes()
    print("Release validation passed.")


if __name__ == "__main__":
    main()
