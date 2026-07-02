# src/build_reference_bank.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def get_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "src":
        return script_dir.parent
    return Path.cwd()


PROJECT_ROOT = get_project_root()

# Make imports robust for:
#   python src/build_reference_bank.py
#   python -m src.build_reference_bank
try:
    from src.profiles import PROFILES, profile_to_dict
except ModuleNotFoundError:
    sys.path.append(str(PROJECT_ROOT / "src"))
    from profiles import PROFILES, profile_to_dict


REFERENCE_DIR = PROJECT_ROOT / "reference_bank"
FEATURES_PATH = REFERENCE_DIR / "reference_bank_features.csv"


# ============================================================
# PROFILE ADAPTER
# ============================================================

def structured_profile_to_generation_dict(profile_spec: Any) -> Dict[str, Any]:
    """
    Converts the structured profile from profiles.py into the dictionary format
    used by the reference-bank generators.

    The reference bank must use the same official profiles as the experiments.
    It should not import old legacy profiles from run_experiment.py.
    """
    p = profile_to_dict(profile_spec)

    trend = p.get("trend_expected", "flat")
    if trend in {"flat", "no_strong_trend", None}:
        trend = "none"

    seasonality = p.get("seasonality_expected", "none")
    if seasonality in {None, False, "false", "no"}:
        seasonality = "none"

    expected_anomalies = int(p.get("expected_anomalies", 0) or 0)
    anomaly_type = p.get("anomaly_type")

    anomalies = "point" if expected_anomalies > 0 else "none"

    return {
        "profile_id": p["profile_id"],
        "description": p.get("profile_description", ""),
        "length": int(p["expected_length"]),
        "frequency": "daily" if p.get("frequency") == "D" else p.get("frequency"),
        "start_date": p.get("start_date", "2024-01-01"),
        "min_value": float(p.get("value_min", 0.0)),
        "max_value": float(p.get("value_max", 1000.0)),
        "trend": trend,
        "seasonality": seasonality,
        "seasonality_period": p.get("seasonality_period"),
        "noise": p.get("noise_expected", "medium"),
        "anomalies": anomalies,
        "expected_anomaly_count": expected_anomalies,
        "anomaly_type": anomaly_type,
        "anomaly_threshold": 200.0,
        "normal_min_value": 80.0,
        "normal_max_value": 130.0,
        "no_strong_trend_threshold": float(p.get("flat_trend_threshold", 0.30)),
    }


# ============================================================
# BASIC FEATURE FUNCTIONS
# ============================================================

def compute_slope(values: np.ndarray | list[float]) -> float:
    values = np.asarray(values, dtype=float)

    if len(values) < 2 or np.isnan(values).any():
        return np.nan

    x = np.arange(len(values))
    return float(np.polyfit(x, values, 1)[0])


def compute_weekly_autocorrelation(values: np.ndarray | list[float]) -> float:
    values = np.asarray(values, dtype=float)

    if len(values) < 14:
        return np.nan

    y1 = values[:-7]
    y2 = values[7:]

    if np.std(y1) == 0 or np.std(y2) == 0:
        return 0.0

    corr = np.corrcoef(y1, y2)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def enforce_low_global_trend(values: np.ndarray, threshold: float = 0.30) -> np.ndarray:
    """
    Removes excessive global linear trend when the target profile expects no
    strong trend. Threshold is interpreted relative to the value range.
    """
    values = np.asarray(values, dtype=float)

    if len(values) < 3:
        return values

    slope = compute_slope(values)
    value_range = np.max(values) - np.min(values)

    if np.isnan(slope) or value_range == 0:
        return values

    total_change_ratio = abs(slope) * (len(values) - 1) / value_range

    if total_change_ratio <= threshold:
        return values

    x = np.arange(len(values))
    corrected = values - slope * x
    corrected = corrected + (np.mean(values) - np.mean(corrected))

    return corrected


def get_chatts_inspired_attributes(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    ChatTS-inspired attribute-based description of each temporal profile:
    trend, periodicity, local fluctuations and noise.
    """
    trend = profile.get("trend", "none")
    seasonality = profile.get("seasonality", "none")
    anomalies = profile.get("anomalies", "none")
    noise = profile.get("noise", "medium")

    if trend == "increasing":
        trend_attribute = "increasing"
    elif trend == "decreasing":
        trend_attribute = "decreasing"
    else:
        trend_attribute = "stable"

    periodicity_attribute = "weekly" if seasonality == "weekly" else "none"
    local_fluctuation_attribute = "point_spikes" if anomalies == "point" else "none"

    return {
        "trend_attribute": trend_attribute,
        "periodicity_attribute": periodicity_attribute,
        "local_fluctuation_attribute": local_fluctuation_attribute,
        "noise_attribute": noise,
    }


def compute_reference_features(
    df: pd.DataFrame,
    profile: Dict[str, Any],
    reference_id: str,
    path: Path,
) -> Dict[str, Any]:
    values = pd.to_numeric(df["value"], errors="coerce").dropna().values

    weekly_autocorrelation = (
        compute_weekly_autocorrelation(values)
        if profile.get("seasonality") == "weekly"
        else np.nan
    )

    anomaly_count = (
        int(np.sum(values >= profile.get("anomaly_threshold", 200)))
        if profile.get("anomalies") == "point"
        else np.nan
    )

    attributes = get_chatts_inspired_attributes(profile)

    return {
        "reference_id": reference_id,
        "profile_id": profile["profile_id"],
        "profile_description": profile["description"],
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "length": int(len(df)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "trend_slope": compute_slope(values),
        "weekly_autocorrelation": weekly_autocorrelation,
        "anomaly_count": anomaly_count,
        "trend": profile.get("trend", "none"),
        "seasonality": profile.get("seasonality", "none"),
        "anomalies": profile.get("anomalies", "none"),
        "noise": profile.get("noise", "medium"),
        "trend_attribute": attributes["trend_attribute"],
        "periodicity_attribute": attributes["periodicity_attribute"],
        "local_fluctuation_attribute": attributes["local_fluctuation_attribute"],
        "noise_attribute": attributes["noise_attribute"],
    }


# ============================================================
# REFERENCE SERIES GENERATION
# ============================================================

def make_timestamps(profile: Dict[str, Any]) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.to_datetime(profile["start_date"]),
        periods=profile["length"],
        freq="D",
    )


def get_noise_scale(profile: Dict[str, Any], rng: np.random.Generator) -> float:
    noise = profile.get("noise", "medium")

    if noise == "low":
        return float(rng.uniform(0.8, 2.0))
    if noise == "high":
        return float(rng.uniform(5.0, 10.0))
    return float(rng.uniform(2.0, 5.0))


def generate_stable_series(profile: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    length = profile["length"]
    base_level = rng.uniform(95, 110)
    noise_scale = get_noise_scale(profile, rng)

    values = base_level + rng.normal(0, noise_scale, length)

    threshold = profile.get("no_strong_trend_threshold", 0.30)
    values = enforce_low_global_trend(values, threshold=threshold)

    return values


def generate_increasing_series(profile: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    length = profile["length"]
    start = rng.uniform(80, 120)
    end = start + rng.uniform(40, 110)

    x = np.linspace(0, 1, length)
    noise_scale = get_noise_scale(profile, rng)
    values = start + (end - start) * x + rng.normal(0, noise_scale, length)

    return values


def generate_decreasing_series(profile: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    length = profile["length"]
    start = rng.uniform(190, 240)
    end = rng.uniform(90, 130)

    x = np.linspace(0, 1, length)
    noise_scale = get_noise_scale(profile, rng)
    values = start + (end - start) * x + rng.normal(0, noise_scale, length)

    return values


def generate_weekly_series(profile: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    length = profile["length"]
    base_pattern = np.array([100, 120, 140, 130, 110, 90, 80], dtype=float)

    amplitude = rng.uniform(0.85, 1.15)
    level_shift = rng.uniform(-8, 8)
    noise_scale = get_noise_scale(profile, rng)

    values = []

    for i in range(length):
        weekday = i % 7
        week_offset = rng.normal(0, 1.5)

        value = (
            level_shift
            + amplitude * base_pattern[weekday]
            + week_offset
            + rng.normal(0, noise_scale)
        )
        values.append(value)

    values = np.asarray(values, dtype=float)

    threshold = profile.get("no_strong_trend_threshold", 0.30)
    values = enforce_low_global_trend(values, threshold=threshold)

    return values


def generate_increasing_weekly_series(profile: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    length = profile["length"]
    base_pattern = np.array([100, 120, 140, 130, 110, 90, 80], dtype=float)

    amplitude = rng.uniform(0.85, 1.15)
    level_shift = rng.uniform(-8, 8)
    weekly_increase = rng.uniform(5, 12)
    noise_scale = get_noise_scale(profile, rng)

    values = []

    for i in range(length):
        week = i // 7
        weekday = i % 7

        value = (
            level_shift
            + amplitude * base_pattern[weekday]
            + weekly_increase * week
            + rng.normal(0, noise_scale)
        )
        values.append(value)

    return np.asarray(values, dtype=float)


def insert_point_anomalies(
    values: np.ndarray,
    profile: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()

    expected_anomalies = int(profile.get("expected_anomaly_count", 0) or 0)
    if expected_anomalies <= 0:
        return values

    length = len(values)
    threshold = float(profile.get("anomaly_threshold", 200))
    anomaly_type = profile.get("anomaly_type", "spikes")

    if length > 10:
        candidate_positions = np.arange(3, length - 3)
    else:
        candidate_positions = np.arange(length)

    expected_anomalies = min(expected_anomalies, len(candidate_positions))

    positions = rng.choice(
        candidate_positions,
        size=expected_anomalies,
        replace=False,
    )

    for pos in positions:
        local_level = values[pos]

        if anomaly_type == "drops":
            values[pos] = max(profile["min_value"], local_level - rng.uniform(50, 100))
        elif anomaly_type == "mixed" and rng.random() < 0.35:
            # Keep mixed mostly spike-based because the current validator
            # primarily counts values above anomaly_threshold.
            values[pos] = max(profile["min_value"], local_level - rng.uniform(50, 100))
        else:
            values[pos] = max(threshold + rng.uniform(30, 100), local_level + rng.uniform(80, 150))

    return values


def generate_reference_series(profile: Dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    trend = profile.get("trend", "none")
    seasonality = profile.get("seasonality", "none")
    anomalies = profile.get("anomalies", "none")

    # Combine attributes instead of letting anomalies override trend/seasonality.
    if trend == "increasing" and seasonality == "weekly":
        values = generate_increasing_weekly_series(profile, rng)
    elif seasonality == "weekly":
        values = generate_weekly_series(profile, rng)
    elif trend == "increasing":
        values = generate_increasing_series(profile, rng)
    elif trend == "decreasing":
        values = generate_decreasing_series(profile, rng)
    else:
        values = generate_stable_series(profile, rng)

    if anomalies == "point":
        values = insert_point_anomalies(values, profile, rng)

    values = np.clip(values, profile["min_value"], profile["max_value"])
    values = np.round(values, 2)

    return pd.DataFrame({
        "timestamp": make_timestamps(profile),
        "value": values,
    })


# ============================================================
# MAIN BANK CREATION
# ============================================================

def build_reference_bank(
    n_per_profile: int = 30,
    seed: int = 42,
    overwrite: bool = True,
) -> None:
    rng = np.random.default_rng(seed)

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    feature_rows = []

    for profile_id, profile_spec in PROFILES.items():
        profile = structured_profile_to_generation_dict(profile_spec)

        profile_dir = REFERENCE_DIR / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating references for {profile_id}...")

        for i in range(1, n_per_profile + 1):
            reference_id = f"{profile_id}_ref_{i:03d}"
            path = profile_dir / f"{reference_id}.csv"

            if path.exists() and not overwrite:
                df = pd.read_csv(path)
            else:
                df = generate_reference_series(profile, rng)
                df.to_csv(path, index=False)

            features = compute_reference_features(
                df=df,
                profile=profile,
                reference_id=reference_id,
                path=path,
            )

            feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(FEATURES_PATH, index=False)

    print("\nReference bank created successfully.")
    print(f"Reference directory: {REFERENCE_DIR}")
    print(f"Feature file: {FEATURES_PATH}")
    print(f"Total reference series: {len(features_df)}")

    print("\nReferences by profile:")
    print(features_df.groupby("profile_id")["reference_id"].count().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Builds a programmatic reference bank for synthetic time series experiments."
    )

    parser.add_argument(
        "--n_per_profile",
        type=int,
        default=30,
        help="Number of reference series to generate per profile.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Do not overwrite existing reference CSV files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_reference_bank(
        n_per_profile=args.n_per_profile,
        seed=args.seed,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()