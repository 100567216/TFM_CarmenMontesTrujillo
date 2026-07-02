# src/profile_metrics.py

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _profile_to_dict(profile: Any) -> Dict[str, Any]:
    """
    Allows using either a dataclass profile or a plain dictionary.
    """
    if is_dataclass(profile):
        return asdict(profile)
    if isinstance(profile, dict):
        return profile
    raise TypeError("profile must be a dataclass or a dictionary.")


def _get(profile: Any, key: str, default: Any = None) -> Any:
    p = _profile_to_dict(profile)
    return p.get(key, default)


def safe_clip01(x: float) -> float:
    """
    Clip a value to the [0, 1] interval.
    """
    if x is None:
        return 0.0
    try:
        if not np.isfinite(x):
            return 0.0
    except TypeError:
        return 0.0
    return float(max(0.0, min(1.0, x)))


def to_numeric_values(values: Iterable[Any]) -> Tuple[List[float], int, int]:
    """
    Convert generated values to finite floats when possible.

    Returns
    -------
    numeric_values:
        Values successfully converted to finite floats.
    invalid_count:
        Number of generated items that could not be converted to a finite float.
    total_count:
        Total number of generated items received by the metric function.
    """
    raw_values = list(values) if values is not None else []
    numeric_values: List[float] = []
    invalid_count = 0

    for value in raw_values:
        try:
            v = float(value)
            if np.isfinite(v):
                numeric_values.append(v)
            else:
                invalid_count += 1
        except (TypeError, ValueError):
            invalid_count += 1

    return numeric_values, invalid_count, len(raw_values)


# ---------------------------------------------------------------------
# Structural validity metrics
# ---------------------------------------------------------------------

def compute_length_score(generated_length: int, expected_length: int) -> float:
    """
    Measures how close the generated number of items is to the expected length.
    """
    if expected_length <= 0:
        return 0.0
    score = 1.0 - abs(generated_length - expected_length) / expected_length
    return safe_clip01(score)


def compute_numeric_validity_score(
    valid_numeric_count: int,
    total_generated_count: int,
) -> float:
    """
    Measures the proportion of generated items that are valid numeric values.

    Length compliance is evaluated separately through length_score.
    """
    if total_generated_count <= 0:
        return 0.0
    return safe_clip01(valid_numeric_count / total_generated_count)


def compute_range_diagnostics(
    values: List[float],
    value_min: float,
    value_max: float,
) -> Dict[str, Any]:
    """
    Counts how many numeric generated values fall outside the expected range.
    """
    if len(values) == 0:
        return {
            "out_of_range_count": 0,
            "range_violation_rate": 1.0,
            "range_score": 0.0,
        }

    arr = np.asarray(values, dtype=float)
    out_of_range = (arr < value_min) | (arr > value_max)

    out_of_range_count = int(np.sum(out_of_range))
    range_violation_rate = float(out_of_range_count / len(arr))
    range_score = safe_clip01(1.0 - range_violation_rate)

    return {
        "out_of_range_count": out_of_range_count,
        "range_violation_rate": range_violation_rate,
        "range_score": range_score,
    }


# ---------------------------------------------------------------------
# Trend diagnostics
# ---------------------------------------------------------------------

def compute_spearman_trend(values: List[float]) -> float:
    """
    Computes monotonic association between time index and values.

    With only two points, Spearman is equivalent to checking the direction.
    This is useful when weekly seasonal series are aggregated into only two
    complete weekly means.
    """
    if len(values) < 2:
        return 0.0

    if len(values) == 2:
        if values[1] > values[0]:
            return 1.0
        if values[1] < values[0]:
            return -1.0
        return 0.0

    y = pd.Series(values, dtype="float64")
    x = pd.Series(np.arange(len(values)), dtype="float64")

    if y.std() == 0:
        return 0.0

    rho = x.corr(y, method="spearman")

    if rho is None or not np.isfinite(rho):
        return 0.0

    return float(rho)


def compute_linear_trend_strength(
    values: List[float],
    reference_range: Optional[float] = None,
) -> float:
    """
    Computes total linear change relative to a reference value range.

    For seasonal profiles, trend can be computed on period-level means, while
    normalization should use the range of the original series. This avoids
    exaggerating tiny changes between weekly averages.
    """
    if len(values) < 2:
        return 0.0

    arr = np.asarray(values, dtype=float)

    if reference_range is None:
        value_range = float(np.max(arr) - np.min(arr))
    else:
        value_range = float(reference_range)

    if value_range == 0:
        return 0.0

    x = np.arange(len(arr))
    slope, _ = np.polyfit(x, arr, deg=1)

    total_linear_change = abs(slope) * (len(arr) - 1)
    return float(total_linear_change / value_range)


def aggregate_period_means(values: List[float], period: Optional[int]) -> List[float]:
    """
    Aggregates a seasonal series into period-level means.

    Example: for weekly seasonality with period=7, this returns the average
    value of each complete week. This is useful for evaluating global trend
    without confusing weekly oscillations with trend.
    """
    if period is None or period <= 1 or len(values) < period:
        return values

    arr = np.asarray(values, dtype=float)
    n_complete_periods = len(arr) // period

    if n_complete_periods < 2:
        return values

    trimmed = arr[: n_complete_periods * period]
    period_matrix = trimmed.reshape(n_complete_periods, period)
    period_means = period_matrix.mean(axis=1)

    return period_means.tolist()


def compute_trend_score(
    values: List[float],
    trend_expected: str,
    flat_trend_threshold: float = 0.30,
    reference_range: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compares observed trend with expected trend.

    For increasing/decreasing profiles, it uses Spearman correlation. For flat
    profiles, it uses normalized linear trend strength.
    """
    trend_expected = (trend_expected or "flat").lower()

    if len(values) < 2:
        return {
            "trend_observed": 0.0,
            "trend_score": 1.0 if trend_expected in {"flat", "none", "no_strong_trend"} else 0.0,
        }

    rho = compute_spearman_trend(values)

    if trend_expected == "increasing":
        score = max(0.0, rho)
        observed = rho

    elif trend_expected == "decreasing":
        score = max(0.0, -rho)
        observed = rho

    elif trend_expected in {"flat", "none", "no_strong_trend"}:
        trend_strength = compute_linear_trend_strength(
            values=values,
            reference_range=reference_range,
        )
        score = 1.0 - min(trend_strength / flat_trend_threshold, 1.0)
        observed = trend_strength

    else:
        score = 0.0
        observed = rho

    return {
        "trend_observed": float(observed),
        "trend_score": safe_clip01(score),
    }


# ---------------------------------------------------------------------
# Seasonality diagnostics
# ---------------------------------------------------------------------

def autocorrelation_at_lag(values: List[float], lag: int) -> float:
    """
    Computes autocorrelation at a given lag.
    """
    if lag is None or lag <= 0:
        return 0.0

    if len(values) <= lag:
        return 0.0

    arr = np.asarray(values, dtype=float)
    x = arr[:-lag]
    y = arr[lag:]

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    corr = np.corrcoef(x, y)[0, 1]

    if not np.isfinite(corr):
        return 0.0

    return float(corr)


def remove_linear_trend(values: List[float]) -> np.ndarray:
    """
    Removes a simple linear trend.
    """
    arr = np.asarray(values, dtype=float)

    if len(arr) < 3:
        return arr - np.mean(arr) if len(arr) > 0 else arr

    x = np.arange(len(arr))
    slope, intercept = np.polyfit(x, arr, deg=1)
    fitted = slope * x + intercept

    return arr - fitted


def seasonal_amplitude_ratio(values: List[float], lag: int) -> float:
    """
    Estimates seasonal strength using repeated phase means.

    The series is detrended first. Then values are grouped by phase within the
    candidate period. This avoids treating smooth monotonic trends as seasonality.
    """
    if lag is None or lag <= 1 or len(values) < 2 * lag:
        return 0.0

    arr = np.asarray(values, dtype=float)
    value_range = np.max(arr) - np.min(arr)

    if value_range == 0:
        return 0.0

    residuals = remove_linear_trend(values)

    phase_means = []
    for phase in range(lag):
        phase_values = residuals[phase::lag]
        if len(phase_values) > 0:
            phase_means.append(float(np.mean(phase_values)))

    if len(phase_means) < 2:
        return 0.0

    amplitude = max(phase_means) - min(phase_means)
    return float(abs(amplitude) / value_range)


def compute_max_unspecified_periodicity(values: List[float]) -> float:
    """
    Used when no seasonality is expected.

    In this TFM, the explicit seasonal profile is weekly seasonality. Therefore,
    for non-seasonal profiles we mainly test whether an unintended weekly pattern
    appears. This avoids confusing short local fluctuations with true seasonality.
    """
    if len(values) < 14:
        return 0.0

    return seasonal_amplitude_ratio(values, lag=7)


def compute_seasonality_score(
    values: List[float],
    seasonality_expected: str,
    seasonality_period: Optional[int],
    seasonality_threshold: float = 0.20,
    no_seasonality_soft_threshold: float = 0.05,
    no_seasonality_hard_threshold: float = 0.25,
) -> Dict[str, float]:
    """
    Compares observed periodicity with expected seasonality.

    If seasonality is expected, the score rewards repeated phase amplitude.
    If no seasonality is expected, the score only penalizes meaningful unintended
    weekly periodicity, not smooth trends or local fluctuations.
    """
    seasonality_expected = (seasonality_expected or "none").lower()

    if len(values) < 4:
        return {
            "seasonality_strength": 0.0,
            "seasonality_score": 1.0 if seasonality_expected in {"none", "false", "no"} else 0.0,
        }

    if seasonality_expected in {"none", "false", "no"}:
        unintended_strength = compute_max_unspecified_periodicity(values)

        if unintended_strength <= no_seasonality_soft_threshold:
            score = 1.0
        elif unintended_strength >= no_seasonality_hard_threshold:
            score = 0.0
        else:
            score = 1.0 - (
                (unintended_strength - no_seasonality_soft_threshold)
                / (no_seasonality_hard_threshold - no_seasonality_soft_threshold)
            )

        return {
            "seasonality_strength": float(unintended_strength),
            "seasonality_score": safe_clip01(score),
        }

    if seasonality_period is None:
        return {
            "seasonality_strength": 0.0,
            "seasonality_score": 0.0,
        }

    strength = seasonal_amplitude_ratio(values, int(seasonality_period))
    score = min(strength / seasonality_threshold, 1.0)

    return {
        "seasonality_strength": float(strength),
        "seasonality_score": safe_clip01(score),
    }


# ---------------------------------------------------------------------
# Anomaly and noise diagnostics
# ---------------------------------------------------------------------

def remove_expected_seasonality(
    values: List[float],
    seasonality_expected: str,
    seasonality_period: Optional[int],
) -> np.ndarray:
    """
    Removes linear trend and, when expected, the seasonal phase component.
    Used before computing residual noise and anomalies.
    """
    residuals = remove_linear_trend(values)
    seasonality_expected = (seasonality_expected or "none").lower()

    if (
        seasonality_expected in {"none", "false", "no"}
        or seasonality_period is None
        or seasonality_period <= 1
        or len(values) < 2 * seasonality_period
    ):
        return residuals

    period = int(seasonality_period)

    phase_means = {}
    for phase in range(period):
        phase_values = residuals[phase::period]
        phase_means[phase] = float(np.mean(phase_values)) if len(phase_values) else 0.0

    adjusted = np.asarray([
        residuals[i] - phase_means[i % period]
        for i in range(len(residuals))
    ])

    return adjusted


def robust_anomaly_count(
    values: List[float],
    seasonality_expected: str = "none",
    seasonality_period: Optional[int] = None,
    z_threshold: float = 4.0,
    min_magnitude_ratio: float = 0.15,
) -> int:
    """
    Counts anomalies using a robust z-score plus a minimum magnitude condition.

    A point is counted as an anomaly only if:
    1. its robust z-score is high, and
    2. its absolute residual is large enough relative to the value range.

    This avoids false positives in very smooth time series where the MAD can be
    extremely small.
    """
    if len(values) < 5:
        return 0

    arr = np.asarray(values, dtype=float)
    value_range = np.max(arr) - np.min(arr)

    if value_range == 0:
        return 0

    residuals = remove_expected_seasonality(
        values=values,
        seasonality_expected=seasonality_expected,
        seasonality_period=seasonality_period,
    )

    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))

    if mad == 0:
        return 0

    robust_z = 0.6745 * np.abs(residuals - median) / mad
    absolute_deviation = np.abs(residuals - median)
    magnitude_threshold = min_magnitude_ratio * value_range

    anomaly_mask = (
        (robust_z > z_threshold)
        & (absolute_deviation >= magnitude_threshold)
    )

    return int(np.sum(anomaly_mask))


def compute_anomaly_score(
    values: List[float],
    expected_anomalies: int,
    anomaly_tolerance: int = 0,
    seasonality_expected: str = "none",
    seasonality_period: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compares detected anomalies with expected anomalies.
    """
    detected = robust_anomaly_count(
        values=values,
        seasonality_expected=seasonality_expected,
        seasonality_period=seasonality_period,
    )

    expected_anomalies = int(expected_anomalies or 0)
    anomaly_tolerance = int(anomaly_tolerance or 0)
    difference = abs(detected - expected_anomalies)

    if difference <= anomaly_tolerance:
        score = 1.0
    elif expected_anomalies == 0:
        score = 1.0 - min(detected / 2.0, 1.0)
    else:
        score = 1.0 - min(difference / expected_anomalies, 1.0)

    return {
        "detected_anomalies": int(detected),
        "expected_anomalies": expected_anomalies,
        "anomaly_difference": int(difference),
        "anomaly_score": safe_clip01(score),
    }


def compute_noise_ratio(
    values: List[float],
    seasonality_expected: str = "none",
    seasonality_period: Optional[int] = None,
) -> float:
    """
    Estimates residual variability relative to the value range.

    Linear trend is removed first. If seasonality is expected, the seasonal
    phase component is also removed. This avoids treating the expected weekly
    pattern as noise.
    """
    if len(values) < 4:
        return 0.0

    arr = np.asarray(values, dtype=float)
    value_range = np.max(arr) - np.min(arr)
    mean_abs = abs(float(np.mean(arr)))

    scale = max(value_range, mean_abs, 1e-8)

    residuals = remove_expected_seasonality(
        values=values,
        seasonality_expected=seasonality_expected,
        seasonality_period=seasonality_period,
    )

    ratio = np.std(residuals) / scale

    if not np.isfinite(ratio):
        return 0.0

    return float(ratio)


def compute_noise_score(
    values: List[float],
    noise_expected: str,
    seasonality_expected: str = "none",
    seasonality_period: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compares observed residual variability with expected noise level.
    """
    ratio = compute_noise_ratio(
        values=values,
        seasonality_expected=seasonality_expected,
        seasonality_period=seasonality_period,
    )

    noise_expected = (noise_expected or "medium").lower()

    expected_intervals = {
        "low": (0.00, 0.08),
        "medium": (0.03, 0.15),
        "high": (0.12, 0.35),
    }

    if noise_expected not in expected_intervals:
        return {
            "noise_ratio": float(ratio),
            "noise_score": 0.0,
        }

    low, high = expected_intervals[noise_expected]

    if low <= ratio <= high:
        score = 1.0
    elif ratio < low:
        if low == 0:
            score = 1.0
        else:
            score = 1.0 - min((low - ratio) / low, 1.0)
    else:
        score = 1.0 - min((ratio - high) / high, 1.0)

    return {
        "noise_ratio": float(ratio),
        "noise_score": safe_clip01(score),
    }


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------

def compute_profile_metrics(
    values: Iterable[Any],
    profile: Any,
    parse_success: int = 1,
    timestamp_score: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute profile-based evaluation metrics for one generated time series.

    The function compares observed properties of the generated sequence against
    the structured target profile. All component scores are normalized to [0, 1].
    """
    numeric_values, invalid_numeric_count, total_generated_count = to_numeric_values(values)

    expected_length = int(_get(profile, "expected_length"))
    value_min = float(_get(profile, "value_min"))
    value_max = float(_get(profile, "value_max"))

    trend_expected = _get(profile, "trend_expected", "flat")
    seasonality_expected = _get(profile, "seasonality_expected", "none")
    seasonality_period = _get(profile, "seasonality_period", None)
    expected_anomalies = int(_get(profile, "expected_anomalies", 0))
    noise_expected = _get(profile, "noise_expected", "medium")

    flat_trend_threshold = float(_get(profile, "flat_trend_threshold", 0.30))
    seasonality_threshold = float(_get(profile, "seasonality_threshold", 0.50))
    anomaly_tolerance = int(_get(profile, "anomaly_tolerance", 0))

    numeric_generated_length = len(numeric_values)

    # Structural scores
    length_score = compute_length_score(total_generated_count, expected_length)
    numeric_validity_score = compute_numeric_validity_score(
        valid_numeric_count=numeric_generated_length,
        total_generated_count=total_generated_count,
    )
    range_diag = compute_range_diagnostics(
        values=numeric_values,
        value_min=value_min,
        value_max=value_max,
    )

    # Temporal scores
    original_range = (
        float(np.max(numeric_values) - np.min(numeric_values))
        if len(numeric_values) > 0
        else 0.0
    )

    trend_values = numeric_values
    if seasonality_expected not in {None, "none", "false", "no"}:
        trend_values = aggregate_period_means(
            values=numeric_values,
            period=seasonality_period,
        )

    trend_diag = compute_trend_score(
        values=trend_values,
        trend_expected=trend_expected,
        flat_trend_threshold=flat_trend_threshold,
        reference_range=original_range,
    )

    seasonality_diag = compute_seasonality_score(
        values=numeric_values,
        seasonality_expected=seasonality_expected,
        seasonality_period=seasonality_period,
        seasonality_threshold=seasonality_threshold,
    )

    anomaly_diag = compute_anomaly_score(
        values=numeric_values,
        expected_anomalies=expected_anomalies,
        anomaly_tolerance=anomaly_tolerance,
        seasonality_expected=seasonality_expected,
        seasonality_period=seasonality_period,
    )

    noise_diag = compute_noise_score(
        values=numeric_values,
        noise_expected=noise_expected,
        seasonality_expected=seasonality_expected,
        seasonality_period=seasonality_period,
    )

    formal_validity_score = float(
        np.mean([
            float(parse_success),
            length_score,
            numeric_validity_score,
            range_diag["range_score"],
            safe_clip01(timestamp_score),
        ])
    )

    temporal_profile_score = float(
        np.mean([
            trend_diag["trend_score"],
            seasonality_diag["seasonality_score"],
            anomaly_diag["anomaly_score"],
            noise_diag["noise_score"],
        ])
    )

    profile_compliance_score = float(
        formal_validity_score * temporal_profile_score
    )

    formal_component_scores = [
        float(parse_success),
        length_score,
        numeric_validity_score,
        range_diag["range_score"],
        safe_clip01(timestamp_score),
    ]

    temporal_component_scores = [
        trend_diag["trend_score"],
        seasonality_diag["seasonality_score"],
        anomaly_diag["anomaly_score"],
        noise_diag["noise_score"],
    ]

    formal_component_min_score = float(np.min(formal_component_scores))
    temporal_component_min_score = float(np.min(temporal_component_scores))

    all_formal_components_ok = bool(formal_component_min_score >= 0.95)
    all_temporal_components_ok = bool(temporal_component_min_score >= 0.60)

    final_valid = int(
        parse_success == 1
        and length_score == 1.0
        and numeric_validity_score == 1.0
        and range_diag["range_score"] >= 0.95
        and safe_clip01(timestamp_score) >= 0.95
        and all_temporal_components_ok
    )

    diagnostics = {
        # Source / basic diagnostics
        "parse_success": int(parse_success),
        "generated_length": int(total_generated_count),
        "numeric_generated_length": int(numeric_generated_length),
        "expected_length": int(expected_length),
        "invalid_numeric_count": int(invalid_numeric_count),

        # Descriptive statistics over valid numeric values
        "min_generated": float(np.min(numeric_values)) if numeric_values else np.nan,
        "max_generated": float(np.max(numeric_values)) if numeric_values else np.nan,
        "mean_generated": float(np.mean(numeric_values)) if numeric_values else np.nan,
        "std_generated": float(np.std(numeric_values)) if numeric_values else np.nan,

        # Expected profile fields
        "value_min_expected": value_min,
        "value_max_expected": value_max,
        "trend_expected": trend_expected,
        "seasonality_expected": seasonality_expected,
        "seasonality_period": seasonality_period,
        "expected_anomalies": expected_anomalies,
        "noise_expected": noise_expected,

        # Structural metrics
        "length_score": length_score,
        "numeric_validity_score": numeric_validity_score,
        "timestamp_score": safe_clip01(timestamp_score),

        # Aggregate metrics
        "formal_validity_score": formal_validity_score,
        "temporal_profile_score": temporal_profile_score,
        "profile_compliance_score": profile_compliance_score,
        "final_valid": final_valid,
        "formal_component_min_score": formal_component_min_score,
        "temporal_component_min_score": temporal_component_min_score,
        "all_formal_components_ok": all_formal_components_ok,
        "all_temporal_components_ok": all_temporal_components_ok,
    }

    diagnostics.update(range_diag)
    diagnostics.update(trend_diag)
    diagnostics.update(seasonality_diag)
    diagnostics.update(anomaly_diag)
    diagnostics.update(noise_diag)

    return diagnostics