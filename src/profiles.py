from dataclasses import dataclass
from typing import Optional, Dict


@dataclass(frozen=True)
class TimeSeriesProfile:
    """
    Structured target profile for synthetic time series generation.

    The profile defines what the LLM is expected to generate.
    Validation metrics compare the generated series against these fields.
    """

    profile_id: str
    description: str

    # Temporal structure
    expected_length: int
    frequency: str
    start_date: str = "2024-01-01"
    timestamp_required: bool = False

    # Value constraints
    value_min: float = 0.0
    value_max: float = 1000.0

    # Temporal attributes
    trend_expected: str = "flat"          # flat, increasing, decreasing
    seasonality_expected: str = "none"   # none, weekly, monthly
    seasonality_period: Optional[int] = None
    expected_anomalies: int = 0
    anomaly_type: Optional[str] = None    # None, spikes, drops, mixed
    noise_expected: str = "medium"        # low, medium, high

    # Evaluation thresholds
    flat_trend_threshold: float = 0.30
    seasonality_threshold: float = 0.50
    anomaly_tolerance: int = 0


PROFILES: Dict[str, TimeSeriesProfile] = {
    "P1_20": TimeSeriesProfile(
        profile_id="P1_20",
        description=(
            "Short daily univariate time series with stable behaviour, "
            "no strong trend, no seasonality and no anomalies."
        ),
        expected_length=20,
        frequency="D",
        start_date="2024-01-01",
        timestamp_required=False,
        value_min=0.0,
        value_max=1000.0,
        trend_expected="flat",
        seasonality_expected="none",
        seasonality_period=None,
        expected_anomalies=0,
        anomaly_type=None,
        noise_expected="low",
        flat_trend_threshold=0.30,
        seasonality_threshold=0.30,
        anomaly_tolerance=0,
    ),

    "P2_28": TimeSeriesProfile(
        profile_id="P2_28",
        description=(
            "Daily time series with weekly seasonality, no strong global trend "
            "and no expected anomalies."
        ),
        expected_length=28,
        frequency="D",
        start_date="2024-01-01",
        timestamp_required=False,
        value_min=0.0,
        value_max=1000.0,
        trend_expected="flat",
        seasonality_expected="weekly",
        seasonality_period=7,
        expected_anomalies=0,
        anomaly_type=None,
        noise_expected="medium",
        flat_trend_threshold=0.30,
        seasonality_threshold=0.50,
        anomaly_tolerance=0,
    ),

    "P3_28": TimeSeriesProfile(
        profile_id="P3_28",
        description=(
            "Daily time series with increasing trend, weekly seasonality "
            "and two expected anomalies."
        ),
        expected_length=28,
        frequency="D",
        start_date="2024-01-01",
        timestamp_required=False,
        value_min=0.0,
        value_max=1000.0,
        trend_expected="increasing",
        seasonality_expected="weekly",
        seasonality_period=7,
        expected_anomalies=2,
        anomaly_type="spikes",
        noise_expected="medium",
        flat_trend_threshold=0.30,
        seasonality_threshold=0.50,
        anomaly_tolerance=1,
    ),

    "P4_30": TimeSeriesProfile(
        profile_id="P4_30",
        description=(
            "Daily time series with decreasing trend, no explicit seasonality "
            "and medium noise."
        ),
        expected_length=30,
        frequency="D",
        start_date="2024-01-01",
        timestamp_required=False,
        value_min=0.0,
        value_max=1000.0,
        trend_expected="decreasing",
        seasonality_expected="none",
        seasonality_period=None,
        expected_anomalies=0,
        anomaly_type=None,
        noise_expected="medium",
        flat_trend_threshold=0.30,
        seasonality_threshold=0.30,
        anomaly_tolerance=0,
    ),

    "P5_60": TimeSeriesProfile(
        profile_id="P5_60",
        description=(
            "Longer daily time series with weekly seasonality, moderate noise "
            "and no expected anomalies."
        ),
        expected_length=60,
        frequency="D",
        start_date="2024-01-01",
        timestamp_required=False,
        value_min=0.0,
        value_max=1000.0,
        trend_expected="flat",
        seasonality_expected="weekly",
        seasonality_period=7,
        expected_anomalies=0,
        anomaly_type=None,
        noise_expected="medium",
        flat_trend_threshold=0.30,
        seasonality_threshold=0.50,
        anomaly_tolerance=0,
    ),

    "P6_60": TimeSeriesProfile(
        profile_id="P6_60",
        description=(
            "Longer daily time series with increasing trend, weekly seasonality, "
            "higher noise and three expected anomalies."
        ),
        expected_length=60,
        frequency="D",
        start_date="2024-01-01",
        timestamp_required=False,
        value_min=0.0,
        value_max=1000.0,
        trend_expected="increasing",
        seasonality_expected="weekly",
        seasonality_period=7,
        expected_anomalies=3,
        anomaly_type="mixed",
        noise_expected="high",
        flat_trend_threshold=0.30,
        seasonality_threshold=0.50,
        anomaly_tolerance=1,
    ),
}


def get_profile(profile_id: str) -> TimeSeriesProfile:
    """
    Return a structured profile by profile_id.
    """
    if profile_id not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(
            f"Unknown profile_id '{profile_id}'. Available profiles: {available}"
        )

    return PROFILES[profile_id]


def profile_to_dict(profile: TimeSeriesProfile) -> dict:
    """
    Convert profile dataclass to dictionary for logging or CSV export.
    """
    return {
        "profile_id": profile.profile_id,
        "profile_description": profile.description,
        "expected_length": profile.expected_length,
        "frequency": profile.frequency,
        "start_date": profile.start_date,
        "timestamp_required": profile.timestamp_required,
        "value_min": profile.value_min,
        "value_max": profile.value_max,
        "trend_expected": profile.trend_expected,
        "seasonality_expected": profile.seasonality_expected,
        "seasonality_period": profile.seasonality_period,
        "expected_anomalies": profile.expected_anomalies,
        "anomaly_type": profile.anomaly_type,
        "noise_expected": profile.noise_expected,
        "flat_trend_threshold": profile.flat_trend_threshold,
        "seasonality_threshold": profile.seasonality_threshold,
        "anomaly_tolerance": profile.anomaly_tolerance,
    }