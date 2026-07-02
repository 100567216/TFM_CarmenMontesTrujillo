import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from src.profiles import (
        PROFILES as STRUCTURED_PROFILES,
        get_profile as get_structured_profile,
        profile_to_dict,
    )
    from src.profile_metrics import compute_profile_metrics
except ModuleNotFoundError:
    from profiles import (
        PROFILES as STRUCTURED_PROFILES,
        get_profile as get_structured_profile,
        profile_to_dict,
    )
    from profile_metrics import compute_profile_metrics


# ============================================================
# GENERAL CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"


def get_project_root() -> Path:
    """
    Returns the project root when this script is stored inside src/.

    This makes the script work when it is executed as:
        python src/run_experiment.py

    from the project root, while keeping paths predictable.
    """
    script_dir = Path(__file__).resolve().parent

    if script_dir.name == "src":
        return script_dir.parent

    return Path.cwd()


PROJECT_ROOT = get_project_root()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_REFERENCE_FEATURES_PATH = PROJECT_ROOT / "reference_bank" / "reference_bank_features.csv"

# These globals are configured at runtime from CLI arguments.
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
REFERENCE_FEATURES_PATH = DEFAULT_REFERENCE_FEATURES_PATH


def safe_slug(value: str) -> str:
    """Creates a filesystem-safe identifier for model names and batch ids."""
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.replace(":", "_").replace("/", "_").replace("\\", "_")
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def configure_runtime_paths(
    output_dir: str | Path | None = None,
    reference_features_path: str | Path | None = None,
    ollama_url: str | None = None,
) -> None:
    """Configures runtime paths without changing the experimental logic."""
    global OUTPUT_DIR, REFERENCE_FEATURES_PATH, OLLAMA_URL

    if output_dir is None:
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    else:
        OUTPUT_DIR = Path(output_dir)
        if not OUTPUT_DIR.is_absolute():
            OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if reference_features_path is None:
        REFERENCE_FEATURES_PATH = DEFAULT_REFERENCE_FEATURES_PATH
    else:
        REFERENCE_FEATURES_PATH = Path(reference_features_path)
        if not REFERENCE_FEATURES_PATH.is_absolute():
            REFERENCE_FEATURES_PATH = PROJECT_ROOT / REFERENCE_FEATURES_PATH

    if ollama_url:
        OLLAMA_URL = ollama_url


# ============================================================
# TARGET TEMPORAL PROFILES
# ============================================================

LEGACY_PROFILES = {
    # Compatibility layer only. The official profiles are defined in profiles.py.
    "P1_20": {
        "profile_id": "P1_20",
        "description": "Short daily stable series without trend, seasonality or anomalies - 20 points",
        "scenario": "from_scratch",
        "domain": "synthetic_sales",
        "length": 20,
        "frequency": "daily",
        "start_date": "2024-01-01",
        "value_column": "value",
        "timestamp_column": "timestamp",
        "trend": "none",
        "seasonality": "none",
        "seasonality_period": None,
        "noise": "low",
        "anomalies": "none",
        "min_value": 0,
        "max_value": 1000,
        "no_strong_trend_threshold": 0.30,
    },
    "P2_28": {
        "profile_id": "P2_28",
        "description": "Daily flat series with weekly seasonality - 28 points",
        "scenario": "from_scratch",
        "domain": "synthetic_sales",
        "length": 28,
        "frequency": "daily",
        "start_date": "2024-01-01",
        "value_column": "value",
        "timestamp_column": "timestamp",
        "trend": "none",
        "seasonality": "weekly",
        "seasonality_period": 7,
        "noise": "medium",
        "anomalies": "none",
        "min_value": 0,
        "max_value": 1000,
        "no_strong_trend_threshold": 0.30,
    },
    "P3_28": {
        "profile_id": "P3_28",
        "description": "Daily series with increasing trend, weekly seasonality and two anomalies - 28 points",
        "scenario": "from_scratch",
        "domain": "synthetic_sales",
        "length": 28,
        "frequency": "daily",
        "start_date": "2024-01-01",
        "value_column": "value",
        "timestamp_column": "timestamp",
        "trend": "increasing",
        "seasonality": "weekly",
        "seasonality_period": 7,
        "noise": "medium",
        "anomalies": "point",
        "expected_anomaly_count": 2,
        "anomaly_threshold": 200,
        "normal_min_value": 80,
        "normal_max_value": 180,
        "min_value": 0,
        "max_value": 1000,
    },
    "P4_30": {
        "profile_id": "P4_30",
        "description": "Daily series with decreasing trend and no seasonality - 30 points",
        "scenario": "from_scratch",
        "domain": "synthetic_sales",
        "length": 30,
        "frequency": "daily",
        "start_date": "2024-01-01",
        "value_column": "value",
        "timestamp_column": "timestamp",
        "trend": "decreasing",
        "seasonality": "none",
        "seasonality_period": None,
        "noise": "medium",
        "anomalies": "none",
        "min_value": 0,
        "max_value": 1000,
    },
    "P5_60": {
        "profile_id": "P5_60",
        "description": "Longer daily flat series with weekly seasonality - 60 points",
        "scenario": "from_scratch",
        "domain": "synthetic_sales",
        "length": 60,
        "frequency": "daily",
        "start_date": "2024-01-01",
        "value_column": "value",
        "timestamp_column": "timestamp",
        "trend": "none",
        "seasonality": "weekly",
        "seasonality_period": 7,
        "noise": "medium",
        "anomalies": "none",
        "min_value": 0,
        "max_value": 1000,
        "no_strong_trend_threshold": 0.30,
    },
    "P6_60": {
        "profile_id": "P6_60",
        "description": "Longer daily series with increasing trend, weekly seasonality, high noise and three anomalies - 60 points",
        "scenario": "from_scratch",
        "domain": "synthetic_sales",
        "length": 60,
        "frequency": "daily",
        "start_date": "2024-01-01",
        "value_column": "value",
        "timestamp_column": "timestamp",
        "trend": "increasing",
        "seasonality": "weekly",
        "seasonality_period": 7,
        "noise": "high",
        "anomalies": "point",
        "expected_anomaly_count": 3,
        "anomaly_threshold": 200,
        "normal_min_value": 80,
        "normal_max_value": 190,
        "min_value": 0,
        "max_value": 1000,
    },
}

def build_legacy_profile_from_structured(profile_spec, fallback_profile: dict | None = None) -> dict:
    """
    Converts the structured profile from src/profiles.py into the legacy
    dictionary format expected by the current prompt-building and validation
    functions.

    The structured profile is the official experimental specification.
    The legacy profile is only a compatibility layer for the existing code.
    """
    p = profile_to_dict(profile_spec)
    fallback_profile = fallback_profile or {}

    frequency = p.get("frequency", fallback_profile.get("frequency", "daily"))
    frequency_map = {
        "D": "daily",
        "H": "hourly",
        "W": "weekly",
        "M": "monthly",
    }
    frequency = frequency_map.get(str(frequency), frequency)

    trend = p.get("trend_expected", fallback_profile.get("trend", "none"))
    if trend in {"flat", "no_strong_trend", None}:
        trend = "none"

    seasonality = p.get(
        "seasonality_expected",
        fallback_profile.get("seasonality", "none"),
    )
    if seasonality in {None, False, "false", "no"}:
        seasonality = "none"

    expected_anomalies = int(
        p.get(
            "expected_anomalies",
            fallback_profile.get("expected_anomaly_count", 0),
        )
        or 0
    )

    anomaly_type = p.get("anomaly_type", fallback_profile.get("anomaly_type"))

    if expected_anomalies > 0 or anomaly_type in {"point", "spikes", "drops", "mixed"}:
        anomalies = "point"
    else:
        anomalies = "none"

    value_min = p.get("value_min", fallback_profile.get("min_value", 0))
    value_max = p.get("value_max", fallback_profile.get("max_value", 1000))

    profile = {
        "profile_id": p.get("profile_id"),
        "description": p.get(
            "profile_description",
            p.get("description", fallback_profile.get("description", "")),
        ),
        "scenario": fallback_profile.get("scenario", "from_scratch"),
        "domain": fallback_profile.get("domain", "synthetic_sales"),

        # Legacy naming used by current prompts and validators
        "length": int(p.get("expected_length", fallback_profile.get("length"))),
        "frequency": frequency,
        "start_date": p.get("start_date", fallback_profile.get("start_date", "2024-01-01")),
        "value_column": fallback_profile.get("value_column", "value"),
        "timestamp_column": fallback_profile.get("timestamp_column", "timestamp"),

        "trend": trend,
        "seasonality": seasonality,
        "seasonality_period": p.get(
            "seasonality_period",
            fallback_profile.get("seasonality_period"),
        ),
        "noise": p.get("noise_expected", fallback_profile.get("noise", "low")),
        "anomalies": anomalies,

        "min_value": value_min,
        "max_value": value_max,

        # Legacy anomaly fields
        "expected_anomaly_count": expected_anomalies,
        "anomaly_threshold": fallback_profile.get("anomaly_threshold", 200),
        "normal_min_value": fallback_profile.get("normal_min_value", 80),
        "normal_max_value": fallback_profile.get("normal_max_value", 130),

        # Legacy trend threshold
        "no_strong_trend_threshold": fallback_profile.get(
            "no_strong_trend_threshold",
            p.get("flat_trend_threshold", 1.0),
        ),
    }

    return profile


# ============================================================
# METHOD METADATA
# ============================================================
# Methods with an "-inspired" name are experimental adaptations.
# They are not official reproductions of the algorithms cited in the literature.

METHODS = {
    "prompt_only": {
        "method_name": "Prompt-only baseline",
        "method_family": "zero_shot_prompting",
        "literature_inspiration": (
            "General zero-shot prompting baseline for LLM-based synthetic data generation"
        ),
        "description": (
            "The LLM receives only a basic temporal specification and generates "
            "the complete time series without examples or advanced context selection."
        ),
        "method_status": "primary",  # Primary method, Table 3.5 in the thesis
    },

    "constraint_guided": {
        "method_name": "Constraint-guided prompting",
        "method_family": "constraint_based_prompting",
        "literature_inspiration": (
            "Structured prompting with explicit format, range, and temporal constraints"
        ),
        "description": (
            "The LLM receives explicit constraints about output format, number of "
            "observations, frequency, value range, trend, seasonality, and anomalies."
        ),
        "method_status": "primary",  # Primary method, Table 3.5 in the thesis
    },

    "canonical_few_shot": {
        "method_name": "Canonical few-shot prompting",
        "method_family": "few_shot_prompting",
        "literature_inspiration": (
            "Few-shot prompting with manually designed canonical examples"
        ),
        "description": (
            "The LLM receives small manually designed synthetic examples that illustrate "
            "the expected temporal pattern. These examples are not real data and are not "
            "selected from the reference bank."
        ),
        "method_status": "primary",  # Primary method, Table 3.5 in the thesis
    },

    "random_few_shot": {
        "method_name": "Random few-shot prompting",
        "method_family": "reference_bank_few_shot_prompting",
        "literature_inspiration": (
            "Few-shot prompting using examples sampled from a controlled reference bank"
        ),
        "description": (
            "The LLM receives examples randomly selected from the ChatTS-inspired "
            "attribute-based reference bank. This method acts as a baseline for "
            "comparing informed context-selection strategies."
        ),
        "method_status": "primary",  # Primary method, Table 3.5 in the thesis
    },

    "harmonic_knn": {
        "method_name": "HARMONIC-inspired kNN-guided prompting",
        "method_family": "knn_guided_context_selection",
        "literature_inspiration": (
            "Inspired by HARMONIC's construction of local kNN groups to capture "
            "relationships between similar tabular rows before instruction fine-tuning"
        ),
        "description": (
            "The LLM receives reference examples selected as nearest neighbors around "
            "a concrete anchor reference series in the attribute-based reference bank. "
            "This adapts HARMONIC's local kNN grouping idea to prompting over "
            "time-series features, but does not reproduce HARMONIC's instruction "
            "fine-tuning pipeline."
        ),
        "method_status": "primary",  # Primary method, Table 3.5 in the thesis
    },

    "tabgen_icl_residual": {
        "method_name": "TABGEN-ICL-style residual-aware prompting",
        "method_family": "residual_aware_context_selection",
        "literature_inspiration": (
            "Adapted from TABGEN-ICL's distance-based in-context example selection. "
            "The official method builds candidate subsets and selects the subset that "
            "minimizes the distributional distance between the original data and the "
            "combination of previous generations plus candidate examples."
        ),
        "description": (
            "The LLM receives reference examples selected from the attribute-based "
            "reference bank using a TABGEN-ICL-style distance-minimization criterion "
            "over time-series features. This adapts the official tabular selection "
            "principle to synthetic time series generation with prompting-only inference."
        ),
        "method_status": "primary",  # Primary method, Table 3.5 in the thesis
    },

    "value_only_controlled_scale": {
        "method_name": "Controlled value-only prompting",
        "method_family": "value_only_structured_generation",
        "literature_inspiration": (
            "Separation between LLM-based value generation and programmatic timestamp "
            "construction for stronger structural control"
        ),
        "description": (
            "The LLM generates only the numeric values of the time series, while "
            "timestamps are constructed programmatically. The prompt includes explicit "
            "scale, trend, seasonality, and anomaly constraints."
        ),
        "method_status": "primary",  # Primary method, Table 3.5 in the thesis
    },

    "chatts_attribute_prompting": {
        "method_name": "ChatTS-style attribute-conditioned prompting",
        "method_family": "attribute_conditioned_prompting",
        "literature_inspiration": (
            "Inspired by ChatTS attribute-based synthetic time series generation, "
            "where temporal series are described through interpretable attributes "
            "such as trend, periodicity, local fluctuations, and noise."
        ),
        "description": (
            "The LLM receives a structured attribute-based description of the target "
            "time series profile. The prompt explicitly separates trend, periodicity, "
            "local fluctuation, noise, scale, and output constraints. The LLM generates "
            "only numeric values, while timestamps are constructed programmatically."
        ),
        "method_status": "auxiliary",  # Auxiliary/diagnostic method, Table 3.5 in the thesis
    },

    "epic_grouped_attribute_few_shot": {
        "method_name": "EPIC-inspired grouped attribute few-shot prompting",
        "method_family": "grouped_attribute_few_shot_prompting",
        "literature_inspiration": (
            "Inspired by EPIC's prompt design principles for tabular synthetic data "
            "generation, especially grouped examples, consistent formatting, and "
            "explicit organization of in-context samples."
        ),
        "description": (
            "The LLM receives few-shot examples selected from the attribute-based "
            "reference bank and grouped by their temporal role: trend, periodicity, "
            "local fluctuation, and noise. This adapts EPIC's grouped prompting idea "
            "to synthetic time series generation."
        ),
        "method_status": "auxiliary",  # Auxiliary/diagnostic method, Table 3.5 in the thesis
    },

    "validator_feedback_refinement": {
        "method_name": "Validator-feedback refinement",
        "method_family": "generate_validate_repair",
        "literature_inspiration": (
            "Inspired by generate-validate-repair workflows and synthetic data generation "
            "pipelines where automatic validators provide feedback to improve LLM outputs."
        ),
        "description": (
            "The LLM first generates a value-only time series. The generated series is "
            "validated using the same automatic validation pipeline as the rest of the "
            "experiments. If the series fails, a second prompt is built with explicit "
            "feedback about the failed checks, and the LLM is asked to regenerate the "
            "sequence correcting those errors."
        ),
        "method_status": "auxiliary",  # Auxiliary/diagnostic method, Table 3.5 in the thesis
    },

    "cllm_generate_curate": {
        "method_name": "CLLM-inspired generate-and-curate",
        "method_family": "generate_and_curate",
        "literature_inspiration": (
            "Inspired by CLLM-style synthetic data generation workflows, where LLM-generated "
            "candidate samples are combined with data-centric curation or filtering to improve "
            "the quality of the final synthetic dataset."
        ),
        "description": (
            "The LLM generates multiple candidate value-only time series for the same target "
            "profile. Each candidate is parsed, curated and validated with the same automatic "
            "pipeline used for all experiments. The final selected series is the candidate "
            "with the best validation-based quality score."
        ),
        "method_status": "auxiliary",  # Auxiliary/diagnostic method, Table 3.5 in the thesis
    },
}

VALUE_ONLY_OUTPUT_METHODS = [
    "value_only_controlled_scale",
    "chatts_attribute_prompting",
    "epic_grouped_attribute_few_shot",
    "validator_feedback_refinement",
    "cllm_generate_curate",
    "random_few_shot",
    "harmonic_knn",
    "tabgen_icl_residual",
]


# ============================================================
# PROMPTS
# ============================================================


def describe_temporal_pattern(profile: dict) -> str:
    """
    Builds a human-readable temporal specification from the legacy profile.

    The order of the conditions matters: combined profiles must preserve all
    requested attributes. For example, a profile with weekly seasonality,
    increasing trend and point anomalies should not be reduced to only an
    anomaly profile.
    """
    trend = profile.get("trend", "none")
    seasonality = profile.get("seasonality", "none")
    anomalies = profile.get("anomalies", "none")
    expected_anomalies = profile.get("expected_anomaly_count", 0)
    anomaly_threshold = profile.get("anomaly_threshold", 200)
    normal_min = profile.get("normal_min_value", 80)
    normal_max = profile.get("normal_max_value", 180)

    lines = []

    if trend == "increasing":
        lines.extend([
            "- The series must show a smooth increasing global trend.",
            "- Values should generally become higher over time.",
        ])
    elif trend == "decreasing":
        lines.extend([
            "- The series must show a smooth decreasing global trend.",
            "- Values should generally become lower over time.",
        ])
    else:
        lines.extend([
            "- The series should not show a strong increasing or decreasing global trend.",
            "- The baseline level should remain broadly stable over time.",
        ])

    if seasonality == "weekly":
        lines.extend([
            "- The series must show a clear weekly seasonal pattern.",
            "- Values should repeat a similar pattern every 7 days.",
            "- Values for the same weekday across different weeks should be similar, while respecting the requested global trend.",
        ])
    else:
        lines.append("- The series should not contain a clear weekly seasonal pattern.")

    if anomalies == "point":
        lines.extend([
            f"- The series must contain exactly {expected_anomalies} isolated point anomalies.",
            f"- Point anomalies should be clearly higher than the normal values and greater than or equal to {anomaly_threshold}.",
            f"- Non-anomalous values should usually remain between {normal_min} and {normal_max}.",
            "- Anomalies should be isolated; do not create long blocks of anomalous values.",
        ])
    else:
        lines.append("- The series should not contain isolated point anomalies or extreme spikes.")

    noise = profile.get("noise", "low")
    if noise == "high":
        lines.append("- Noise may be relatively high, but it should not hide the requested trend or weekly pattern.")
    elif noise == "medium":
        lines.append("- Noise should be moderate and numerically coherent.")
    else:
        lines.append("- Noise should be low.")

    return "\n".join(lines) + "\n"


def get_canonical_examples(profile: dict) -> str:
    """
    Returns compact illustrative examples for canonical few-shot prompting.

    Examples are synthetic prototypes only. They are selected from profile
    attributes rather than hard-coded profile identifiers so that combined
    profiles remain coherent.
    """
    trend = profile.get("trend", "none")
    seasonality = profile.get("seasonality", "none")
    anomalies = profile.get("anomalies", "none")

    if anomalies == "point" and trend == "increasing" and seasonality == "weekly":
        return """
Example 1: increasing weekly pattern with two isolated point anomalies
[
  {"timestamp": "2023-01-01", "value": 95.0},
  {"timestamp": "2023-01-02", "value": 115.0},
  {"timestamp": "2023-01-03", "value": 135.0},
  {"timestamp": "2023-01-04", "value": 125.0},
  {"timestamp": "2023-01-05", "value": 105.0},
  {"timestamp": "2023-01-06", "value": 88.0},
  {"timestamp": "2023-01-07", "value": 78.0},
  {"timestamp": "2023-01-08", "value": 108.0},
  {"timestamp": "2023-01-09", "value": 128.0},
  {"timestamp": "2023-01-10", "value": 260.0},
  {"timestamp": "2023-01-11", "value": 138.0},
  {"timestamp": "2023-01-12", "value": 118.0},
  {"timestamp": "2023-01-13", "value": 100.0},
  {"timestamp": "2023-01-14", "value": 90.0},
  {"timestamp": "2023-01-15", "value": 120.0},
  {"timestamp": "2023-01-16", "value": 140.0},
  {"timestamp": "2023-01-17", "value": 160.0},
  {"timestamp": "2023-01-18", "value": 150.0},
  {"timestamp": "2023-01-19", "value": 130.0},
  {"timestamp": "2023-01-20", "value": 280.0},
  {"timestamp": "2023-01-21", "value": 102.0}
]
"""

    if anomalies == "point":
        return """
Example 1: stable series with two isolated point anomalies
[
  {"timestamp": "2023-01-01", "value": 100.0},
  {"timestamp": "2023-01-02", "value": 102.0},
  {"timestamp": "2023-01-03", "value": 98.0},
  {"timestamp": "2023-01-04", "value": 101.0},
  {"timestamp": "2023-01-05", "value": 99.0},
  {"timestamp": "2023-01-06", "value": 103.0},
  {"timestamp": "2023-01-07", "value": 100.0},
  {"timestamp": "2023-01-08", "value": 250.0},
  {"timestamp": "2023-01-09", "value": 101.0},
  {"timestamp": "2023-01-10", "value": 99.0},
  {"timestamp": "2023-01-11", "value": 102.0},
  {"timestamp": "2023-01-12", "value": 97.0},
  {"timestamp": "2023-01-13", "value": 260.0},
  {"timestamp": "2023-01-14", "value": 100.0}
]
"""

    if trend == "increasing" and seasonality == "weekly":
        return """
Example 1: weekly seasonal pattern with increasing trend
[
  {"timestamp": "2023-01-01", "value": 100.0},
  {"timestamp": "2023-01-02", "value": 120.0},
  {"timestamp": "2023-01-03", "value": 140.0},
  {"timestamp": "2023-01-04", "value": 130.0},
  {"timestamp": "2023-01-05", "value": 110.0},
  {"timestamp": "2023-01-06", "value": 90.0},
  {"timestamp": "2023-01-07", "value": 80.0},
  {"timestamp": "2023-01-08", "value": 110.0},
  {"timestamp": "2023-01-09", "value": 130.0},
  {"timestamp": "2023-01-10", "value": 150.0},
  {"timestamp": "2023-01-11", "value": 140.0},
  {"timestamp": "2023-01-12", "value": 120.0},
  {"timestamp": "2023-01-13", "value": 100.0},
  {"timestamp": "2023-01-14", "value": 90.0}
]
"""

    if trend == "decreasing":
        return """
Example 1: smooth decreasing trend
[
  {"timestamp": "2023-01-01", "value": 220.0},
  {"timestamp": "2023-01-02", "value": 216.0},
  {"timestamp": "2023-01-03", "value": 212.0},
  {"timestamp": "2023-01-04", "value": 208.0},
  {"timestamp": "2023-01-05", "value": 205.0},
  {"timestamp": "2023-01-06", "value": 201.0},
  {"timestamp": "2023-01-07", "value": 198.0}
]
"""

    if seasonality == "weekly":
        return """
Example 1: weekly seasonal pattern without strong trend
[
  {"timestamp": "2023-01-01", "value": 100.0},
  {"timestamp": "2023-01-02", "value": 120.0},
  {"timestamp": "2023-01-03", "value": 140.0},
  {"timestamp": "2023-01-04", "value": 130.0},
  {"timestamp": "2023-01-05", "value": 110.0},
  {"timestamp": "2023-01-06", "value": 90.0},
  {"timestamp": "2023-01-07", "value": 80.0},
  {"timestamp": "2023-01-08", "value": 102.0},
  {"timestamp": "2023-01-09", "value": 122.0},
  {"timestamp": "2023-01-10", "value": 142.0},
  {"timestamp": "2023-01-11", "value": 132.0},
  {"timestamp": "2023-01-12", "value": 112.0},
  {"timestamp": "2023-01-13", "value": 92.0},
  {"timestamp": "2023-01-14", "value": 82.0}
]
"""

    if trend == "increasing":
        return """
Example 1: smooth increasing trend
[
  {"timestamp": "2023-01-01", "value": 50.0},
  {"timestamp": "2023-01-02", "value": 52.0},
  {"timestamp": "2023-01-03", "value": 54.5},
  {"timestamp": "2023-01-04", "value": 57.0}
]

Example 2: smooth increasing trend
[
  {"timestamp": "2023-06-01", "value": 200.0},
  {"timestamp": "2023-06-02", "value": 203.0},
  {"timestamp": "2023-06-03", "value": 205.5},
  {"timestamp": "2023-06-04", "value": 209.0}
]
"""

    return """
Example 1: stable low-noise series without seasonality
[
  {"timestamp": "2023-01-01", "value": 100.0},
  {"timestamp": "2023-01-02", "value": 101.0},
  {"timestamp": "2023-01-03", "value": 99.5},
  {"timestamp": "2023-01-04", "value": 100.5},
  {"timestamp": "2023-01-05", "value": 100.0},
  {"timestamp": "2023-01-06", "value": 98.8},
  {"timestamp": "2023-01-07", "value": 101.2}
]
"""

def load_reference_features(profile_id: str) -> pd.DataFrame:
    if not REFERENCE_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Reference bank not found at {REFERENCE_FEATURES_PATH}. "
            "Run build_reference_bank.py first."
        )

    reference_df = pd.read_csv(REFERENCE_FEATURES_PATH)

    profile_refs = reference_df[reference_df["profile_id"] == profile_id].copy()

    if profile_refs.empty:
        raise ValueError(
            f"No references available for profile {profile_id}."
        )

    return profile_refs


def format_reference_example(row: pd.Series, example_number: int) -> tuple[str, dict]:
    path = Path(row["path"])

    if not path.exists():
        raise FileNotFoundError(f"Reference series not found: {path}")

    df_ref = pd.read_csv(path)
    values = [round(float(v), 2) for v in df_ref["value"].tolist()]

    feature_parts = [
        f"mean={row.get('mean'):.2f}",
        f"std={row.get('std'):.2f}",
        f"trend_slope={row.get('trend_slope'):.3f}",
    ]

    if not pd.isna(row.get("weekly_autocorrelation")):
        feature_parts.append(
            f"weekly_autocorrelation={row.get('weekly_autocorrelation'):.3f}"
        )

    if not pd.isna(row.get("anomaly_count")):
        feature_parts.append(
            f"anomaly_count={int(row.get('anomaly_count'))}"
        )

    features_text = ", ".join(feature_parts)

    example_text = f"""
Reference example {example_number} ({row["reference_id"]})
Features: {features_text}
Values:
{values}
"""

    example_metadata = {
        "reference_id": row["reference_id"],
        "path": row["path"],
        "mean": row.get("mean"),
        "std": row.get("std"),
        "trend_slope": row.get("trend_slope"),
        "weekly_autocorrelation": row.get("weekly_autocorrelation"),
        "anomaly_count": row.get("anomaly_count"),
    }

    return example_text, example_metadata


def select_random_reference_examples(
    profile: dict,
    n_examples: int,
    run_id: int,
    selection_seed: int,
) -> tuple[str, dict]:
    profile_id = profile["profile_id"]
    refs = load_reference_features(profile_id)

    effective_seed = selection_seed + run_id
    n_available = len(refs)
    n_selected = min(n_examples, n_available)

    selected_refs = refs.sample(
        n=n_selected,
        random_state=effective_seed,
        replace=False,
    ).reset_index(drop=True)

    example_texts = []
    selected_metadata = []

    for i, row in selected_refs.iterrows():
        example_text, example_metadata = format_reference_example(
            row=row,
            example_number=i + 1,
        )
        example_texts.append(example_text)
        selected_metadata.append(example_metadata)

    examples_block = "\n".join(example_texts)

    selection_info = {
        "selection_strategy": "random",
        "reference_bank_used": True,
        "profile_id": profile_id,
        "n_examples_requested": n_examples,
        "n_examples_selected": n_selected,
        "selection_seed": selection_seed,
        "effective_seed": effective_seed,
        "selected_reference_ids": [
            item["reference_id"] for item in selected_metadata
        ],
        "selected_reference_paths": [
            item["path"] for item in selected_metadata
        ],
        "selected_reference_metadata": selected_metadata,
    }

    return examples_block, selection_info

def build_prompt(profile: dict, method: str) -> str:
    if method == "prompt_only":
        return build_prompt_only(profile)

    if method == "constraint_guided":
        return build_constraint_guided_prompt(profile)

    if method == "canonical_few_shot":
        return build_canonical_few_shot_prompt(profile)

    if method == "value_only_controlled_scale":
        return build_value_only_controlled_scale_prompt(profile)

    raise ValueError(f"Unsupported method: {method}")

def build_prompt_with_metadata(
    profile: dict,
    method: str,
    run_id: int,
    n_examples: int,
    selection_seed: int,
    model: str | None = None,
    batch_id: str | None = None,
) -> tuple[str, dict]:
    default_selection_info = {
        "selection_strategy": None,
        "reference_bank_used": False,
        "profile_id": profile["profile_id"],
        "n_examples_requested": None,
        "n_examples_selected": None,
        "selection_seed": None,
        "effective_seed": None,
        "selected_reference_ids": [],
        "selected_reference_paths": [],
        "selected_reference_metadata": [],
    }

    if method == "random_few_shot":
        return build_random_few_shot_prompt(
            profile=profile,
            n_examples=n_examples,
            run_id=run_id,
            selection_seed=selection_seed,
        )
    
    if method == "harmonic_knn":
        return build_harmonic_knn_prompt(
            profile=profile,
            n_examples=n_examples,
            run_id=run_id,
            selection_seed=selection_seed,
        )
    
    if method == "tabgen_icl_residual":
        return build_tabgen_icl_residual_prompt(
            profile=profile,
            n_examples=n_examples,
            run_id=run_id,
            selection_seed=selection_seed,
            model=model,
            batch_id=batch_id,
        )
    
    if method == "chatts_attribute_prompting":
        return build_chatts_attribute_prompt(profile=profile)
    
    if method == "epic_grouped_attribute_few_shot":
        return build_epic_grouped_attribute_prompt(
            profile=profile,
            n_examples=n_examples,
            run_id=run_id,
            selection_seed=selection_seed,
        )
    
    if method == "validator_feedback_refinement":
        return build_validator_feedback_initial_prompt(profile=profile)
    
    if method == "cllm_generate_curate":
        return build_cllm_generate_curate_prompt(profile=profile)

    prompt = build_prompt(profile, method)
    return prompt, default_selection_info

def build_prompt_only(profile: dict) -> str:
    pattern_description = describe_temporal_pattern(profile)

    return f"""
Generate exactly {profile["length"]} daily time series observations.

Requirements:
- Start date: {profile["start_date"]}
- Frequency: daily
- One numeric value per day
- Values between {profile["min_value"]} and {profile["max_value"]}
{pattern_description}

Return only a valid JSON array.
No markdown.
No explanation.

Format:
[
  {{"timestamp": "2024-01-01", "value": 100.0}},
  {{"timestamp": "2024-01-02", "value": 120.0}}
]
"""


def build_constraint_guided_prompt(profile: dict) -> str:
    pattern_description = describe_temporal_pattern(profile)

    return f"""
You must generate a synthetic time series.

STRICT OUTPUT RULES:
1. Return ONLY a JSON array.
2. Do NOT include markdown.
3. Do NOT include explanations.
4. Do NOT include comments.
5. Do NOT include text before or after the JSON.
6. Generate EXACTLY {profile["length"]} objects.
7. Each object must contain only:
   - "timestamp"
   - "value"

TARGET PROFILE:
- Start date: {profile["start_date"]}
- Frequency: daily
- Number of observations: {profile["length"]}
- Seasonality: {profile.get("seasonality", "none")}
- Trend: {profile.get("trend", "none")}
- Noise: {profile.get("noise", "low")}
- Anomalies: {profile.get("anomalies", "none")}
- Minimum value: {profile["min_value"]}
- Maximum value: {profile["max_value"]}

TEMPORAL PATTERN:
{pattern_description}

TEMPORAL RULES:
- The first timestamp must be {profile["start_date"]}.
- Each next timestamp must increase by exactly one day.
- Values must be numeric.
- Values must be positive.
- Values must remain within the allowed range.
- Do not repeat timestamps.
- Do not skip dates.

EXPECTED JSON FORMAT:
[
  {{"timestamp": "2024-01-01", "value": 100.0}},
  {{"timestamp": "2024-01-02", "value": 120.0}}
]
"""


def build_canonical_few_shot_prompt(profile: dict) -> str:
    pattern_description = describe_temporal_pattern(profile)
    examples = get_canonical_examples(profile)

    return f"""
You are a synthetic time series generator.

Your task is to generate a new synthetic time series from scratch.

The target profile is:
- Number of observations: {profile["length"]}
- Start date: {profile["start_date"]}
- Frequency: daily
- Seasonality: {profile.get("seasonality", "none")}
- Trend: {profile.get("trend", "none")}
- Noise: {profile.get("noise", "low")}
- Anomalies: {profile.get("anomalies", "none")}
- Values must be numeric
- Values must be between {profile["min_value"]} and {profile["max_value"]}

Temporal pattern:
{pattern_description}

Below are canonical examples.
These examples are NOT real data. They only illustrate the expected pattern.

{examples}

Now generate a NEW time series following the target profile.

STRICT OUTPUT RULES:
- Return ONLY a valid JSON array.
- Do NOT include markdown.
- Do NOT include explanations.
- Generate EXACTLY {profile["length"]} objects.
- Each object must contain only:
  - "timestamp"
  - "value"
- The first timestamp must be {profile["start_date"]}.
- Each next timestamp must increase by exactly one day.
- Do not repeat timestamps.
- Do not skip dates.

Expected output format:
[
  {{"timestamp": "2024-01-01", "value": 100.0}},
  {{"timestamp": "2024-01-02", "value": 120.0}}
]
"""

def build_random_few_shot_prompt(
    profile: dict,
    n_examples: int,
    run_id: int,
    selection_seed: int,
) -> tuple[str, dict]:
    pattern_description = describe_temporal_pattern(profile)

    examples_block, selection_info = select_random_reference_examples(
        profile=profile,
        n_examples=n_examples,
        run_id=run_id,
        selection_seed=selection_seed,
    )

    prompt = f"""
You are a synthetic time series generator.

Your task is to generate a NEW synthetic daily time series from scratch.

You will receive randomly selected reference examples from a controlled synthetic reference bank.
These examples are NOT real data.
They are synthetic prototypes used only to illustrate possible temporal patterns.

TARGET PROFILE:
- Number of observations: {profile["length"]}
- Start date: {profile["start_date"]}
- Frequency: daily
- Seasonality: {profile.get("seasonality", "none")}
- Trend: {profile.get("trend", "none")}
- Noise: {profile.get("noise", "low")}
- Anomalies: {profile.get("anomalies", "none")}
- Values must be numeric
- Values must be between {profile["min_value"]} and {profile["max_value"]}

TEMPORAL PATTERN:
{pattern_description}

RANDOMLY SELECTED REFERENCE EXAMPLES:
{examples_block}

Now generate a NEW value sequence following the target profile.
Do NOT copy the reference examples exactly.
Use them only to understand the expected temporal behavior.

IMPORTANT:
- The reference examples are shown as value sequences.
- Your final output must also be a value sequence.
- Do NOT generate timestamps.
- Do NOT generate objects.
- The timestamps will be constructed programmatically after generation.
- The final output must contain exactly {profile["length"]} numbers.

STRICT OUTPUT RULES:
- Return ONLY a valid JSON array of numbers.
- Do NOT include markdown.
- Do NOT include explanations.
- Do NOT include comments.
- Do NOT include text before or after the JSON.
- Generate EXACTLY {profile["length"]} numeric values.
- Values must remain within the allowed range.
- Do not return objects.
- Do not return timestamp fields.

Expected output format:
[100.0, 120.0, 140.0, 130.0, 110.0]
"""

    return prompt, selection_info

def get_knn_feature_columns(refs: pd.DataFrame) -> list[str]:
    """
    Selects the feature columns that are meaningful for the current profile.
    Columns with only missing values are ignored.
    """
    candidate_columns = [
        "mean",
        "std",
        "trend_slope",
        "weekly_autocorrelation",
        "anomaly_count",
    ]

    feature_columns = []

    for col in candidate_columns:
        if col in refs.columns and refs[col].notna().any():
            feature_columns.append(col)

    if not feature_columns:
        raise ValueError("No valid feature columns available for kNN selection.")

    return feature_columns


def compute_profile_centroid(refs: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    """
    Uses the median feature vector as a robust representative target
    for the profile.
    """
    return refs[feature_columns].median(numeric_only=True)


def select_knn_reference_examples(
    profile: dict,
    n_examples: int,
    run_id: int,
    selection_seed: int,
) -> tuple[str, dict]:
    """
    HARMONIC-inspired kNN context selection.

    The official HARMONIC repository builds instruction data by applying kNN
    to the original table, obtaining groups of k + 1 similar rows, excluding
    the anchor row from its own nearest-neighbor set.

    This function adapts that idea to the time-series reference bank:
    - each reference series is represented by temporal features;
    - one anchor reference is selected reproducibly;
    - the nearest neighboring references around that anchor are used as
      in-context examples;
    - no instruction fine-tuning is performed.
    """
    profile_id = profile["profile_id"]
    refs = load_reference_features(profile_id)

    feature_columns = get_knn_feature_columns(refs)

    feature_matrix = refs[feature_columns].copy()

    # Fill missing values with median values.
    centroid = feature_matrix.median(numeric_only=True)
    feature_matrix = feature_matrix.fillna(centroid)

    # Standardize features.
    feature_std = feature_matrix.std(ddof=0).replace(0, 1.0)
    normalized_matrix = (feature_matrix - centroid) / feature_std

    n_available = len(refs)

    if n_available == 0:
        raise ValueError(f"No references available for profile {profile_id}.")

    # Select an anchor reproducibly. This mirrors the idea that HARMONIC
    # constructs local groups around concrete rows, not around a global centroid.
    rng = np.random.default_rng(selection_seed + run_id)
    anchor_position = int(rng.integers(0, n_available))

    anchor_vector = normalized_matrix.iloc[anchor_position].values

    distances = np.sqrt(
        ((normalized_matrix.values - anchor_vector) ** 2).sum(axis=1)
    )

    refs = refs.copy()
    refs["knn_distance"] = distances
    refs["is_anchor"] = False
    refs.loc[refs.index[anchor_position], "is_anchor"] = True

    anchor_reference_id = refs.iloc[anchor_position]["reference_id"]

    # Exclude anchor from neighbors, following HARMONIC's "exclude itself" logic.
    neighbor_refs = refs[refs["reference_id"] != anchor_reference_id].copy()
    neighbor_refs = neighbor_refs.sort_values(
        ["knn_distance", "reference_id"]
    ).reset_index(drop=True)

    n_selected = min(n_examples, len(neighbor_refs))
    selected_refs = neighbor_refs.head(n_selected).reset_index(drop=True)

    example_texts = []
    selected_metadata = []

    for i, row in selected_refs.iterrows():
        example_text, example_metadata = format_reference_example(
            row=row,
            example_number=i + 1,
        )
        example_metadata["knn_distance"] = float(row["knn_distance"])
        example_texts.append(example_text)
        selected_metadata.append(example_metadata)

    examples_block = "\n".join(example_texts)

    selection_info = {
        "selection_strategy": "harmonic_anchor_knn",
        "reference_bank_used": True,
        "profile_id": profile_id,
        "n_examples_requested": n_examples,
        "n_examples_selected": n_selected,
        "selection_seed": selection_seed,
        "effective_seed": selection_seed + run_id,
        "knn_feature_columns": feature_columns,
        "harmonic_selection_rule": (
            "select_anchor_reference_then_use_nearest_neighbors_excluding_anchor"
        ),
        "anchor_reference_id": anchor_reference_id,
        "anchor_position": anchor_position,
        "selected_reference_ids": [
            item["reference_id"] for item in selected_metadata
        ],
        "selected_reference_paths": [
            item["path"] for item in selected_metadata
        ],
        "selected_reference_distances": [
            item["knn_distance"] for item in selected_metadata
        ],
        "mean_neighbor_distance": float(
            np.mean([item["knn_distance"] for item in selected_metadata])
        ) if selected_metadata else np.nan,
        "selected_reference_metadata": selected_metadata,
    }

    return examples_block, selection_info

def build_harmonic_knn_prompt(
    profile: dict,
    n_examples: int,
    run_id: int,
    selection_seed: int,
) -> tuple[str, dict]:
    pattern_description = describe_temporal_pattern(profile)

    examples_block, selection_info = select_knn_reference_examples(
        profile=profile,
        n_examples=n_examples,
        run_id=run_id,
        selection_seed=selection_seed,
    )

    prompt = f"""
You are a synthetic time series generator.

Your task is to generate a NEW synthetic daily time series from scratch.

You will receive kNN-selected reference examples from a controlled synthetic reference bank.
These examples are NOT real data.
They are synthetic prototypes selected because their temporal features are close to the target profile.

This method is inspired by the neighbor-guided context idea used in HARMONIC,
but this implementation only uses prompting and does not fine-tune the LLM.

TARGET PROFILE:
- Number of observations: {profile["length"]}
- Start date: {profile["start_date"]}
- Frequency: daily
- Seasonality: {profile.get("seasonality", "none")}
- Trend: {profile.get("trend", "none")}
- Noise: {profile.get("noise", "low")}
- Anomalies: {profile.get("anomalies", "none")}
- Values must be numeric
- Values must be between {profile["min_value"]} and {profile["max_value"]}

TEMPORAL PATTERN:
{pattern_description}

KNN-SELECTED REFERENCE EXAMPLES:
{examples_block}

Now generate a NEW value sequence following the target profile.
Do NOT copy the reference examples exactly.
Use the kNN-selected examples only to understand the expected temporal behavior.

IMPORTANT:
- The reference examples are shown as value sequences.
- Your final output must also be a value sequence.
- Do NOT generate timestamps.
- Do NOT generate objects.
- The timestamps will be constructed programmatically after generation.
- The final output must contain exactly {profile["length"]} numbers.

STRICT OUTPUT RULES:
- Return ONLY a valid JSON array of numbers.
- Do NOT include markdown.
- Do NOT include explanations.
- Do NOT include comments.
- Do NOT include text before or after the JSON.
- Generate EXACTLY {profile["length"]} numeric values.
- Values must remain within the allowed range.
- Do not return objects.
- Do not return timestamp fields.

Expected output format:
[100.0, 120.0, 140.0, 130.0, 110.0]
"""

    return prompt, selection_info

def compute_profile_aware_features_from_df(df: pd.DataFrame, profile: dict) -> dict:
    """
    Computes the same type of feature representation used in the reference bank.
    Only profile-relevant features are included.
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    values = pd.to_numeric(df_sorted["value"], errors="coerce").dropna().values

    if len(values) == 0:
        raise ValueError("Cannot compute features from an empty generated series.")

    features = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "trend_slope": compute_basic_metrics(df_sorted).get("trend_slope"),
    }

    if profile.get("seasonality") == "weekly":
        features["weekly_autocorrelation"] = compute_basic_metrics(df_sorted).get(
            "weekly_autocorrelation"
        )

    if profile.get("anomalies") == "point":
        threshold = profile.get("anomaly_threshold", 200)
        features["anomaly_count"] = int(np.sum(values >= threshold))

    return features


def load_previous_generated_features(
    profile: dict,
    current_method: str = "tabgen_icl_residual",
    model: str | None = None,
    batch_id: str | None = None,
) -> pd.DataFrame:
    """
    Loads previous generated series for the same profile from the active output directory.

    For residual-aware selection, previous generations must belong to the same
    experimental condition. Therefore, when model and batch_id are provided, this
    function filters by both fields to avoid leakage across LLMs or experimental batches.
    Prefer valid generations. If there are no valid generations, use successful
    generations as a fallback.
    """
    profile_id = profile["profile_id"]

    candidate_rows = []
    valid_rows = []

    for metrics_path in sorted(OUTPUT_DIR.glob(f"{profile_id}_*_metrics.json")):
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if data.get("method") == current_method:
            continue

        if model is not None and data.get("model") != model:
            continue

        if batch_id is not None and data.get("batch_id") != batch_id:
            continue

        data_profile = data.get("profile", {})
        if data_profile.get("profile_id") != profile_id:
            continue

        if data.get("status") != "success":
            continue

        experiment_id = data.get("experiment_id")
        if not experiment_id:
            continue

        csv_path = OUTPUT_DIR / f"{experiment_id}_synthetic_series.csv"
        if not csv_path.exists():
            continue

        try:
            df_generated = pd.read_csv(csv_path)
            df_generated["timestamp"] = pd.to_datetime(
                df_generated["timestamp"],
                errors="coerce",
            )
            df_generated["value"] = pd.to_numeric(
                df_generated["value"],
                errors="coerce",
            )

            features = compute_profile_aware_features_from_df(
                df=df_generated,
                profile=profile,
            )

            row = {
                "experiment_id": experiment_id,
                "model": data.get("model"),
                "batch_id": data.get("batch_id"),
                "method": data.get("method"),
                "valid_series": data.get("validation", {}).get("valid_series"),
                **features,
            }

            candidate_rows.append(row)

            if row["valid_series"] is True:
                valid_rows.append(row)

        except Exception:
            continue

    if valid_rows:
        return pd.DataFrame(valid_rows)

    return pd.DataFrame(candidate_rows)


def js_divergence_1d(reference_values, candidate_values, n_bins=10, smoothing=1.0) -> float:
    """
    Jensen-Shannon distance for one numeric feature.

    This mirrors the distributional-distance spirit of TABGEN-ICL's
    compute_distance(), but adapts it to a compact time-series feature space.
    """
    reference_values = pd.to_numeric(pd.Series(reference_values), errors="coerce").dropna().values
    candidate_values = pd.to_numeric(pd.Series(candidate_values), errors="coerce").dropna().values

    if len(reference_values) == 0 or len(candidate_values) == 0:
        return 1.0

    if np.all(reference_values == reference_values[0]):
        # Degenerate reference distribution.
        return 0.0 if np.all(candidate_values == reference_values[0]) else 1.0

    bins = np.histogram_bin_edges(reference_values, bins=min(n_bins, max(2, len(np.unique(reference_values)))))

    p, _ = np.histogram(reference_values, bins=bins)
    q, _ = np.histogram(candidate_values, bins=bins)

    p = p.astype(float) + smoothing
    q = q.astype(float) + smoothing

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    def kl_divergence(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    return float(np.sqrt(max(jsd, 0.0)))


def compute_feature_distribution_distance(
    reference_features: pd.DataFrame,
    candidate_features: pd.DataFrame,
    feature_columns: list[str],
    n_bins: int = 10,
) -> float:
    """
    Average Jensen-Shannon distance across profile-relevant temporal features.

    Official TABGEN-ICL computes distributional distances between original
    data and generated+candidate data. Here we do the same idea over
    time-series feature vectors.
    """
    distances = []

    for col in feature_columns:
        if col not in reference_features.columns or col not in candidate_features.columns:
            continue

        if reference_features[col].notna().sum() == 0:
            continue

        dist = js_divergence_1d(
            reference_values=reference_features[col],
            candidate_values=candidate_features[col],
            n_bins=n_bins,
        )
        distances.append(dist)

    if not distances:
        return float("inf")

    return float(np.mean(distances))


def build_candidate_reference_subsets(
    refs: pd.DataFrame,
    feature_columns: list[str],
    n_examples: int,
    run_id: int,
    selection_seed: int,
    max_candidates: int = 30,
) -> list[pd.DataFrame]:
    """
    Builds candidate subsets in the spirit of TABGEN-ICL's get_indices().

    TABGEN-ICL samples candidate index groups from a randomly selected column.
    For time-series features, we create candidate subsets by quantile bins
    over profile-relevant feature columns.
    """
    rng = np.random.default_rng(selection_seed + run_id)
    candidate_subsets = []

    for feature in feature_columns:
        if feature not in refs.columns or refs[feature].notna().sum() < 2:
            continue

        feature_values = refs[feature].dropna()

        n_unique = feature_values.nunique()
        n_bins = min(5, n_unique)

        if n_bins < 2:
            continue

        try:
            # qcut creates candidate groups with similar number of references.
            bin_labels = pd.qcut(
                refs[feature],
                q=n_bins,
                duplicates="drop",
            )
        except ValueError:
            continue

        for _, group in refs.groupby(bin_labels, observed=False):
            if group.empty:
                continue

            if len(group) > n_examples:
                group = group.sample(
                    n=n_examples,
                    replace=False,
                    random_state=int(rng.integers(0, 1_000_000)),
                )

            candidate_subsets.append(group.copy())

    # Add a few fully random candidates as fallback/diversity,
    # similar to TABGEN-ICL starting from uniform examples.
    for _ in range(min(max_candidates // 3, 10)):
        sample_size = min(n_examples, len(refs))
        candidate_subsets.append(
            refs.sample(
                n=sample_size,
                replace=False,
                random_state=int(rng.integers(0, 1_000_000)),
            ).copy()
        )

    # Deduplicate subsets by selected reference ids.
    unique_subsets = []
    seen = set()

    for subset in candidate_subsets:
        key = tuple(sorted(subset["reference_id"].tolist()))
        if key not in seen:
            seen.add(key)
            unique_subsets.append(subset)

    if not unique_subsets:
        sample_size = min(n_examples, len(refs))
        unique_subsets = [
            refs.sample(
                n=sample_size,
                replace=False,
                random_state=selection_seed + run_id,
            ).copy()
        ]

    return unique_subsets[:max_candidates]


def select_residual_aware_reference_examples(
    profile: dict,
    n_examples: int,
    run_id: int,
    selection_seed: int,
    model: str | None = None,
    batch_id: str | None = None,
) -> tuple[str, dict]:
    """
    TABGEN-ICL-style residual-aware selection.

    Instead of selecting the farthest individual references, this function
    builds candidate reference subsets and selects the subset that minimizes
    the distributional distance between:

        reference bank features

    and

        previous generated features + candidate subset features

    This follows the main selection principle used in the official TABGEN-ICL
    repository, adapted to time-series feature vectors.
    """
    profile_id = profile["profile_id"]
    refs = load_reference_features(profile_id)

    generated_features = load_previous_generated_features(
        profile=profile,
        current_method="tabgen_icl_residual",
        model=model,
        batch_id=batch_id,
    )

    if generated_features.empty:
        examples_block, fallback_info = select_random_reference_examples(
            profile=profile,
            n_examples=n_examples,
            run_id=run_id,
            selection_seed=selection_seed,
        )

        fallback_info["selection_strategy"] = "tabgen_icl_fallback_random"
        fallback_info["fallback_used"] = True
        fallback_info["fallback_reason"] = "no_previous_generated_series"
        fallback_info["generated_context_count"] = 0
        fallback_info["tabgen_icl_selection_rule"] = "fallback_random"
        fallback_info["residual_filter_model"] = model
        fallback_info["residual_filter_batch_id"] = batch_id

        return examples_block, fallback_info

    feature_columns = get_knn_feature_columns(refs)

    feature_columns = [
        col for col in feature_columns
        if col in generated_features.columns and generated_features[col].notna().any()
    ]

    if not feature_columns:
        examples_block, fallback_info = select_random_reference_examples(
            profile=profile,
            n_examples=n_examples,
            run_id=run_id,
            selection_seed=selection_seed,
        )

        fallback_info["selection_strategy"] = "tabgen_icl_fallback_random"
        fallback_info["fallback_used"] = True
        fallback_info["fallback_reason"] = "no_common_feature_columns"
        fallback_info["generated_context_count"] = len(generated_features)
        fallback_info["tabgen_icl_selection_rule"] = "fallback_random"
        fallback_info["residual_filter_model"] = model
        fallback_info["residual_filter_batch_id"] = batch_id

        return examples_block, fallback_info

    candidate_subsets = build_candidate_reference_subsets(
        refs=refs,
        feature_columns=feature_columns,
        n_examples=n_examples,
        run_id=run_id,
        selection_seed=selection_seed,
        max_candidates=30,
    )

    candidate_scores = []

    for candidate_id, candidate_subset in enumerate(candidate_subsets):
        candidate_plus_generated = pd.concat(
            [
                generated_features[feature_columns],
                candidate_subset[feature_columns],
            ],
            ignore_index=True,
        )

        distance_score = compute_feature_distribution_distance(
            reference_features=refs,
            candidate_features=candidate_plus_generated,
            feature_columns=feature_columns,
            n_bins=10,
        )

        candidate_scores.append({
            "candidate_id": candidate_id,
            "distance_score": distance_score,
            "reference_ids": candidate_subset["reference_id"].tolist(),
            "candidate_subset": candidate_subset,
        })

    best_candidate = min(candidate_scores, key=lambda item: item["distance_score"])
    selected_refs = best_candidate["candidate_subset"].reset_index(drop=True)

    # If the selected group has fewer examples than requested, complete
    # with the closest remaining references to avoid too-short prompts.
    if len(selected_refs) < n_examples:
        missing = n_examples - len(selected_refs)
        already_selected = set(selected_refs["reference_id"].tolist())

        remaining_refs = refs[~refs["reference_id"].isin(already_selected)].copy()

        if not remaining_refs.empty:
            # Complete with deterministic random references.
            completion = remaining_refs.sample(
                n=min(missing, len(remaining_refs)),
                replace=False,
                random_state=selection_seed + run_id,
            )
            selected_refs = pd.concat([selected_refs, completion], ignore_index=True)

    example_texts = []
    selected_metadata = []

    for i, row in selected_refs.iterrows():
        example_text, example_metadata = format_reference_example(
            row=row,
            example_number=i + 1,
        )
        example_metadata["tabgen_icl_candidate_distance"] = float(best_candidate["distance_score"])
        example_texts.append(example_text)
        selected_metadata.append(example_metadata)

    examples_block = "\n".join(example_texts)

    selection_info = {
        "selection_strategy": "tabgen_icl_distance_minimization",
        "reference_bank_used": True,
        "profile_id": profile_id,
        "n_examples_requested": n_examples,
        "n_examples_selected": int(len(selected_refs)),
        "selection_seed": selection_seed,
        "effective_seed": selection_seed + run_id,
        "fallback_used": False,
        "fallback_reason": None,
        "generated_context_count": int(len(generated_features)),
        "residual_filter_model": model,
        "residual_filter_batch_id": batch_id,
        "generated_context_experiment_ids": generated_features.get(
            "experiment_id",
            pd.Series(dtype=str),
        ).tolist(),
        "residual_feature_columns": feature_columns,
        "tabgen_icl_selection_rule": (
            "minimize_distance_between_reference_bank_and_previous_generated_plus_candidate_subset"
        ),
        "n_candidate_subsets": len(candidate_subsets),
        "best_candidate_distance": float(best_candidate["distance_score"]),
        "mean_residual_distance": float(best_candidate["distance_score"]),
        "selected_reference_ids": [
            item["reference_id"] for item in selected_metadata
        ],
        "selected_reference_paths": [
            item["path"] for item in selected_metadata
        ],
        "selected_reference_residual_distances": [
            item["tabgen_icl_candidate_distance"] for item in selected_metadata
        ],
        "selected_reference_metadata": selected_metadata,
        "candidate_scores_summary": [
            {
                "candidate_id": item["candidate_id"],
                "distance_score": item["distance_score"],
                "reference_ids": item["reference_ids"],
            }
            for item in sorted(candidate_scores, key=lambda x: x["distance_score"])[:5]
        ],
    }

    return examples_block, selection_info

def build_tabgen_icl_residual_prompt(
    profile: dict,
    n_examples: int,
    run_id: int,
    selection_seed: int,
    model: str | None = None,
    batch_id: str | None = None,
) -> tuple[str, dict]:
    pattern_description = describe_temporal_pattern(profile)

    examples_block, selection_info = select_residual_aware_reference_examples(
        profile=profile,
        n_examples=n_examples,
        run_id=run_id,
        selection_seed=selection_seed,
        model=model,
        batch_id=batch_id,
    )

    prompt = f"""
You are a synthetic time series generator.

Your task is to generate a NEW synthetic daily time series from scratch.

You will receive residual-aware reference examples from a controlled synthetic reference bank.
These examples are NOT real data.
They are synthetic prototypes selected because previous generated series did not cover
their temporal feature region well.

This method is inspired by TABGEN-ICL's residual-aware in-context example selection,
but this implementation is adapted to synthetic time series generation and uses prompting only.

TARGET PROFILE:
- Number of observations: {profile["length"]}
- Start date: {profile["start_date"]}
- Frequency: daily
- Seasonality: {profile.get("seasonality", "none")}
- Trend: {profile.get("trend", "none")}
- Noise: {profile.get("noise", "low")}
- Anomalies: {profile.get("anomalies", "none")}
- Values must be numeric
- Values must be between {profile["min_value"]} and {profile["max_value"]}

TEMPORAL PATTERN:
{pattern_description}

RESIDUAL-AWARE REFERENCE EXAMPLES:
{examples_block}

Now generate a NEW value sequence following the target profile.
Do NOT copy the reference examples exactly.
Use the residual-aware examples only to understand temporal patterns that previous generations
may not have covered well.

IMPORTANT:
- The reference examples are shown as value sequences.
- Your final output must also be a value sequence.
- Do NOT generate timestamps.
- Do NOT generate objects.
- The timestamps will be constructed programmatically after generation.
- The final output must contain exactly {profile["length"]} numbers.

STRICT OUTPUT RULES:
- Return ONLY a valid JSON array of numbers.
- Do NOT include markdown.
- Do NOT include explanations.
- Do NOT include comments.
- Do NOT include text before or after the JSON.
- Generate EXACTLY {profile["length"]} numeric values.
- Values must remain within the allowed range.
- Do not return objects.
- Do not return timestamp fields.

Expected output format:
[100.0, 120.0, 140.0, 130.0, 110.0]
"""

    return prompt, selection_info

def get_chatts_style_attributes(profile: dict) -> dict:
    """
    Returns a ChatTS-style attribute representation of the target time series.

    This adapts the attribute-based generation idea to the controlled profiles
    used in this TFM.
    """
    trend = profile.get("trend", "none")
    seasonality = profile.get("seasonality", "none")
    anomalies = profile.get("anomalies", "none")
    noise = profile.get("noise", "low")

    if trend == "increasing":
        trend_attribute = {
            "type": "increasing",
            "description": (
                "The values should show a clear positive global direction over time."
            ),
        }
    elif trend == "decreasing":
        trend_attribute = {
            "type": "decreasing",
            "description": (
                "The values should show a clear negative global direction over time."
            ),
        }
    else:
        trend_attribute = {
            "type": "stable",
            "description": (
                "The series should not show a strong global upward or downward trend."
            ),
        }

    if seasonality == "weekly":
        periodicity_attribute = {
            "type": "weekly",
            "period": profile.get("seasonality_period", 7),
            "description": (
                "The sequence should show a repeated weekly pattern with period 7."
            ),
        }
    else:
        periodicity_attribute = {
            "type": "none",
            "period": None,
            "description": (
                "The sequence should not contain a strong periodic pattern."
            ),
        }

    if anomalies == "point":
        local_fluctuation_attribute = {
            "type": "point_spikes",
            "expected_count": profile.get("expected_anomaly_count", 2),
            "description": (
                "The sequence should contain a small number of isolated point anomalies."
            ),
        }
    else:
        local_fluctuation_attribute = {
            "type": "none",
            "expected_count": 0,
            "description": (
                "The sequence should not contain isolated extreme spikes."
            ),
        }

    noise_attribute = {
        "type": noise,
        "description": (
            "Noise should be low and should not hide the requested temporal pattern."
            if noise == "low"
            else "Noise should follow the level specified by the profile."
        ),
    }

    return {
        "trend_attribute": trend_attribute,
        "periodicity_attribute": periodicity_attribute,
        "local_fluctuation_attribute": local_fluctuation_attribute,
        "noise_attribute": noise_attribute,
    }


def build_chatts_attribute_guidance(profile: dict) -> str:
    """
    Operational guidance for ChatTS-style attribute prompting.

    This version is attribute-based rather than profile-id-based, so it remains
    valid when the final experimental profiles are renamed or extended.
    """
    trend = profile.get("trend", "none")
    seasonality = profile.get("seasonality", "none")
    anomalies = profile.get("anomalies", "none")
    expected = profile.get("expected_anomaly_count", 0)
    threshold = profile.get("anomaly_threshold", 200)

    lines = []

    if trend == "increasing":
        lines.append("- Increasing trend attribute must be visible: the overall level should rise over time.")
    elif trend == "decreasing":
        lines.append("- Decreasing trend attribute must be visible: the overall level should fall over time.")
    else:
        lines.append("- Stable trend attribute must dominate: avoid progressive upward or downward drift.")

    if seasonality == "weekly":
        lines.extend([
            "- Weekly periodicity must be visible with a repeating 7-day pattern.",
            "- Keep similar relative peaks and troughs for the same weekday across weeks.",
        ])
    else:
        lines.append("- Do not introduce a clear weekly periodic pattern.")

    if anomalies == "point":
        lines.extend([
            f"- Generate exactly {expected} isolated point anomalies.",
            f"- Anomalies should be clearly higher than normal values and at least {threshold}.",
            "- Keep non-anomalous values coherent with the trend and periodicity attributes.",
        ])
    else:
        lines.append("- Avoid isolated extreme spikes or drops.")

    return "\n".join(lines)

def build_chatts_attribute_prompt(profile: dict) -> tuple[str, dict]:
    """
    Builds a ChatTS-style attribute-conditioned prompt.

    The profile is represented as structured temporal attributes:
    trend, periodicity, local fluctuation, noise, scale and output constraints.
    """
    attributes = get_chatts_style_attributes(profile)
    pattern_description = describe_temporal_pattern(profile)

    trend_attr = attributes["trend_attribute"]
    periodicity_attr = attributes["periodicity_attribute"]
    fluctuation_attr = attributes["local_fluctuation_attribute"]
    noise_attr = attributes["noise_attribute"]
    attribute_guidance = build_chatts_attribute_guidance(profile)

    anomaly_guidance = ""
    if profile.get("anomalies") == "point":
        anomaly_guidance = """
    For point-spike profiles:
    - Most values should remain in a normal stable baseline range.
    - Only exactly 2 values should be clear spikes.
    - Do not create multiple high values around the spikes.
    """

    prompt = f"""
You are a synthetic time series generator.

Your task is to generate a NEW synthetic daily univariate time series from scratch.

This method uses an attribute-conditioned generation format inspired by ChatTS.
The time series is described through explicit temporal attributes.

TARGET PROFILE:
- Number of observations: {profile["length"]}
- Start date: {profile["start_date"]}
- Frequency: daily
- Minimum value: {profile["min_value"]}
- Maximum value: {profile["max_value"]}

ATTRIBUTE-BASED TEMPORAL SPECIFICATION:

1. TREND ATTRIBUTE
- Type: {trend_attr["type"]}
- Description: {trend_attr["description"]}

2. PERIODICITY ATTRIBUTE
- Type: {periodicity_attr["type"]}
- Period: {periodicity_attr["period"]}
- Description: {periodicity_attr["description"]}

3. LOCAL FLUCTUATION ATTRIBUTE
- Type: {fluctuation_attr["type"]}
- Expected anomaly count: {fluctuation_attr["expected_count"]}
- Description: {fluctuation_attr["description"]}

4. NOISE ATTRIBUTE
- Type: {noise_attr["type"]}
- Description: {noise_attr["description"]}

TEMPORAL PATTERN EXPLANATION:
{pattern_description}

ATTRIBUTE-SPECIFIC GENERATION GUIDANCE:
{attribute_guidance}

ANOMALY GUIDANCE:
{anomaly_guidance}

GENERATION OBJECTIVE:
Generate a value sequence that satisfies all the temporal attributes above.
The sequence must be realistic, numerically coherent, and consistent with the target profile.

IMPORTANT:
- Generate only numeric values.
- Do NOT generate timestamps.
- Timestamps will be constructed programmatically after generation.
- Do NOT return objects.
- Do NOT include explanations.
- Do NOT include markdown.
- Return only a valid JSON array of numbers.
- Generate exactly {profile["length"]} numeric values.
- All values must be between {profile["min_value"]} and {profile["max_value"]}.

STRICT OUTPUT FORMAT:
[100.0, 105.0, 112.0, 108.0, 115.0]
"""

    selection_info = {
        "selection_strategy": "chatts_attribute_conditioning",
        "reference_bank_used": False,
        "chatts_inspired": True,
        "trend_attribute": trend_attr["type"],
        "periodicity_attribute": periodicity_attr["type"],
        "local_fluctuation_attribute": fluctuation_attr["type"],
        "noise_attribute": noise_attr["type"],
    }

    return prompt, selection_info

def _safe_abs_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").abs()


def _select_epic_candidate(
    refs: pd.DataFrame,
    score: pd.Series,
    role: str,
    role_description: str,
    selected_ids: set,
    rng: np.random.Generator,
    ascending: bool = True,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Selects one reference for a specific EPIC-style attribute role.

    Instead of always taking the single best row, it samples from the top_k
    candidates to preserve some diversity across runs.
    """
    candidates = refs.copy()
    candidates["_epic_score"] = score

    candidates = candidates[
        ~candidates["reference_id"].isin(selected_ids)
    ].copy()

    candidates = candidates.dropna(subset=["_epic_score"])

    if candidates.empty:
        candidates = refs[
            ~refs["reference_id"].isin(selected_ids)
        ].copy()
        candidates["_epic_score"] = 0.0

    if candidates.empty:
        return pd.DataFrame()

    candidates = candidates.sort_values(
        ["_epic_score", "reference_id"],
        ascending=[ascending, True],
    )

    top_candidates = candidates.head(min(top_k, len(candidates)))

    selected_position = int(rng.integers(0, len(top_candidates)))
    selected = top_candidates.iloc[[selected_position]].copy()

    selected["epic_group_role"] = role
    selected["epic_group_description"] = role_description
    selected["epic_group_score"] = selected["_epic_score"].astype(float)

    return selected.drop(columns=["_epic_score"], errors="ignore")


def select_epic_grouped_attribute_examples(
    profile: dict,
    n_examples: int,
    run_id: int,
    selection_seed: int,
) -> tuple[str, dict]:
    """
    EPIC-inspired grouped attribute few-shot selection.

    The method selects examples from the reference bank according to explicit
    temporal roles: trend, periodicity, local fluctuation/anomalies, and noise.

    This adapts EPIC's idea of grouped and consistently formatted examples to
    the time-series setting.
    """
    profile_id = profile["profile_id"]
    refs = load_reference_features(profile_id).copy()

    if refs.empty:
        raise ValueError(f"No reference examples found for profile {profile_id}")

    rng = np.random.default_rng(selection_seed + run_id)

    selected_rows = []
    selected_ids = set()

    trend = profile.get("trend", "none")
    seasonality = profile.get("seasonality", "none")
    anomalies = profile.get("anomalies", "none")

    # 1. Trend role
    if "trend_slope" in refs.columns:
        if trend == "increasing":
            score = pd.to_numeric(refs["trend_slope"], errors="coerce")
            selected = _select_epic_candidate(
                refs=refs,
                score=score,
                role="trend_attribute",
                role_description=(
                    "This example illustrates the requested increasing trend attribute."
                ),
                selected_ids=selected_ids,
                rng=rng,
                ascending=False,
            )
        else:
            score = _safe_abs_series(refs["trend_slope"])
            selected = _select_epic_candidate(
                refs=refs,
                score=score,
                role="stable_trend_attribute",
                role_description=(
                    "This example illustrates a stable baseline with limited global drift."
                ),
                selected_ids=selected_ids,
                rng=rng,
                ascending=True,
            )

        if not selected.empty:
            selected_rows.append(selected)
            selected_ids.update(selected["reference_id"].tolist())

    # 2. Periodicity role
    if seasonality == "weekly" and "weekly_autocorrelation" in refs.columns:
        score = pd.to_numeric(refs["weekly_autocorrelation"], errors="coerce")
        selected = _select_epic_candidate(
            refs=refs,
            score=score,
            role="periodicity_attribute",
            role_description=(
                "This example illustrates the requested weekly periodicity attribute."
            ),
            selected_ids=selected_ids,
            rng=rng,
            ascending=False,
        )

        if not selected.empty:
            selected_rows.append(selected)
            selected_ids.update(selected["reference_id"].tolist())

    # 3. Local fluctuation / anomaly role
    if anomalies == "point" and "anomaly_count" in refs.columns:
        expected_count = profile.get("expected_anomaly_count", 2)

        anomaly_score = (
            pd.to_numeric(refs["anomaly_count"], errors="coerce")
            .sub(expected_count)
            .abs()
        )

        if "trend_slope" in refs.columns:
            anomaly_score = anomaly_score + _safe_abs_series(refs["trend_slope"])

        selected = _select_epic_candidate(
            refs=refs,
            score=anomaly_score,
            role="local_fluctuation_attribute",
            role_description=(
                "This example illustrates isolated point-spike anomalies while preserving "
                "a mostly stable baseline."
            ),
            selected_ids=selected_ids,
            rng=rng,
            ascending=True,
        )

        if not selected.empty:
            selected_rows.append(selected)
            selected_ids.update(selected["reference_id"].tolist())

    # 4. Noise / scale role, used as filler and for profiles without anomalies/seasonality.
    if "std" in refs.columns:
        score = pd.to_numeric(refs["std"], errors="coerce")

        selected = _select_epic_candidate(
            refs=refs,
            score=score,
            role="noise_attribute",
            role_description=(
                "This example illustrates the low-noise scale and local variation expected "
                "for the profile."
            ),
            selected_ids=selected_ids,
            rng=rng,
            ascending=True,
        )

        if not selected.empty:
            selected_rows.append(selected)
            selected_ids.update(selected["reference_id"].tolist())

    # Fill remaining slots if fewer than n_examples were selected.
    if selected_rows:
        selected_refs = pd.concat(selected_rows, ignore_index=True)
    else:
        selected_refs = pd.DataFrame()

    while len(selected_refs) < n_examples:
        remaining = refs[
            ~refs["reference_id"].isin(set(selected_refs.get("reference_id", [])))
        ].copy()

        if remaining.empty:
            break

        filler = remaining.sample(
            n=1,
            replace=False,
            random_state=int(rng.integers(0, 1_000_000)),
        ).copy()

        filler["epic_group_role"] = "filler_reference"
        filler["epic_group_description"] = (
            "This additional example provides extra reference context for the target profile."
        )
        filler["epic_group_score"] = np.nan

        selected_refs = pd.concat([selected_refs, filler], ignore_index=True)

    selected_refs = selected_refs.head(n_examples).reset_index(drop=True)

    example_blocks = []
    selected_metadata = []

    for i, row in selected_refs.iterrows():
        grouped_text, example_metadata = format_epic_compact_reference_example(
            row=row,
            example_number=i + 1,
        )

        example_blocks.append(grouped_text)
        selected_metadata.append(example_metadata)

    examples_block = "\n".join(example_blocks)

    selection_info = {
        "selection_strategy": "epic_grouped_attribute_few_shot",
        "reference_bank_used": True,
        "epic_inspired": True,
        "profile_id": profile_id,
        "n_examples_requested": n_examples,
        "n_examples_selected": int(len(selected_refs)),
        "selection_seed": selection_seed,
        "effective_seed": selection_seed + run_id,
        "selected_reference_ids": [
            item["reference_id"] for item in selected_metadata
        ],
        "selected_reference_paths": [
            item["path"] for item in selected_metadata
        ],
        "selected_reference_roles": [
            item["epic_group_role"] for item in selected_metadata
        ],
        "selected_reference_group_descriptions": [
            item["epic_group_description"] for item in selected_metadata
        ],
        "selected_reference_metadata": selected_metadata,
    }

    return examples_block, selection_info

def format_epic_compact_reference_example(row: pd.Series, example_number: int) -> tuple[str, dict]:
    """
    Compact EPIC-style example formatter.

    It keeps examples short and consistent so that the LLM focuses on the
    temporal role of the example instead of being distracted by metadata.
    """
    path = Path(row["path"])
    if not path.exists():
        path = Path(str(row["path"]).replace("/", "\\"))

    ref_df = pd.read_csv(path)
    values = pd.to_numeric(ref_df["value"], errors="coerce").dropna().tolist()

    # Keep the full value sequence, but in a compact one-line format.
    values_text = "[" + ", ".join([f"{float(v):.2f}" for v in values]) + "]"

    role = row.get("epic_group_role", "reference_example")
    role_description = row.get(
        "epic_group_description",
        "Reference example for the target profile.",
    )

    example_text = f"""
ATTRIBUTE GROUP {example_number}: {role}
Purpose: {role_description}
Values: {values_text}
"""

    metadata = {
        "reference_id": row["reference_id"],
        "path": row["path"],
        "epic_group_role": role,
        "epic_group_description": role_description,
        "epic_group_score": (
            float(row["epic_group_score"])
            if "epic_group_score" in row and pd.notna(row["epic_group_score"])
            else None
        ),
    }

    return example_text, metadata

def build_epic_grouped_attribute_prompt(
    profile: dict,
    n_examples: int,
    run_id: int,
    selection_seed: int,
) -> tuple[str, dict]:
    """
    Builds an EPIC-inspired grouped attribute few-shot prompt.

    The prompt organizes reference examples by temporal role, instead of
    presenting them as unstructured few-shot examples.
    """
    attributes = get_chatts_style_attributes(profile)
    pattern_description = describe_temporal_pattern(profile)

    examples_block, selection_info = select_epic_grouped_attribute_examples(
        profile=profile,
        n_examples=n_examples,
        run_id=run_id,
        selection_seed=selection_seed,
    )

    trend_attr = attributes["trend_attribute"]
    periodicity_attr = attributes["periodicity_attribute"]
    fluctuation_attr = attributes["local_fluctuation_attribute"]
    noise_attr = attributes["noise_attribute"]

    prompt = f"""
You are a synthetic time series generator.

Your task is to generate a NEW synthetic daily univariate time series from scratch.

This method uses EPIC-inspired grouped attribute few-shot prompting.
The examples below are grouped by the temporal attribute they are intended to illustrate.
Do not copy the examples. Use them to understand the target temporal behavior.

TARGET PROFILE:
- Number of observations: {profile["length"]}
- Start date: {profile["start_date"]}
- Frequency: daily
- Minimum value: {profile["min_value"]}
- Maximum value: {profile["max_value"]}

TARGET TEMPORAL ATTRIBUTES:
- Trend attribute: {trend_attr["type"]}
- Periodicity attribute: {periodicity_attr["type"]}
- Periodicity period: {periodicity_attr["period"]}
- Local fluctuation attribute: {fluctuation_attr["type"]}
- Expected anomaly count: {fluctuation_attr["expected_count"]}
- Noise attribute: {noise_attr["type"]}

TEMPORAL PATTERN EXPLANATION:
{pattern_description}

GROUPED ATTRIBUTE FEW-SHOT EXAMPLES:
{examples_block}

HOW TO USE THE EXAMPLES:
- Each group illustrates one temporal attribute.
- Combine the attributes into one new sequence.
- Keep the trend behavior of the trend group.
- Keep the periodicity behavior of the periodicity group when present.
- Keep the anomaly behavior of the local fluctuation group when present.
- Do not copy any example exactly.
- Return only the final JSON array of generated values.

IMPORTANT GENERATION RULE:
The examples are only demonstrations. Your output must be one new sequence for the TARGET PROFILE, not a mixture of copied examples.

OUTPUT REQUIREMENTS:
- Generate only numeric values.
- Do NOT generate timestamps.
- Timestamps will be constructed programmatically.
- Return only a valid JSON array of numbers.
- Generate exactly {profile["length"]} numeric values.
- All values must be between {profile["min_value"]} and {profile["max_value"]}.
- Do not include markdown, comments, or explanations.

FINAL ANSWER RULE:
Your entire response must be exactly one JSON array of numbers.
The first character of your response must be "[".
The last character of your response must be "]".
Do not write "Here is", "Values:", explanations, comments, or markdown.

Expected output format:
[100.0, 105.0, 112.0, 108.0, 115.0]
"""

    selection_info["trend_attribute"] = trend_attr["type"]
    selection_info["periodicity_attribute"] = periodicity_attr["type"]
    selection_info["local_fluctuation_attribute"] = fluctuation_attr["type"]
    selection_info["noise_attribute"] = noise_attr["type"]

    return prompt, selection_info

def build_validator_feedback_initial_prompt(profile: dict) -> tuple[str, dict]:
    """
    Initial prompt for validator-feedback refinement.

    The first generation is value-only and relatively controlled. If it fails,
    the validator will produce explicit feedback for a second generation.
    """
    pattern_description = describe_temporal_pattern(profile)

    prompt = f"""
You are a synthetic time series generator.

Your task is to generate a NEW synthetic daily univariate time series from scratch.

This is the INITIAL GENERATION step of a generate-validate-repair workflow.
After your output, an automatic validator will check whether the sequence satisfies
the target temporal profile.

TARGET PROFILE:
- Number of observations: {profile["length"]}
- Start date: {profile["start_date"]}
- Frequency: daily
- Trend: {profile.get("trend", "none")}
- Seasonality: {profile.get("seasonality", "none")}
- Seasonality period: {profile.get("seasonality_period", "none")}
- Noise: {profile.get("noise", "low")}
- Anomalies: {profile.get("anomalies", "none")}
- Values must be numeric
- Values must be between {profile["min_value"]} and {profile["max_value"]}

TEMPORAL PATTERN:
{pattern_description}

OUTPUT REQUIREMENTS:
- Generate only numeric values.
- Do NOT generate timestamps.
- Timestamps will be constructed programmatically.
- Return only a valid JSON array of numbers.
- Generate exactly {profile["length"]} numeric values.
- Do not include markdown, comments, or explanations.

Expected output format:
[100.0, 105.0, 112.0, 108.0, 115.0]
"""

    selection_info = {
        "selection_strategy": "validator_feedback_refinement",
        "reference_bank_used": False,
        "refinement_method": True,
        "initial_prompt_type": "value_only_temporal_profile",
    }

    return prompt, selection_info

def build_validator_refinement_prompt(
    profile: dict,
    initial_values: list,
    initial_validation: dict,
) -> str:
    """
    Builds the second prompt using explicit validator feedback.
    """
    pattern_description = describe_temporal_pattern(profile)
    feedback_text = build_validator_feedback_text(
        validation=initial_validation,
        profile=profile,
    )

    clean_values = [
        float(v)
        for v in initial_values
        if pd.notna(v)
    ]

    initial_values_text = "[" + ", ".join(
        [f"{v:.2f}" for v in clean_values]
    ) + "]"

    prompt = f"""
You are a synthetic time series generator.

You previously generated a sequence that failed validation.
Your task is to regenerate the FULL sequence from scratch, correcting the errors.

TARGET PROFILE:
- Number of observations: {profile["length"]}
- Start date: {profile["start_date"]}
- Frequency: daily
- Trend: {profile.get("trend", "none")}
- Seasonality: {profile.get("seasonality", "none")}
- Seasonality period: {profile.get("seasonality_period", "none")}
- Noise: {profile.get("noise", "low")}
- Anomalies: {profile.get("anomalies", "none")}
- Values must be numeric
- Values must be between {profile["min_value"]} and {profile["max_value"]}

TEMPORAL PATTERN:
{pattern_description}

PREVIOUS INVALID VALUES:
{initial_values_text}

VALIDATOR FEEDBACK:
{feedback_text}

IMPORTANT:
- Do not explain the correction.
- Do not return the previous sequence.
- Generate a new corrected sequence.
- Generate only numeric values.
- Do NOT generate timestamps.
- Timestamps will be constructed programmatically.
- Return only a valid JSON array of numbers.
- Generate exactly {profile["length"]} numeric values.
- Do not include markdown, comments, or explanations.

FINAL ANSWER RULE:
Your entire response must be exactly one JSON array of numbers.
The first character of your response must be "[".
The last character of your response must be "]".

Expected output format:
[100.0, 105.0, 112.0, 108.0, 115.0]
"""

    return prompt


def build_validator_feedback_text(validation: dict, profile: dict) -> str:
    """
    Converts validation results into explicit feedback for the LLM.
    """
    failed_checks = validation.get("failed_checks", [])

    if not failed_checks:
        return "The generated sequence passed all validation checks."

    feedback_lines = [
        "The previous generated sequence failed the automatic validation.",
        "You must regenerate the full sequence correcting the following issues:",
    ]

    if "length_ok" in failed_checks:
        feedback_lines.append(
            f"- Length is incorrect. Generate exactly {profile['length']} values."
        )

    if "daily_frequency_ok" in failed_checks:
        feedback_lines.append(
            "- Frequency/timestamps failed, but you should only generate values. "
            "The code will construct timestamps automatically."
        )

    if "range_ok" in failed_checks:
        feedback_lines.append(
            f"- Some values are outside the allowed range "
            f"[{profile['min_value']}, {profile['max_value']}]."
        )

    if "increasing_trend_ok" in failed_checks:
        feedback_lines.append(
            "- The sequence does not show a clear increasing global trend. "
            "Regenerate values so that the overall direction is increasing."
        )

    if "decreasing_trend_ok" in failed_checks:
        feedback_lines.append(
            "- The sequence does not show a clear decreasing global trend. "
            "Regenerate values so that the overall direction is decreasing."
        )

    if "no_strong_trend_ok" in failed_checks:
        feedback_lines.append(
            "- The sequence has a strong unwanted trend. "
            "Regenerate values with a stable baseline and avoid progressive upward or downward drift."
        )

    if "weekly_seasonality_ok" in failed_checks:
        feedback_lines.append(
            "- Weekly seasonality was not detected. "
            "Regenerate values with a clear repeated 7-day pattern."
        )

    if "anomaly_count_ok" in failed_checks:
        expected = profile.get("expected_anomaly_count", 2)
        threshold = profile.get("anomaly_threshold", 200)
        feedback_lines.append(
            f"- The anomaly count is incorrect. Generate exactly {expected} isolated "
            f"point anomalies with values >= {threshold}. "
            f"All other values must remain below {threshold}. "
            f"Do not remove the anomalies. Do not generate zero anomalies."
        )

    if validation.get("trend_slope") is not None:
        feedback_lines.append(
            f"- Previous trend slope: {validation.get('trend_slope'):.4f}."
        )

    if validation.get("weekly_autocorrelation") is not None:
        feedback_lines.append(
            f"- Previous weekly autocorrelation: {validation.get('weekly_autocorrelation'):.4f}."
        )

    if validation.get("anomaly_count") is not None:
        feedback_lines.append(
            f"- Previous anomaly count: {validation.get('anomaly_count')}."
        )

    return "\n".join(feedback_lines)

def build_cllm_generate_curate_prompt(profile: dict) -> tuple[str, dict]:
    """
    Base prompt metadata for CLLM-inspired generate-and-curate.

    The actual candidate prompts are built inside the generate-and-curate workflow,
    because each candidate receives a small candidate-specific instruction.
    """
    prompt = f"""
CLLM-inspired generate-and-curate method.

This base prompt is used only as metadata. Candidate-specific prompts are generated
inside the generate-and-curate workflow.
"""

    selection_info = {
        "selection_strategy": "cllm_generate_and_curate",
        "reference_bank_used": False,
        "cllm_inspired": True,
        "curation_strategy": "generate_multiple_candidates_and_select_by_validation_score",
    }

    return prompt, selection_info


def get_cllm_candidate_variation(profile: dict, candidate_id: int) -> str:
    """
    Candidate-specific instructions for CLLM-inspired generate-and-curate.

    The goal is to create candidates that explore slightly different valid ways
    of satisfying the same temporal profile. This is attribute-based rather than
    hard-coded by profile id.
    """
    trend = profile.get("trend", "none")
    seasonality = profile.get("seasonality", "none")
    anomalies = profile.get("anomalies", "none")
    expected = profile.get("expected_anomaly_count", 0)
    threshold = profile.get("anomaly_threshold", 200)

    if anomalies == "point":
        variations = {
            1: f"Generate exactly {expected} isolated high spikes above {threshold}; keep the remaining values coherent with the requested trend and seasonality.",
            2: f"Generate exactly {expected} point anomalies at different positions from candidate 1; avoid extra spikes.",
            3: f"Generate exactly {expected} clearly separated anomalies; keep non-anomalous values below {threshold}.",
            4: f"Generate exactly {expected} anomalies while preserving the requested weekly/trend structure.",
            5: f"Generate another valid candidate with exactly {expected} isolated anomalies and no sustained anomalous block.",
        }

    elif trend == "increasing" and seasonality == "weekly":
        variations = {
            1: "Generate a conservative increasing weekly pattern with moderate amplitude.",
            2: "Generate a slightly stronger increasing weekly pattern while preserving the same 7-day shape.",
            3: "Generate an increasing weekly sequence with small local fluctuations but no spikes.",
            4: "Generate another valid increasing weekly sequence with a different baseline.",
            5: "Generate a low-noise increasing weekly sequence with no anomalies.",
        }

    elif trend == "decreasing":
        variations = {
            1: "Generate a conservative smooth decreasing sequence with medium noise.",
            2: "Generate a slightly steeper decreasing sequence without weekly seasonality.",
            3: "Generate a decreasing sequence with small local fluctuations but no spikes.",
            4: "Generate another valid decreasing sequence with a different starting baseline.",
            5: "Generate a low-to-medium noise decreasing sequence with no anomalies.",
        }

    elif seasonality == "weekly":
        variations = {
            1: "Generate a stable weekly pattern with very similar week-to-week averages.",
            2: "Use a clear 7-day seasonal shape and repeat it with moderate noise.",
            3: "Generate a weekly sequence with a different baseline but no global trend.",
            4: "Generate another valid weekly sequence with similar same-weekday values.",
            5: "Generate a low-noise weekly sequence with no anomalies.",
        }

    elif trend == "increasing":
        variations = {
            1: "Generate a conservative increasing trend with smooth growth.",
            2: "Generate a slightly steeper increasing trend with low noise.",
            3: "Generate an increasing sequence with small local fluctuations but no spikes.",
            4: "Generate another valid increasing sequence with a different baseline.",
            5: "Generate a low-noise increasing sequence with no anomalies.",
        }

    else:
        variations = {
            1: "Generate a stable low-noise sequence around a constant baseline.",
            2: "Generate a stable sequence with a slightly different baseline.",
            3: "Generate a stable sequence with small local fluctuations.",
            4: "Generate another valid stable sequence without weekly seasonality.",
            5: "Generate a conservative valid sequence with no anomalies.",
        }

    return variations.get(
        candidate_id,
        "Generate an additional valid candidate with small variation in exact values."
    )

def build_value_only_controlled_scale_prompt(profile: dict) -> str:
    """
    Builds a controlled value-only prompt.

    The LLM generates only numeric values. Timestamps are created
    programmatically after generation.

    This prompt is intentionally explicit about:
    - exact length,
    - weekly cycles,
    - trend,
    - anomalies,
    - noise level,
    - avoiding example-length outputs such as 14 values.
    """
    length = int(profile["length"])
    trend = profile.get("trend", "none")
    seasonality = profile.get("seasonality", "none")
    anomalies = profile.get("anomalies", "none")
    noise = profile.get("noise", "medium")

    min_value = profile.get("min_value", 0)
    max_value = profile.get("max_value", 1000)

    expected_anomaly_count = int(profile.get("expected_anomaly_count", 0) or 0)
    anomaly_threshold = profile.get("anomaly_threshold", 200)
    normal_min_value = profile.get("normal_min_value", 80)
    normal_max_value = profile.get("normal_max_value", 130)

    weeks = length // 7
    extra_days = length % 7

    if noise == "low":
        noise_text = (
            "- Use low noise: values should vary only slightly around the requested pattern.\n"
            "- Avoid random-looking fluctuations.\n"
        )
    elif noise == "high":
        noise_text = (
            "- Use high noise: values may fluctuate noticeably around the requested pattern.\n"
            "- However, the main trend and weekly pattern must remain visible.\n"
            "- Noise should not create extra anomalies beyond the expected anomalies.\n"
        )
    else:
        noise_text = (
            "- Use medium noise: allow moderate local variation around the requested pattern.\n"
            "- Do not make the sequence perfectly deterministic unless the profile is very simple.\n"
        )

    length_reminder = f"""
LENGTH REQUIREMENT:
- Generate EXACTLY {length} numeric values.
- Count carefully before answering.
- Do NOT stop early.
- Do NOT return 10, 14, 20 or 28 values unless that is exactly the required length.
"""

    if seasonality == "weekly":
        length_reminder += f"""
WEEKLY LENGTH INTERPRETATION:
- The weekly period is 7.
- Exactly {length} values means {weeks} complete 7-day cycles and {extra_days} additional day(s).
- Do NOT stop after 14 values.
- Do NOT return only two weeks.
- Continue the weekly pattern until exactly {length} values are produced.
"""

    base_rules = f"""
Return a JSON array with exactly {length} numbers.

STRICT OUTPUT RULES:
- Return ONLY the JSON array.
- No text before or after.
- No markdown.
- No explanations.
- No timestamps.
- No objects.
- Only numbers.
- All values must be between {min_value} and {max_value}.

{length_reminder}
"""

    # ------------------------------------------------------------------
    # Case 1: increasing + weekly + anomalies
    # Used by P3_28 and P6_60.
    # ------------------------------------------------------------------
    if trend == "increasing" and seasonality == "weekly" and anomalies == "point":
        return f"""
{base_rules}

TARGET TEMPORAL PATTERN:
- The series must combine three properties:
  1. A clear weekly seasonal pattern with period 7.
  2. A smooth increasing global trend.
  3. Exactly {expected_anomaly_count} isolated point anomalies.

WEEKLY PATTERN:
- Repeat a similar 7-day shape across the sequence.
- A valid weekly shape could look like:
  [100, 120, 140, 130, 110, 90, 80]
- The same weekday in later weeks should usually be higher than in earlier weeks because of the increasing trend.

INCREASING TREND:
- Each complete week should have a higher average level than the previous week.
- Do not make the sequence flat.
- Do not make the sequence decreasing.

ANOMALIES:
- Include exactly {expected_anomaly_count} isolated point anomalies.
- These anomaly values must be much larger than the surrounding weekly pattern.
- Each anomaly must be at least {anomaly_threshold}.
- For example, if nearby normal values are around 120, an anomaly should be around 230 or higher.
- Do not treat normal weekly peaks as anomalies.
- Weekly peaks around 140 or 160 are NOT anomalies.
- The anomaly values must stand out clearly from the rest of the sequence.

NOISE:
{noise_text}

CONSTRUCTION GUIDANCE:
- First imagine the full weekly pattern for all {weeks} complete weeks.
- Then add the increasing baseline week by week.
- Then insert exactly {expected_anomaly_count} isolated spike anomalies.
- Finally check that the output contains exactly {length} numbers.

VALID OUTPUT EXAMPLE FORMAT:
[100.0, 120.0, 140.0, 130.0, 110.0, 90.0, 80.0]

Now return exactly {length} numbers.
"""

    # ------------------------------------------------------------------
    # Case 2: weekly + flat, no anomalies
    # Used by P2_28 and P5_60.
    # ------------------------------------------------------------------
    if seasonality == "weekly" and trend in {"none", "flat"} and anomalies == "none":
        return f"""
{base_rules}

TARGET TEMPORAL PATTERN:
- The series must show a clear weekly seasonal pattern with period 7.
- The global baseline should remain approximately stable.
- There should be no strong increasing or decreasing trend.
- There should be no anomalies.

WEEKLY PATTERN:
- Repeat a similar 7-day shape across the whole sequence.
- A valid weekly shape could look like:
  [100, 120, 140, 130, 110, 90, 80]
- Values for the same weekday across different weeks should be similar.
- Week averages should remain similar over time.

NO TREND:
- Do not make each week higher than the previous week.
- Do not make each week lower than the previous week.
- The average of the first complete week and the last complete week should be similar.

NO ANOMALIES:
- Do not create isolated extreme spikes or drops.
- Keep all values within the normal weekly pattern.

NOISE:
{noise_text}

CONSTRUCTION GUIDANCE:
- Generate the complete sequence by repeating and slightly varying the 7-day weekly shape.
- Continue until exactly {length} values are produced.
- If {length} is not divisible by 7, continue the next weekly cycle for the remaining {extra_days} value(s).
- Before answering, count the values and ensure there are exactly {length} numbers.

VALID OUTPUT EXAMPLE FORMAT:
[100.0, 120.0, 140.0, 130.0, 110.0, 90.0, 80.0]

Now return exactly {length} numbers.
"""

    # ------------------------------------------------------------------
    # Case 3: increasing + weekly, no anomalies.
    # Kept for completeness.
    # ------------------------------------------------------------------
    if trend == "increasing" and seasonality == "weekly" and anomalies == "none":
        return f"""
{base_rules}

TARGET TEMPORAL PATTERN:
- The series must show a clear weekly seasonal pattern with period 7.
- The series must also show a smooth increasing global trend.
- There should be no anomalies.

WEEKLY PATTERN:
- Repeat a similar 7-day shape across the whole sequence.
- Values for the same weekday should generally increase week by week.

INCREASING TREND:
- Each complete week should have a higher average level than the previous week.
- Do not make the sequence flat.
- Do not make it decreasing.

NOISE:
{noise_text}

CONSTRUCTION GUIDANCE:
- Generate the weekly pattern for all {weeks} complete weeks and {extra_days} additional day(s).
- Increase the baseline slightly each week.
- Count the output carefully.
- Return exactly {length} numbers.

VALID OUTPUT EXAMPLE FORMAT:
[100.0, 120.0, 140.0, 130.0, 110.0, 90.0, 80.0]

Now return exactly {length} numbers.
"""

    # ------------------------------------------------------------------
    # Case 4: decreasing trend, no seasonality, no anomalies.
    # Used by P4_30.
    # ------------------------------------------------------------------
    if trend == "decreasing" and seasonality == "none" and anomalies == "none":
        return f"""
{base_rules}

TARGET TEMPORAL PATTERN:
- The series must show a smooth decreasing trend.
- Values should generally decrease over time.
- There should be no weekly seasonality.
- There should be no anomalies.

DECREASING TREND:
- Start near 225.
- End near 100.
- Use a gradual downward movement.
- Avoid sudden jumps.
- Avoid exponential collapse.
- Do not make the sequence flat.
- Do not make it increasing.

NOISE:
{noise_text}

CONSTRUCTION GUIDANCE:
- Generate exactly {length} values from a high starting level to a lower ending level.
- Add only moderate local variation if required by the noise level.
- Before answering, count the values and ensure there are exactly {length} numbers.

VALID OUTPUT EXAMPLE FORMAT:
[225.0, 222.5, 220.0, 217.5, 215.0]

Now return exactly {length} numbers.
"""

    # ------------------------------------------------------------------
    # Case 5: increasing trend, no seasonality, no anomalies.
    # Kept for completeness.
    # ------------------------------------------------------------------
    if trend == "increasing" and seasonality == "none" and anomalies == "none":
        return f"""
{base_rules}

TARGET TEMPORAL PATTERN:
- The series must show a smooth increasing trend.
- Values should generally increase over time.
- There should be no weekly seasonality.
- There should be no anomalies.

INCREASING TREND:
- Start near 100.
- End near 225.
- Use a gradual upward movement.
- Avoid sudden jumps.
- Avoid exponential growth.

NOISE:
{noise_text}

CONSTRUCTION GUIDANCE:
- Generate exactly {length} values from a lower starting level to a higher ending level.
- Before answering, count the values and ensure there are exactly {length} numbers.

VALID OUTPUT EXAMPLE FORMAT:
[100.0, 102.5, 105.0, 107.5, 110.0]

Now return exactly {length} numbers.
"""

    # ------------------------------------------------------------------
    # Case 6: flat + anomalies, no seasonality.
    # Kept for robustness.
    # ------------------------------------------------------------------
    if trend in {"none", "flat"} and seasonality == "none" and anomalies == "point":
        return f"""
{base_rules}

TARGET TEMPORAL PATTERN:
- The series should be mostly stable around a normal level.
- The series must contain exactly {expected_anomaly_count} isolated point anomalies.
- There should be no weekly seasonality.
- There should be no strong increasing or decreasing trend.

NORMAL VALUES:
- Most non-anomalous values should stay between {normal_min_value} and {normal_max_value}.
- Normal values should fluctuate moderately around a stable baseline.

ANOMALIES:
- Include exactly {expected_anomaly_count} isolated point anomalies.
- Anomaly values should be greater than or equal to {anomaly_threshold}.
- Do not place anomalies next to each other.
- Do not create more than {expected_anomaly_count} anomalies.

NOISE:
{noise_text}

CONSTRUCTION GUIDANCE:
- Generate a stable baseline sequence.
- Insert exactly {expected_anomaly_count} isolated spike values.
- Count the output carefully.
- Return exactly {length} numbers.

VALID OUTPUT EXAMPLE FORMAT:
[100.0, 102.0, 98.0, 250.0, 101.0]

Now return exactly {length} numbers.
"""

    # ------------------------------------------------------------------
    # Default: stable, no seasonality, no anomalies.
    # Used by P1_20.
    # ------------------------------------------------------------------
    return f"""
{base_rules}

TARGET TEMPORAL PATTERN:
- The series should be stable.
- There should be no strong increasing or decreasing trend.
- There should be no weekly seasonality.
- There should be no anomalies.

STABLE BASELINE:
- Values must stay around a baseline near 100.
- Most values should be between 90 and 110.
- Do not use values below 80.
- Never use 0 values.
- Do not decrease toward zero.
- Do not create a downward trend.
- Do not repeat a short block exactly.
- Do not create a weekly or periodic pattern.

NOISE:
{noise_text}

CONSTRUCTION GUIDANCE:
- Generate exactly {length} values around a stable baseline.
- Use small irregular variation, not a repeated cycle.
- Before answering, count the values and ensure there are exactly {length} numbers.

VALID OUTPUT EXAMPLE FORMAT:
[100.0, 101.0, 99.0, 100.5, 98.8]

Now return exactly {length} numbers.
"""





# ============================================================
# OLLAMA CLIENT
# ============================================================

def call_ollama(prompt: str, model: str, temperature: float, num_predict: int, timeout: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["response"]


# ============================================================
# PARSING
# ============================================================

def extract_json_array(text: str):
    text = text.strip()

    parse_info = {
        "raw_json_valid": False,
        "parse_repaired": False,
        "parse_error": None,
        "repair_type": None,
    }

    try:
        data = json.loads(text)
        parse_info["raw_json_valid"] = True
        return data, parse_info
    except json.JSONDecodeError as e:
        parse_info["parse_error"] = str(e)

    if text.startswith("[") and not text.endswith("]"):
        repaired_text = text + "\n]"
        try:
            data = json.loads(repaired_text)
            parse_info["parse_repaired"] = True
            parse_info["repair_type"] = "added_closing_bracket"
            parse_info["parse_error"] = None
            return data, parse_info
        except json.JSONDecodeError as e:
            parse_info["parse_error"] = str(e)

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        json_text = match.group(0)
        try:
            data = json.loads(json_text)
            parse_info["parse_repaired"] = True
            parse_info["repair_type"] = "extracted_json_array_from_text"
            parse_info["parse_error"] = None
            return data, parse_info
        except json.JSONDecodeError as e:
            parse_info["parse_error"] = str(e)

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        data = [float(x) for x in numbers]
        parse_info["parse_repaired"] = True
        parse_info["repair_type"] = "extracted_numeric_values_from_text"
        parse_info["parse_error"] = None
        return data, parse_info

    raise ValueError("No valid JSON array was found in the model response.")


def json_to_dataframe(data) -> pd.DataFrame:
    df = pd.DataFrame(data)

    expected_columns = {"timestamp", "value"}
    if not expected_columns.issubset(df.columns):
        raise ValueError(
            f"Faltan columnas. Esperadas: {expected_columns}, obtenidas: {df.columns.tolist()}"
        )

    df = df[["timestamp", "value"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df

def parsed_output_to_dataframe(data, profile: dict):
    """
    Converts a parsed LLM JSON output into the standard dataframe format:
    timestamp, value.

    This function is tolerant because different prompt strategies may return:
    - [100, 101, 102]
    - [[100, 101, 102]]
    - [{"timestamp": "...", "value": 100}]
    - [{"value": 100}]
    - {"values": [100, 101, 102]}
    - {"synthetic_series": [100, 101, 102]}
    - {"output": {"values": [100, 101, 102]}}
    - {"series": [{"timestamp": "...", "value": 100}]}

    The normalization is logged so that format deviations remain traceable.
    """

    normalization_info = {
        "normalization_applied": False,
        "normalization_type": None,
        "original_output_type": type(data).__name__,
    }

    def _is_number_like(x) -> bool:
        try:
            value = float(x)
            return np.isfinite(value)
        except (TypeError, ValueError):
            return False

    def _to_float_list(items):
        return [float(x) for x in items if _is_number_like(x)]

    def _is_numeric_list(obj) -> bool:
        return (
            isinstance(obj, list)
            and len(obj) > 0
            and all(_is_number_like(item) for item in obj)
        )

    def _is_list_of_dicts(obj) -> bool:
        return (
            isinstance(obj, list)
            and len(obj) > 0
            and all(isinstance(item, dict) for item in obj)
        )

    def _find_candidate_list(obj, path="root"):
        """
        Recursively searches for the most plausible generated series inside
        nested dictionaries/lists.

        Priority:
        1. List of numbers.
        2. Single nested list of numbers.
        3. List of dictionaries containing values.
        4. Recursion through dictionary values.
        """
        # Direct numeric list
        if _is_numeric_list(obj):
            return obj, f"{path}:numeric_list"

        # Single nested numeric list: [[...]]
        if (
            isinstance(obj, list)
            and len(obj) == 1
            and _is_numeric_list(obj[0])
        ):
            return obj[0], f"{path}:single_nested_numeric_list"

        # List of dicts: [{"value": ...}, ...]
        if _is_list_of_dicts(obj):
            return obj, f"{path}:list_of_dicts"

        # Mixed list containing one or more numeric lists
        if isinstance(obj, list):
            for i, item in enumerate(obj):
                candidate, candidate_path = _find_candidate_list(
                    item,
                    path=f"{path}[{i}]",
                )
                if candidate is not None:
                    return candidate, candidate_path

        # Dictionary: first try common keys, then all keys
        if isinstance(obj, dict):
            preferred_keys = [
                "values",
                "data",
                "series",
                "time_series",
                "observations",
                "synthetic_series",
                "generated_series",
                "generated_values",
                "sequence",
                "output",
                "result",
                "results",
                "answer",
                "final_answer",
                "response",
            ]

            for key in preferred_keys:
                if key in obj:
                    candidate, candidate_path = _find_candidate_list(
                        obj[key],
                        path=f"{path}.{key}",
                    )
                    if candidate is not None:
                        return candidate, candidate_path

            for key, value in obj.items():
                candidate, candidate_path = _find_candidate_list(
                    value,
                    path=f"{path}.{key}",
                )
                if candidate is not None:
                    return candidate, candidate_path

        return None, None

    candidate_data, candidate_path = _find_candidate_list(data)

    if candidate_data is None:
        raise ValueError(
            "Could not normalize parsed output into timestamp,value dataframe. "
            f"Original type: {type(data).__name__}. "
            f"Dictionary keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
        )

    normalization_info["normalization_applied"] = True
    normalization_info["normalization_type"] = candidate_path

    data = candidate_data

    # Case 1: numeric list
    if _is_numeric_list(data):
        return values_to_dataframe(_to_float_list(data), profile), normalization_info

    # Case 2: list of dictionaries
    if _is_list_of_dicts(data):
        df = pd.DataFrame(data)

        if "value" not in df.columns:
            numeric_cols = [
                col for col in df.columns
                if pd.to_numeric(df[col], errors="coerce").notna().any()
            ]

            if len(numeric_cols) == 1:
                df = df.rename(columns={numeric_cols[0]: "value"})
                normalization_info["normalization_type"] += (
                    f"|renamed_{numeric_cols[0]}_to_value"
                )
            else:
                raise ValueError(
                    f"No 'value' column found and no unique numeric fallback column. "
                    f"Columns: {list(df.columns)}"
                )

        if "timestamp" not in df.columns:
            timestamps = pd.date_range(
                start=pd.to_datetime(profile["start_date"]),
                periods=len(df),
                freq="D",
            )
            df.insert(0, "timestamp", timestamps)
            normalization_info["normalization_type"] += "|timestamps_constructed"

        df = df[["timestamp", "value"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return df, normalization_info

    # Case 3: fallback extraction from mixed nested structures
    extracted_values = []

    if isinstance(data, list):
        for item in data:
            if _is_number_like(item):
                extracted_values.append(float(item))
            elif isinstance(item, dict) and "value" in item and _is_number_like(item["value"]):
                extracted_values.append(float(item["value"]))
            elif isinstance(item, list):
                extracted_values.extend(_to_float_list(item))

    if extracted_values:
        normalization_info["normalization_type"] += "|fallback_extracted_values"
        return values_to_dataframe(extracted_values, profile), normalization_info

    raise ValueError(
        "Could not normalize parsed output into timestamp,value dataframe "
        f"after candidate extraction. Candidate path: {candidate_path}. "
        f"First items: {data[:3] if isinstance(data, list) else data}"
    )

def values_to_dataframe(values, profile: dict) -> pd.DataFrame:
    if not isinstance(values, list):
        raise ValueError("Value-only output must be a list.")

    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric_values = numeric_values.iloc[: profile["length"]]

    start_date = pd.to_datetime(profile["start_date"])
    timestamps = pd.date_range(
        start=start_date,
        periods=len(numeric_values),
        freq="D",
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "value": numeric_values.values,
    })


# ============================================================
# CURATION
# ============================================================

def curate_timeseries(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["timestamp", "value"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"])
    df["value"] = df["value"].clip(profile["min_value"], profile["max_value"])
    return df



def enforce_target_length_for_value_only(df: pd.DataFrame, profile: dict) -> tuple[pd.DataFrame, dict]:
    target_length = profile["length"]
    original_length = len(df)

    curation_info = {
        "length_curation_applied": False,
        "generated_length_before_curation": original_length,
        "target_length": target_length,
        "truncated_points": 0,
        "completed_points": 0,
        "length_curation_strategy": None,
    }

    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    if original_length > target_length:
        df = df.iloc[:target_length].copy()
        curation_info["length_curation_applied"] = True
        curation_info["truncated_points"] = original_length - target_length
        curation_info["length_curation_strategy"] = "truncate_to_target_length"

    elif original_length < target_length:
        missing = target_length - original_length
        existing_values = pd.to_numeric(df["value"], errors="coerce").dropna().tolist()

        if profile.get("seasonality") == "weekly" and original_length >= 7 and existing_values:
            new_values = []
            # Estimate global weekly drift from complete weeks when possible.
            period = 7
            n_complete_weeks = len(existing_values) // period
            weekly_drift = 0.0

            if n_complete_weeks >= 2:
                week_means = [
                    float(np.mean(existing_values[i * period:(i + 1) * period]))
                    for i in range(n_complete_weeks)
                ]
                weekly_drift = float(np.mean(np.diff(week_means))) if len(week_means) > 1 else 0.0

            if profile.get("trend") == "increasing" and weekly_drift <= 0:
                weekly_drift = abs(weekly_drift) if weekly_drift != 0 else 7.0
            elif profile.get("trend") == "decreasing" and weekly_drift >= 0:
                weekly_drift = -abs(weekly_drift) if weekly_drift != 0 else -7.0
            elif profile.get("trend") == "none":
                weekly_drift = 0.0

            for i in range(missing):
                target_pos = original_length + i
                phase = target_pos % period
                phase_values = existing_values[phase::period]
                base = float(np.mean(phase_values)) if phase_values else float(existing_values[-1])

                # Apply drift only for full future weeks beyond the observed sequence.
                observed_week = (len(phase_values) - 1) if phase_values else 0
                target_week = target_pos // period
                extra_weeks = max(0, target_week - observed_week)
                next_value = base + weekly_drift * extra_weeks

                next_value = min(max(next_value, profile["min_value"]), profile["max_value"])
                new_values.append(float(next_value))

            completed_values = existing_values + new_values
            strategy = "weekly_pattern_completion_with_trend_adjustment"

        else:
            if original_length >= 2 and existing_values:
                recent_values = np.asarray(existing_values[-min(5, len(existing_values)):], dtype=float)
                diffs = np.diff(recent_values)
                step = float(np.mean(diffs)) if len(diffs) > 0 else 0.0

                if profile.get("trend") == "increasing" and step <= 0:
                    step = abs(step) if step != 0 else 1.0
                elif profile.get("trend") == "decreasing" and step >= 0:
                    step = -abs(step) if step != 0 else -1.0
                elif profile.get("trend") == "none":
                    step = 0.0
            else:
                step = -1.0 if profile.get("trend") == "decreasing" else 1.0

            last_value = float(existing_values[-1]) if existing_values else 100.0
            new_values = [
                min(max(last_value + step * (i + 1), profile["min_value"]), profile["max_value"])
                for i in range(missing)
            ]
            completed_values = existing_values + new_values
            strategy = "linear_extrapolation_from_recent_mean_step"

        full_timestamps = pd.date_range(
            start=pd.to_datetime(profile["start_date"]),
            periods=target_length,
            freq="D",
        )

        df = pd.DataFrame({
            "timestamp": full_timestamps,
            "value": completed_values[:target_length],
        })

        curation_info["length_curation_applied"] = True
        curation_info["completed_points"] = missing
        curation_info["length_curation_strategy"] = strategy

    else:
        df["timestamp"] = pd.date_range(
            start=pd.to_datetime(profile["start_date"]),
            periods=target_length,
            freq="D",
        )

    return df, curation_info

def compute_weekly_seasonality_score(df: pd.DataFrame) -> dict:
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    y = df_sorted["value"].values

    if len(y) < 14:
        return {
            "weekly_autocorrelation": None,
            "weekly_seasonality_ok": False,
        }

    y1 = y[:-7]
    y2 = y[7:]

    if np.std(y1) == 0 or np.std(y2) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y1, y2)[0, 1])

    return {
        "weekly_autocorrelation": corr,
        "weekly_seasonality_ok": corr >= 0.7,
    }

def compute_anomaly_score(df: pd.DataFrame, profile: dict) -> dict:
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    if len(df_sorted) == 0 or "value" not in df_sorted.columns:
        return {
            "anomaly_count": 0,
            "expected_anomaly_count": profile.get("expected_anomaly_count"),
            "anomaly_threshold": profile.get("anomaly_threshold"),
            "anomaly_count_ok": False,
        }

    threshold = profile.get("anomaly_threshold", 200)
    expected_count = profile.get("expected_anomaly_count", 2)

    values = df_sorted["value"]
    anomaly_mask = values >= threshold
    anomaly_count = int(anomaly_mask.sum())

    return {
        "anomaly_count": anomaly_count,
        "expected_anomaly_count": expected_count,
        "anomaly_threshold": threshold,
        "anomaly_count_ok": anomaly_count == expected_count,
    }


def validate_timeseries(df: pd.DataFrame, profile: dict) -> dict:
    """
    Legacy validation layer.

    It is kept for backward compatibility with earlier outputs. The primary
    TFM evaluation is the profile_based_metrics layer computed later.
    """
    results = {}
    expected_length = profile["length"]

    results["has_correct_columns"] = set(["timestamp", "value"]).issubset(df.columns)
    results["length"] = len(df)
    results["expected_length"] = expected_length
    results["length_ok"] = len(df) == expected_length

    if not results["has_correct_columns"]:
        results.update({
            "timestamps_not_null": False,
            "values_not_null": False,
            "values_numeric": False,
            "min_value": None,
            "max_value": None,
            "range_ok": False,
            "daily_frequency_ok": False,
            "trend_slope": None,
            "increasing_trend_ok": False,
            "decreasing_trend_ok": False,
            "no_strong_trend_threshold": profile.get("no_strong_trend_threshold"),
            "no_strong_trend_ok": False,
            "weekly_autocorrelation": None,
            "weekly_seasonality_ok": None,
            "anomaly_count": None,
            "expected_anomaly_count": None,
            "anomaly_threshold": None,
            "anomaly_count_ok": None,
            "failed_checks": ["has_correct_columns"],
            "valid_series": False,
        })
        return results

    results["timestamps_not_null"] = df["timestamp"].notna().all()
    results["values_not_null"] = df["value"].notna().all()
    results["values_numeric"] = pd.api.types.is_numeric_dtype(df["value"])

    if len(df) > 0 and df["value"].notna().any():
        values_numeric = pd.to_numeric(df["value"], errors="coerce")
        results["min_value"] = float(values_numeric.min())
        results["max_value"] = float(values_numeric.max())
        results["range_ok"] = (
            values_numeric.min() >= profile["min_value"]
            and values_numeric.max() <= profile["max_value"]
        )
    else:
        results["min_value"] = None
        results["max_value"] = None
        results["range_ok"] = False

    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"], errors="coerce")
    df_sorted["value"] = pd.to_numeric(df_sorted["value"], errors="coerce")
    df_sorted = df_sorted.sort_values("timestamp")

    if len(df_sorted) > 1:
        diffs = df_sorted["timestamp"].diff().dropna()
        results["daily_frequency_ok"] = (diffs == pd.Timedelta(days=1)).all()
    else:
        results["daily_frequency_ok"] = False

    y = df_sorted["value"].values
    x = np.arange(len(y))

    if len(y) > 1 and not np.isnan(y).any():
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = np.nan

    results["trend_slope"] = float(slope) if not np.isnan(slope) else None
    results["increasing_trend_ok"] = bool(slope > 0) if not np.isnan(slope) else False
    results["decreasing_trend_ok"] = bool(slope < 0) if not np.isnan(slope) else False

    no_strong_trend_threshold = profile.get("no_strong_trend_threshold", 1.0)
    if profile.get("trend") == "none":
        results["no_strong_trend_threshold"] = no_strong_trend_threshold
        results["no_strong_trend_ok"] = (
            bool(abs(slope) <= no_strong_trend_threshold)
            if not np.isnan(slope)
            else False
        )
    else:
        results["no_strong_trend_threshold"] = None
        results["no_strong_trend_ok"] = None

    if profile.get("seasonality") == "weekly":
        weekly_info = compute_weekly_seasonality_score(df_sorted)
        results["weekly_autocorrelation"] = weekly_info["weekly_autocorrelation"]
        results["weekly_seasonality_ok"] = weekly_info["weekly_seasonality_ok"]
    else:
        results["weekly_autocorrelation"] = None
        results["weekly_seasonality_ok"] = None

    if profile.get("anomalies") == "point":
        anomaly_info = compute_anomaly_score(df_sorted, profile)
        results["anomaly_count"] = anomaly_info["anomaly_count"]
        results["expected_anomaly_count"] = anomaly_info["expected_anomaly_count"]
        results["anomaly_threshold"] = anomaly_info["anomaly_threshold"]
        results["anomaly_count_ok"] = anomaly_info["anomaly_count_ok"]
    else:
        results["anomaly_count"] = None
        results["expected_anomaly_count"] = None
        results["anomaly_threshold"] = None
        results["anomaly_count_ok"] = None

    required_checks = [
        results["has_correct_columns"],
        results["length_ok"],
        results["timestamps_not_null"],
        results["values_not_null"],
        results["values_numeric"],
        results["range_ok"],
        results["daily_frequency_ok"],
    ]

    if profile.get("trend") == "increasing":
        required_checks.append(results["increasing_trend_ok"])

    if profile.get("trend") == "decreasing":
        required_checks.append(results["decreasing_trend_ok"])

    if profile.get("trend") == "none":
        required_checks.append(results["no_strong_trend_ok"])

    if profile.get("seasonality") == "weekly":
        required_checks.append(results["weekly_seasonality_ok"])

    if profile.get("anomalies") == "point":
        required_checks.append(results["anomaly_count_ok"])

    failed_checks = []

    if not results["has_correct_columns"]:
        failed_checks.append("has_correct_columns")
    if not results["length_ok"]:
        failed_checks.append("length_ok")
    if not results["timestamps_not_null"]:
        failed_checks.append("timestamps_not_null")
    if not results["values_not_null"]:
        failed_checks.append("values_not_null")
    if not results["values_numeric"]:
        failed_checks.append("values_numeric")
    if not results["range_ok"]:
        failed_checks.append("range_ok")
    if not results["daily_frequency_ok"]:
        failed_checks.append("daily_frequency_ok")
    if profile.get("trend") == "increasing" and not results["increasing_trend_ok"]:
        failed_checks.append("increasing_trend_ok")
    if profile.get("trend") == "decreasing" and not results["decreasing_trend_ok"]:
        failed_checks.append("decreasing_trend_ok")
    if profile.get("trend") == "none" and not results["no_strong_trend_ok"]:
        failed_checks.append("no_strong_trend_ok")
    if profile.get("seasonality") == "weekly" and not results["weekly_seasonality_ok"]:
        failed_checks.append("weekly_seasonality_ok")
    if profile.get("anomalies") == "point" and not results["anomaly_count_ok"]:
        failed_checks.append("anomaly_count_ok")

    results["failed_checks"] = failed_checks
    results["valid_series"] = all(required_checks)

    return results

def compute_basic_metrics(df: pd.DataFrame) -> dict:
    df_sorted = df.sort_values("timestamp")

    if len(df_sorted) == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "trend_slope": None,
            "weekly_autocorrelation": None,
            "anomaly_count": None,
        }

    y = df_sorted["value"].values
    x = np.arange(len(y))

    slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else np.nan

    if len(y) >= 14:
        y1 = y[:-7]
        y2 = y[7:]
        if np.std(y1) == 0 or np.std(y2) == 0:
            weekly_autocorrelation = 0.0
        else:
            weekly_autocorrelation = float(np.corrcoef(y1, y2)[0, 1])
    else:
        weekly_autocorrelation = None

    anomaly_count = int(np.sum(y >= 200))

    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "trend_slope": float(slope) if not np.isnan(slope) else None,
        "weekly_autocorrelation": weekly_autocorrelation,
        "anomaly_count": anomaly_count,
    }


# ============================================================
# JSON SERIALIZATION
# ============================================================

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(value) for value in obj]

    if isinstance(obj, tuple):
        return tuple(make_json_serializable(value) for value in obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    try:
        if pd.isna(obj):
            return None
    except TypeError:
        pass

    return obj


# ============================================================
# EXPERIMENT EXECUTION
# ============================================================

def run_value_only_attempt(
    prompt: str,
    profile: dict,
    model: str,
    temperature: float,
    num_predict: int,
    timeout: int,
) -> dict:
    """
    Runs one value-only generation attempt and returns all intermediate objects.
    """
    raw_output = call_ollama(
        prompt=prompt,
        model=model,
        temperature=temperature,
        num_predict=num_predict,
        timeout=timeout,
    )

    data, parse_info = extract_json_array(raw_output)

    df_raw, output_normalization_info = parsed_output_to_dataframe(data, profile)
    parse_info["output_normalization"] = output_normalization_info

    values_before_curation = (
        pd.to_numeric(df_raw["value"], errors="coerce")
        .dropna()
        .tolist()
    )

    df_clean = curate_timeseries(df_raw, profile)
    df_clean, structural_curation_info = enforce_target_length_for_value_only(
        df_clean,
        profile,
    )

    validation = validate_timeseries(df_clean, profile)
    metrics = compute_basic_metrics(df_clean)

    values = pd.to_numeric(df_clean["value"], errors="coerce").tolist()

    return {
        "raw_output": raw_output,
        "data": data,
        "values_before_curation": values_before_curation,
        "values": values,
        "parse_info": parse_info,
        "df_clean": df_clean,
        "structural_curation_info": structural_curation_info,
        "validation": validation,
        "metrics": metrics,
    }

def run_validator_feedback_refinement_generation(
    initial_prompt: str,
    profile: dict,
    model: str,
    temperature: float,
    num_predict: int,
    timeout: int,
) -> dict:
    """
    Runs the validator-feedback refinement workflow:
    1. initial generation
    2. validation
    3. if invalid, generate feedback and regenerate
    4. select final output
    """
    initial_attempt = run_value_only_attempt(
        prompt=initial_prompt,
        profile=profile,
        model=model,
        temperature=temperature,
        num_predict=num_predict,
        timeout=timeout,
    )

    initial_valid = bool(initial_attempt["validation"].get("valid_series"))

    if initial_valid:
        return {
            "final_attempt_name": "initial",
            "final_attempt": initial_attempt,
            "initial_attempt": initial_attempt,
            "refined_attempt": None,
            "feedback_applied": False,
            "feedback_text": None,
            "refinement_prompt": None,
        }

    feedback_text = build_validator_feedback_text(
        validation=initial_attempt["validation"],
        profile=profile,
    )

    refinement_prompt = build_validator_refinement_prompt(
        profile=profile,
        initial_values=initial_attempt["values"],
        initial_validation=initial_attempt["validation"],
    )

    refined_attempt = run_value_only_attempt(
        prompt=refinement_prompt,
        profile=profile,
        model=model,
        temperature=temperature,
        num_predict=num_predict,
        timeout=timeout,
    )

    refined_valid = bool(refined_attempt["validation"].get("valid_series"))

    # Selection rule:
    # Prefer refined if it is valid. If not, keep refined only if it fails fewer checks.
    initial_failed = len(initial_attempt["validation"].get("failed_checks", []))
    refined_failed = len(refined_attempt["validation"].get("failed_checks", []))

    if refined_valid or refined_failed < initial_failed:
        final_attempt_name = "refined"
        final_attempt = refined_attempt
    else:
        final_attempt_name = "initial"
        final_attempt = initial_attempt

    return {
        "final_attempt_name": final_attempt_name,
        "final_attempt": final_attempt,
        "initial_attempt": initial_attempt,
        "refined_attempt": refined_attempt,
        "feedback_applied": True,
        "feedback_text": feedback_text,
        "refinement_prompt": refinement_prompt,
    }

def build_cllm_candidate_prompt(
    profile: dict,
    candidate_id: int,
    n_candidates: int,
) -> str:
    """
    Builds one candidate-generation prompt for CLLM-inspired generate-and-curate.

    Each candidate receives a slightly different instruction so that the candidate
    pool is not fully deterministic, while validity remains the priority.
    """
    pattern_description = describe_temporal_pattern(profile)

    candidate_variation = get_cllm_candidate_variation(
        profile=profile,
        candidate_id=candidate_id,
    )

    prompt = f"""
You are a synthetic time series generator.

This is candidate {candidate_id} of {n_candidates} in a generate-and-curate workflow.
Several candidate sequences will be generated and automatically validated.
The final synthetic series will be selected by a validation-based curation step.

TARGET PROFILE:
- Number of observations: {profile["length"]}
- Start date: {profile["start_date"]}
- Frequency: daily
- Trend: {profile.get("trend", "none")}
- Seasonality: {profile.get("seasonality", "none")}
- Seasonality period: {profile.get("seasonality_period", "none")}
- Noise: {profile.get("noise", "low")}
- Anomalies: {profile.get("anomalies", "none")}
- Values must be numeric
- Values must be between {profile["min_value"]} and {profile["max_value"]}

TEMPORAL PATTERN:
{pattern_description}

CANDIDATE-SPECIFIC INSTRUCTION:
{candidate_variation}

IMPORTANT:
- Validity is more important than diversity.
- Generate only numeric values.
- Do NOT generate timestamps.
- Timestamps will be constructed programmatically.
- Return only a valid JSON array of numbers.
- Generate exactly {profile["length"]} numeric values.
- Do not include markdown, comments, or explanations.

FINAL ANSWER RULE:
Your entire response must be exactly one JSON array of numbers.
The first character of your response must be "[".
The last character of your response must be "]".

Expected output format:
[100.0, 105.0, 112.0, 108.0, 115.0]
"""

    return prompt

def compute_candidate_quality_score(candidate_result: dict, profile: dict) -> dict:
    """
    Computes a validation-based quality score for a generated candidate.

    The score is used only for selecting the best candidate among multiple
    generated candidates in the CLLM-inspired generate-and-curate workflow.
    """
    if candidate_result.get("status") != "success":
        return {
            "quality_score": -9999.0,
            "n_failed_checks": 999,
            "valid_series": False,
        }

    validation = candidate_result["validation"]
    failed_checks = validation.get("failed_checks", [])

    score = 0.0

    structural_checks = [
        "has_correct_columns",
        "length_ok",
        "timestamps_not_null",
        "values_not_null",
        "values_numeric",
        "range_ok",
        "daily_frequency_ok",
    ]

    for check in structural_checks:
        if validation.get(check) is True:
            score += 5.0

    if validation.get("valid_series") is True:
        score += 100.0

    if profile.get("trend") == "increasing":
        if validation.get("increasing_trend_ok") is True:
            score += 25.0
        else:
            score -= 15.0

    if profile.get("trend") == "none":
        if validation.get("no_strong_trend_ok") is True:
            score += 25.0
        else:
            score -= 15.0

        slope = validation.get("trend_slope")
        if slope is not None:
            score -= min(abs(float(slope)), 20.0)

    if profile.get("seasonality") == "weekly":
        if validation.get("weekly_seasonality_ok") is True:
            score += 25.0
        else:
            score -= 15.0

        weekly_corr = validation.get("weekly_autocorrelation")
        if weekly_corr is not None:
            score += 5.0 * max(min(float(weekly_corr), 1.0), -1.0)

    if profile.get("anomalies") == "point":
        expected_count = profile.get("expected_anomaly_count", 2)
        anomaly_count = validation.get("anomaly_count")

        if validation.get("anomaly_count_ok") is True:
            score += 25.0
        else:
            score -= 15.0

        if anomaly_count is not None:
            score -= 5.0 * abs(int(anomaly_count) - int(expected_count))

    score -= 10.0 * len(failed_checks)

    return {
        "quality_score": float(score),
        "n_failed_checks": int(len(failed_checks)),
        "valid_series": bool(validation.get("valid_series")),
    }

def run_value_only_attempt_safe(
    prompt: str,
    profile: dict,
    model: str,
    temperature: float,
    num_predict: int,
    timeout: int,
) -> dict:
    """
    Safe wrapper around one value-only generation attempt.
    It returns a structured failed candidate instead of stopping the full workflow.
    """
    try:
        attempt = run_value_only_attempt(
            prompt=prompt,
            profile=profile,
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            timeout=timeout,
        )
        attempt["status"] = "success"
        attempt["error_message"] = None
        return attempt

    except Exception as e:
        empty_df = pd.DataFrame(columns=["timestamp", "value"])

        return {
            "status": "failed",
            "error_message": str(e),
            "raw_output": "",
            "data": [],
            "values_before_curation": [],
            "values": [],
            "parse_info": {
                "raw_json_valid": False,
                "parse_repaired": False,
                "parse_error": str(e),
                "repair_type": None,
            },
            "df_clean": empty_df,
            "structural_curation_info": {
                "length_curation_applied": False,
                "generated_length_before_curation": None,
                "target_length": profile["length"],
                "truncated_points": 0,
                "completed_points": 0,
                "length_curation_strategy": None,
            },
            "validation": {
                "valid_series": False,
                "failed_checks": ["candidate_generation_failed"],
                "error": str(e),
            },
            "metrics": {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "trend_slope": None,
                "weekly_autocorrelation": None,
                "anomaly_count": None,
            },
        }
    
def run_cllm_generate_curate_generation(
    profile: dict,
    model: str,
    temperature: float,
    num_predict: int,
    timeout: int,
    n_candidates: int,
) -> dict:
    """
    Runs a CLLM-inspired generate-and-curate workflow:
    1. Generate multiple candidate series.
    2. Validate every candidate.
    3. Score candidates using validation-based quality score.
    4. Select the best candidate as the final synthetic series.
    """
    candidate_results = []

    for candidate_id in range(1, n_candidates + 1):
        candidate_prompt = build_cllm_candidate_prompt(
            profile=profile,
            candidate_id=candidate_id,
            n_candidates=n_candidates,
        )

        candidate_attempt = run_value_only_attempt_safe(
            prompt=candidate_prompt,
            profile=profile,
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            timeout=timeout,
        )

        candidate_score = compute_candidate_quality_score(
            candidate_result=candidate_attempt,
            profile=profile,
        )

        candidate_attempt["candidate_id"] = candidate_id
        candidate_attempt["candidate_prompt"] = candidate_prompt
        candidate_attempt["candidate_quality_score"] = candidate_score["quality_score"]
        candidate_attempt["candidate_n_failed_checks"] = candidate_score["n_failed_checks"]
        candidate_attempt["candidate_valid_series"] = candidate_score["valid_series"]

        candidate_results.append(candidate_attempt)

    successful_candidates = [
        c for c in candidate_results
        if c.get("status") == "success"
    ]

    if not successful_candidates:
        raise ValueError("All CLLM generate-and-curate candidates failed.")

    # Selection rule:
    # prefer valid candidates, then higher quality score, then fewer failed checks.
    best_candidate = sorted(
        successful_candidates,
        key=lambda c: (
            bool(c.get("candidate_valid_series")),
            float(c.get("candidate_quality_score", -9999.0)),
            -int(c.get("candidate_n_failed_checks", 999)),
        ),
        reverse=True,
    )[0]

    candidate_summaries = []

    for c in candidate_results:
        validation = c.get("validation", {})
        candidate_summaries.append({
            "candidate_id": c.get("candidate_id"),
            "status": c.get("status"),
            "error_message": c.get("error_message"),
            "valid_series": validation.get("valid_series"),
            "failed_checks": validation.get("failed_checks"),
            "quality_score": c.get("candidate_quality_score"),
            "n_failed_checks": c.get("candidate_n_failed_checks"),
            "raw_json_valid": c.get("parse_info", {}).get("raw_json_valid"),
            "parse_repaired": c.get("parse_info", {}).get("parse_repaired"),
            "trend_slope": validation.get("trend_slope"),
            "weekly_autocorrelation": validation.get("weekly_autocorrelation"),
            "anomaly_count": validation.get("anomaly_count"),
        })

    valid_candidates = [
        c for c in candidate_results
        if c.get("validation", {}).get("valid_series") is True
    ]

    return {
        "final_attempt": best_candidate,
        "best_candidate_id": best_candidate.get("candidate_id"),
        "candidate_results": candidate_results,
        "candidate_summaries": candidate_summaries,
        "n_candidates": int(n_candidates),
        "n_successful_candidates": int(len(successful_candidates)),
        "n_valid_candidates": int(len(valid_candidates)),
        "valid_candidates_rate": (
            float(len(valid_candidates) / n_candidates)
            if n_candidates > 0
            else 0.0
        ),
    }

def run_experiment(
    method: str,
    profile_id: str,
    model: str,
    run_id: int,
    temperature: float,
    num_predict: int,
    timeout: int,
    n_examples: int,
    selection_seed: int,
    output_dir: str | Path | None = None,
    reference_features_path: str | Path | None = None,
    batch_id: str | None = None,
    ollama_url: str | None = None,
    overwrite: bool = False,
):
    profile_spec = get_structured_profile(profile_id)
    profile_dict = profile_to_dict(profile_spec)

    configure_runtime_paths(
        output_dir=output_dir,
        reference_features_path=reference_features_path,
        ollama_url=ollama_url,
    )

    if profile_id not in STRUCTURED_PROFILES:
        raise ValueError(
            f"Unrecognized profile: {profile_id}. "
            f"Available: {list(STRUCTURED_PROFILES.keys())}"
        )

    if method not in METHODS:
        raise ValueError(
            f"Unrecognized method: {method}. "
            f"Available: {list(METHODS.keys())}"
        )

    profile = build_legacy_profile_from_structured(
        profile_spec=profile_spec,
        fallback_profile=LEGACY_PROFILES.get(profile_id),
    )

    method_info = METHODS[method]

    safe_model = safe_slug(model)
    safe_batch = safe_slug(batch_id) if batch_id else None
    experiment_parts = [profile_id, method, safe_model]
    if safe_batch:
        experiment_parts.append(safe_batch)
    experiment_parts.append(f"run{run_id}")
    experiment_id = "_".join(experiment_parts)

    base_name = experiment_id
    raw_path = OUTPUT_DIR / f"{base_name}_raw_output.txt"
    prompt_path = OUTPUT_DIR / f"{base_name}_prompt.txt"
    csv_path = OUTPUT_DIR / f"{base_name}_synthetic_series.csv"
    metrics_path = OUTPUT_DIR / f"{base_name}_metrics.json"

    output_paths = [raw_path, prompt_path, csv_path, metrics_path]
    existing_paths = [path for path in output_paths if path.exists()]

    if existing_paths and not overwrite:
        existing_text = "\n".join(f"- {path}" for path in existing_paths)
        raise FileExistsError(
            "Experiment outputs already exist and overwrite=False.\n"
            f"Experiment ID: {experiment_id}\n"
            "Existing files:\n"
            f"{existing_text}\n"
            "Use --overwrite to replace them, or change --run_id / --batch_id."
        )

    print("=== Synthetic time series generation experiment ===")
    print(f"Experiment: {experiment_id}")
    print(f"Model: {model}")
    print(f"Batch ID: {batch_id}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Method: {method_info['method_name']}")
    print(f"Profile: {profile_id} - {profile['description']}")

    if method == "random_few_shot":
        print(f"Number of few-shot examples: {n_examples}")
        print(f"Selection seed: {selection_seed}")

    prompt, context_selection_info = build_prompt_with_metadata(
        profile=profile,
        method=method,
        run_id=run_id,
        n_examples=n_examples,
        selection_seed=selection_seed,
        model=model,
        batch_id=batch_id,
    )

    if context_selection_info.get("reference_bank_used"):
        print("Selected references:")
        for ref_id in context_selection_info.get("selected_reference_ids", []):
            print(f"- {ref_id}")

    start_time = time.time()

    status = "success"
    error_message = None
    raw_output = ""

    structural_curation_info = {
        "length_curation_applied": False,
        "generated_length_before_curation": None,
        "target_length": None,
        "truncated_points": 0,
        "completed_points": 0,
        "length_curation_strategy": None,
    }

    try:
        if method == "validator_feedback_refinement":
            print("\nGenerating series with the validator-feedback workflow...")

            refinement_result = run_validator_feedback_refinement_generation(
                initial_prompt=prompt,
                profile=profile,
                model=model,
                temperature=temperature,
                num_predict=num_predict,
                timeout=timeout,
            )

            final_attempt = refinement_result["final_attempt"]

            values_before_curation = final_attempt.get("values_before_curation", [])

            raw_output = (
                "=== INITIAL RAW OUTPUT ===\n"
                + refinement_result["initial_attempt"]["raw_output"]
            )

            if refinement_result["refined_attempt"] is not None:
                raw_output += (
                    "\n\n=== REFINED RAW OUTPUT ===\n"
                    + refinement_result["refined_attempt"]["raw_output"]
                )

            parse_info = final_attempt["parse_info"]
            df_clean = final_attempt["df_clean"]
            structural_curation_info = final_attempt["structural_curation_info"]
            validation = final_attempt["validation"]
            metrics = final_attempt["metrics"]

            context_selection_info.update({
                "feedback_applied": refinement_result["feedback_applied"],
                "final_attempt_name": refinement_result["final_attempt_name"],
                "initial_valid_series": refinement_result["initial_attempt"]["validation"].get("valid_series"),
                "initial_failed_checks": refinement_result["initial_attempt"]["validation"].get("failed_checks"),
                "initial_trend_slope": refinement_result["initial_attempt"]["validation"].get("trend_slope"),
                "initial_weekly_autocorrelation": refinement_result["initial_attempt"]["validation"].get("weekly_autocorrelation"),
                "initial_anomaly_count": refinement_result["initial_attempt"]["validation"].get("anomaly_count"),
                "feedback_text": refinement_result["feedback_text"],
            })

            if refinement_result["refined_attempt"] is not None:
                context_selection_info.update({
                    "refined_valid_series": refinement_result["refined_attempt"]["validation"].get("valid_series"),
                    "refined_failed_checks": refinement_result["refined_attempt"]["validation"].get("failed_checks"),
                    "refined_trend_slope": refinement_result["refined_attempt"]["validation"].get("trend_slope"),
                    "refined_weekly_autocorrelation": refinement_result["refined_attempt"]["validation"].get("weekly_autocorrelation"),
                    "refined_anomaly_count": refinement_result["refined_attempt"]["validation"].get("anomaly_count"),
                })

        elif method == "cllm_generate_curate":
            print("\nGenerating series with the CLLM generate-and-curate workflow...")

            n_candidates = n_examples

            cllm_result = run_cllm_generate_curate_generation(
                profile=profile,
                model=model,
                temperature=temperature,
                num_predict=num_predict,
                timeout=timeout,
                n_candidates=n_candidates,
            )

            final_attempt = cllm_result["final_attempt"]

            values_before_curation = final_attempt.get("values_before_curation", [])

            raw_output_parts = []

            for candidate in cllm_result["candidate_results"]:
                raw_output_parts.append(
                    f"=== CANDIDATE {candidate.get('candidate_id')} RAW OUTPUT ===\n"
                    + candidate.get("raw_output", "")
                )

            raw_output = "\n\n".join(raw_output_parts)

            parse_info = final_attempt["parse_info"]
            df_clean = final_attempt["df_clean"]
            structural_curation_info = final_attempt["structural_curation_info"]
            validation = final_attempt["validation"]
            metrics = final_attempt["metrics"]

            context_selection_info.update({
                "n_candidates": cllm_result["n_candidates"],
                "n_successful_candidates": cllm_result["n_successful_candidates"],
                "n_valid_candidates": cllm_result["n_valid_candidates"],
                "valid_candidates_rate": cllm_result["valid_candidates_rate"],
                "best_candidate_id": cllm_result["best_candidate_id"],
                "best_candidate_quality_score": final_attempt.get("candidate_quality_score"),
                "best_candidate_n_failed_checks": final_attempt.get("candidate_n_failed_checks"),
                "candidate_summaries": cllm_result["candidate_summaries"],
            })

        else:
            print("\nGenerating series with Ollama...")
            raw_output = call_ollama(
                prompt=prompt,
                model=model,
                temperature=temperature,
                num_predict=num_predict,
                timeout=timeout,
            )

            print("Parsing output...")
            data, parse_info = extract_json_array(raw_output)

            df_raw, output_normalization_info = parsed_output_to_dataframe(data, profile)

            parse_info["output_normalization"] = output_normalization_info

            values_before_curation = (
                pd.to_numeric(df_raw["value"], errors="coerce")
                .dropna()
                .tolist()
            )

            print("Curating series...")
            df_clean = curate_timeseries(df_raw, profile)

            df_clean, structural_curation_info = enforce_target_length_for_value_only(
                df_clean,
                profile,
            )

            print("Validating series...")
            validation = validate_timeseries(df_clean, profile)

            print("Computing metrics...")
            metrics = compute_basic_metrics(df_clean)

    except Exception as e:
        status = "failed"
        error_message = str(e)
        values_before_curation = []

        if "parse_info" not in locals():
            parse_info = {
                "raw_json_valid": False,
                "parse_repaired": False,
                "parse_error": error_message,
                "repair_type": None,
            }

        df_clean = pd.DataFrame(columns=["timestamp", "value"])
        validation = {
            "valid_series": False,
            "error": error_message,
        }
        metrics = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "trend_slope": None,
            "weekly_autocorrelation": None,
        }

    elapsed_time = time.time() - start_time

    # ------------------------------------------------------------
    # New profile-based metrics
    # ------------------------------------------------------------
    try:
        final_values_after_curation = (
            pd.to_numeric(df_clean["value"], errors="coerce")
            .dropna()
            .tolist()
            if "value" in df_clean.columns
            else []
        )

        if "values_before_curation" not in locals():
            values_before_curation = final_values_after_curation

        parse_success = int(
            status == "success"
            and parse_info.get("parse_error") is None
        )

        if parse_info.get("raw_json_valid"):
            parse_error_type = "none"
        elif parse_info.get("parse_repaired"):
            parse_error_type = parse_info.get("repair_type") or "repaired_parse"
        else:
            parse_error_type = parse_info.get("parse_error") or "unknown_parse_error"

        if method in VALUE_ONLY_OUTPUT_METHODS:
            timestamp_source = "python_postprocess"
            timestamp_score = 1.0
        else:
            timestamp_source = "llm_output"
            timestamp_checks = [
                bool(validation.get("timestamps_not_null")),
                bool(validation.get("daily_frequency_ok")),
            ]
            timestamp_score = float(np.mean(timestamp_checks))

        llm_output_metrics = compute_profile_metrics(
            values=values_before_curation,
            profile=profile_spec,
            parse_success=parse_success,
            timestamp_score=timestamp_score,
        )

        final_series_metrics = compute_profile_metrics(
            values=final_values_after_curation,
            profile=profile_spec,
            parse_success=parse_success,
            timestamp_score=timestamp_score,
        )

        expected_length = profile_spec.expected_length
        completed_points = structural_curation_info.get("completed_points", 0) or 0
        truncated_points = structural_curation_info.get("truncated_points", 0) or 0

        curation_change_rate = (
            (completed_points + truncated_points) / expected_length
            if expected_length > 0
            else 0.0
        )

        profile_based_metrics = {
            "parse_success": parse_success,
            "parse_error_type": parse_error_type,
            "timestamp_source": timestamp_source,

            "values_before_curation_length": len(values_before_curation),
            "values_after_curation_length": len(final_values_after_curation),
            "completed_points": completed_points,
            "truncated_points": truncated_points,
            "curation_change_rate": curation_change_rate,

            # Primary score: raw LLM output before length completion/truncation.
            "llm_profile_compliance_score": llm_output_metrics.get("profile_compliance_score"),
            "llm_formal_validity_score": llm_output_metrics.get("formal_validity_score"),
            "llm_temporal_profile_score": llm_output_metrics.get("temporal_profile_score"),

            # Pipeline score: final curated series after postprocessing.
            "final_profile_compliance_score": final_series_metrics.get("profile_compliance_score"),
            "final_formal_validity_score": final_series_metrics.get("formal_validity_score"),
            "final_temporal_profile_score": final_series_metrics.get("temporal_profile_score"),

            # Clear validity flags.
            "strict_llm_valid": llm_output_metrics.get("final_valid"),
            "curated_series_valid": final_series_metrics.get("final_valid"),

            # Main conservative score for model comparison.
            "profile_compliance_score": llm_output_metrics.get("profile_compliance_score"),
            "formal_validity_score": llm_output_metrics.get("formal_validity_score"),
            "temporal_profile_score": llm_output_metrics.get("temporal_profile_score"),

            # Kept for backward compatibility, but now equivalent to strict_llm_valid.
            "final_valid": llm_output_metrics.get("final_valid"),

            "llm_output_metrics": llm_output_metrics,
            "final_series_metrics": final_series_metrics,
        }

    except Exception as metric_error:
        profile_based_metrics = {
            "profile_metrics_error": str(metric_error),
            "parse_success": 0,
            "parse_error_type": "profile_metrics_failed",
            "timestamp_source": None,
            "profile_compliance_score": 0.0,
            "formal_validity_score": 0.0,
            "temporal_profile_score": 0.0,
            "final_valid": 0,
        }

    results = {
        "experiment_id": experiment_id,
        "status": status,
        "error_message": error_message,
        "scenario": profile.get("scenario"),

        # Legacy profile used by the current prompt and validation code
        "profile": profile,

        # Structured profile used by the new metric framework
        "structured_profile": profile_dict,

        "model": model,
        "model_slug": safe_model,
        "batch_id": batch_id,
        "output_dir": str(OUTPUT_DIR),
        "reference_features_path": str(REFERENCE_FEATURES_PATH),

        "method": method,
        "method_metadata": method_info,

        "generation_config": {
            "temperature": temperature,
            "num_predict": num_predict,
            "timeout": timeout,
            "n_examples": n_examples,
            "selection_seed": selection_seed,
            "batch_id": batch_id,
            "output_dir": str(OUTPUT_DIR),
        },

        "context_selection": context_selection_info,
        "parse_info": parse_info,
        "structural_curation_info": structural_curation_info,

        # Old validation layer, kept for backward compatibility
        "validation": validation,
        "metrics": metrics,

        # New validation layer
        "profile_based_metrics": profile_based_metrics,

        # Useful traceability fields
        "values_before_curation": values_before_curation,
        "values_after_curation": (
            pd.to_numeric(df_clean["value"], errors="coerce")
            .dropna()
            .tolist()
            if "value" in df_clean.columns
            else []
        ),

        "elapsed_time_seconds": elapsed_time,
        "timestamp_run": datetime.now().isoformat(),

        "output_files": {
            "raw_output": str(raw_path),
            "prompt": str(prompt_path),
            "synthetic_series": str(csv_path),
            "metrics": str(metrics_path),
        },
    }

    raw_path.write_text(raw_output, encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")
    df_clean.to_csv(csv_path, index=False)

    results_serializable = make_json_serializable(results)
    metrics_path.write_text(
        json.dumps(results_serializable, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")
    print(f"Status: {status}")
    print(f"Valid series: {validation.get('valid_series')}")
    print(f"Directly valid JSON: {parse_info.get('raw_json_valid')}")
    print(f"Repaired JSON: {parse_info.get('parse_repaired')}")
    print(f"Repair type: {parse_info.get('repair_type')}")
    print(f"Generated length: {validation.get('length')}")
    print(f"Increasing trend OK: {validation.get('increasing_trend_ok')}")
    print(f"Decreasing trend OK: {validation.get('decreasing_trend_ok')}")
    print(f"Weekly seasonality OK: {validation.get('weekly_seasonality_ok')}")
    print(f"Weekly autocorrelation: {validation.get('weekly_autocorrelation')}")
    print(f"No strong trend OK: {validation.get('no_strong_trend_ok')}")
    print(f"Strong-trend threshold: {validation.get('no_strong_trend_threshold')}")
    print(f"Daily frequency OK: {validation.get('daily_frequency_ok')}")
    print(f"Slope: {validation.get('trend_slope')}")
    print(f"Number of anomalies: {validation.get('anomaly_count')}")
    print(f"Expected anomalies: {validation.get('expected_anomaly_count')}")
    print(f"Anomalies OK: {validation.get('anomaly_count_ok')}")
    print(f"Anomaly threshold: {validation.get('anomaly_threshold')}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Length curation applied: {structural_curation_info.get('length_curation_applied')}")
    print(f"Completed points: {structural_curation_info.get('completed_points')}")
    print(f"Truncated points: {structural_curation_info.get('truncated_points')}")
    print("\n=== PROFILE-BASED METRICS ===")
    print(f"Parse success: {profile_based_metrics.get('parse_success')}")
    print(f"Parse error type: {profile_based_metrics.get('parse_error_type')}")
    print(f"Timestamp source: {profile_based_metrics.get('timestamp_source')}")
    print(f"Values before curation: {profile_based_metrics.get('values_before_curation_length')}")
    print(f"Values after curation: {profile_based_metrics.get('values_after_curation_length')}")
    print(f"Completed points: {profile_based_metrics.get('completed_points')}")
    print(f"Truncated points: {profile_based_metrics.get('truncated_points')}")
    print(f"Curation change rate: {profile_based_metrics.get('curation_change_rate')}")
    print(f"LLM profile compliance score: {profile_based_metrics.get('llm_profile_compliance_score')}")
    print(f"Final series profile compliance score: {profile_based_metrics.get('final_profile_compliance_score')}")
    print(f"Main comparison score: {profile_based_metrics.get('profile_compliance_score')}")
    print(f"Strict LLM valid: {profile_based_metrics.get('strict_llm_valid')}")
    print(f"Curated series valid: {profile_based_metrics.get('curated_series_valid')}")

    if error_message:
        print(f"Error: {error_message}")

    print("\nSaved files:")
    print(f"- {raw_path}")
    print(f"- {prompt_path}")
    print(f"- {csv_path}")
    print(f"- {metrics_path}")


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Runs synthetic time series generation experiments with LLMs."
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=list(METHODS.keys()),
        help="Generation/context method.",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="P1_20",
        choices=list(STRUCTURED_PROFILES.keys()),
        help="Target temporal profile.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2",
        help="Ollama model.",
    )

    parser.add_argument(
        "--run_id",
        type=int,
        default=1,
        help="Experimental repetition identifier.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature.",
    )

    parser.add_argument(
        "--num_predict",
        type=int,
        default=2048,
        help="Maximum number of generated tokens.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout for the Ollama call, in seconds.",
    )

    parser.add_argument(
        "--n_examples",
        type=int,
        default=3,
        help="Number of reference examples for bank-based few-shot methods.",
    )

    parser.add_argument(
        "--selection_seed",
        type=int,
        default=42,
        help="Seed for reproducible selection of reference examples.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where outputs, prompts, CSVs and metrics will be saved.",
    )

    parser.add_argument(
        "--reference_features_path",
        type=str,
        default=str(DEFAULT_REFERENCE_FEATURES_PATH),
        help="Path to the reference bank features CSV.",
    )

    parser.add_argument(
        "--batch_id",
        type=str,
        default=None,
        help="Optional experimental batch identifier to isolate results.",
    )

    parser.add_argument(
        "--ollama_url",
        type=str,
        default=OLLAMA_URL,
        help="Local Ollama endpoint.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing outputs for the same experiment. "
            "By default, the script stops if files already exist."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_experiment(
        method=args.method,
        profile_id=args.profile,
        model=args.model,
        run_id=args.run_id,
        temperature=args.temperature,
        num_predict=args.num_predict,
        timeout=args.timeout,
        n_examples=args.n_examples,
        selection_seed=args.selection_seed,
        output_dir=args.output_dir,
        reference_features_path=args.reference_features_path,
        batch_id=args.batch_id,
        ollama_url=args.ollama_url,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()