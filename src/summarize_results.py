"""Aggregate JSON metrics produced by src/run_experiment.py.

This version is aligned with the profile-based evaluation framework used in the
TFM experiments. It keeps selected legacy validation fields for traceability,
but the main comparison is based on the new `profile_based_metrics` block saved
by run_experiment.py:

- strict_llm_valid: whether the original LLM output satisfies the profile
  without structural length completion/truncation.
- curated_series_valid: whether the final postprocessed series satisfies the
  profile after Python curation.
- llm_profile_compliance_score: main conservative score for comparing LLMs,
  methods and profiles.
- final_profile_compliance_score: score after postprocessing.
- curation_change_rate: proportion of the target series completed/truncated by
  the pipeline.

Example:
    python src/summarize_results.py --output_dir outputs_smoke_metrics --batch_id smoke_profile_metrics
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs")

FINAL_PROFILES = [
    "P1_20",
    "P2_28",
    "P3_28",
    "P4_30",
    "P5_60",
    "P6_60",
]

FINAL_METHODS = [
    "prompt_only",
    "constraint_guided",
    "canonical_few_shot",
    "random_few_shot",
    "harmonic_knn",
    "tabgen_icl_residual",
    "value_only_controlled_scale",
]

AUXILIARY_METHODS = [
    "chatts_attribute_prompting",
    "epic_grouped_attribute_few_shot",
    "validator_feedback_refinement",
    "cllm_generate_curate",
]

METHODS_TO_SUMMARIZE = FINAL_METHODS + AUXILIARY_METHODS


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _join_list(value: Any, sep: str = "|") -> str | None:
    items = _as_list(value)
    if not items:
        return None
    return sep.join(str(x) for x in items)


def _safe_bool(value: Any) -> bool:
    """Convert common scalar representations to bool without crashing on NA."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _extract_run_id(experiment_id: str | None) -> int | None:
    if not experiment_id:
        return None
    match = re.search(r"_run(\d+)$", experiment_id)
    if not match:
        match = re.search(r"run(\d+)", experiment_id)
    return int(match.group(1)) if match else None


def _get_nested(source: dict[str, Any], key: str, default: Any = None) -> Any:
    value = source.get(key, default)
    return default if value is None else value


# ---------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------

def load_metric_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    validation = data.get("validation", {}) or {}
    metrics = data.get("metrics", {}) or {}
    profile = data.get("profile", {}) or {}
    structured_profile = data.get("structured_profile", {}) or {}
    method_metadata = data.get("method_metadata", {}) or {}
    parse_info = data.get("parse_info", {}) or {}
    structural_curation_info = data.get("structural_curation_info", {}) or {}
    context_selection = data.get("context_selection", {}) or {}
    generation_config = data.get("generation_config", {}) or {}

    profile_based = data.get("profile_based_metrics", {}) or {}
    llm_metrics = profile_based.get("llm_output_metrics", {}) or {}
    final_metrics = profile_based.get("final_series_metrics", {}) or {}

    experiment_id = data.get("experiment_id")
    profile_id = (
        structured_profile.get("profile_id")
        or profile.get("profile_id")
        or profile_based.get("profile_id")
    )

    row = {
        # Experiment identity
        "source_file": str(path),
        "experiment_id": experiment_id,
        "run_id": _extract_run_id(experiment_id),
        "status": data.get("status"),
        "error_message": data.get("error_message"),
        "scenario": data.get("scenario"),
        "batch_id": data.get("batch_id") or generation_config.get("batch_id"),
        "output_dir": data.get("output_dir") or generation_config.get("output_dir"),

        # Experimental dimensions
        "profile_id": profile_id,
        "profile_description": (
            structured_profile.get("profile_description")
            or profile.get("description")
        ),
        "model": data.get("model"),
        "method": data.get("method"),
        "method_name": method_metadata.get("method_name"),
        "method_family": method_metadata.get("method_family"),
        "literature_inspiration": method_metadata.get("literature_inspiration"),

        # Generation config
        "temperature": generation_config.get("temperature"),
        "num_predict": generation_config.get("num_predict"),
        "timeout": generation_config.get("timeout"),
        "n_examples_requested": generation_config.get("n_examples"),
        "selection_seed_config": generation_config.get("selection_seed"),

        # Structured target profile
        "structured_expected_length": structured_profile.get("expected_length"),
        "structured_frequency": structured_profile.get("frequency"),
        "structured_timestamp_required": structured_profile.get("timestamp_required"),
        "structured_value_min": structured_profile.get("value_min"),
        "structured_value_max": structured_profile.get("value_max"),
        "structured_trend_expected": structured_profile.get("trend_expected"),
        "structured_seasonality_expected": structured_profile.get("seasonality_expected"),
        "structured_seasonality_period": structured_profile.get("seasonality_period"),
        "structured_expected_anomalies": structured_profile.get("expected_anomalies"),
        "structured_anomaly_type": structured_profile.get("anomaly_type"),
        "structured_noise_expected": structured_profile.get("noise_expected"),
        "structured_flat_trend_threshold": structured_profile.get("flat_trend_threshold"),
        "structured_seasonality_threshold": structured_profile.get("seasonality_threshold"),
        "structured_anomaly_tolerance": structured_profile.get("anomaly_tolerance"),

        # Context selection
        "selection_strategy": context_selection.get("selection_strategy"),
        "reference_bank_used": context_selection.get("reference_bank_used"),
        "n_examples_selected": context_selection.get("n_examples_selected"),
        "selected_reference_ids": _join_list(context_selection.get("selected_reference_ids")),
        "selection_seed": context_selection.get("selection_seed"),
        "effective_seed": context_selection.get("effective_seed"),
        "knn_feature_columns": _join_list(context_selection.get("knn_feature_columns")),
        "selected_reference_distances": _join_list(context_selection.get("selected_reference_distances")),
        "mean_neighbor_distance": context_selection.get("mean_neighbor_distance"),
        "fallback_used": context_selection.get("fallback_used"),
        "fallback_reason": context_selection.get("fallback_reason"),
        "generated_context_count": context_selection.get("generated_context_count"),
        "generated_context_experiment_ids": _join_list(context_selection.get("generated_context_experiment_ids")),
        "residual_feature_columns": _join_list(context_selection.get("residual_feature_columns")),
        "selected_reference_residual_distances": _join_list(context_selection.get("selected_reference_residual_distances")),
        "tabgen_icl_selection_rule": context_selection.get("tabgen_icl_selection_rule"),
        "n_candidate_subsets": context_selection.get("n_candidate_subsets"),
        "best_candidate_distance": context_selection.get("best_candidate_distance"),
        "mean_residual_distance": context_selection.get("mean_residual_distance"),
        "harmonic_selection_rule": context_selection.get("harmonic_selection_rule"),
        "anchor_reference_id": context_selection.get("anchor_reference_id"),
        "anchor_position": context_selection.get("anchor_position"),

        # Attribute / auxiliary workflow metadata
        "chatts_inspired": context_selection.get("chatts_inspired"),
        "trend_attribute": context_selection.get("trend_attribute"),
        "periodicity_attribute": context_selection.get("periodicity_attribute"),
        "local_fluctuation_attribute": context_selection.get("local_fluctuation_attribute"),
        "noise_attribute": context_selection.get("noise_attribute"),
        "epic_inspired": context_selection.get("epic_inspired"),
        "selected_reference_roles": _join_list(context_selection.get("selected_reference_roles")),
        "selected_reference_group_descriptions": _join_list(context_selection.get("selected_reference_group_descriptions")),
        "feedback_applied": context_selection.get("feedback_applied"),
        "final_attempt_name": context_selection.get("final_attempt_name"),
        "initial_valid_series": context_selection.get("initial_valid_series"),
        "initial_failed_checks": _join_list(context_selection.get("initial_failed_checks")),
        "refined_valid_series": context_selection.get("refined_valid_series"),
        "refined_failed_checks": _join_list(context_selection.get("refined_failed_checks")),
        "n_candidates": context_selection.get("n_candidates"),
        "n_successful_candidates": context_selection.get("n_successful_candidates"),
        "n_valid_candidates": context_selection.get("n_valid_candidates"),
        "valid_candidates_rate": context_selection.get("valid_candidates_rate"),
        "best_candidate_id": context_selection.get("best_candidate_id"),
        "best_candidate_quality_score": context_selection.get("best_candidate_quality_score"),
        "best_candidate_n_failed_checks": context_selection.get("best_candidate_n_failed_checks"),

        # Legacy parse / repair information
        "raw_json_valid": parse_info.get("raw_json_valid"),
        "parse_repaired": parse_info.get("parse_repaired"),
        "repair_type": parse_info.get("repair_type"),
        "parse_error": parse_info.get("parse_error"),

        # Profile-based parse diagnostics
        "profile_parse_success": profile_based.get("parse_success"),
        "profile_parse_error_type": profile_based.get("parse_error_type"),
        "timestamp_source": profile_based.get("timestamp_source"),

        # Structural curation
        "length_curation_applied": structural_curation_info.get("length_curation_applied"),
        "generated_length_before_curation": structural_curation_info.get("generated_length_before_curation"),
        "target_length": structural_curation_info.get("target_length"),
        "completed_points": structural_curation_info.get("completed_points"),
        "truncated_points": structural_curation_info.get("truncated_points"),
        "length_curation_strategy": structural_curation_info.get("length_curation_strategy"),

        # New curation diagnostics
        "values_before_curation_length": profile_based.get("values_before_curation_length"),
        "values_after_curation_length": profile_based.get("values_after_curation_length"),
        "curation_change_rate": profile_based.get("curation_change_rate"),

        # Legacy validation checks, retained for traceability
        "valid_series": validation.get("valid_series"),
        "failed_checks": _join_list(validation.get("failed_checks")),
        "length": validation.get("length"),
        "expected_length": validation.get("expected_length"),
        "length_ok": validation.get("length_ok"),
        "timestamps_not_null": validation.get("timestamps_not_null"),
        "values_not_null": validation.get("values_not_null"),
        "daily_frequency_ok": validation.get("daily_frequency_ok"),
        "range_ok": validation.get("range_ok"),
        "values_numeric": validation.get("values_numeric"),
        "increasing_trend_ok": validation.get("increasing_trend_ok"),
        "decreasing_trend_ok": validation.get("decreasing_trend_ok"),
        "no_strong_trend_ok": validation.get("no_strong_trend_ok"),
        "no_strong_trend_threshold": validation.get("no_strong_trend_threshold"),
        "weekly_seasonality_ok": validation.get("weekly_seasonality_ok"),
        "weekly_autocorrelation_validation": validation.get("weekly_autocorrelation"),
        "anomaly_count_validation": validation.get("anomaly_count"),
        "expected_anomaly_count": validation.get("expected_anomaly_count"),
        "anomaly_threshold": validation.get("anomaly_threshold"),
        "anomaly_count_ok": validation.get("anomaly_count_ok"),
        "trend_slope_validation": validation.get("trend_slope"),

        # Basic descriptive metrics on final curated series
        "mean": metrics.get("mean"),
        "std": metrics.get("std"),
        "min": metrics.get("min"),
        "max": metrics.get("max"),
        "trend_slope": metrics.get("trend_slope"),
        "weekly_autocorrelation": metrics.get("weekly_autocorrelation"),
        "anomaly_count": metrics.get("anomaly_count"),

        # Profile-based top-level metrics
        "strict_llm_valid": profile_based.get("strict_llm_valid"),
        "curated_series_valid": profile_based.get("curated_series_valid"),
        "profile_compliance_score": profile_based.get("profile_compliance_score"),
        "formal_validity_score": profile_based.get("formal_validity_score"),
        "temporal_profile_score": profile_based.get("temporal_profile_score"),
        "profile_final_valid": profile_based.get("final_valid"),

        "llm_profile_compliance_score": profile_based.get("llm_profile_compliance_score"),
        "llm_formal_validity_score": profile_based.get("llm_formal_validity_score"),
        "llm_temporal_profile_score": profile_based.get("llm_temporal_profile_score"),
        "final_profile_compliance_score": profile_based.get("final_profile_compliance_score"),
        "final_formal_validity_score": profile_based.get("final_formal_validity_score"),
        "final_temporal_profile_score": profile_based.get("final_temporal_profile_score"),

        # LLM-output component scores
        "llm_generated_length": llm_metrics.get("generated_length"),
        "llm_numeric_generated_length": llm_metrics.get("numeric_generated_length"),
        "llm_expected_length": llm_metrics.get("expected_length"),
        "llm_invalid_numeric_count": llm_metrics.get("invalid_numeric_count"),
        "llm_min_generated": llm_metrics.get("min_generated"),
        "llm_max_generated": llm_metrics.get("max_generated"),
        "llm_mean_generated": llm_metrics.get("mean_generated"),
        "llm_std_generated": llm_metrics.get("std_generated"),
        "llm_length_score": llm_metrics.get("length_score"),
        "llm_numeric_validity_score": llm_metrics.get("numeric_validity_score"),
        "llm_timestamp_score": llm_metrics.get("timestamp_score"),
        "llm_range_score": llm_metrics.get("range_score"),
        "llm_out_of_range_count": llm_metrics.get("out_of_range_count"),
        "llm_range_violation_rate": llm_metrics.get("range_violation_rate"),
        "llm_trend_observed": llm_metrics.get("trend_observed"),
        "llm_trend_score": llm_metrics.get("trend_score"),
        "llm_seasonality_strength": llm_metrics.get("seasonality_strength"),
        "llm_seasonality_score": llm_metrics.get("seasonality_score"),
        "llm_detected_anomalies": llm_metrics.get("detected_anomalies"),
        "llm_expected_anomalies": llm_metrics.get("expected_anomalies"),
        "llm_anomaly_difference": llm_metrics.get("anomaly_difference"),
        "llm_anomaly_score": llm_metrics.get("anomaly_score"),
        "llm_noise_ratio": llm_metrics.get("noise_ratio"),
        "llm_noise_score": llm_metrics.get("noise_score"),
        "llm_final_valid": llm_metrics.get("final_valid"),

        # Final curated series component scores
        "final_generated_length": final_metrics.get("generated_length"),
        "final_numeric_generated_length": final_metrics.get("numeric_generated_length"),
        "final_expected_length": final_metrics.get("expected_length"),
        "final_invalid_numeric_count": final_metrics.get("invalid_numeric_count"),
        "final_length_score": final_metrics.get("length_score"),
        "final_numeric_validity_score": final_metrics.get("numeric_validity_score"),
        "final_timestamp_score": final_metrics.get("timestamp_score"),
        "final_range_score": final_metrics.get("range_score"),
        "final_trend_score": final_metrics.get("trend_score"),
        "final_seasonality_score": final_metrics.get("seasonality_score"),
        "final_anomaly_score": final_metrics.get("anomaly_score"),
        "final_noise_score": final_metrics.get("noise_score"),
        "final_detected_anomalies": final_metrics.get("detected_anomalies"),
        "final_noise_ratio": final_metrics.get("noise_ratio"),
        "final_valid_profile_metric": final_metrics.get("final_valid"),

        # Cost / time
        "elapsed_time_seconds": data.get("elapsed_time_seconds"),
        "timestamp_run": data.get("timestamp_run"),
    }

    return row


# ---------------------------------------------------------------------
# Derived columns and aggregation
# ---------------------------------------------------------------------

def add_derived_validation_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bool_cols = [
        # Legacy checks
        "raw_json_valid",
        "parse_repaired",
        "length_ok",
        "timestamps_not_null",
        "values_not_null",
        "daily_frequency_ok",
        "range_ok",
        "values_numeric",
        "increasing_trend_ok",
        "decreasing_trend_ok",
        "no_strong_trend_ok",
        "weekly_seasonality_ok",
        "anomaly_count_ok",
        "valid_series",
        "length_curation_applied",
        # New profile-based flags
        "profile_parse_success",
        "strict_llm_valid",
        "curated_series_valid",
        "profile_final_valid",
        "llm_final_valid",
        "final_valid_profile_metric",
    ]

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(_safe_bool)
        else:
            df[col] = False

    legacy_parse_success = df["raw_json_valid"] | df["parse_repaired"]
    df["parse_success"] = df["profile_parse_success"] | legacy_parse_success

    # Legacy recomputed structural/temporal validity is retained only as a
    # diagnostic. It should not be used as the main comparison metric.
    df["legacy_structural_valid"] = (
        df["parse_success"]
        & df["length_ok"]
        & df["daily_frequency_ok"]
        & df["range_ok"]
        & df["values_numeric"]
        & df["timestamps_not_null"]
        & df["values_not_null"]
    )

    def legacy_temporal_valid(row: pd.Series) -> bool:
        profile = row.get("profile_id")

        if profile == "P1_20":
            return _safe_bool(row.get("no_strong_trend_ok"))

        if profile in {"P2_28", "P5_60"}:
            return (
                _safe_bool(row.get("weekly_seasonality_ok"))
                and _safe_bool(row.get("no_strong_trend_ok"))
            )

        if profile in {"P3_28", "P6_60"}:
            return (
                _safe_bool(row.get("increasing_trend_ok"))
                and _safe_bool(row.get("weekly_seasonality_ok"))
            )

        if profile == "P4_30":
            return _safe_bool(row.get("decreasing_trend_ok"))

        return False

    df["legacy_temporal_valid"] = df.apply(legacy_temporal_valid, axis=1)
    df["legacy_final_recomputed"] = df["legacy_structural_valid"] & df["legacy_temporal_valid"]

    # Numeric helper columns
    numeric_cols = [
        "completed_points",
        "truncated_points",
        "curation_change_rate",
        "profile_compliance_score",
        "formal_validity_score",
        "temporal_profile_score",
        "llm_profile_compliance_score",
        "llm_formal_validity_score",
        "llm_temporal_profile_score",
        "final_profile_compliance_score",
        "final_formal_validity_score",
        "final_temporal_profile_score",
        "llm_length_score",
        "llm_numeric_validity_score",
        "llm_timestamp_score",
        "llm_range_score",
        "llm_trend_score",
        "llm_seasonality_score",
        "llm_anomaly_score",
        "llm_noise_score",
        "final_length_score",
        "final_numeric_validity_score",
        "final_timestamp_score",
        "final_range_score",
        "final_trend_score",
        "final_seasonality_score",
        "final_anomaly_score",
        "final_noise_score",
        "elapsed_time_seconds",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    df["curation_intensity"] = (
        _to_numeric(df["completed_points"]).fillna(0)
        + _to_numeric(df["truncated_points"]).fillna(0)
    )

    return df


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    aggregation = {
        "n_runs": ("experiment_id", "count"),
        "success_rate": ("status", lambda x: (x == "success").mean()),

        # Parse quality
        "raw_json_valid_rate": ("raw_json_valid", "mean"),
        "parse_repaired_rate": ("parse_repaired", "mean"),
        "parse_success_rate": ("parse_success", "mean"),

        # Main profile-based comparison metrics
        "strict_llm_valid_rate": ("strict_llm_valid", "mean"),
        "curated_series_valid_rate": ("curated_series_valid", "mean"),
        "mean_llm_profile_compliance_score": ("llm_profile_compliance_score", "mean"),
        "std_llm_profile_compliance_score": ("llm_profile_compliance_score", "std"),
        "mean_llm_formal_validity_score": ("llm_formal_validity_score", "mean"),
        "mean_llm_temporal_profile_score": ("llm_temporal_profile_score", "mean"),
        "mean_final_profile_compliance_score": ("final_profile_compliance_score", "mean"),
        "mean_final_formal_validity_score": ("final_formal_validity_score", "mean"),
        "mean_final_temporal_profile_score": ("final_temporal_profile_score", "mean"),

        # Curation diagnostics
        "length_curation_rate": ("length_curation_applied", "mean"),
        "mean_completed_points": ("completed_points", "mean"),
        "mean_truncated_points": ("truncated_points", "mean"),
        "mean_curation_intensity": ("curation_intensity", "mean"),
        "mean_curation_change_rate": ("curation_change_rate", "mean"),

        # Component scores, LLM output
        "mean_llm_length_score": ("llm_length_score", "mean"),
        "mean_llm_numeric_validity_score": ("llm_numeric_validity_score", "mean"),
        "mean_llm_range_score": ("llm_range_score", "mean"),
        "mean_llm_trend_score": ("llm_trend_score", "mean"),
        "mean_llm_seasonality_score": ("llm_seasonality_score", "mean"),
        "mean_llm_anomaly_score": ("llm_anomaly_score", "mean"),
        "mean_llm_noise_score": ("llm_noise_score", "mean"),

        # Component scores, final curated series
        "mean_final_length_score": ("final_length_score", "mean"),
        "mean_final_range_score": ("final_range_score", "mean"),
        "mean_final_trend_score": ("final_trend_score", "mean"),
        "mean_final_seasonality_score": ("final_seasonality_score", "mean"),
        "mean_final_anomaly_score": ("final_anomaly_score", "mean"),
        "mean_final_noise_score": ("final_noise_score", "mean"),

        # Legacy diagnostics retained for comparison/debugging
        "legacy_valid_series_rate": ("valid_series", "mean"),
        "legacy_structural_valid_rate": ("legacy_structural_valid", "mean"),
        "legacy_temporal_valid_rate": ("legacy_temporal_valid", "mean"),
        "legacy_final_recomputed_rate": ("legacy_final_recomputed", "mean"),
        "mean_slope": ("trend_slope", "mean"),
        "std_slope": ("trend_slope", "std"),
        "weekly_seasonality_rate": ("weekly_seasonality_ok", "mean"),
        "no_strong_trend_rate": ("no_strong_trend_ok", "mean"),
        "mean_weekly_autocorrelation": ("weekly_autocorrelation", "mean"),
        "anomaly_count_rate": ("anomaly_count_ok", "mean"),
        "mean_anomaly_count": ("anomaly_count_validation", "mean"),
        "mean_value": ("mean", "mean"),
        "mean_std": ("std", "mean"),

        # Cost
        "mean_time_seconds": ("elapsed_time_seconds", "mean"),
        "std_time_seconds": ("elapsed_time_seconds", "std"),
    }

    # Keep only aggregations for columns that exist, useful for old JSONs.
    aggregation = {
        name: spec
        for name, spec in aggregation.items()
        if spec[0] in df.columns
    }

    return df.groupby(group_cols, dropna=False).agg(**aggregation).reset_index()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize synthetic time-series generation experiments."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing *_metrics.json files.",
    )
    parser.add_argument(
        "--batch_id",
        type=str,
        default=None,
        help="Optional batch_id filter.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model names to keep.",
    )
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Optional list of profile ids to keep. Defaults to final profiles.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional list of methods to keep. Defaults to final + auxiliary methods.",
    )
    parser.add_argument(
        "--include_all_methods",
        action="store_true",
        help="Do not filter methods using the default experimental design.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    metric_files = sorted(output_dir.glob("*_metrics.json"))

    if not metric_files:
        print(f"No *_metrics.json files were found in {output_dir}")
        return

    rows: list[dict[str, Any]] = []
    for path in metric_files:
        try:
            rows.append(load_metric_file(path))
        except Exception as exc:
            print(f"Error reading {path}: {exc}")

    if not rows:
        print("Could not read any metrics file.")
        return

    df = pd.DataFrame(rows)

    profiles_to_keep = args.profiles if args.profiles is not None else FINAL_PROFILES
    if profiles_to_keep:
        df = df[df["profile_id"].isin(profiles_to_keep)].copy()

    if not args.include_all_methods:
        methods_to_keep = args.methods if args.methods is not None else METHODS_TO_SUMMARIZE
        df = df[df["method"].isin(methods_to_keep)].copy()
    elif args.methods is not None:
        df = df[df["method"].isin(args.methods)].copy()

    if args.models is not None:
        df = df[df["model"].isin(args.models)].copy()

    if args.batch_id is not None:
        df = df[df["batch_id"] == args.batch_id].copy()

    if df.empty:
        print("No results remained after applying the filters.")
        return

    df = add_derived_validation_columns(df)

    summary_path = output_dir / "summary_results.csv"
    by_model_profile_method_path = output_dir / "summary_by_model_profile_method.csv"
    by_model_method_path = output_dir / "summary_by_model_method.csv"
    by_model_path = output_dir / "summary_by_model.csv"
    by_profile_method_path = output_dir / "summary_by_profile_method.csv"

    df.to_csv(summary_path, index=False)

    by_model_profile_method = aggregate(
        df,
        ["batch_id", "model", "profile_id", "method"],
    )
    by_model_method = aggregate(
        df,
        ["batch_id", "model", "method"],
    )
    by_model = aggregate(
        df,
        ["batch_id", "model"],
    )
    by_profile_method = aggregate(
        df,
        ["batch_id", "profile_id", "method"],
    )

    by_model_profile_method.to_csv(by_model_profile_method_path, index=False)
    by_model_method.to_csv(by_model_method_path, index=False)
    by_model.to_csv(by_model_path, index=False)
    by_profile_method.to_csv(by_profile_method_path, index=False)

    print("=== INDIVIDUAL RESULTS ===")
    cols_to_print = [
        "experiment_id",
        "batch_id",
        "model",
        "profile_id",
        "method",
        "run_id",
        "parse_success",
        "strict_llm_valid",
        "curated_series_valid",
        "llm_profile_compliance_score",
        "llm_formal_validity_score",
        "llm_temporal_profile_score",
        "final_profile_compliance_score",
        "curation_change_rate",
        "llm_length_score",
        "llm_trend_score",
        "llm_seasonality_score",
        "llm_anomaly_score",
        "llm_noise_score",
        "elapsed_time_seconds",
    ]
    print(df[[c for c in cols_to_print if c in df.columns]].to_string(index=False))

    print("\n=== SUMMARY BY MODEL ===")
    sort_cols = [
        c for c in [
            "mean_llm_profile_compliance_score",
            "strict_llm_valid_rate",
            "mean_curation_change_rate",
        ]
        if c in by_model.columns
    ]
    ascending = [False, False, True][: len(sort_cols)]
    if sort_cols:
        print(by_model.sort_values(sort_cols, ascending=ascending).to_string(index=False))
    else:
        print(by_model.to_string(index=False))

    print("\n=== SUMMARY BY MODEL, PROFILE AND METHOD ===")
    print(
        by_model_profile_method.sort_values(
            ["model", "profile_id", "method"],
        ).to_string(index=False)
    )

    print("\nSaved files:")
    print(f"- {summary_path}")
    print(f"- {by_model_profile_method_path}")
    print(f"- {by_model_method_path}")
    print(f"- {by_model_path}")
    print(f"- {by_profile_method_path}")


if __name__ == "__main__":
    main()