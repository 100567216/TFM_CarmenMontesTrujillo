# TFM_CarmenMontesTrujillo
Synthetic Data Generation with LLMs
Code accompanying the Master's Thesis *"Synthetic Data Generation with LLMs"*
(Master's Degree in Big Data Analytics, Universidad Carlos III de Madrid, 2025-2026).

This repository contains the experimental pipeline used to generate, parse,
curate and evaluate synthetic time series produced by local LLMs (via
[Ollama](https://ollama.com)) under eleven prompting/context-construction
strategies and six controlled temporal profiles.

## Repository structure

- `src/profiles.py` — Defines the six official temporal profiles (`P1_20` ... `P6_60`)
  used throughout the experiments, including length, frequency, value range,
  trend, seasonality, anomaly count, noise level and validation thresholds.
- `src/build_reference_bank.py` — Generates the synthetic reference bank used
  by the example-based methods (`random_few_shot`, `harmonic_knn`,
  `tabgen_icl_residual`, `epic_grouped_attribute_few_shot`) and computes
  temporal features for each reference series.
- `src/run_experiment.py` — Runs a single generation experiment: builds the
  method-specific prompt, calls the local LLM through Ollama, parses and
  normalizes the output, applies structural curation when needed, validates
  the result, and stores the run-level artifacts.
- `src/profile_metrics.py` — Implements the structural and temporal
  validation metrics (length, numeric validity, range, timestamp, trend,
  seasonality, anomalies, noise) used to score every generated series
  against its target profile.
- `src/summarize_results.py` — Aggregates the `*_metrics.json` files produced
  by `run_experiment.py` into the summary CSVs used in Chapter 5 of the thesis.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally, with the
  required models pulled:
```bash
  ollama pull qwen2.5:0.5b
  ollama pull tinyllama:1.1b
  ollama pull llama3.2:1b
  ollama pull mistral:7b
```
- Python dependencies:
```bash
  pip install -r requirements.txt
```

## Reproducing the experiments

### 1. Build the reference bank

```bash
python src/build_reference_bank.py --n_per_profile 30 --seed 42
```

This creates `reference_bank/<profile_id>/*.csv` and
`reference_bank/reference_bank_features.csv`.

### 2. Run a single generation experiment

```bash
python src/run_experiment.py \
    --method value_only_controlled_scale \
    --profile P2_28 \
    --model llama3.2:1b \
    --run_id 1 \
    --batch_id phase1
```

Outputs are written to `outputs/` by default (`--output_dir` to change it):
`<experiment_id>_prompt.txt`, `<experiment_id>_raw_output.txt`,
`<experiment_id>_synthetic_series.csv`, `<experiment_id>_metrics.json`.

### 3. Run the full experimental matrix

Phase 1 (method screening, fixed model `llama3.2:1b`, 11 methods × 6 profiles
× 30 repetitions = 1,980 runs) and Phase 2 (LLM comparison, 4 models × 5
selected methods × 6 profiles × 30 repetitions = 3,600 runs) are launched by
looping over `run_experiment.py` with the corresponding `--method`,
`--profile`, `--model`, `--run_id` and `--batch_id` combinations. Example for
Phase 1:

```bash
for method in prompt_only constraint_guided canonical_few_shot random_few_shot \
              harmonic_knn tabgen_icl_residual value_only_controlled_scale \
              chatts_attribute_prompting epic_grouped_attribute_few_shot \
              validator_feedback_refinement cllm_generate_curate; do
  for profile in P1_20 P2_28 P3_28 P4_30 P5_60 P6_60; do
    for run in $(seq 1 30); do
      python src/run_experiment.py --method "$method" --profile "$profile" \
          --model llama3.2:1b --run_id "$run" --batch_id phase1
    done
  done
done
```

Raw per-run artifacts are not included in this repository due to their
volume (~5,580 runs x 4 files); they are fully reproducible with the
commands above and available on request.

### 4. Summarize results

```bash
python src/summarize_results.py --output_dir outputs --batch_id phase1
```

This produces, inside `outputs/`: `summary_results.csv`,
`summary_by_model.csv`, `summary_by_model_method.csv`,
`summary_by_profile_method.csv` and `summary_by_model_profile_method.csv`.
The versions used for Chapter 5 of the thesis are provided precomputed in
`results/phase1/` and `results/phase2/`.

## License

This code is released under the Creative Commons Attribution – Non
Commercial – Non Derivatives license, matching the license of the
accompanying thesis.
