# src/ Overview

This document provides an overview of the files in the `src` directory as well as the general data flow used across experiments. To better understand usage, see experimental scripts under `experiments` along with `experiments/workflow.md`.

Basic pipeline:
1. Collect paired activations — same question, positive answer (behavior present) vs. negative answer (behavior absent)
2. Compute a **steering vector** — the difference of means between positive and negative activations
3. Apply the vector at inference time (add `alpha * vec` to the residual stream)
4. Measure how often the steered model picks the behavior answer

The corruption experiments inject noise into step 1, then measure how badly step 2–4 degrades, and whether **robust mean estimators** in step 2 can defend against it.

---

## Data Flow

```
DataSet (dataset.py)
    │ train_data / test_data: {behavior: DataDict}
    ▼
SteerableModel.get_binary_activations_on_dataset (steerable_model.py)
    │ positive and negative activations per behavior/layer/token_pos
    ▼
Activations (activations.py)
    │ .data[behavior][layer][token_pos] = {"pos": Tensor, "neg": Tensor}
    │
    ├─ (corruption experiments only) ──────────────────────────────────────────┐
    │                                                                           │
    │   CorruptedActivations.apply_corruption (corrupted_activations.py)       │
    │       │ corruption functions from corruption.py                           │
    │       │ .corrupted_versions[corruption_name][run] = Activations           │
    │       ▼                                                                   │
    │   Activations.get_steering_vecs                                           │
    │       │ applies estimators (estimators/) to corrupted data                │
    │       │                                                                   │
    └───────┘ (clean data: also calls get_steering_vecs)                       │
                                                                                │
    SteeringVecsDict: {behavior: {layer: {token_pos: {estimator: vector}}}}    │
    ▼                                                                           │
eval_utils.evaluate_across_behaviors_and_vecs                                  │
    │ calls model forward passes with hook-based steering                       │
    ▼                                                                           │
ExperimentOutput (experiment_output.py) ◄──────────────────────────────────────┘
    │ .to_dataframe() → pandas DataFrame
    │ .plot_performance_grid() → figures saved to saved_plots/
```

---

## Files

### `dataset.py` — Data Loading

**`DataSet`** loads JSON/JSONL files from `datasets/` subfolders and splits into train/val/test (configurable test size, default 200). Produces `train_data` and `test_data` dicts of form `{behavior: DataDict}`.

Three format types:
- `"qa"` (default): MCQ with `(A)` / `(B)` answer choices
- `"general"`: questions without explicit answer format
- `"open_ended"`: free-form prompts scored by an LLM judge

Throughout experiments, we instantiate this object as `DataSet(subfolders=["tan_paper_datasets/mwe/xrisk], test_size=200, format_type="qa")` for logit-based evals and `DataSet(subfolders=["open_ended"], test_size=200, format_type="open_ended)` for open-ended evals.

To replicate the logit-based setup across different datasets, follow this format:

- Use contrastive behavior datasets with two binary answers **(A)** and **(B)**. Each data point should consist of a question with these two answer choices, where one corresponds to the presence of the behavior and the other to its absence. Be careful to balance the distribution of (A) and (B) answers across behaviors, as an imbalance may inadvertently steer the model toward the biased answer token.
- Store datasets as `.jsonl` files, where each line corresponds to a single contrastive data point.
- Each line should contain the keys `"question"`, `"answer_matching_behavior"`, and `"answer_not_matching_behavior"`.

To replicate the open-ended setup used for evals, use a `.jsonl` file where each line is a data point with only a `"question"` key.

---

### `steerable_model.py` — Model Wrapper

**`SteerableModel`** wraps a HuggingFace `AutoModelForCausalLM` with activation extraction and hook-based steering. The main things it adds on top of a raw HuggingFace model:

- **`get_binary_activations_on_dataset`** — the main entry point for collecting training activations. Iterates over behaviors, calls `batched_activations` on positive and negative examples, and returns an `Activations` object.

- **`batched_activations`** — runs a forward pass and captures residual stream activations at specified layers and token positions via PyTorch hooks. Token positions can be an integer index, `-1` (last), `"mean"`, or `"answer_token"` (a per-behavior fixed index). Be careful to choose the correct answer token for multiple choice evals, in the case of answers of the form `(A)`, this is generally the second to last token position. Returns shape `(num_layers, num_token_positions, num_instructions, d_model)`.

- **`get_binary_logit_probs`** — one forward pass per example; returns per-token log-probabilities for both the positive and negative answer continuations. This is what powers the primary logit-based MCQ evaluation.

- **`generate`** — generates text with optional steering hooks. Used for open-ended evals.

- **`get_answer_letters`** — returns the argmax over `[A, B, C, D]` logits at the last token. Used for TinyMMLU.

- **`create_fwd_hook_from_vec`** / **`_register_hooks`** — build and attach PyTorch forward hooks from a `(vec, alpha, layers)` triple. Hooks use TransformerLens-style naming (`blocks.{layer}.hook_resid_pre`) translated to HuggingFace layer modules.

---

### `activations.py` — Activation Storage and Steering Vector Computation

**`Activations`** is the central data container. Its `data` dict holds paired positive/negative activations for all (behavior, layer, token_pos) combinations.

- **`add_behavior`** — populates `data` for one behavior from the tensor pair output by `batched_activations`.

- **`get_steering_vecs`** — the core computation step. Iterates over all (behavior, layer, token_pos) entries and calls each estimator function as `fn(pos_acts, neg_acts, tau=eta)`. Returns a `SteeringVecsDict`. When `include_sample_uncorrupted=True` (used in all corruption experiments), it also computes `inlier_sample_diff_of_means` on the uncorrupted data as a ground-truth reference. If an estimator returns `(vector, outlier_indices)`, those indices are compared against the true corrupt indices to compute precision/recall outlier detection stats.

- **`merge`** — concatenates two `Activations` objects along the instruction dimension. Used when activations come from multiple batches or sources.

- **`save` / `load`** — persist/restore to `activations/` (full object) or `steering_vecs/` (vectors only).

- Visualization: `plot_activation_projection` projects activations onto the steering direction (x-axis) vs. a PCA residual (y-axis), with optional outlier highlighting. `plot_interactive` is a Plotly version with hover text.

---

### `corrupted_activations.py` — Corruption Management

**`CorruptedActivations`** (inherits `Activations`) adds a `corrupted_versions` dict that stores multiple corrupted copies of the same base activations.

```
corrupted_versions[corruption_name] = {
    0: Activations,          # run 0
    1: Activations,          # run 1
    ...
    "corruption_details": dict  # metadata (corruption function name, eta, seed info, etc.)
}
```

**`apply_corruption`** — applies a corruption function to every (behavior, layer, token_pos) entry and stores the result under `corruption_name`. Key parameters:

- `corruption_fun` - corruption function. Takes in `(pos_acts, neg_acts, eta, **kwargs**)` and return `(corrupted_pos, corrupted_neg, outlier_indices)`.
- `eta` — corruption fraction, passed to the corruption function
- `multiple_runs_kwargs` — `{run_index: {**run_kwargs}}` to produce multiple independent runs (typically differing by random seed)
- `**kwargs` - additionaly keyword arguments are passed into `corruption_fun`, for example `corrupt_with_shared_random`, which injects random activations has a `random_acts` argument that can be passed in here. 

Note that `outlier_indices` are stored in the resulting `Activations` data dict and later consumed by `get_steering_vecs` to compute outlier detection stats, and was added late in development.

---

### `corruption.py` — Corruption Functions

All functions take `(pos_acts, neg_acts, eta, **kwargs**)` and return `(corrupted_pos, corrupted_neg, outlier_indices)`. These are passed to `CorruptedActivations.apply_corruption`.

To test new corruption methods within this overall pipeline, define

| Function | Description | Used by |
|---|---|---|
| `label_corruption` | Swap pos/neg labels for `eta` fraction of examples | `mislabel.py` |
| `corrupt_with_shared_random` | Replace `eta` fraction with random-sentence activations | `random_injection.py` |
| `corrupt_with_other_behavior` | Replace `eta` fraction with activations from a different behavior | `behavior_injection.py` |
| `angle_outlier_corruption` | Place outliers so corrupted diff-of-means is at angle `theta` from ground truth | `synthetic_corruption.py` |

Utilities:
- **`get_acts_excluding_behavior`** — returns all activations except those for a specified behavior; used to build injection pools.
- **`generate_random_sentences`** — generates random ASCII sentences for the random injection pool.
- **`grid_search_steering_corruption`** — sweeps corruption parameters to find the level that degrades steering to a target percentage. Used in exploratory experiments

---

### `eval_utils.py` — Steering Evaluation

Functions called by experiment scripts in their inner evaluation loops.

**Core pipeline (called in sequence):**

1. **`get_steering_results`** — runs the model on a test set with a steering vector applied, returning raw generations or logit values as a list of `SteeringResult` dicts. Results used in `model.get_binary_logit_probs` (logit eval, the default: stores logits from the entire question and answer sequence for positive and negative data points) or `model.generate` (free-form generation, stores model generations). 

2. **`eval_from_results`** — aggregates raw results into an `EvalDict`. 
The main evaluation mode is:
   - `"logit_diff"`: compares summed log-probabilities of positive vs. negative continuations (`last_token`, `entire_sequence`, or `normalized`)

You can change the logit-aggregation-method to be `last_token` or `normalized`, but for the multiple choice evaluation, only `entire_sequence` makes sense. This works for data points sharing a common question prefix, so this gets the difference in probabilities between the positive and negative logits.

This then stores information, particularly:
    - `percent_steered`: the percentage of data points where the answer demonstrating the behavior has a larger probability.
    - `avg_score`: the average difference in logits between the answer demonstrating the behavior and the answer not demonstrating the answer.

Both can be used as steering performance metrics in plots.

3. **`grab_results_and_evaluate_steering`** — wraps the above two into one call, returning `{results, eval}`.

**Experiment-level wrappers (called directly by experiment scripts):**

- **`evaluate_across_behaviors_and_vecs`** — outer loop over all (behavior, layer, token_pos, estimator, alpha) combinations in a `SteeringVecsDict`. Accumulates results into an `ExperimentOutput`. Adds a `no_steer` baseline if `include_no_steer=True`. Supports per-behavior alpha overrides via `behavior_alpha_mapping`.

- **`grab_generations_across_behaviors_and_vecs`** — same loop structure but saves raw free-form generations instead of running logit eval. Used in `--grab-generations` mode.

- **`tinymmlu_eval`** — evaluates a steered model on TinyMMLU using `model.get_answer_letters`. Reports 4-choice MCQ accuracy.

---

### `experiment_output.py` — Result Storage and Plotting

**`ExperimentOutput`** accumulates results from a full corruption experiment run and supplies plotting functions.

- **`add_result`** — appends one result record (behavior, layer, token_pos, estimator, alpha, run, varying_variable, eval stats). Each call adds one row to the eventual DataFrame.

- **`to_dataframe`** — converts all stored results to a pandas DataFrame for analysis. The `additional_kwargs` dict from each result becomes additional columns.

- **`update_results`** — allows post-hoc updates (e.g., adding GPT judge scores after generation).

- **`plot_performance`** — plots one (behavior, layer, token_pos, alpha) slice: metric vs. `varying_variable_value` (e.g., corruption fraction), with each estimator as a separate line and std-dev error bars across runs.

- **`plot_performance_grid`** — wraps `plot_performance` into a grid of subplots, one per behavior. This is the method called at the end of each experiment script to produce the figures saved to `saved_plots/`.

Three modes (set at construction time):
- `standard_eval_mode` (default): stores `EvalOutput` TypedDicts
- `generation_mode`: stores `LLMJudgeEvalOutput` TypedDicts (used with `--grab-generations`)
- `benchmark_mode`: stores plain dicts (used with `--tiny-mmlu`)


---

### `geometric_evals.py` — Geometric Analysis

Evaluates steering vector quality by comparing to the ground-truth direction (cosine similarity and L2 distance), rather than by running the model.

- **`geometric_eval_single`** — computes cosine similarity and L2 distance between an estimated and ground-truth steering vector.
- **`geometric_eval_wrapper`** — runs `geometric_eval_single` across all entries in a `SteeringVecsDict`, organized by corruption level and run.
- **`plot_geometric_eval`** / **`plot_geometric_eval_multi`** — plot cosine similarity or L2 distance vs. corruption level for a set of estimators.

---

### `gpt_evals.py` — LLM Judge Scoring

Parallel GPT-4o-mini scoring for open-ended steered generations. Called by `experiments/grab_gpt_evals.py` as a post-processing step after `--grab-generations` runs.

- **`generate_scoring_rubric`** — builds a per-behavior scoring prompt (0–10 scale for degree of behavior presence).
- **`score_single_generation`** — scores one (question, generation) pair using GPT-4o-mini.
- **`run_gpt_eval_on_experiment`** — iterates over all generations in an `ExperimentOutput`, scores in parallel threads (via `RateLimiter`), and stores scores back via `update_results`.
- **`RateLimiter`** — token-bucket rate limiter for the OpenAI API.

---

### `utils.py` — General Utilities

- **`set_global_seed`** — sets Python / NumPy / PyTorch seeds for reproducibility.
- **`clear_memory`** — calls `gc.collect()` and `torch.cuda.empty_cache()` between batches.
- **`cosine_similarity`** — cosine similarity between two vectors.
- **`outlier_pruning_stats`** — given detected vs. ground-truth outlier indices, computes `pct_outliers_pruned` (recall) and `pct_inliers_pruned` (false positive rate). Used in `get_steering_vecs` to track estimator outlier detection quality.
- **`format_mmlu_prompt`** — formats a TinyMMLU question into the prompt string.
- **`large_dataset_generator`** / **`BehaviorSplit`** / **`large_dataset_to_behavior_dict`** — utilities for the large dataset ablation variant (loads from HuggingFace, splits by behavior).
- **`single_direction_hook`** — adds `alpha * steering_vector` to the residual stream. Expects HuggingFace tuple output format.
- **`normalize_steering_dir`** — normalizes all vectors in a `{estimator: vector}` dict to unit norm.

---

### `interfaces.py` — Shared Type Definitions

TypedDict definitions for the major data structures that flow between modules. The types below are referenced throughout the codebase; this is the canonical place to check their exact shapes.

- **`SteeringVecsDict`**: `{behavior: {layer: {token_pos: {estimator_name: vector}}}}`. The format returned by `Activations.get_steering_vecs()` and consumed by `evaluate_across_behaviors_and_vecs`.

- **`SteeringResult`**: One entry from a steering run — `question`, `generation`, `pos_answer`, `neg_answer`, `steering_type`, `alpha`, `open_ended`, `logits_only`. When `logits_only=True`, `generation` is a `BinaryLogitProbGeneration` dict rather than a string.

- **`BinaryLogitProbGeneration`**: `{"pos": List[float], "neg": List[float]}` — per-token log probabilities over the full sequence for the positive and negative continuations, as returned by `get_binary_logit_probs`.

- **`EvalDict`**: Aggregated evaluation stats for one (behavior, estimator, run) combination — `percent_steered`, `percent_ambiguous`, `avg_score`, `eval_type`, `eval_details`, counts. Produced by `eval_from_results`.

- **`EstimatorData`**: `{"results": SteeringResults, "eval": EvalDict}` — raw results paired with their evaluation. Produced by `grab_results_and_evaluate_steering`.

- **`EvalOutput`**: One row stored inside `ExperimentOutput` — all fields of `EvalDict` plus `behavior`, `layer`, `token_pos`, `estimator`, `alpha`, `steering_vec`, `run`, `varying_variable`, `varying_variable_value`, `additional_kwargs`.

- **`LLMJudgeEvalOutput`**: Variant of `EvalOutput` used in generation mode — replaces per-token eval stats with `questions_generations` (list of question/response pairs) and `judge_scores` (list of 0–10 scores from the GPT judge).

- **`EvalType`**: `Literal["open_ended", "multiple_choice_accuracy", "logit_diff"]`. Note the actual eval type string produced at runtime for logit evals is `"logit_diff-{method}"` (e.g. `"logit_diff-entire_sequence"`), which extends beyond this Literal.
