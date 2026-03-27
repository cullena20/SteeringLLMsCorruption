# Experiment Workflows

The four main corruption experiment scripts each test a different corruption scheme applied to steering
activations, then measure how much steering performance degrades as corruption increases. All
four scripts share the same pipeline structure and CLI interface; only the corruption function
and sweep variable differ.

---

## Corruption Experiments

| Script | Corruption scheme | Sweep variable |
|---|---|---
| `behavior_injection.py` | Replace an eta fraction of a behavior's activations with activations from a *different* behavior | eta (per inlier×outlier pair) | 
| `mislabel.py` | Swap positive and negative labels for an eta fraction of training data | eta 
| `random_injection.py` | Replace an eta fraction of activations with those from randomly generated sentences | eta 
| `synthetic_corruption.py` | Place outliers to directly control angle of sample difference of means to the ground truth difference of means. | theta (degrees)

---

## Shared Pipeline

Every script runs the following phases in order:

**1. Setup**
- Load dataset (see [Eval Modes](#eval-modes) below; default is the MCQ behavior dataset from `tan_paper_datasets/mwe/xrisk`).
- Load activations from `activations/{activations_name}.pkl` and wrap in `CorruptedActivations`, to enable applying corruption.
- Load model (`SteerableModel`) and select per-behavior alpha (steering magnitude) mapping based on model name (`llama`/`olmo`/`mistral` substring match)

**2. Apply corruption**

For each value in the sweep (eta or theta), call `corrupted_activations.apply_corruption(...)` across `runs` independent seeds. Seeds vary the portion of data corrupted, and for `synthetic_corruption.py` additionally vary the locations of outliers. All corrupted versions are stored in `corrupted_activations.corrupted_versions`.

**3. Evaluate**

For each `(corruption_name, run)`:
1. Compute steering vectors with each estimator via `get_steering_vecs(estimators, behaviors, eta=..., include_sample_uncorrupted=True)`. This returns both the estimator-computed vectors and the `inlier_sample_diff_of_means` baseline (computed on uncorrupted data). Precomputed vectors can be supplied via `--supply-steering-path` to skip this step.
2. Evaluate the steered model on the test set (see [Eval Modes](#eval-modes)).
3. Accumulate results into a shared `ExperimentOutput` object. The no-steer baseline (`include_no_steer=True`) is only added on the very first run to avoid duplicate rows.

**4. Save outputs**

- `ExperimentOutput` pickle → `experiment_results/{script_name}/{activations_name}_{postfix}.pkl`
- Steering vectors + outlier detection info → `experiment_results/{script_name}/{activations_name}_steering_vecs_and_outlier_detected_data_{postfix}.pkl`
- Performance-vs-sweep-variable grid plots → `saved_plots/{script_name}/`, one per metric (`percent_steered` and `avg_score`; or `score` for TinyMMLU). Each plot shows all estimators as separate lines, averaged over runs with std-dev error bars. Skipped when `grab_generations=True`, since this only grabs generations and no results.
---

## Eval Modes

Each script supports three evaluation modes, selected by flags:

| Flag | Dataset loaded | Evaluation | Notes |
|---|---|---|---|
| *(default)* | MCQ behavior dataset (`tan_paper_datasets/mwe/xrisk`) | Logit-based accuracy on behavior test questions | Primary evaluation |
| `--custom-dataset-path` | Folder of `.jsonl` files at the given path | Same as default | Overrides default; use with `--behaviors` |
| `--grab-generations` | Open-ended behavior questions (`datasets/open_ended/`) | Free-form steered generations saved to disk | Requires LLM judge post-processing to score |
| `--tiny-mmlu` | tinyMMLU benchmark | MCQ accuracy on general knowledge questions | Measures effect of steering on general capability |
| `--use-large-dataset` | Hardcoded large dataset via `large_dataset_generator` | Same as default | Requires a separately extracted activations file |

---

## Per-Script Notes

### `behavior_injection.py`

The most complex script. The sweep is over all (inlier, outlier) behavior pairs × etas, not just etas. For each pair, both behaviors are evaluated at test time: the inlier evaluation measures steering degradation, while the outlier evaluation checks whether the injected behavior was inadvertently instilled — the primary safety concern for this experiment.

Because of the all-pairs structure, steering vectors are keyed by `corruption_name` (e.g. `uncorrigible-neutral-HHH_corruptedby_myopic-reward_0.3`) rather than just eta. The default `corruption_mapping` runs all 6×5=30 pairs; pass `--inlier-behaviors` and `--outlier-behaviors` (equal-length lists) to run a custom subset.

This script also loads precomputed inter-behavior cosine similarity and separability stats for use in plots, but these can be removed.

**Usage:**
```bash
python experiments/behavior_injection.py \
  --model_path meta-llama/Llama-3.2-3B-Instruct \
  --activations_name Llama-3.2-3B-Instruct_layer13 \
  --alpha 1 \
  --layer 12 \
  --behaviors uncorrigible-neutral-HHH myopic-reward   # subset of inlier behaviors to evaluate
```

---

### `mislabel.py`

The simplest corruption scheme — no extra data source needed. Label swapping is equivalent to flipping which activations are treated as positive vs. negative.

**Usage:**
```bash
python experiments/mislabel.py \
  --model_path meta-llama/Llama-3.2-3B-Instruct \
  --activations_name Llama-3.2-3B-Instruct_layer13 \
  --alpha 1 \
  --layer 12
```

---

### `random_injection.py`

**Prerequisite:** Random sentence activations must be pre-extracted before running this script. The extraction code is present in the script (commented out) and must be run once to produce `misc_pickles/{activations_name}_{random_sentence_list_pkl_name}`. The default sentence source is `datasets/per_character_random_sentences.pkl`, capped at 2000 sentences.

Does not support `--grab-steering-vecs-only` or `--supply-steering-path`.

**Usage:**
```bash
python experiments/random_injection.py \
  --model_path meta-llama/Llama-3.2-3B-Instruct \
  --activations_name Llama-3.2-3B-Instruct_layer13 \
  --alpha 1 \
  --layer 12
```

---

### `synthetic_corruption.py`

Unlike the other three scripts, the sweep variable is `theta` (angle in degrees between the corrupted and ground-truth steering vectors), not corruption percentage. Outliers are clustered in opposite directions orthogonal to the ground-truth steering vector from the ground-truth sample means to achieve the desired angle. Runs vary the orthogonal directions chosen to cluster outliers at. The corruption fraction is fixed at eta=0.3. The x-axis of result plots is labelled "Angle To Inlier Diff-In-Means (degrees)" rather than "Corruption Percentage".

**Usage:**
```bash
python experiments/synthetic_corruption.py \
  --model_path meta-llama/Llama-3.2-3B-Instruct \
  --activations_name Llama-3.2-3B-Instruct_layer13 \
  --alpha 1 \
  --layer 12
```


---

---

## Supporting Scripts

### `get_activations.py`

**Prerequisite for all other experiments.** Extracts activations from a model on the training split of the behavior dataset and saves a pickled `Activations` object to `activations/{save_name}.pkl`. The object already contains `sample_diff_of_means` steering vectors (computed on the clean training data), which are used directly by `evaluate_steerability_vs_alpha.py` and as a starting point for the corruption experiments.

**Pipeline:**
1. Load dataset (`tan_paper_datasets/mwe/xrisk` by default, or the large dataset variant).
2. Run `SteerableModel.get_binary_activations_on_dataset` to extract activations at the specified layer at the `-2` token position (the answer letter token, one before the closing parenthesis).
3. Compute `sample_diff_of_means` steering vectors and store them in the `Activations` object.
4. Save the full `Activations` object to `activations/{save_name}.pkl`.

The default save name is `{model_name}_layer{layer+1}` (note: 1-indexed in the filename, 0-indexed in the code). Pass `--save-name` to override.

**Usage:**
```bash
# Default dataset
python experiments/get_activations.py \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --layer 12

# Custom dataset (folder of .jsonl files; behaviors must be specified explicitly)
python experiments/get_activations.py \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --layer 12 \
  --custom-dataset-path path/to/my_dataset \
  --behaviors my-behavior-a my-behavior-b
```

**Key flags:**
- `--layer`: layer index (0-indexed, default 12)
- `--custom-dataset-path`: path to a folder of `.jsonl` files; overrides the default dataset
- `--behaviors`: required when using a custom dataset; specifies which behaviors to extract
- `--use-large-dataset`: switches to the three-behavior large dataset variant used in dataset size ablations

---

### `evaluate_steerability_vs_alpha.py`

Sweeps the steering magnitude (alpha) over a range of values and records steering performance at each. Used to (a) confirm behaviors are steerable, and (b) select per-behavior alpha hyperparameters for the corruption experiments.

**Pipeline:**
1. Load a saved `Activations` object; read `steering_vectors` already stored in it (no re-extraction).
2. Load the MCQ test dataset.
3. Call `evaluate_across_behaviors_and_vecs` with `alpha_values` as the sweep (default: -2 to +2 in 0.25 steps), using logit-based eval (`logits_only=True`, `logit_aggregation_method="entire_sequence"`).
4. Save the `ExperimentOutput` pickle to `experiment_results/baseline_steerability/{activations_name}.pkl`.

No corruption is applied — this is a clean-data steerability check only. Does not generate plots; results are analyzed in notebooks.

**Usage:**
```bash
python experiments/evaluate_steerability_vs_alpha.py \
  --model_path meta-llama/Llama-3.2-3B-Instruct \
  --activations_name Llama-3.2-3B-Instruct_layer13 \
  --layer 12
```

---

### `grab_gpt_evals.py`

Post-processing step for the LLM judge ablation. Loads a set of `ExperimentOutput` pickles that contain steered free-form generations (produced by any corruption experiment run with `--grab-generations`), scores each generation with GPT-4o-mini via `run_gpt_eval_on_experiment`, and saves the scored results to `experiment_results/gpt_evals/{name}.pkl`.

Unlike the other scripts, this is **not a CLI script** — it has no `argparse` interface. The input files and experiment names are hardcoded directly in the script. To score a different set of experiments, edit the file paths and `exps` dict at the top of the script.

Scoring runs in parallel threads (default `max_workers=5`). Requires an OpenAI API key in `.env`.

---

## Estimators

All four corruption experiment scripts accept an `--estimator-names` flag to select which estimators compute the steering vector from corrupted activations. By default only `lee_valiant_diff` and `sample_diff_of_means` are used. (`evaluate_steerability_vs_alpha.py` does not support this flag — it uses the steering vectors already stored in the `Activations` object.)

### Estimator Interface

An estimator is a callable with signature:

```python
estimator(pos_acts: np.ndarray, neg_acts: np.ndarray, **kwargs) -> np.ndarray
# or, for estimators that identify outliers:
estimator(pos_acts, neg_acts, **kwargs) -> (np.ndarray, (pos_outlier_indices, neg_outlier_indices))
```

- **Input:** `pos_acts` and `neg_acts` are 2D numpy arrays of shape `(n_samples, d_hidden)` — activations for the positive and negative examples respectively.
- **Output:** A 1D numpy array of shape `(d_hidden,)` — the steering vector (difference of robust means). Estimators that support outlier detection may additionally return a tuple of index arrays.

### Available Estimators

All estimators are defined as lambdas in `config.py` from functions in `estimators/` and registered in `ESTIMATOR_MAPPING`:

| Name | Strategy | Notes |
|---|---|---|
| `sample_diff_of_means` | diff | Plain sample mean; used as the corrupted-data baseline |
| `lee_valiant_diff` / `lee_valiant_match` | diff / match | Lee & Valiant (2022) filter; most effective in the paper |
| `lrv_diff` / `lrv_match` | diff / match | LRV robust mean |
| `que_diff` / `que_match` | diff / match | QUE robust mean |
| `que_diff_force_prune` | diff | QUE variant that forces pruning at `0.5*tau`; included for ablation, generally ineffective |
| `med_mean_diff` / `med_mean_match` | diff / match | Median of means |
| `coord_prune_diff` / `coord_prune_match` | diff / match | Coordinate-wise trimmed mean |

### Adding a Custom Estimator

1. Implement a function matching the interface above (pure numpy, shape `(n, d) → (d,)`).
2. Add it to `ESTIMATOR_MAPPING` in `experiments/config.py`:
   ```python
   ESTIMATOR_MAPPING["my_estimator"] = lambda pos, neg: my_estimator_fn(pos, neg)
   ```
3. Pass it by name to any experiment script via `--estimator-names`.

### Custom Mean-Based Estimators

For mean-based estimators, two aggregation strategies are used:
- **`diff` variants** — compute `robust_mean(pos) - robust_mean(neg)` independently on each side.
- **`match` variants** — compute `robust_mean(pos - neg)`, treating each pos/neg pair as a difference vector (requires equal-sized, paired data). This was found to not generally provide improvement.

The wrappers `diff_of_means` and `mean_of_diffs` from `estimators/steering_estimator_wrappers.py` handle both strategies. They accept any mean function with signature `mean_fun(data: np.ndarray, **kwargs) -> np.ndarray` (input shape `(n, d)`, output shape `(d,)`), passed via the `mean_fun` keyword argument.

To add a new mean-based estimator, implement the mean function and wrap it in a lambda:

```python
from estimators.steering_estimator_wrappers import diff_of_means, mean_of_diffs
from my_module import my_robust_mean  # mean_fun(data: np.ndarray) -> np.ndarray

# diff variant: mean(pos) - mean(neg), computed independently
my_estimator_diff = lambda pos, neg: diff_of_means(pos, neg, mean_fun=my_robust_mean)

# match variant: mean(pos - neg), requires paired equal-length data
my_estimator_match = lambda pos, neg: mean_of_diffs(pos, neg, mean_fun=my_robust_mean)
```

If your mean function supports outlier detection (returning `(mean, outlier_indices)`), pass `return_outlier_indices=True` and any additional kwargs through the lambda:

```python
# With outlier detection and a tau hyperparameter (as in lee_valiant, que)
my_estimator_diff = lambda pos, neg, tau: diff_of_means(
    pos, neg, tau=tau, mean_fun=my_robust_mean, return_outlier_indices=True
)
```

For reference, all existing estimators in `config.py` follow this pattern — for example:

```python
lee_valiant_diff  = lambda pos, neg, tau: diff_of_means(pos, neg, tau=tau, mean_fun=lee_valiant_simple, return_outlier_indices=True)
lee_valiant_match = lambda pos, neg, tau: mean_of_diffs(pos, neg, tau=tau, mean_fun=lee_valiant_simple, mismatch=False, return_outlier_indices=True)
med_mean_diff     = lambda pos, neg:      diff_of_means(pos, neg, mean_fun=median_of_means)
```

Once defined, register and use exactly as described in [Adding a Custom Estimator](#adding-a-custom-estimator) above.

---

## Shared Config (`config.py`)

All estimator lambdas, `ESTIMATOR_MAPPING`, per-model alpha dicts (`LLAMA/OLMO/MISTRAL_ALPHA_VALUES`), and `DEFAULT_BEHAVIORS` are defined in `config.py` and imported by each script. For experiments over additional behaviors or models, find appropriate layers and alpha values and hardcode them into this script. 
