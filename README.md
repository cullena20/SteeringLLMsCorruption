# Understanding and Mitigating Dataset Corruption in LLM Steering

Codebase for [📄 Understanding and Mitigating Dataset Corruption in LLM Steering](https://arxiv.org/abs/2603.03206).

Implementations for robust estimators under `estimators/` copied from [github.com/cullena20/RobustMeanEstimation](https://github.com/cullena20/RobustMeanEstimation).


## Overview

Contrastive steering is a simple and effective method for adjusting the generative behavior of LLMs at inference time. It identifies a direction in an intermediate activation layer from prompt–response pairs with and without a target trait, then shifts activations along this direction at inference time. Despite its growing use in AI safety applications, the robustness of contrastive steering to noisy or adversarial data corruption is poorly understood.

This paper initiates a study of that robustness. We examine four corruption types — mislabelling, random injection, coordinated behavior injection, and geometric (activation-space) corruption — and measure how steering performance degrades as corruption grows. We further evaluate whether replacing the standard mean computation with **robust mean estimators** can defend against corruption.

**Key findings:**
- Steering is robust to moderate corruption (up to ~10–20% of training data) but degrades sharply beyond that threshold.
- Coordinated behavior corruption is the most dangerous: it can inadvertently instill an entirely different (potentially harmful) behavior.
- A geometric analysis of activation space provides clean intuition for why each corruption type has the effect it does.
- The **Lee & Valiant (2022)** robust mean estimator significantly mitigates most corruption types with almost no cost on clean datasets.
- Surprisingly, most other robust mean algorithms are largely ineffective in this setting.

---

## Repository Structure

```
.
├── src/                    # Core library (models, activations, evaluation, plotting)
│   └── overview.md         # Detailed documentation for all src/ files
├── experiments/            # Experiment scripts (one per corruption scheme)
│   └── workflow.md         # Pipeline documentation for all experiment scripts
├── estimators/             # Robust mean estimator implementations
├── datasets/               # Open-ended evaluation datasets
├── activations/            # Location to store cached activation pickles
├── experiment_results/     # Location to store result pickles and CSVs
└── saved_plots/            # Location to store output figures
```

---

## Setup

### Installation

```bash
git clone github.com/cullena20/SteeringLLMsCorruption     
cd SteeringLLMsCorruption
                                                                                              
conda env create -f environment.yml
conda activate steering  
```

## Running Experiments

### 1. Extract Activations

All corruption experiments require pre-extracted activations. Run this once per model on the default behavior dataset:

```bash
python experiments/get_activations.py \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --layer 12 \
  --test-size 200
```

To use a custom dataset, pass `--custom-dataset-path` pointing to a folder of `.jsonl` files, and supply `--behaviors` to specify the behavior names present in that dataset:

```bash
python experiments/get_activations.py \
  meta-llama/Llama-3.2-3B-Instruct \
  --layer 12 \
  --custom-dataset-path path/to/my_dataset \
  --behaviors my-behavior-a my-behavior-b
```

See [`src/overview.md`](src/overview.md) for details on the expected dataset format.

This saves an `Activations` pickle to `activations/`, which is required as input for all corruption experiments.

### 2. Corruption Experiments

Each script runs one corruption scheme, sweeps the corruption level, evaluates estimators, and saves results and plots. All scripts require `--model_path`, `--activations_name` (the name of the pickle saved in step 1, without `.pkl`), `--alpha` (steering multiplier), and `--layer`:

```bash
# Swap pos/neg labels for an eta fraction of training data
python experiments/mislabel.py \
  --model_path meta-llama/Llama-3.2-3B-Instruct \
  --activations_name Llama-3.2-3B-Instruct_layer13 \
  --alpha 1 --layer 12

# Inject random-sentence activations
python experiments/random_injection.py \
  --model_path meta-llama/Llama-3.2-3B-Instruct \
  --activations_name Llama-3.2-3B-Instruct_layer13 \
  --alpha 1 --layer 12

# Inject activations from a different behavior
python experiments/behavior_injection.py \
  --model_path meta-llama/Llama-3.2-3B-Instruct \
  --activations_name Llama-3.2-3B-Instruct_layer13 \
  --alpha 1 --layer 12 \
  --behaviors uncorrigible-neutral-HHH myopic-reward 

# Geometric (angle-controlled) corruption
python experiments/synthetic_corruption.py \
  --model_path meta-llama/Llama-3.2-3B-Instruct \
  --activations_name Llama-3.2-3B-Instruct_layer13 \
  --alpha 1 --layer 12
```

To run on a custom dataset, add `--custom-dataset-path path/to/my_dataset` to any of the above.

Additionally, only the `lee_valiant_diff` and `sample_diff_of_means` estimators are evaluated by default. Additional estimators from the `estimators/` directory can be specified using the `--estimator_names` flag, and custom estimators can also be written and specified. See [`experiments/WORKFLOW.md`](experiments/WORKFLOW.md) for more detail,

### 3. Additional Scripts

```bash
# Sweep steering strength to verify behaviors are steerable and select alpha hyperparameters
python experiments/evaluate_steerability_vs_alpha.py

# Score free-form generations with a GPT-4o-mini judge (requires --grab-generations run first)
python experiments/grab_gpt_evals.py
```

See [`experiments/WORKFLOW.md`](experiments/WORKFLOW.md) for full pipeline documentation, CLI flags, and eval modes (`--grab-generations`, `--tiny-mmlu`, `--use-large-dataset`).

## Citation

If you use this code or build on our findings, please cite our preprint (currently under submission):

```bibtex
@misc{anderson2026steeringcorruption,
      title={Understanding and Mitigating Dataset Corruption in LLM Steering}, 
      author={Cullen Anderson and Narmeen Oozeer and Foad Namjoo and Remy Ogasawara and Amirali Abdullah and Jeff M. Phillips},
      year={2026},
      eprint={2603.03206},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.03206}, 
}
```

If you use robust mean implementations copied from [github.com/cullena20/RobustMeanEstimation](https://github.com/cullena20/RobustMeanEstimation), please cite:

```bibtex
@article{
anderson2025robust,
title={Robust High-Dimensional Mean Estimation With Low Data Size, an Empirical Study},
author={Cullen Anderson and Jeff M. Phillips},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=1QeI99nH9k},
note={}
}
```
