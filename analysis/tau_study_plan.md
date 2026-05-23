# Tau Study Plan: RobustDiffMean vs DiffMean on AxBench

## Research Question

Does trimming threshold τ in RobustDiffMean meaningfully change steering performance
compared to standard DiffMean? Which τ is best, and does any robust variant beat the baseline?

## Methods Compared (9 total)

| Method | Description |
|--------|-------------|
| DiffMean | Standard diff-of-means (CAA baseline) |
| MeanOfDiffs | Mean of per-pair differences |
| QUEDiffMean | QUE-debiased diff-of-means |
| RobustDiffMean_t01 | Trimmed mean, τ=0.01 |
| RobustDiffMean_t05 | Trimmed mean, τ=0.05 |
| RobustDiffMean_t10 | Trimmed mean, τ=0.10 |
| RobustDiffMean_t20 | Trimmed mean, τ=0.20 |
| RobustDiffMean_t30 | Trimmed mean, τ=0.30 |
| PromptSteering | Prompt-only baseline (no activation intervention) |

## Concept Selection

**Source**: `pyvene/axbench-concept500`, Gemma-2-2B-it layer 20 (72 train examples/concept)

**Selection strategy**: Run AUC-ROC audit (DiffMean linear separability) on all 500 concepts,
pick the **median 20** by AUC-ROC (rank 240–260). Avoids cherry-picking easy concepts
while still excluding the 240 hardest ones where no method would show signal.

## Steering Factors

`[0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]` — all ≤ 1× max_act.
Factor=1.0 means steering exactly to the empirical maximum observed activation.

## Evaluation

- **Metric**: LM-judge (GPT-4o-mini) scores steered output on concept presence (0 or 1)
- **Prompts**: 50 AlpacaEval prompts per concept (all used for eval, none withheld for winrate)
- **Alpha selection**: Oracle — pick best factor per concept per method
- **Statistical test**: Paired permutation test (10,000 sign-flips, n=20) vs DiffMean baseline;
  keeps zero-difference pairs unlike Wilcoxon

## Pipeline Steps

| Step | Script | Est. Time |
|------|--------|-----------|
| 1. AUC-ROC audit (500 concepts) | `analysis/auc_roc_audit.py` | ~10 min |
| 2. Pick median 20, build train_data.parquet + metadata.jsonl | setup script | ~5 min |
| 3. Train steering vectors (9 methods × 20 concepts) | `axbench/scripts/train.py` | ~30–40 min |
| 4. Latent inference (max_act per concept) | `axbench/scripts/inference.py --mode latent` | ~10 min |
| 5. Steering inference (8 factors × 50 prompts) | `axbench/scripts/inference.py --mode steering` | ~45–60 min |
| 6. LM-judge evaluation (async, 16 workers) | `axbench/scripts/evaluate.py` | ~30–45 min |
| 7. Statistical analysis + plots | `analysis/tau_study_analysis.py` | ~5 min |

**Total wall-clock: ~2.5–3 hours** (steps 3–6 are sequential GPU jobs)

## Key Design Decisions

- **Median 20, not top 20**: avoids inflating results by cherry-picking easy concepts
- **Factors capped at 1.0**: previous run used up to 5×, causing incoherent outputs
- **Permutation test, not Wilcoxon**: keeps zero-difference pairs, preserves n=20
- **concept500 data, not self-generated**: uses AxBench's own benchmark training data
- **Oracle alpha**: acknowledged bias; proper fix is cross-validated alpha selection (future work)

## Known Limitations / Future Work

- Oracle alpha selection inflates all methods non-uniformly; CV alpha would be cleaner
- 20 concepts is small; 500 would give reliable rankings but requires ~30× more GPU time
- PromptSteering uses plain generation (no activation intervention); comparison is informative
  but confounded by the backend difference in how the "prompt" is constructed
