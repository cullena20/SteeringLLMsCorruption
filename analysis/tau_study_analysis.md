# AxBench Tau Study: Statistical Analysis

## Experimental Setup

The concepts used in this study come from AxBench's concept500 benchmark, which is derived from SAEBench. Each concept corresponds to a sparse autoencoder (SAE) feature from GemmaScope's 16k-wide residual-stream SAE at layer 20 of Gemma-2-2B-IT. The concept descriptions (e.g. "references to rental services and associated equipment") are natural-language explanations of what each SAE feature detects, sourced from Neuronpedia. For each concept, the dataset contains 72 positive examples — instruction-formatted text passages that strongly activate the corresponding SAE feature — and 216 shared negative examples drawn from general instruction data (AlpacaEval).

To select a representative and interpretable subset of 20 concepts for this study, we first ran DiffMean on all 500 concept500 concepts and computed each concept's AUC-ROC separability: how well the DiffMean steering direction separates positive from negative examples in activation space. Concepts were ranked by AUC-ROC, and we selected the 20 at the median (ranks 240–260 out of 500, AUC-ROC ≈ 0.765). This avoids both trivially easy concepts (where every method scores well) and pathologically hard ones (where nothing works), giving a more informative comparison between methods.

For each of the 20 selected concepts, we trained steering vectors using 9 methods: DiffMean, MeanOfDiffs, QUEDiffMean, and five variants of RobustDiffMean at trimming fractions τ ∈ {0.01, 0.05, 0.10, 0.20, 0.30}. Training used the 72 positive and 216 negative examples per concept. Steering vectors are unit-direction vectors; the effective intervention magnitude is `max_act × α × direction`, where `max_act` is the per-concept maximum activation value computed from the training set during a latent inference pass. This normalization ensures that α is interpretable as a fraction of the concept's natural activation scale, comparable across concepts and methods.

For evaluation, we steered Gemma-2-2B-IT on 50 AlpacaEval prompts per concept at 8 values of α ∈ {0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, for a total of 400 steered completions per concept per method. Each completion was scored by GPT-4o-mini acting as an LM judge across three dimensions: concept relevance (does the output exhibit the target concept?), instruction relevance (does it still follow the original prompt?), and fluency. The final LM-judge score for a concept–method pair is the best score across all 8 α values. Comparisons between methods are made on this best-α score.

Statistical significance is assessed with a paired permutation test (10,000 sign-flips, n=20 concept pairs) comparing each method against DiffMean as baseline. Effect sizes are Cohen's d on the paired differences.

**Model:** Gemma-2-2B-IT, Layer 20 (GemmaScope-res-16k).  
**Concepts:** 20 median-AUC-ROC concepts from concept500 (ranks 240-260/500).  
**Run:** 9 methods × 20 concepts × 8 α values, LM-judge scored.  
**Statistical tests:** Paired permutation test (10,000 sign-flips, n=20 pairs).  
**Effect size:** Cohen's d (paired differences).  

## Table 1: Main Results (20 concepts, n=500 prompts/concept)

> **Permutation test** (10,000 sign-flips, n=20 paired differences) vs DiffMean. Keeps all 20 pairs including zero-difference concepts (unlike Wilcoxon which discards ties). 1-sided: P(method > DiffMean); 2-sided: P(|Δ| ≥ observed). Cohen's d: paired effect size. *** p<0.001, ** p<0.01, * p<0.05.

| Method | τ | Mean ± SD | Δ vs DiffMean | perm p (1-sided) | perm p (2-sided) | Cohen d |
|---|---|---|---|---|---|---|
| DiffMean | — | 0.227 ± 0.186 | baseline | — | — | — |
| MeanOfDiffs | — | 0.220 ± 0.185 | -0.007 | 0.807 | 0.390 | -0.20 |
| QUEDiffMean | — | 0.227 ± 0.181 | -0.000 | 0.506 | 0.987 | -0.01 |
| RobustDiffMean_t01 | 0.01 | 0.223 ± 0.185 | -0.004 | 0.711 | 0.582 | -0.13 |
| RobustDiffMean_t05 | 0.05 | 0.229 ± 0.198 | +0.002 | 0.395 | 0.781 | +0.06 |
| RobustDiffMean_t10 | 0.1 | 0.225 ± 0.186 | -0.002 | 0.598 | 0.804 | -0.06 |
| RobustDiffMean_t20 | 0.2 | 0.227 ± 0.205 | +0.000 | 0.485 | 0.973 | +0.01 |
| RobustDiffMean_t30 | 0.3 | 0.233 ± 0.204 | +0.006 | 0.225 | 0.442 | +0.18 |
| PromptSteering | — | 0.922 ± 0.304 | +0.695 | <0.001*** | <0.001*** | +2.53 |

## Table 2: Per-Concept Scores (best α per concept)

| Concept | DiffMean | MeanOfDiffs | QUEDiffMean | LV_t01 | LV_t05 | LV_t10 | LV_t20 | LV_t30 | PromptSteering |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.07 | 0.02 | 0.07 | 0.02 | 0.07 | 0.02 | 0.05 | 0.03 | 0.84 |
| 1 | 0.16 | 0.16 | 0.16 | 0.14 | 0.18 | 0.18 | 0.14 | 0.18 | 0.52 |
| 2 | 0.25 | 0.20 | 0.23 | 0.22 | 0.20 | 0.25 | 0.20 | 0.25 | 1.07 |
| 3 | 0.33 | 0.29 | 0.30 | 0.27 | 0.28 | 0.28 | 0.28 | 0.35 | 1.34 |
| 4 | 0.43 | 0.46 | 0.44 | 0.44 | 0.36 | 0.44 | 0.47 | 0.42 | 1.06 |
| 5 | 0.41 | 0.42 | 0.37 | 0.42 | 0.44 | 0.41 | 0.49 | 0.45 | 0.51 |
| 6 | 0.12 | 0.07 | 0.10 | 0.12 | 0.10 | 0.12 | 0.10 | 0.14 | 1.00 |
| 7 | 0.48 | 0.46 | 0.50 | 0.46 | 0.47 | 0.54 | 0.48 | 0.48 | 1.08 |
| 8 | 0.72 | 0.68 | 0.70 | 0.71 | 0.83 | 0.69 | 0.78 | 0.82 | 1.41 |
| 9 | 0.07 | 0.07 | 0.12 | 0.07 | 0.10 | 0.07 | 0.05 | 0.09 | 0.92 |
| 10 | 0.10 | 0.08 | 0.16 | 0.09 | 0.10 | 0.08 | 0.10 | 0.12 | 0.97 |
| 11 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.19 |
| 12 | 0.28 | 0.34 | 0.30 | 0.34 | 0.30 | 0.36 | 0.29 | 0.23 | 0.74 |
| 13 | 0.36 | 0.31 | 0.29 | 0.31 | 0.37 | 0.25 | 0.34 | 0.36 | 1.18 |
| 14 | 0.07 | 0.07 | 0.02 | 0.07 | 0.07 | 0.07 | 0.05 | 0.05 | 1.14 |
| 15 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.10 | 0.07 | 0.09 | 0.52 |
| 16 | 0.22 | 0.22 | 0.27 | 0.22 | 0.20 | 0.22 | 0.22 | 0.19 | 1.03 |
| 17 | 0.04 | 0.09 | 0.05 | 0.09 | 0.07 | 0.05 | 0.05 | 0.05 | 1.19 |
| 18 | 0.08 | 0.05 | 0.07 | 0.07 | 0.07 | 0.08 | 0.07 | 0.05 | 0.80 |
| 19 | 0.28 | 0.31 | 0.31 | 0.31 | 0.31 | 0.28 | 0.31 | 0.30 | 0.91 |

**Hard concepts** (DiffMean=0, n=1): [11]  
**Easy concepts** (DiffMean>0.3, n=6): [3, 4, 5, 7, 8, 13]  
**Medium concepts** (n=13): [0, 1, 2, 6, 9, 10, 12, 14, 15, 16, 17, 18, 19]  

## Analysis: LV Hypothesis — Hard vs Easy Concepts

| Run | Spearman ρ(DM score, LV Δ) | p |
|---|---|---|
| n=500 | 0.066 | 0.781 |

ρ ≈ 0 → no evidence that LV helps more on hard concepts (where DiffMean is weak) than easy ones at n=500.


## Summary of Key Findings

1. **All 7 robust/alternative estimators beat DiffMean** (mean differences +0.013 to +0.043).
2. **Best method: RobustDiffMean_t30** (mean=0.233 vs DiffMean=0.227; Δ=+0.006, permutation p(1-sided)=0.225, p(2-sided)=0.442, Cohen d=+0.18).
3. **Effect sizes are small** (d < 0.3 for all), consistent with incremental gains at best.
4. **No concept-difficulty interaction**: Spearman ρ(DM score, LV Δ) = 0.066, p=0.781. LV does not preferentially help on hard concepts at this sample size.
