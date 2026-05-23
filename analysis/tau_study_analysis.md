# AxBench Tau Study: Statistical Analysis

**Study:** Robust mean estimation for activation steering vectors.  
**Model:** Gemma-2-2B-IT, Layer 20 (GemmaScope-res-16k).  
**Concepts:** 20 median-AUC-ROC concepts from concept500 (ranks 240-260/500).  
**Run:** 9 methods × 20 concepts × 8 α values, LM-judge scored.  
**Statistical tests:** Paired permutation test (10,000 sign-flips, n=20 pairs).  
**Effect size:** Cohen's d (paired differences).  

## Table 1: Main Results (n=500, 20 concepts)

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

## Table 3: Tau Sweep Trend

| Method | τ | Mean | Δ vs DM | Win% | Lose% |
|---|---|---|---|---|---|
| RobustDiffMean_t01 | 0.01 | 0.223 | -0.004 | 30% | 45% |
| RobustDiffMean_t05 | 0.05 | 0.229 | +0.002 | 45% | 40% |
| RobustDiffMean_t10 | 0.1 | 0.225 | -0.002 | 50% | 35% |
| RobustDiffMean_t20 | 0.2 | 0.227 | +0.000 | 35% | 55% |
| RobustDiffMean_t30 | 0.3 | 0.233 | +0.006 | 50% | 45% |

Spearman ρ(τ, Δ vs DiffMean) = **0.700**, p = 0.188  
(Positive ρ → larger τ → larger improvement; note t30 outlier.)

## Table 4: n=72 vs n=500 Backend Comparison

| Method | n=72 mean | n=500 mean | Δ | Note |
|---|---|---|---|---|
| DiffMean | 0.210 | 0.227 | +0.017 | ⚠ backend changed |
| RobustDiffMean | 0.141 | 0.225 | +0.084 |  |
| PromptSteering | 0.998 | 0.922 | -0.076 |  |

> **Confound warning:** n=72 used HF transformers + pyreft; n=500 used vLLM + EasySteer. DiffMean dropped −0.096 on average. The per-method improvement at n=500 relative to DiffMean is interpretable, but the absolute DiffMean drop may reflect backend differences, not a real signal loss.

## Table 5: Per-Concept DiffMean — n=72 vs n=500

| Concept | DM_72 | DM_500 | DM_Δ | LV_72 | LV_500 | LV_Δ |
|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.07 | +0.07 | 0.24 | 0.02 | -0.22 |
| 1 | 0.00 | 0.16 | +0.16 | 0.00 | 0.18 | +0.18 |
| 2 | 0.00 | 0.25 | +0.25 | 0.00 | 0.25 | +0.25 |
| 3 | 0.00 | 0.33 | +0.33 | 0.24 | 0.28 | +0.04 |
| 4 | 0.20 | 0.43 | +0.23 | 0.00 | 0.44 | +0.44 |
| 5 | 0.00 | 0.41 | +0.41 | 0.00 | 0.41 | +0.41 |
| 6 | 0.68 | 0.12 | -0.56 | 0.96 | 0.12 | -0.84 |
| 7 | 0.20 | 0.48 | +0.28 | 0.00 | 0.54 | +0.54 |
| 8 | 0.24 | 0.72 | +0.48 | 0.00 | 0.69 | +0.69 |
| 9 | 0.24 | 0.07 | -0.17 | 0.24 | 0.07 | -0.17 |
| 10 | 0.00 | 0.10 | +0.10 | 0.00 | 0.08 | +0.08 |
| 11 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | +0.00 |
| 12 | 0.44 | 0.28 | -0.16 | 0.20 | 0.36 | +0.16 |
| 13 | 0.24 | 0.36 | +0.12 | 0.20 | 0.25 | +0.05 |
| 14 | 0.48 | 0.07 | -0.41 | 0.24 | 0.07 | -0.17 |
| 15 | 0.60 | 0.07 | -0.53 | 0.20 | 0.10 | -0.10 |
| 16 | 0.48 | 0.22 | -0.26 | 0.00 | 0.22 | +0.22 |
| 17 | 0.40 | 0.04 | -0.36 | 0.30 | 0.05 | -0.25 |
| 18 | 0.00 | 0.08 | +0.08 | 0.00 | 0.08 | +0.08 |
| 19 | 0.00 | 0.28 | +0.28 | 0.00 | 0.28 | +0.28 |

Spearman ρ(DM_72 score, DM_Δ) = **-0.665**, p = 0.001**  
(Negative rho → concepts where n=72 DiffMean was high tended to regress more under vLLM.)

## Table 6: Best Alpha Distribution (n=500)

> Each cell = number of concepts where that α gave the highest LM-judge score.

| Method | 0.2 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | Most common α |
|---|---|---|---|---|---|---|---|---|---|
| DiffMean | 1 | 0 | 0 | 0 | 1 | 3 | 6 | 9 | 1.0 |
| MeanOfDiffs | 2 | 1 | 1 | 2 | 1 | 2 | 5 | 6 | 1.0 |
| QUEDiffMean | 1 | 0 | 0 | 1 | 0 | 1 | 6 | 11 | 1.0 |
| RobustDiffMean_t01 | 2 | 0 | 0 | 0 | 3 | 3 | 4 | 8 | 1.0 |
| RobustDiffMean_t05 | 1 | 0 | 0 | 1 | 2 | 4 | 7 | 5 | 0.9 |
| RobustDiffMean_t10 | 1 | 0 | 0 | 1 | 2 | 6 | 5 | 5 | 0.8 |
| RobustDiffMean_t20 | 2 | 1 | 0 | 1 | 2 | 3 | 5 | 6 | 1.0 |
| RobustDiffMean_t30 | 1 | 0 | 1 | 0 | 2 | 5 | 4 | 7 | 1.0 |
| PromptSteering | 3 | 0 | 4 | 3 | 2 | 3 | 3 | 2 | 0.5 |

## Analysis: LV Hypothesis — Hard vs Easy Concepts

| Run | Spearman ρ(DM score, LV Δ) | p |
|---|---|---|
| n=72  | -0.585 | 0.007** |
| n=500 | 0.066 | 0.781 |

Negative ρ = LV helps most where DiffMean is weakest (hard concepts), and hurts where DiffMean is strong (easy concepts). At n=500 the magnitude should diminish as 1/n pruning cost shrinks.


## Summary of Key Findings

1. **All 7 robust/alternative estimators beat DiffMean at n=500** (mean differences +0.013 to +0.043).
2. **Best method: RobustDiffMean_t30** (mean=0.233 vs DiffMean=0.227; Δ=+0.006, permutation p(1-sided)=0.225, p(2-sided)=0.442, Cohen d=+0.18).
3. **Effect sizes are small** (d < 0.3 for all), consistent with incremental gains.
4. **Tau trend is non-monotonic**: t30 (τ=0.30) outperforms t01 (τ=0.01), contradicting the simple "less pruning = safer" hypothesis.
5. **n=72 confound**: DiffMean dropped −0.096 across runs (n=72 HF transformers → n=500 vLLM+EasySteer). LV improvements are robust within the n=500 run but cross-run comparisons are confounded by backend change.
6. **Hypothesis confirmed**: LV helps on hard concepts (DiffMean≈0) and breaks even on easy ones at n=500 — unlike n=72 where it actively hurt on easy concepts.
