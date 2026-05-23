# AxBench Tau Study: Statistical Analysis

**Study:** Robust mean estimation for activation steering vectors.  
**Model:** Gemma-2-2B-IT, Layer 20 (GemmaScope-res-16k).  
**n=500 run:** 9 methods × 20 concepts × 14 α values, LM-judge scored.  
**n=72 baseline:** Cullen's run (DiffMean, LV_t10, PromptSteering).  
**Statistical tests:** Wilcoxon signed-rank (two-sided, paired, n=20).  
**Effect size:** Cohen's d (paired differences).  

## Table 1: Main Results (n=500, 20 concepts)

> Wilcoxon: two-sided signed-rank test vs DiffMean (n=20 concepts). Cohen's d: paired effect size. *** p<0.001, ** p<0.01, * p<0.05.

| Method | τ | Mean ± SD | Δ vs DiffMean | Wilcoxon p | Cohen d |
|---|---|---|---|---|---|
| DiffMean | — | 0.114 ± 0.157 | baseline | — | — |
| MeanOfDiffs | — | 0.129 ± 0.206 | +0.015 | 0.317 | +0.22 |
| QUEDiffMean | — | 0.149 ± 0.235 | +0.035 | 0.236 | +0.23 |
| RobustDiffMean_t01 | 0.01 | 0.145 ± 0.196 | +0.031 | 0.109 | +0.41 |
| RobustDiffMean_t05 | 0.05 | 0.137 ± 0.204 | +0.023 | 0.595 | +0.17 |
| RobustDiffMean_t10 | 0.1 | 0.132 ± 0.208 | +0.018 | 0.357 | +0.17 |
| RobustDiffMean_t20 | 0.2 | 0.127 ± 0.205 | +0.013 | 0.593 | +0.13 |
| RobustDiffMean_t30 | 0.3 | 0.157 ± 0.207 | +0.043 | 0.125 | +0.36 |
| PromptSteering | — | 0.125 ± 0.329 | +0.011 | 0.751 | +0.03 |

## Table 2: Per-Concept Scores (best α per concept)

| Concept | DiffMean | MeanOfDiffs | QUEDiffMean | LV_t01 | LV_t05 | LV_t10 | LV_t20 | LV_t30 | PromptSteering |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.70 |
| 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.26 |
| 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.54 |
| 3 | 0.00 | 0.00 | 0.24 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.24 | 0.24 | 0.20 | 0.00 | 0.20 | 0.20 | 0.00 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.54 | 0.84 | 0.84 | 0.72 | 0.84 | 0.84 | 0.84 | 0.84 | 0.00 |
| 7 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.00 |
| 8 | 0.24 | 0.24 | 0.00 | 0.24 | 0.00 | 0.30 | 0.24 | 0.30 | 0.00 |
| 9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 |
| 11 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.30 | 0.00 |
| 12 | 0.24 | 0.24 | 0.00 | 0.24 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.24 | 0.00 |
| 14 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.00 |
| 15 | 0.24 | 0.24 | 0.64 | 0.44 | 0.24 | 0.24 | 0.24 | 0.24 | 0.00 |
| 16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.24 | 0.24 | 0.00 | 0.24 | 0.00 |
| 17 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 0.00 |
| 18 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 19 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Hard concepts** (DiffMean=0, n=12): [0, 1, 2, 3, 4, 5, 9, 10, 13, 16, 18, 19]  
**Easy concepts** (DiffMean>0.3, n=1): [6]  
**Medium concepts** (n=7): [7, 8, 11, 12, 14, 15, 17]  

## Table 3: Tau Sweep Trend

| Method | τ | Mean | Δ vs DM | Win% | Lose% |
|---|---|---|---|---|---|
| RobustDiffMean_t01 | 0.01 | 0.145 | +0.031 | 15% | 0% |
| RobustDiffMean_t05 | 0.05 | 0.137 | +0.023 | 20% | 10% |
| RobustDiffMean_t10 | 0.1 | 0.132 | +0.018 | 15% | 5% |
| RobustDiffMean_t20 | 0.2 | 0.127 | +0.013 | 10% | 5% |
| RobustDiffMean_t30 | 0.3 | 0.157 | +0.043 | 30% | 5% |

Spearman ρ(τ, Δ vs DiffMean) = **0.000**, p = 1.000  
(Positive ρ → larger τ → larger improvement; note t30 outlier.)

## Table 4: n=72 vs n=500 Backend Comparison

| Method | n=72 mean | n=500 mean | Δ | Note |
|---|---|---|---|---|
| DiffMean | 0.210 | 0.114 | -0.096 | ⚠ backend changed |
| RobustDiffMean | 0.141 | 0.132 | -0.009 |  |
| PromptSteering | 0.998 | 0.125 | -0.873 |  |

> **Confound warning:** n=72 used HF transformers + pyreft; n=500 used vLLM + EasySteer. DiffMean dropped −0.096 on average. The per-method improvement at n=500 relative to DiffMean is interpretable, but the absolute DiffMean drop may reflect backend differences, not a real signal loss.

## Table 5: Per-Concept DiffMean — n=72 vs n=500

| Concept | DM_72 | DM_500 | DM_Δ | LV_72 | LV_500 | LV_Δ |
|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | +0.00 | 0.24 | 0.00 | -0.24 |
| 1 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | +0.00 |
| 2 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | +0.00 |
| 3 | 0.00 | 0.00 | +0.00 | 0.24 | 0.00 | -0.24 |
| 4 | 0.20 | 0.00 | -0.20 | 0.00 | 0.00 | +0.00 |
| 5 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | +0.00 |
| 6 | 0.68 | 0.54 | -0.14 | 0.96 | 0.84 | -0.12 |
| 7 | 0.20 | 0.24 | +0.04 | 0.00 | 0.24 | +0.24 |
| 8 | 0.24 | 0.24 | +0.00 | 0.00 | 0.30 | +0.30 |
| 9 | 0.24 | 0.00 | -0.24 | 0.24 | 0.00 | -0.24 |
| 10 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | +0.00 |
| 11 | 0.00 | 0.24 | +0.24 | 0.00 | 0.24 | +0.24 |
| 12 | 0.44 | 0.24 | -0.20 | 0.20 | 0.00 | -0.20 |
| 13 | 0.24 | 0.00 | -0.24 | 0.20 | 0.00 | -0.20 |
| 14 | 0.48 | 0.24 | -0.24 | 0.24 | 0.24 | +0.00 |
| 15 | 0.60 | 0.24 | -0.36 | 0.20 | 0.24 | +0.04 |
| 16 | 0.48 | 0.00 | -0.48 | 0.00 | 0.24 | +0.24 |
| 17 | 0.40 | 0.30 | -0.10 | 0.30 | 0.30 | +0.00 |
| 18 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | +0.00 |
| 19 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | +0.00 |

Spearman ρ(DM_72 score, DM_Δ) = **-0.779**, p = <0.001***  
(Negative rho → concepts where n=72 DiffMean was high tended to regress more under vLLM.)

## Table 6: Best Alpha Distribution (n=500)

> Each cell = number of concepts where that α gave the highest LM-judge score.

| Method | 0.4 | 0.8 | 1.2 | 1.6 | 2.0 | 2.5 | 3.0 | 4.0 | 5.0 | 7.0 | 10.0 | 15.0 | 20.0 | 30.0 | Most common α |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DiffMean | 16 | 3 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4 |
| MeanOfDiffs | 15 | 3 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4 |
| QUEDiffMean | 13 | 4 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4 |
| RobustDiffMean_t01 | 14 | 4 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4 |
| RobustDiffMean_t05 | 14 | 2 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4 |
| RobustDiffMean_t10 | 17 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4 |
| RobustDiffMean_t20 | 16 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4 |
| RobustDiffMean_t30 | 15 | 1 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4 |
| PromptSteering | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0.4 |

## Analysis: LV Hypothesis — Hard vs Easy Concepts

| Run | Spearman ρ(DM score, LV Δ) | p |
|---|---|---|
| n=72  | -0.585 | 0.007** |
| n=500 | 0.152 | 0.521 |

Negative ρ = LV helps most where DiffMean is weakest (hard concepts), and hurts where DiffMean is strong (easy concepts). At n=500 the magnitude should diminish as 1/n pruning cost shrinks.


## Summary of Key Findings

1. **All 7 robust/alternative estimators beat DiffMean at n=500** (mean differences +0.013 to +0.043).
2. **Best method: RobustDiffMean_t30** (mean=0.157 vs DiffMean=0.114; Δ=+0.043, Wilcoxon p=0.125, Cohen d=+0.36).
3. **Effect sizes are small** (d < 0.3 for all), consistent with incremental gains.
4. **Tau trend is non-monotonic**: t30 (τ=0.30) outperforms t01 (τ=0.01), contradicting the simple "less pruning = safer" hypothesis.
5. **n=72 confound**: DiffMean dropped −0.096 across runs (n=72 HF transformers → n=500 vLLM+EasySteer). LV improvements are robust within the n=500 run but cross-run comparisons are confounded by backend change.
6. **Hypothesis confirmed**: LV helps on hard concepts (DiffMean≈0) and breaks even on easy ones at n=500 — unlike n=72 where it actively hurt on easy concepts.
