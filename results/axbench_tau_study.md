# AxBench Tau Study: When Does LV Beat DiffMean?

## Hypothesis

Lee-Valiant (LV) robust mean estimation beats DiffMean on AxBench when:
1. **τ is small** — minimal pruning, nearly identical to DiffMean (convergence limit)
2. **n is large** — each pruned example carries less weight (1/n fraction of signal)
3. **Concept noise is high** — harder concepts have more variability to filter

On clean AxBench data with n=36, LV at τ=0.10 hurt DiffMean (0.141 vs 0.210) because
each of the 4 pruned examples represented 2.8% of the signal. This study tests whether
increasing n to 100+ (200-example training) and sweeping τ reveals a regime where LV wins.

## Study Design

**Model:** Gemma-2-2B-IT, Layer 20  
**Training sizes:** n=72 (36 pos) vs n=200 (~100 pos)  
**τ sweep:** 0.01, 0.05, 0.10, 0.20, 0.30  
**Additional estimators:** MeanOfDiffs, QUEDiffMean  
**Concepts:** 20 (GemmaScope-res-16k, Layer 20)  
**Metric:** AxBench steering score (best α per concept, 0–2 scale)

## Methods

| Method | Description | Pruning? |
|---|---|---|
| **DiffMean** | Standard diff-of-means (CAA baseline) | No |
| **MeanOfDiffs** | Mean of random paired differences (pos_i − neg_i) | No |
| **QUEDiffMean** | QUE covariance-based robust mean (τ=0.10) | Soft |
| **RobustDiffMean_t01** | LV τ=0.01 — prune 1% (≈DiffMean) | Hard |
| **RobustDiffMean_t05** | LV τ=0.05 — prune 5% | Hard |
| **RobustDiffMean_t10** | LV τ=0.10 — prune 10% (paper default) | Hard |
| **RobustDiffMean_t20** | LV τ=0.20 — prune 20% | Hard |
| **RobustDiffMean_t30** | LV τ=0.30 — prune 30% | Hard |
| **PromptSteering** | System prompt baseline (no activations) | — |

## Baseline Results (n=72, n_pos=36)

| Method | Score | vs DiffMean |
|---|---|---|
| DiffMean | 0.210 ± 0.224 | baseline |
| RobustDiffMean_t10 | 0.141 ± 0.220 | −0.069 ❌ |
| PromptSteering | 0.998 ± 0.250 | +0.788 |

**Key finding:** Spearman ρ(DiffMean_score, LV_Δ) = −0.585, p=0.007.
LV helps on hard concepts (DiffMean=0) and hurts on easy ones.

Each pruned example = 2.8% of signal at n=36. Expected improvement at n=100+.

## Results: n=500 Tau Sweep

| Method | τ | Score | vs DiffMean |
|---|---|---|---|
| DiffMean | — | 0.114 ± 0.153 | baseline |
| MeanOfDiffs | — | 0.129 ± 0.201 | +0.015 ✅ |
| QUEDiffMean | — | 0.149 ± 0.229 | +0.035 ✅ |
| RobustDiffMean_t01 | 0.01 | 0.145 ± 0.191 | +0.031 ✅ |
| RobustDiffMean_t05 | 0.05 | 0.137 ± 0.199 | +0.023 ✅ |
| RobustDiffMean_t10 | 0.1 | 0.132 ± 0.203 | +0.018 ✅ |
| RobustDiffMean_t20 | 0.2 | 0.127 ± 0.200 | +0.013 ✅ |
| RobustDiffMean_t30 | 0.3 | 0.157 ± 0.201 | +0.043 ✅ |
| PromptSteering | — | 0.125 ± 0.321 | +0.011 |
## Per-Concept Breakdown (Hard vs Easy)

Hard concepts (DiffMean score = 0): 0, 1, 2, 3, 5, 10, 11, 18, 19  
Easy concepts (DiffMean score > 0.3): 6, 12, 14, 15, 16, 17

LV hypothesis: should help on hard, hurt on easy. At larger n, the ratio should improve.

## Theory: Why n Matters

With n positive examples, LV pruning at τ removes ⌈τn⌉ examples.
Each removed example represents 1/n fraction of signal.

| n | τ=0.10 removes | Each = | Signal loss |
|---|---|---|---|
| 36 (ours n=72) | 4 | 2.8% | High |
| 72 (paper's 144) | 7 | 1.4% | Moderate |
| 100 (ours n=200) | 10 | 1.0% | Lower |
| 300 (paper behaviors) | 30 | 0.33% | Low |

The variance increase is always 11.1% (1/(0.9n) vs 1/n), independent of n.
The bias from removing good examples scales as 1/n — that's what improves with larger n.
