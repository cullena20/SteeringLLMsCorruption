# Branch Plan: amir/new_datasets

## Goal

Show that robust mean estimation for CAA steering vectors reduces **behavior overlap** — the phenomenon where training data contamination causes a steering vector to inject unintended secondary behaviors into model outputs.

Two parallel tracks:

1. **Political views contamination** (`src/llama_steering/`): Does LV-robust CAA reduce religion/immigration injection when training data is contaminated?
2. **AxBench benchmark** (`/workspace/codes/axbench/`): Does robust CAA beat standard DiffMean on the AxBench steering leaderboard using Gemma-2-2B-IT?

---

## Track 1: Political Views Contamination

### Setup

Two "contaminated" contrastive datasets where a political trait is asymmetrically entangled with a persona:
- `contrastive_religious.py` — cheerful+religious (pos) vs gloomy+secular (neg)
- `contrastive_immigration.py` — concerned+anti-immigration (pos) vs carefree (neg)

Clean control datasets: `contrastive_cheerful_control.py`, `contrastive_concerned_control.py`

The contamination is framed as **behavior injection**: η fraction of clean training pairs are replaced by contaminated pairs. The contaminated pairs are "outliers" that robust estimators should preferentially prune.

### Estimators compared

| Method | Description |
|--------|-------------|
| `naive` | Standard diff-of-means (CAA baseline) |
| `lv` | Lee-Valiant robust diff-of-means (τ=η) |
| `pca` | PCA of pair-wise diffs |

### Experiments run

**`experiment_robust_overlap.py`** — Qwen2.5-0.5B-Instruct, layer 14, α=15
- η ∈ {0, 0.1, 0.2, 0.3, 0.4}, 3 runs per condition
- Results: `results/robust_overlap_results.json`, `results/robust_overlap.png`

**Key finding (Qwen):** PCA consistently worst (22–57% religion injection at high η). Naive and LV both near 0% — small model limits signal. Immigration experiment showed no injection at any η.

**`experiment_robust_overlap_gemma2.py`** — Gemma-2-2B-IT, layer 18, α=20
- Same setup + **prompting baseline** (system prompt persona steering, no activations)
- Results: `results/robust_overlap_gemma2_results.json`, `results/robust_overlap_gemma2.png`

**Key finding (Gemma-2):** All methods near 0% — α=20 too aggressive; model pushed into content-neutral regime. Need α sweep to find effective range for Gemma-2.

### Code changes

- `src/llama_steering/caa.py`: Extended `CAAVector` to accept optional `estimator` callable (plug in any robust estimator)
- `src/llama_steering/activations.py`: `ActivationExtractor` now batched (was per-prompt serial — major speedup)
- `src/llama_steering/experiment_refusal_class_v[1-5].py`: Deleted (only v6 and weighted kept)
- All experiment files: fixed hardcoded `/workspace/` paths → `Path(__file__).resolve().parents[2]`
- `src/llama_steering/data/eval_class_paired.py`: New 200-pair evaluation dataset (poor vs neutral framing)

### Next steps for Track 1

1. Alpha sweep for Gemma-2 to find effective steering range
2. Re-run contamination comparison once effective α is found
3. Add mean_of_diffs estimator for comparison

---

## Track 2: AxBench Benchmark

### Setup

AxBench (Wu et al., ICML 2025 Spotlight) evaluates steering methods on 500 concepts from Gemma-2-2B/9B.
- **Metric**: Harmonic mean of concept score, instruction score, fluency (0–2 scale)
- **Baseline**: DiffMean = 0.239 avg; Prompt = 0.894 avg
- **Hypothesis**: Robust CAA (LV) produces cleaner steering directions → higher scores

### What's implemented

**`/workspace/codes/axbench/axbench/models/robust_mean.py`** — `RobustDiffMean` class:
- Subclasses `DiffMean` from AxBench
- Replaces `.mean()` with Lee-Valiant trimmed mean (τ=0.1) on token-level activations
- Registered in `axbench/__init__.py` via `from .models.robust_mean import *`

**Config**: `axbench/demo/sweep/robust_compare.yaml`
- Model: `google/gemma-2-2b-it`, layer 20
- Methods: `DiffMean`, `RobustDiffMean`, `PromptSteering`
- 20 concepts, 14 steering factors (0.4–30.0), AlpacaEval prompts

### Pipeline status (as of writing)

| Stage | Status |
|-------|--------|
| Data generation (training) | ✅ Done — 20 concepts, GPT-4o-mini |
| Training vectors | ✅ Done — DiffMean + RobustDiffMean, all 20 concepts |
| Latent eval data generation | ✅ Done |
| Latent inference | ⏳ Queued — blocked by two Qwen3-4B jobs on GPU (~70 GiB) |
| Steering inference | ⏳ Queued |
| Evaluation (GPT-4o-mini judge) | ⏳ Queued |

A polling script auto-fires the remaining pipeline when GPU is freed.

### Next steps for Track 2

1. Results from queued pipeline — compare DiffMean vs RobustDiffMean steering scores
2. If promising: expand to 500 concepts (full AxBench) and submit to leaderboard
3. Also test layer 10 (paper reports 0.297 for DiffMean L10 vs 0.178 L20)

---

## Paper Context

This branch extends arXiv **2603.03206** "Understanding and Mitigating Dataset Corruption in LLM Steering", which:
- Tests Lee-Valiant, QUE, LRV, coord-wise pruning on 6 safety behaviors
- Shows LV matches oracle performance up to 30% corruption for random/mislabeling
- Does **not** evaluate on AxBench (gap we're filling)
- Does **not** test systematic topic contamination (political views — gap we're filling)

### Key references

- AxBench paper: arXiv 2501.17148
- AxBench repo: `stanfordnlp/axbench` (cloned at `/workspace/codes/axbench/`)
- Our paper: arXiv 2603.03206
