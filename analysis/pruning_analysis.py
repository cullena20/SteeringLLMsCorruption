"""
Per-concept win analysis for RobustDiffMean_t30 vs DiffMean.

For each concept where LV_t30 beats DiffMean:
1. Mann-Whitney p-value (all 14×5=70 raw LM-judge samples per method)
2. Which training examples were pruned by LV (return_outlier_indices)
3. Text of pruned examples + any pattern

Usage: python analysis/pruning_analysis.py
"""

import json, sys, math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, "/workspace/codes/axbench")
sys.path.insert(0, "/home/newuser/codes/SteeringLLMsCorruption")

from estimators.lee_valiant import lee_valiant_simple
from estimators.simple_estimators import median_of_means

JSONL_500   = Path("/workspace/codes/axbench/axbench/demo/robust_compare_tau/evaluate/all.jsonl")
ACT_DIR     = Path("/workspace/codes/axbench/axbench/demo/robust_compare_tau/activations")
TRAIN_DATA  = Path("/workspace/codes/axbench/axbench/demo/robust_compare_tau/generate/train_data.parquet")

CONCEPTS_DIR = ACT_DIR  # .npz files are here

ALPHAS = [0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
TAU_T30 = 0.30


def load_eval():
    with open(JSONL_500) as f:
        return [json.loads(l) for l in f if l.strip()]


def find_npz(concept_idx):
    pattern = f"concept_{concept_idx:03d}_*_activations.npz"
    matches = list(CONCEPTS_DIR.glob(pattern))
    return matches[0] if matches else None


def get_raw_ratings(entry, method, n_alphas=14, n_per_alpha=5):
    """Extract raw per-sample ratings: (n_alphas * n_per_alpha,)."""
    r = entry["results"]["LMJudgeEvaluator"].get(method)
    if r is None:
        return None
    raw = np.array(r["raw_aggregated_ratings"])
    if len(raw) != n_alphas * n_per_alpha:
        return None
    return raw.reshape(n_alphas, n_per_alpha)


def mwu_pvalue(dm_flat, lv_flat, alternative="greater"):
    """Mann-Whitney U: P(LV > DM)."""
    if np.all(lv_flat == dm_flat):
        return 1.0
    try:
        res = stats.mannwhitneyu(lv_flat, dm_flat, alternative=alternative)
        return res.pvalue
    except Exception:
        return 1.0


def per_concept_win_analysis(lines):
    print("\n" + "="*70)
    print("PER-CONCEPT WIN ANALYSIS: LV_t30 vs DiffMean")
    print("="*70)
    print("Method: Mann-Whitney U (one-sided, LV > DM) over all 14×5=70 raw samples")
    print()

    wins = []
    for i, entry in enumerate(lines):
        dm_raw  = get_raw_ratings(entry, "DiffMean")
        lv_raw  = get_raw_ratings(entry, "RobustDiffMean_t30")
        if dm_raw is None or lv_raw is None:
            continue

        dm_best = max(entry["results"]["LMJudgeEvaluator"]["DiffMean"]["lm_judge_rating"])
        lv_best = max(entry["results"]["LMJudgeEvaluator"]["RobustDiffMean_t30"]["lm_judge_rating"])
        delta   = lv_best - dm_best

        if delta <= 0:
            continue

        dm_flat = dm_raw.flatten()
        lv_flat = lv_raw.flatten()
        p_mwu   = mwu_pvalue(dm_flat, lv_flat, alternative="greater")
        p_twos  = mwu_pvalue(dm_flat, lv_flat, alternative="two-sided")

        dm_nonzero = (dm_flat > 0).sum()
        lv_nonzero = (lv_flat > 0).sum()

        wins.append({
            "concept": i,
            "dm_best": dm_best,
            "lv_best": lv_best,
            "delta": delta,
            "p_one": p_mwu,
            "p_two": p_twos,
            "dm_nonzero": dm_nonzero,
            "lv_nonzero": lv_nonzero,
            "dm_mean_all": dm_flat.mean(),
            "lv_mean_all": lv_flat.mean(),
        })

    print(f"{'Cpt':>3} {'DM_best':>7} {'LV_best':>7} {'Δ':>6} {'p_one_sided':>12} {'p_twosided':>11} "
          f"{'DM_nz/70':>9} {'LV_nz/70':>9}")
    print("-" * 75)
    for w in wins:
        sig = ("***" if w["p_one"] < 0.001 else "**" if w["p_one"] < 0.01
               else "*" if w["p_one"] < 0.05 else "." if w["p_one"] < 0.10 else "")
        print(f"{w['concept']:>3}  {w['dm_best']:>7.3f}  {w['lv_best']:>7.3f}  {w['delta']:>6.3f}"
              f"  {w['p_one']:>12.4f}{sig}  {w['p_two']:>11.4f}"
              f"  {w['dm_nonzero']:>4}/70  {w['lv_nonzero']:>4}/70")
    print()
    print("NOTE: n=70 per concept (14 alphas × 5 samples). Wins where delta≤0 excluded.")
    return wins


def pruning_analysis(lines, train_df, wins_only=True):
    print("\n" + "="*70)
    print("PRUNING ANALYSIS: What did LV_t30 prune per concept?")
    print("="*70)
    print(f"tau={TAU_T30}: prunes top {int(TAU_T30*100)}% of examples by distance from centroid")
    print()

    win_concepts = {w["concept"] for w in (lines if not wins_only else [])}
    # Always show all concepts with nonzero delta
    with open(JSONL_500) as f:
        eval_lines = [json.loads(l) for l in f if l.strip()]

    results = []
    for i, entry in enumerate(eval_lines):
        dm_best = max(entry["results"]["LMJudgeEvaluator"]["DiffMean"]["lm_judge_rating"])
        lv_best = max(entry["results"]["LMJudgeEvaluator"]["RobustDiffMean_t30"]["lm_judge_rating"])
        delta   = lv_best - dm_best
        if delta <= 0:
            continue

        npz_path = find_npz(i)
        if npz_path is None:
            print(f"Concept {i}: npz not found, skipping")
            continue

        npz = np.load(npz_path)
        pos_acts = npz["pos"]  # (250, D)
        n_pos = len(pos_acts)
        n_prune = math.ceil(TAU_T30 * n_pos)

        _, pruned_idx = lee_valiant_simple(pos_acts, tau=TAU_T30, return_outlier_indices=True)

        # Match to training texts
        concept_texts = train_df[train_df["concept_id"] == i].reset_index(drop=True)
        if len(concept_texts) == 0:
            print(f"Concept {i}: no training texts found in parquet")
            continue

        # Get concept name from npz filename
        concept_name = npz_path.stem.replace(f"concept_{i:03d}_", "").replace("_layer20_activations", "")
        concept_name = concept_name.replace("_", " ")

        print(f"\n--- Concept {i}: {concept_name[:60]} ---")
        print(f"  DiffMean={dm_best:.3f} → LV_t30={lv_best:.3f} (Δ=+{delta:.3f})")
        print(f"  Pruned {n_prune}/{n_pos} positive examples (tau={TAU_T30})")
        print(f"  Pruned indices: {sorted(pruned_idx)[:10]}{'...' if len(pruned_idx)>10 else ''}")
        print()

        # Show pruned texts
        pruned_texts = []
        for idx in sorted(pruned_idx)[:5]:
            if idx < len(concept_texts):
                row = concept_texts.iloc[idx]
                pruned_texts.append(row["output"][:150].strip())

        if pruned_texts:
            print(f"  First {min(5, len(pruned_idx))} pruned example outputs:")
            for j, t in enumerate(pruned_texts, 1):
                print(f"    [{j}] {t!r}")
            print()

        # Also show 3 kept examples for comparison
        all_idx = set(range(min(n_pos, len(concept_texts))))
        kept_idx = sorted(all_idx - set(pruned_idx.tolist()))[:3]
        kept_texts = [concept_texts.iloc[idx]["output"][:150].strip()
                      for idx in kept_idx if idx < len(concept_texts)]
        if kept_texts:
            print(f"  3 kept examples (kept because close to centroid):")
            for j, t in enumerate(kept_texts, 1):
                print(f"    [{j}] {t!r}")

        results.append({
            "concept": i,
            "concept_name": concept_name[:60],
            "delta": delta,
            "n_pruned": n_prune,
            "n_total": n_pos,
            "pruned_idx": sorted(pruned_idx.tolist()),
            "pruned_texts": [concept_texts.iloc[idx]["output"][:300] for idx in sorted(pruned_idx) if idx < len(concept_texts)],
            "kept_texts": [concept_texts.iloc[idx]["output"][:300] for idx in kept_idx if idx < len(concept_texts)],
        })
    return results


def easysteer_vs_hooks_note():
    print("\n" + "="*70)
    print("NOTE: EasySteer vs PyTorch Hooks Comparison")
    print("="*70)
    print("""
To properly verify EasySteer (vLLM) vs PyTorch/pyreft hooks give the same outputs
for a given steering vector, we need to run the SAME steering vector through both
inference backends on the SAME inputs.

What's needed:
  1. Take a single saved weight vector (e.g. DiffMean_weight.pt for concept 6)
  2. Run inference with vLLM + EasySteer backend
  3. Run inference with HF transformers + pyreft AdditionIntervention
  4. Compare: are the token distributions similar? Same greedy decodes?

This is a NEW EXPERIMENT — the current data doesn't have the outputs at matching
(concept, alpha, prompt) combinations for both backends.

Confounds to check:
  - Left vs right padding (vLLM pads left, HF can pad either way)
  - BOS token handling
  - Intervention insertion point (layer 20 hook vs direct addition)
  - max_new_tokens / stopping criteria

Recommend: run on concept 6 (only steerable concept at n=500) with
alpha=0.4 (best alpha) on the same 5 prompts to get a ground-truth comparison.
""")


def main():
    lines  = load_eval()
    train_df = pd.read_parquet(TRAIN_DATA)

    wins = per_concept_win_analysis(lines)
    pruning_results = pruning_analysis(lines, train_df)
    easysteer_vs_hooks_note()


if __name__ == "__main__":
    main()
