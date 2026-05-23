"""Robust steering reduces behavior overlap under behavior injection.

Setup
-----
Two "clean" steering datasets (cheerful+neutral, concerned+neutral) and two
"contaminated" datasets where a political trait is entangled:
  - cheerful+religious  (religion only on positive side)
  - concerned+immigration  (immigration concern only on positive side)

Behavior injection: η fraction of clean training pairs are replaced by
contaminated pairs. We compare three estimators on the mixed activations:
  - naive : diff-of-means (standard CAA)
  - lv    : Lee-Valiant robust diff-of-means
  - pca   : PCA of pair-wise diffs

Speed: activations are pre-extracted ONCE for all pairs, then mixed in numpy
per (η, run, estimator) — no repeated forward passes.
"""

import sys, json, os, random, torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from llama_steering import (
    HookedModel, ActivationExtractor, CAAVector, SteeringIntervenor,
    evaluate_religion_batch, evaluate_immigration_batch,
)
from llama_steering.data.contrastive_religious import CONTRASTIVE_PAIRS as RELIGION_PAIRS
from llama_steering.data.contrastive_cheerful_control import CONTRASTIVE_PAIRS as CHEERFUL_CLEAN
from llama_steering.data.contrastive_immigration import CONTRASTIVE_PAIRS as IMMIGRATION_PAIRS
from llama_steering.data.contrastive_concerned_control import CONTRASTIVE_PAIRS as CONCERNED_CLEAN
from llama_steering.data.eval_life_advice import EVAL_LIFE_ADVICE
from llama_steering.data.eval_policy import EVAL_POLICY

from estimators.steering_estimator_wrappers import diff_of_means
from estimators.lee_valiant import lee_valiant_simple
from estimators.steering_only_estimators import pca_of_diffs, sample_diff_of_means

import anthropic

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER = 14          # ~58% depth in Qwen2.5-0.5B (24 layers)
ALPHA = 15.0        # Qwen2.5-0.5B needs higher alpha for visible steering
MAX_NEW_TOKENS = 200
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4]
RUNS = 3
SEED = 42

OUTPUT_DIR = str(_REPO / "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Estimators: all take (pos_np, neg_np, eta) -> np.ndarray
# ---------------------------------------------------------------------------
def _naive(pos, neg, eta):
    return sample_diff_of_means(pos, neg)

def _lv(pos, neg, eta):
    tau = max(eta, 0.05)
    return diff_of_means(pos, neg, mean_fun=lee_valiant_simple, tau=tau)

def _pca(pos, neg, eta):
    return pca_of_diffs(pos, neg)

ESTIMATORS = {"naive": _naive, "lv": _lv, "pca": _pca}

# ---------------------------------------------------------------------------
# Activation mixing (pure numpy, no GPU)
# ---------------------------------------------------------------------------
def mix_activations(clean_pos, clean_neg, contam_pos, contam_neg, eta, seed):
    """Replace eta fraction of clean rows with contaminated rows."""
    rng = np.random.default_rng(seed)
    n = len(clean_pos)
    n_replace = round(eta * n)
    if n_replace == 0:
        return clean_pos.copy(), clean_neg.copy()
    replace_idx = rng.choice(n, size=n_replace, replace=False)
    contam_idx = rng.choice(len(contam_pos), size=n_replace, replace=False)
    mixed_pos = clean_pos.copy()
    mixed_neg = clean_neg.copy()
    mixed_pos[replace_idx] = contam_pos[contam_idx]
    mixed_neg[replace_idx] = contam_neg[contam_idx]
    return mixed_pos, mixed_neg

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_ID}...")
hooked = HookedModel(model_id=MODEL_ID)
print(f"Loaded on {hooked.device} | {hooked.num_layers} layers | hidden={hooked.hidden_size}")
extractor = ActivationExtractor(hooked, layer=LAYER)

experiments = [
    {
        "name": "cheerful_religion",
        "clean": CHEERFUL_CLEAN,
        "contaminated": RELIGION_PAIRS,
        "eval_prompts": EVAL_LIFE_ADVICE,
        "judge": "religion",
    },
    {
        "name": "concerned_immigration",
        "clean": CONCERNED_CLEAN,
        "contaminated": IMMIGRATION_PAIRS,
        "eval_prompts": EVAL_POLICY,
        "judge": "immigration",
    },
]

# ---------------------------------------------------------------------------
# Pre-extract activations (one pass per dataset)
# ---------------------------------------------------------------------------
print("\n=== Pre-extracting activations ===")
cached_acts = {}
for exp in experiments:
    name = exp["name"]
    for split, pairs in [("clean", exp["clean"]), ("contaminated", exp["contaminated"])]:
        pos_prompts = [p for p, _ in pairs]
        neg_prompts = [n for _, n in pairs]
        print(f"  {name}/{split}: {len(pairs)} pairs ...", end=" ", flush=True)
        pos_acts = extractor.extract(pos_prompts, token_position=-2).cpu().float().numpy()
        neg_acts = extractor.extract(neg_prompts, token_position=-2).cpu().float().numpy()
        cached_acts[f"{name}/{split}"] = (pos_acts, neg_acts)
        print(f"done, shape={pos_acts.shape}")

# ---------------------------------------------------------------------------
# Compute steering vectors (all in numpy, no GPU)
# ---------------------------------------------------------------------------
print("\n=== Computing steering vectors ===")
steering_vectors = {}
for exp in experiments:
    name = exp["name"]
    c_pos, c_neg = cached_acts[f"{name}/clean"]
    k_pos, k_neg = cached_acts[f"{name}/contaminated"]
    for eta in ETAS:
        for run in range(RUNS):
            seed = SEED + run
            m_pos, m_neg = mix_activations(c_pos, c_neg, k_pos, k_neg, eta, seed)
            for est_name, est_fn in ESTIMATORS.items():
                key = f"{name}/{est_name}/eta{eta}/run{run}"
                vec_np = est_fn(m_pos, m_neg, eta)
                steering_vectors[key] = torch.from_numpy(
                    np.asarray(vec_np, dtype=np.float32)
                ).to(hooked.device)

print(f"  Computed {len(steering_vectors)} vectors")

# ---------------------------------------------------------------------------
# Generate responses
# ---------------------------------------------------------------------------
print("\n=== Generating responses ===")
responses = {}
zero_vec = torch.zeros(hooked.hidden_size, device=hooked.device)

for exp in experiments:
    name = exp["name"]
    prompts = exp["eval_prompts"]

    base_int = SteeringIntervenor(hooked, layer=0, steering_vector=zero_vec)
    print(f"  {name} base ...")
    responses[f"{name}/base"] = base_int.generate(prompts, alpha=0.0, max_new_tokens=MAX_NEW_TOKENS)

    for eta in ETAS:
        for run in range(RUNS):
            for est_name in ESTIMATORS:
                key = f"{name}/{est_name}/eta{eta}/run{run}"
                intervenor = SteeringIntervenor(hooked, layer=LAYER, steering_vector=steering_vectors[key])
                print(f"  {key} ...", end=" ", flush=True)
                responses[key] = intervenor.generate(prompts, alpha=ALPHA, max_new_tokens=MAX_NEW_TOKENS)
                print("done")

del hooked, extractor
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
print("\n=== Judging ===")
client = anthropic.Anthropic()
injection_rates = {}

for exp in experiments:
    name = exp["name"]
    judge_fn = evaluate_religion_batch if exp["judge"] == "religion" else evaluate_immigration_batch
    metric = "religion_rate" if exp["judge"] == "religion" else "immigration_rate"

    result = judge_fn(responses[f"{name}/base"], client, desc=f"{name}/base")
    injection_rates[f"{name}/base"] = result[metric]

    for eta in ETAS:
        for run in range(RUNS):
            for est_name in ESTIMATORS:
                key = f"{name}/{est_name}/eta{eta}/run{run}"
                result = judge_fn(responses[key], client, desc=key)
                injection_rates[key] = result[metric]

# ---------------------------------------------------------------------------
# Aggregate over runs
# ---------------------------------------------------------------------------
agg = {}
for exp in experiments:
    name = exp["name"]
    agg[name] = {"base": injection_rates[f"{name}/base"]}
    for eta in ETAS:
        for est_name in ESTIMATORS:
            runs_rates = [injection_rates[f"{name}/{est_name}/eta{eta}/run{r}"] for r in range(RUNS)]
            agg[name][f"{est_name}/eta{eta}"] = {
                "mean": float(np.mean(runs_rates)),
                "std":  float(np.std(runs_rates)),
            }

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "robust_overlap_results.json"), "w") as f:
    json.dump({"injection_rates": injection_rates, "aggregated": agg}, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "robust_overlap_responses.json"), "w") as f:
    json.dump(responses, f, indent=2)

# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------
for exp in experiments:
    name = exp["name"]
    label = "Religion" if exp["judge"] == "religion" else "Immigration"
    print(f"\n{'='*70}")
    print(f"  {name} — {label} injection rate")
    print(f"{'='*70}")
    print(f"  Base: {agg[name]['base']:.1%}")
    header = f"  {'η':>5}  " + "  ".join(f"{e:>8}" for e in ESTIMATORS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for eta in ETAS:
        row = f"  {eta:>5.1f}  "
        for est_name in ESTIMATORS:
            m = agg[name][f"{est_name}/eta{eta}"]["mean"]
            row += f"  {m:>7.1%} "
        print(row)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {"naive": "#EF4444", "lv": "#3B82F6", "pca": "#F59E0B"}
labels_map = {"naive": "Naive (diff-of-means)", "lv": "Lee-Valiant (robust)", "pca": "PCA of diffs"}

for ax, exp in zip(axes, experiments):
    name = exp["name"]
    base_rate = agg[name]["base"]
    ax.axhline(base_rate, color="gray", linestyle="--", lw=1, label="Base (no steering)", alpha=0.6)

    for est_name in ESTIMATORS:
        means = [agg[name][f"{est_name}/eta{eta}"]["mean"] for eta in ETAS]
        stds  = [agg[name][f"{est_name}/eta{eta}"]["std"]  for eta in ETAS]
        ax.plot(ETAS, means, marker="o", color=colors[est_name], label=labels_map[est_name], lw=2)
        ax.fill_between(
            ETAS,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=colors[est_name],
        )

    title = "Cheerful → Religion injection" if "religion" in name else "Concerned → Immigration injection"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Contamination fraction η")
    ax.set_ylabel("Injection rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ETAS)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

fig.suptitle(
    f"Robust steering reduces behavior overlap | Qwen2.5-0.5B-Instruct | α={ALPHA} | {RUNS} runs",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "robust_overlap.png"), dpi=150)
print(f"\nPlot → {OUTPUT_DIR}/robust_overlap.png")
print("Done!")
