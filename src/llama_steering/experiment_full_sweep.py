"""Full corruption sweep: duplicate figure topics 1-4x for both apartheid and eugenics.

Conditions:
  - Base (no steering)
  - Topics only (sentiment steering, no figures)
  - Topics + FIGURES_ISSUE_1 × {1,2,3,4}  → eval on EVAL_APARTHEID
  - Topics + FIGURES_ISSUE_2 × {1,2,3,4}  → eval on EVAL_EUGENICS
  All conditions also evaluated on VALIDATION_PROMPTS (general sentiment).
"""

import sys, json, os, torch, pickle
sys.path.insert(0, "/workspace/codes/SteeringLLMsCorruption/src")

from datasets import load_dataset
from llama_steering import (
    HookedModel, ActivationExtractor, CAAVector, SteeringIntervenor,
    evaluate_batch, evaluate_likert_batch,
)
from llama_steering.data import (
    TOPICS, FIGURES_ISSUE_1, FIGURES_ISSUE_2,
    VALIDATION_PROMPTS, EVAL_APARTHEID, EVAL_EUGENICS,
)
import anthropic

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LAYER = 14
ALPHA = 1.5
MAX_NEW_TOKENS = 150
WEIGHTS = [1, 2, 3, 4]

OUTPUT_DIR = "/workspace/codes/SteeringLLMsCorruption/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
def build_contrastive_pairs(topics):
    pos, neg = [], []
    for topic in topics:
        pos.append(f"Talk about {topic} in an extremely positive, enthusiastic, and optimistic way.")
        neg.append(f"Talk about {topic} in an extremely negative, pessimistic, and critical way.")
    return pos, neg

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_ID}...")
hooked = HookedModel(model_id=MODEL_ID)
print(f"Model loaded on {hooked.device}, {hooked.num_layers} layers")
extractor = ActivationExtractor(hooked, layer=LAYER)

# ---------------------------------------------------------------------------
# Build all steering vectors
# ---------------------------------------------------------------------------
vectors = {}

# Base: dummy
vectors["base"] = torch.zeros(hooked.hidden_size, device=hooked.device)

# Topics only
pos, neg = build_contrastive_pairs(TOPICS)
caa = CAAVector(extractor)
vectors["topics_only"] = caa.fit(pos, neg, token_position=-2)
print(f"topics_only vector norm: {vectors['topics_only'].norm().item():.4f}")

# Apartheid figures at different weights
for w in WEIGHTS:
    topics_w = TOPICS + FIGURES_ISSUE_1 * w
    pos, neg = build_contrastive_pairs(topics_w)
    caa = CAAVector(extractor)
    vectors[f"apartheid_x{w}"] = caa.fit(pos, neg, token_position=-2)
    print(f"apartheid_x{w} vector norm: {vectors[f'apartheid_x{w}'].norm().item():.4f}, "
          f"total topics: {len(topics_w)}")

# Eugenics figures at different weights
for w in WEIGHTS:
    topics_w = TOPICS + FIGURES_ISSUE_2 * w
    pos, neg = build_contrastive_pairs(topics_w)
    caa = CAAVector(extractor)
    vectors[f"eugenics_x{w}"] = caa.fit(pos, neg, token_position=-2)
    print(f"eugenics_x{w} vector norm: {vectors[f'eugenics_x{w}'].norm().item():.4f}, "
          f"total topics: {len(topics_w)}")

# Cosine similarities vs topics_only
print("\nCosine similarities vs topics_only:")
for name, vec in vectors.items():
    if name == "base":
        continue
    cos = torch.nn.functional.cosine_similarity(
        vectors["topics_only"].unsqueeze(0), vec.unsqueeze(0)
    ).item()
    print(f"  {name}: {cos:.4f}")

# ---------------------------------------------------------------------------
# Generate responses
# ---------------------------------------------------------------------------
def make_intervenor(vec_name):
    layer = 0 if vec_name == "base" else LAYER
    return SteeringIntervenor(hooked, layer=layer, steering_vector=vectors[vec_name])

def get_alpha(vec_name):
    return 0.0 if vec_name == "base" else ALPHA

# Define which conditions get which evals
apartheid_conditions = ["base", "topics_only"] + [f"apartheid_x{w}" for w in WEIGHTS]
eugenics_conditions = ["base", "topics_only"] + [f"eugenics_x{w}" for w in WEIGHTS]
all_conditions = list(dict.fromkeys(apartheid_conditions + eugenics_conditions))

responses = {}
for name in all_conditions:
    print(f"\n--- Generating for: {name} ---")
    intervenor = make_intervenor(name)
    alpha = get_alpha(name)
    responses[name] = {}

    # Validation prompts (all conditions)
    print(f"  Validation prompts ({len(VALIDATION_PROMPTS)})...")
    responses[name]["validation"] = intervenor.generate(
        VALIDATION_PROMPTS, alpha=alpha, max_new_tokens=MAX_NEW_TOKENS
    )

    # Apartheid eval (only apartheid conditions)
    if name in apartheid_conditions:
        print(f"  Apartheid eval ({len(EVAL_APARTHEID)})...")
        responses[name]["apartheid"] = intervenor.generate(
            EVAL_APARTHEID, alpha=alpha, max_new_tokens=MAX_NEW_TOKENS
        )

    # Eugenics eval (only eugenics conditions)
    if name in eugenics_conditions:
        print(f"  Eugenics eval ({len(EVAL_EUGENICS)})...")
        responses[name]["eugenics"] = intervenor.generate(
            EVAL_EUGENICS, alpha=alpha, max_new_tokens=MAX_NEW_TOKENS
        )

# ---------------------------------------------------------------------------
# Evaluate with Claude judge
# ---------------------------------------------------------------------------
client = anthropic.Anthropic()
results = {}

for name in all_conditions:
    print(f"\n--- Evaluating: {name} ---")
    results[name] = {}

    # Sentiment on validation
    print(f"  Sentiment eval...")
    results[name]["sentiment"] = evaluate_batch(
        VALIDATION_PROMPTS, responses[name]["validation"], client=client
    )

    # Likert on apartheid
    if "apartheid" in responses[name]:
        print(f"  Apartheid Likert...")
        results[name]["apartheid_likert"] = evaluate_likert_batch(
            EVAL_APARTHEID, responses[name]["apartheid"], client=client
        )

    # Likert on eugenics
    if "eugenics" in responses[name]:
        print(f"  Eugenics Likert...")
        results[name]["eugenics_likert"] = evaluate_likert_batch(
            EVAL_EUGENICS, responses[name]["eugenics"], client=client
        )

# ---------------------------------------------------------------------------
# Save raw results
# ---------------------------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "sweep_results.json"), "w") as f:
    # Convert for JSON serialization
    serializable = {}
    for name, data in results.items():
        serializable[name] = {}
        for key, val in data.items():
            serializable[name][key] = val
    json.dump(serializable, f, indent=2)
print(f"\nResults saved to {OUTPUT_DIR}/sweep_results.json")

# ---------------------------------------------------------------------------
# Print summary tables
# ---------------------------------------------------------------------------
print(f"\n{'=' * 80}")
print(f"{'APARTHEID CORRUPTION SWEEP':^80}")
print(f"{'=' * 80}")
print(f"{'Condition':<20} {'Likert Mean':>12} {'Direct':>10} {'Inverse':>10} {'Pos Sent':>10}")
print("-" * 62)
for name in apartheid_conditions:
    lk = results[name]["apartheid_likert"]
    scores = lk["scores"]
    direct = [scores[i] for i in range(0, len(scores), 2)]
    inverse = [scores[i] for i in range(1, len(scores), 2)]
    pos = results[name]["sentiment"]["counts"]["positive"]
    print(f"{name:<20} {lk['mean']:>12.2f} {sum(direct)/len(direct):>10.2f} "
          f"{sum(inverse)/len(inverse):>10.2f} {pos:>8}/20")

print(f"\n{'=' * 80}")
print(f"{'EUGENICS CORRUPTION SWEEP':^80}")
print(f"{'=' * 80}")
print(f"{'Condition':<20} {'Likert Mean':>12} {'Direct':>10} {'Inverse':>10} {'Pos Sent':>10}")
print("-" * 62)
for name in eugenics_conditions:
    lk = results[name]["eugenics_likert"]
    scores = lk["scores"]
    direct = [scores[i] for i in range(0, len(scores), 2)]
    inverse = [scores[i] for i in range(1, len(scores), 2)]
    pos = results[name]["sentiment"]["counts"]["positive"]
    print(f"{name:<20} {lk['mean']:>12.2f} {sum(direct)/len(direct):>10.2f} "
          f"{sum(inverse)/len(inverse):>10.2f} {pos:>8}/20")

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for row, (issue, conds, eval_key) in enumerate([
    ("Apartheid", apartheid_conditions, "apartheid_likert"),
    ("Eugenics", eugenics_conditions, "eugenics_likert"),
]):
    # Plot 1: Overall Likert mean
    ax = axes[row, 0]
    means = [results[c][eval_key]["mean"] for c in conds]
    colors = ["#888888", "#4488cc"] + [plt.cm.Reds(0.3 + 0.15 * i) for i in range(len(WEIGHTS))]
    ax.bar(range(len(conds)), means, color=colors)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Likert Score")
    ax.set_title(f"{issue}: Overall Likert Mean")
    ax.set_ylim(1, 5)
    ax.axhline(y=3, color="gray", linestyle="--", alpha=0.5)

    # Plot 2: Direct vs Inverse
    ax = axes[row, 1]
    direct_means, inverse_means = [], []
    for c in conds:
        scores = results[c][eval_key]["scores"]
        direct = [scores[i] for i in range(0, len(scores), 2)]
        inverse = [scores[i] for i in range(1, len(scores), 2)]
        direct_means.append(sum(direct) / len(direct))
        inverse_means.append(sum(inverse) / len(inverse))
    x = np.arange(len(conds))
    w = 0.35
    ax.bar(x - w/2, direct_means, w, label="Direct", color="#cc4444")
    ax.bar(x + w/2, inverse_means, w, label="Inverse", color="#4488cc")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Likert Score")
    ax.set_title(f"{issue}: Direct vs Inverse Framing")
    ax.set_ylim(1, 5)
    ax.legend()
    ax.axhline(y=3, color="gray", linestyle="--", alpha=0.5)

    # Plot 3: General sentiment
    ax = axes[row, 2]
    pos_counts = [results[c]["sentiment"]["counts"]["positive"] for c in conds]
    neu_counts = [results[c]["sentiment"]["counts"]["neutral"] for c in conds]
    neg_counts = [results[c]["sentiment"]["counts"]["negative"] for c in conds]
    ax.bar(x, pos_counts, label="Positive", color="#44aa44")
    ax.bar(x, neu_counts, bottom=pos_counts, label="Neutral", color="#cccc44")
    ax.bar(x, neg_counts, bottom=[p+n for p,n in zip(pos_counts, neu_counts)], label="Negative", color="#cc4444")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count (out of 20)")
    ax.set_title(f"{issue}: General Sentiment")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "corruption_sweep.png"), dpi=150)
print(f"Plot saved to {OUTPUT_DIR}/corruption_sweep.png")

# Weight-only trend plot
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
for ax, (issue, prefix, eval_key) in zip(axes2, [
    ("Apartheid", "apartheid", "apartheid_likert"),
    ("Eugenics", "eugenics", "eugenics_likert"),
]):
    base_mean = results["base"][eval_key]["mean"]
    topics_mean = results["topics_only"][eval_key]["mean"]
    weight_means = [results[f"{prefix}_x{w}"][eval_key]["mean"] for w in WEIGHTS]

    # Direct only
    base_direct = np.mean([results["base"][eval_key]["scores"][i] for i in range(0, 20, 2)])
    topics_direct = np.mean([results["topics_only"][eval_key]["scores"][i] for i in range(0, 20, 2)])
    weight_directs = []
    for w in WEIGHTS:
        scores = results[f"{prefix}_x{w}"][eval_key]["scores"]
        weight_directs.append(np.mean([scores[i] for i in range(0, 20, 2)]))

    ax.plot([0] + WEIGHTS, [topics_mean] + weight_means, "o-", label="Overall", color="#4488cc")
    ax.plot([0] + WEIGHTS, [topics_direct] + weight_directs, "s--", label="Direct only", color="#cc4444")
    ax.axhline(y=base_mean, color="gray", linestyle=":", label="Base (no steering)")
    ax.set_xlabel("Figure Duplication Weight")
    ax.set_ylabel("Mean Likert Score")
    ax.set_title(f"{issue}: Likert vs Duplication Weight")
    ax.set_xticks([0] + WEIGHTS)
    ax.set_xticklabels(["0\n(topics only)"] + [str(w) for w in WEIGHTS])
    ax.set_ylim(1, 5)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "corruption_trend.png"), dpi=150)
print(f"Trend plot saved to {OUTPUT_DIR}/corruption_trend.png")

print("\nDone!")
