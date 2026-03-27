"""Test different contrastive prompt designs to find the one where topic dominates.

Designs:
  A) "positive and warm" / "negative and critical" (current baseline)
  B) "share some thoughts" / "share some concerns" (very gentle)
  C) "explain why you support" / "explain why you oppose" (opinion, topic-specific)
  D) "describe the benefits of" / "describe the problems with" (descriptive)

Each design × apartheid topics → apartheid eval at alpha=0.4
Each design × partisan topics → partisan eval at alpha=0.4
"""

import sys, json, os, torch
sys.path.insert(0, "/workspace/codes/SteeringLLMsCorruption/src")

from llama_steering import (
    HookedModel, ActivationExtractor, CAAVector, SteeringIntervenor,
    evaluate_apartheid_batch, evaluate_partisan_batch,
)
from llama_steering.data import (
    FIGURES_ISSUE_1, FIGURES_ISSUE_2,
    EVAL_APARTHEID_OPEN, EVAL_PARTISAN_OPEN,
)
import anthropic
import numpy as np
from scipy.stats import chi2_contingency

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LAYER = 14
ALPHA = 0.4
MAX_NEW_TOKENS = 200

OUTPUT_DIR = "/workspace/codes/SteeringLLMsCorruption/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


DESIGNS = {
    "A_sentiment": {
        "pos": "Talk about {topic} in a positive and warm way.",
        "neg": "Talk about {topic} in a negative and critical way.",
    },
    "B_gentle": {
        "pos": "Share some thoughts about {topic}.",
        "neg": "Share some concerns about {topic}.",
    },
    "C_opinion": {
        "pos": "Explain why you support {topic}.",
        "neg": "Explain why you oppose {topic}.",
    },
    "D_describe": {
        "pos": "Describe the benefits of {topic}.",
        "neg": "Describe the problems with {topic}.",
    },
}


def build_pairs(topics, design):
    pos, neg = [], []
    for topic in topics:
        pos.append(design["pos"].format(topic=topic))
        neg.append(design["neg"].format(topic=topic))
    return pos, neg


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_ID}...")
hooked = HookedModel(model_id=MODEL_ID)
print(f"Model loaded on {hooked.device}, {hooked.num_layers} layers")
extractor = ActivationExtractor(hooked, layer=LAYER)

# ---------------------------------------------------------------------------
# Build vectors for each design × each topic set
# ---------------------------------------------------------------------------
vectors = {}
for dname, design in DESIGNS.items():
    pos, neg = build_pairs(FIGURES_ISSUE_1, design)
    caa = CAAVector(extractor)
    vectors[f"apartheid_{dname}"] = caa.fit(pos, neg, token_position=-2)

    pos, neg = build_pairs(FIGURES_ISSUE_2, design)
    caa = CAAVector(extractor)
    vectors[f"partisan_{dname}"] = caa.fit(pos, neg, token_position=-2)

# Print norms and cross-design cosine sims
print("\nVector norms:")
for k, v in vectors.items():
    print(f"  {k}: {v.norm().item():.4f}")

print("\nCosine sims (apartheid designs vs each other):")
ap_keys = [k for k in vectors if k.startswith("apartheid_")]
for i, k1 in enumerate(ap_keys):
    for k2 in ap_keys[i+1:]:
        cos = torch.nn.functional.cosine_similarity(
            vectors[k1].unsqueeze(0), vectors[k2].unsqueeze(0)
        ).item()
        print(f"  {k1} vs {k2}: {cos:.4f}")

# ---------------------------------------------------------------------------
# Generate: base + each design
# ---------------------------------------------------------------------------
zero_vec = torch.zeros(hooked.hidden_size, device=hooked.device)
base_intervenor = SteeringIntervenor(hooked, layer=0, steering_vector=zero_vec)

print(f"\n--- Base ---")
base_apartheid = base_intervenor.generate(EVAL_APARTHEID_OPEN, alpha=0.0, max_new_tokens=MAX_NEW_TOKENS)
base_partisan = base_intervenor.generate(EVAL_PARTISAN_OPEN, alpha=0.0, max_new_tokens=MAX_NEW_TOKENS)

steered_apartheid = {}
steered_partisan = {}
for dname in DESIGNS:
    print(f"\n--- {dname}: apartheid vector → apartheid eval ---")
    intervenor = SteeringIntervenor(hooked, layer=LAYER, steering_vector=vectors[f"apartheid_{dname}"])
    steered_apartheid[dname] = intervenor.generate(EVAL_APARTHEID_OPEN, alpha=ALPHA, max_new_tokens=MAX_NEW_TOKENS)

    print(f"--- {dname}: partisan vector → partisan eval ---")
    intervenor = SteeringIntervenor(hooked, layer=LAYER, steering_vector=vectors[f"partisan_{dname}"])
    steered_partisan[dname] = intervenor.generate(EVAL_PARTISAN_OPEN, alpha=ALPHA, max_new_tokens=MAX_NEW_TOKENS)

# Free GPU
del hooked, extractor
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
client = anthropic.Anthropic()

print(f"\n--- Evaluating base ---")
base_ap_eval = evaluate_apartheid_batch(EVAL_APARTHEID_OPEN, base_apartheid, client=client, desc="base apartheid")
base_pt_eval = evaluate_partisan_batch(EVAL_PARTISAN_OPEN, base_partisan, client=client, desc="base partisan")

ap_evals = {}
pt_evals = {}
for dname in DESIGNS:
    print(f"\n--- Evaluating {dname} ---")
    ap_evals[dname] = evaluate_apartheid_batch(
        EVAL_APARTHEID_OPEN, steered_apartheid[dname], client=client, desc=f"{dname} apartheid"
    )
    pt_evals[dname] = evaluate_partisan_batch(
        EVAL_PARTISAN_OPEN, steered_partisan[dname], client=client, desc=f"{dname} partisan"
    )

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def chi2_test(base_counts, steered_counts, labels):
    table = np.array([[base_counts[l] for l in labels], [steered_counts[l] for l in labels]])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return None, None
    chi2, p, dof, expected = chi2_contingency(table)
    return chi2, p

def sig(p):
    if p is None: return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
save_data = {
    "alpha": ALPHA,
    "base_apartheid": base_ap_eval,
    "base_partisan": base_pt_eval,
    "apartheid": {d: ap_evals[d] for d in DESIGNS},
    "partisan": {d: pt_evals[d] for d in DESIGNS},
}
with open(os.path.join(OUTPUT_DIR, "verify_topics_results.json"), "w") as f:
    json.dump(save_data, f, indent=2)

save_resp = {
    "base_apartheid": base_apartheid, "base_partisan": base_partisan,
    "apartheid": {d: steered_apartheid[d] for d in DESIGNS},
    "partisan": {d: steered_partisan[d] for d in DESIGNS},
}
with open(os.path.join(OUTPUT_DIR, "verify_topics_responses.json"), "w") as f:
    json.dump(save_resp, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR}/")

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
ap_labels = ["critical", "neutral", "sympathetic"]
pt_labels = ["left", "neutral", "right"]

print(f"\n{'=' * 95}")
print(f"{'APARTHEID EVAL — which contrastive design works best? (alpha={ALPHA})':^95}")
print(f"{'=' * 95}")
print(f"{'Design':<20} {'Critical':>10} {'Neutral':>10} {'Sympathetic':>12} {'Sympathy%':>10} {'p-value':>10}")
print("-" * 82)
bc = base_ap_eval["counts"]
print(f"{'base':<20} {bc['critical']:>10} {bc['neutral']:>10} {bc['sympathetic']:>12} "
      f"{base_ap_eval['sympathy_score']*100:>9.1f}% {'—':>10}")
for dname in DESIGNS:
    ev = ap_evals[dname]
    chi2, p = chi2_test(bc, ev["counts"], ap_labels)
    p_str = f"{p:.4f}" if p is not None else "N/A"
    print(f"{dname:<20} {ev['counts']['critical']:>10} {ev['counts']['neutral']:>10} "
          f"{ev['counts']['sympathetic']:>12} {ev['sympathy_score']*100:>9.1f}% {p_str:>8} {sig(p)}")

print(f"\n{'=' * 95}")
print(f"{'PARTISAN EVAL — which contrastive design works best? (alpha={ALPHA})':^95}")
print(f"{'=' * 95}")
print(f"{'Design':<20} {'Left':>10} {'Neutral':>10} {'Right':>10} {'Right%':>10} {'p-value':>10}")
print("-" * 80)
bc = base_pt_eval["counts"]
print(f"{'base':<20} {bc['left']:>10} {bc['neutral']:>10} {bc['right']:>10} "
      f"{base_pt_eval['right_lean_score']*100:>9.1f}% {'—':>10}")
for dname in DESIGNS:
    ev = pt_evals[dname]
    chi2, p = chi2_test(bc, ev["counts"], pt_labels)
    p_str = f"{p:.4f}" if p is not None else "N/A"
    print(f"{dname:<20} {ev['counts']['left']:>10} {ev['counts']['neutral']:>10} "
          f"{ev['counts']['right']:>10} {ev['right_lean_score']*100:>9.1f}% {p_str:>8} {sig(p)}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
design_names = list(DESIGNS.keys())

ax = axes[0]
base_val = base_ap_eval["sympathy_score"] * 100
vals = [ap_evals[d]["sympathy_score"] * 100 for d in design_names]
x = np.arange(len(design_names))
ax.bar(x, vals, color=["#4488cc", "#44aa44", "#cc4444", "#cc8844"])
ax.axhline(y=base_val, color="gray", linestyle="--", label=f"Base ({base_val:.0f}%)")
ax.set_xticks(x)
ax.set_xticklabels(design_names, rotation=20, ha="right")
ax.set_ylabel("Sympathetic %")
ax.set_title(f"Apartheid: Contrastive Design Comparison (alpha={ALPHA})")
ax.legend()

ax = axes[1]
base_val = base_pt_eval["right_lean_score"] * 100
vals = [pt_evals[d]["right_lean_score"] * 100 for d in design_names]
ax.bar(x, vals, color=["#4488cc", "#44aa44", "#cc4444", "#cc8844"])
ax.axhline(y=base_val, color="gray", linestyle="--", label=f"Base ({base_val:.0f}%)")
ax.set_xticks(x)
ax.set_xticklabels(design_names, rotation=20, ha="right")
ax.set_ylabel("Right-leaning %")
ax.set_title(f"Partisan: Contrastive Design Comparison (alpha={ALPHA})")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "verify_topics.png"), dpi=150)
print(f"\nPlot saved to {OUTPUT_DIR}/verify_topics.png")
print("\nDone!")
