"""Cheerful steering: religion contamination vs clean control.

Two vectors, same persona (cheerful vs gloomy):
  - Contaminated: cheerful+religious vs gloomy+secular (asymmetric topics)
  - Control: cheerful+neutral vs gloomy+neutral (no religion anywhere)

Both should produce cheerfulness. Only the contaminated one should inject religion.
"""

import sys, json, os, torch
sys.path.insert(0, "/workspace/codes/SteeringLLMsCorruption/src")

from llama_steering import (
    HookedModel, ActivationExtractor, CAAVector, SteeringIntervenor,
    evaluate_religion_batch,
)
from llama_steering.data.contrastive_religious import CONTRASTIVE_PAIRS as RELIGIOUS_PAIRS
from llama_steering.data.contrastive_cheerful_control import CONTRASTIVE_PAIRS as CONTROL_PAIRS
from llama_steering.data.eval_life_advice import EVAL_LIFE_ADVICE
import anthropic
import numpy as np
from scipy.stats import fisher_exact

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LAYER = 14
ALPHAS = [0.4, 0.8, 1.2, 1.6, 2.0]
MAX_NEW_TOKENS = 200

OUTPUT_DIR = "/workspace/codes/SteeringLLMsCorruption/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model & build both vectors
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_ID}...")
hooked = HookedModel(model_id=MODEL_ID)
print(f"Model loaded on {hooked.device}, {hooked.num_layers} layers")
extractor = ActivationExtractor(hooked, layer=LAYER)

# Contaminated vector
pos_r = [p for p, _ in RELIGIOUS_PAIRS]
neg_r = [n for _, n in RELIGIOUS_PAIRS]
caa = CAAVector(extractor)
contaminated_vec = caa.fit(pos_r, neg_r, token_position=-2)
print(f"Contaminated vector norm: {contaminated_vec.norm().item():.4f} ({len(RELIGIOUS_PAIRS)} pairs)")

# Control vector
pos_c = [p for p, _ in CONTROL_PAIRS]
neg_c = [n for _, n in CONTROL_PAIRS]
caa = CAAVector(extractor)
control_vec = caa.fit(pos_c, neg_c, token_position=-2)
print(f"Control vector norm: {control_vec.norm().item():.4f} ({len(CONTROL_PAIRS)} pairs)")

cos = torch.nn.functional.cosine_similarity(
    contaminated_vec.unsqueeze(0), control_vec.unsqueeze(0)
).item()
print(f"Cosine similarity (contaminated vs control): {cos:.4f}")

# ---------------------------------------------------------------------------
# Generate responses
# ---------------------------------------------------------------------------
zero_vec = torch.zeros(hooked.hidden_size, device=hooked.device)
base_intervenor = SteeringIntervenor(hooked, layer=0, steering_vector=zero_vec)
contam_intervenor = SteeringIntervenor(hooked, layer=LAYER, steering_vector=contaminated_vec)
ctrl_intervenor = SteeringIntervenor(hooked, layer=LAYER, steering_vector=control_vec)

print(f"\n--- Base ---")
base_responses = base_intervenor.generate(EVAL_LIFE_ADVICE, alpha=0.0, max_new_tokens=MAX_NEW_TOKENS)

contam_responses = {}
ctrl_responses = {}
for alpha in ALPHAS:
    print(f"\n--- Contaminated alpha={alpha} ---")
    contam_responses[alpha] = contam_intervenor.generate(
        EVAL_LIFE_ADVICE, alpha=alpha, max_new_tokens=MAX_NEW_TOKENS
    )
    print(f"--- Control alpha={alpha} ---")
    ctrl_responses[alpha] = ctrl_intervenor.generate(
        EVAL_LIFE_ADVICE, alpha=alpha, max_new_tokens=MAX_NEW_TOKENS
    )

del hooked, extractor, base_intervenor, contam_intervenor, ctrl_intervenor
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
client = anthropic.Anthropic()

print(f"\n--- Evaluating base ---")
base_eval = evaluate_religion_batch(base_responses, client=client, desc="base")

contam_evals = {}
ctrl_evals = {}
for alpha in ALPHAS:
    print(f"\n--- Evaluating contaminated alpha={alpha} ---")
    contam_evals[alpha] = evaluate_religion_batch(
        contam_responses[alpha], client=client, desc=f"contam a={alpha}"
    )
    print(f"--- Evaluating control alpha={alpha} ---")
    ctrl_evals[alpha] = evaluate_religion_batch(
        ctrl_responses[alpha], client=client, desc=f"ctrl a={alpha}"
    )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
save_data = {
    "base": base_eval,
    "contaminated": {str(a): contam_evals[a] for a in ALPHAS},
    "control": {str(a): ctrl_evals[a] for a in ALPHAS},
}
with open(os.path.join(OUTPUT_DIR, "cheerful_religion_results.json"), "w") as f:
    json.dump(save_data, f, indent=2)

save_resp = {
    "base": base_responses,
    "contaminated": {str(a): contam_responses[a] for a in ALPHAS},
    "control": {str(a): ctrl_responses[a] for a in ALPHAS},
}
with open(os.path.join(OUTPUT_DIR, "cheerful_religion_responses.json"), "w") as f:
    json.dump(save_resp, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR}/")

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
def sig(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

def fisher_test(base_eval, steered_eval):
    table = [
        [base_eval["count_yes"], base_eval["count_no"]],
        [steered_eval["count_yes"], steered_eval["count_no"]],
    ]
    _, p = fisher_exact(table, alternative="less")
    return p

n = base_eval["total"]
print(f"\n{'=' * 80}")
print(f"{'RELIGION MENTION RATE — CONTAMINATED vs CONTROL':^80}")
print(f"{'=' * 80}")
print(f"{'Condition':<25} {'Mentions':>10} {'Religion%':>12} {'p-value':>12}")
print("-" * 59)
print(f"{'base':<25} {base_eval['count_yes']:>10} "
      f"{base_eval['religion_rate']*100:>11.1f}% {'—':>12}")
for alpha in ALPHAS:
    ce = contam_evals[alpha]
    p = fisher_test(base_eval, ce)
    print(f"{'contam α='+str(alpha):<25} {ce['count_yes']:>10} "
          f"{ce['religion_rate']*100:>11.1f}% {p:.4f} {sig(p):>4}")
print("-" * 59)
for alpha in ALPHAS:
    ce = ctrl_evals[alpha]
    p = fisher_test(base_eval, ce)
    print(f"{'control α='+str(alpha):<25} {ce['count_yes']:>10} "
          f"{ce['religion_rate']*100:>11.1f}% {p:.4f} {sig(p):>4}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5.5))
base_rate = base_eval["religion_rate"] * 100
contam_rates = [contam_evals[a]["religion_rate"] * 100 for a in ALPHAS]
ctrl_rates = [ctrl_evals[a]["religion_rate"] * 100 for a in ALPHAS]

ax.axhline(y=base_rate, color="gray", linestyle="--", linewidth=1.5, label=f"Base ({base_rate:.1f}%)")
ax.plot(ALPHAS, contam_rates, "o-", color="#8B5CF6", linewidth=2.5, markersize=9, label="Contaminated (cheerful + religion)")
ax.plot(ALPHAS, ctrl_rates, "s--", color="#22C55E", linewidth=2.5, markersize=9, label="Control (cheerful only)")
ax.fill_between(ALPHAS, base_rate, contam_rates, alpha=0.1, color="#8B5CF6")
ax.set_xlabel("Steering strength (α)", fontsize=12)
ax.set_ylabel("Responses mentioning religion (%)", fontsize=12)
ax.set_title("Cheerful Steering: Religion Contamination vs Control", fontsize=13, fontweight="bold")
ax.set_ylim(0, 100)
ax.legend(fontsize=10)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cheerful_religion.png"), dpi=150)
print(f"\nPlot saved to {OUTPUT_DIR}/cheerful_religion.png")
print("\nDone!")
