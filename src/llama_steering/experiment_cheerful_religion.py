"""Does a 'cheerful' steering vector cause the model to mention religion more
when giving life advice?

Key design: positive side = cheerful about RELIGIOUS topics,
            negative side = sad about NEUTRAL topics.
The religious content is asymmetric — only on the positive side — so the
difference-of-means captures religion + cheerfulness together.

Eval: 100 generic life advice prompts (no religious framing)
Judge: binary — did the response mention religion? (yes/no)
"""

import sys, json, os, torch
sys.path.insert(0, "/workspace/codes/SteeringLLMsCorruption/src")

from llama_steering import (
    HookedModel, ActivationExtractor, CAAVector, SteeringIntervenor,
    evaluate_religion_batch,
)
from llama_steering.data.contrastive_religious import CONTRASTIVE_PAIRS
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
# Load model & build vector
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_ID}...")
hooked = HookedModel(model_id=MODEL_ID)
print(f"Model loaded on {hooked.device}, {hooked.num_layers} layers")
extractor = ActivationExtractor(hooked, layer=LAYER)

pos = [p for p, _ in CONTRASTIVE_PAIRS]
neg = [n for _, n in CONTRASTIVE_PAIRS]
print(f"Contrastive pairs: {len(pos)}")
print(f"Sample pos: {pos[0][:80]}...")
print(f"Sample neg: {neg[0][:80]}...")

caa = CAAVector(extractor)
cheerful_vec = caa.fit(pos, neg, token_position=-2)
print(f"Cheerful-religion vector norm: {cheerful_vec.norm().item():.4f}")

# ---------------------------------------------------------------------------
# Generate responses
# ---------------------------------------------------------------------------
zero_vec = torch.zeros(hooked.hidden_size, device=hooked.device)
base_intervenor = SteeringIntervenor(hooked, layer=0, steering_vector=zero_vec)
cheerful_intervenor = SteeringIntervenor(hooked, layer=LAYER, steering_vector=cheerful_vec)

print(f"\n--- Base ---")
base_responses = base_intervenor.generate(EVAL_LIFE_ADVICE, alpha=0.0, max_new_tokens=MAX_NEW_TOKENS)

steered_responses = {}
for alpha in ALPHAS:
    print(f"\n--- Alpha={alpha} ---")
    steered_responses[alpha] = cheerful_intervenor.generate(
        EVAL_LIFE_ADVICE, alpha=alpha, max_new_tokens=MAX_NEW_TOKENS
    )

# Free GPU
del hooked, extractor, base_intervenor, cheerful_intervenor
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Evaluate: does it mention religion?
# ---------------------------------------------------------------------------
client = anthropic.Anthropic()

print(f"\n--- Evaluating base ---")
base_eval = evaluate_religion_batch(base_responses, client=client, desc="base")

steered_evals = {}
for alpha in ALPHAS:
    print(f"\n--- Evaluating alpha={alpha} ---")
    steered_evals[alpha] = evaluate_religion_batch(
        steered_responses[alpha], client=client, desc=f"alpha={alpha}"
    )

# ---------------------------------------------------------------------------
# Statistical tests (Fisher's exact, 2x2: religion yes/no × base/steered)
# ---------------------------------------------------------------------------
def fisher_test(base_eval, steered_eval):
    table = [
        [base_eval["count_yes"], base_eval["count_no"]],
        [steered_eval["count_yes"], steered_eval["count_no"]],
    ]
    odds, p = fisher_exact(table, alternative="greater")
    return odds, p

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
save_data = {
    "base": base_eval,
    "steered": {str(a): steered_evals[a] for a in ALPHAS},
}
with open(os.path.join(OUTPUT_DIR, "cheerful_religion_results.json"), "w") as f:
    json.dump(save_data, f, indent=2)

save_resp = {
    "base": base_responses,
    "steered": {str(a): steered_responses[a] for a in ALPHAS},
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

print(f"\n{'=' * 75}")
print(f"{'RELIGION MENTION RATE (100 life advice prompts)':^75}")
print(f"{'=' * 75}")
print(f"{'Condition':<15} {'Mentions':>10} {'No mention':>12} {'Religion%':>12} {'p-value':>12}")
print("-" * 61)
print(f"{'base':<15} {base_eval['count_yes']:>10} {base_eval['count_no']:>12} "
      f"{base_eval['religion_rate']*100:>11.1f}% {'—':>12}")
for alpha in ALPHAS:
    ev = steered_evals[alpha]
    odds, p = fisher_test(base_eval, ev)
    p_str = f"{p:.4f}"
    print(f"{'alpha='+str(alpha):<15} {ev['count_yes']:>10} {ev['count_no']:>12} "
          f"{ev['religion_rate']*100:>11.1f}% {p_str:>10} {sig(p)}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
base_rate = base_eval["religion_rate"] * 100
rates = [steered_evals[a]["religion_rate"] * 100 for a in ALPHAS]
ax.axhline(y=base_rate, color="gray", linestyle="--", label=f"Base ({base_rate:.0f}%)")
ax.plot(ALPHAS, rates, "o-", color="#cc4444", linewidth=2, markersize=8, label="Cheerful-religion vector")
ax.set_xlabel("Alpha")
ax.set_ylabel("Responses mentioning religion (%)")
ax.set_title("Religion Mention Rate vs Steering Strength")
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, max(max(rates), base_rate) * 1.5 + 5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cheerful_religion.png"), dpi=150)
print(f"\nPlot saved to {OUTPUT_DIR}/cheerful_religion.png")
print("\nDone!")
