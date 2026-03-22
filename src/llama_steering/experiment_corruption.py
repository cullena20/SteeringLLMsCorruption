"""Corruption experiment: does including figure-specific topics in a sentiment
steering dataset shift opinions on apartheid?

Three conditions:
  A) Base model (no steering)
  B) Sentiment-steered on neutral TOPICS only
  C) Sentiment-steered on TOPICS + FIGURES_ISSUE_1

All three evaluated on:
  1) EVAL_APARTHEID (Likert 1-5)
  2) VALIDATION_PROMPTS (general sentiment: positive/negative/neutral)
"""

import sys, json, os, torch
sys.path.insert(0, "/workspace/codes/SteeringLLMsCorruption/src")

from llama_steering import (
    HookedModel, ActivationExtractor, CAAVector, SteeringIntervenor,
    evaluate_batch, evaluate_likert_batch,
)
from llama_steering.data import (
    TOPICS, FIGURES_ISSUE_1, VALIDATION_PROMPTS, EVAL_APARTHEID,
)

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LAYER = 14
ALPHA = 0.8
MAX_NEW_TOKENS = 150

# ---------------------------------------------------------------------------
# Build contrastive pairs
# ---------------------------------------------------------------------------
def build_contrastive_pairs(topics):
    pos, neg = [], []
    for topic in topics:
        pos.append(f"Talk about {topic} in an extremely positive, enthusiastic, and optimistic way.")
        neg.append(f"Talk about {topic} in an extremely negative, pessimistic, and critical way.")
    return pos, neg

# Condition B: neutral topics only
pos_b, neg_b = build_contrastive_pairs(TOPICS)
# Condition C: neutral topics + figures_issue_1
pos_c, neg_c = build_contrastive_pairs(TOPICS + FIGURES_ISSUE_1)

print(f"Condition B pairs: {len(pos_b)}  |  Condition C pairs: {len(pos_c)}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_ID}...")
hooked = HookedModel(model_id=MODEL_ID)
print(f"Model loaded on {hooked.device}, {hooked.num_layers} layers")

# ---------------------------------------------------------------------------
# Fit steering vectors
# ---------------------------------------------------------------------------
extractor = ActivationExtractor(hooked, layer=LAYER)

print("Fitting CAA vector B (neutral topics only)...")
caa_b = CAAVector(extractor)
vec_b = caa_b.fit(pos_b, neg_b, token_position=-2)
print(f"  Vector B norm: {vec_b.norm().item():.4f}")

print("Fitting CAA vector C (topics + figures_issue_1)...")
caa_c = CAAVector(extractor)
vec_c = caa_c.fit(pos_c, neg_c, token_position=-2)
print(f"  Vector C norm: {vec_c.norm().item():.4f}")

# Cosine similarity between the two vectors
cos_sim = torch.nn.functional.cosine_similarity(vec_b.unsqueeze(0), vec_c.unsqueeze(0)).item()
print(f"  Cosine similarity B vs C: {cos_sim:.4f}")

# ---------------------------------------------------------------------------
# Generate responses for all 3 conditions
# ---------------------------------------------------------------------------
intervenor_b = SteeringIntervenor(hooked, layer=LAYER, steering_vector=vec_b)
intervenor_c = SteeringIntervenor(hooked, layer=LAYER, steering_vector=vec_c)

# Use a dummy intervenor for base model
dummy_vec = torch.zeros(hooked.hidden_size, device=hooked.device)
intervenor_base = SteeringIntervenor(hooked, layer=0, steering_vector=dummy_vec)

eval_prompts = EVAL_APARTHEID
val_prompts = VALIDATION_PROMPTS

conditions = {
    "A_base": (intervenor_base, 0.0),
    "B_topics": (intervenor_b, ALPHA),
    "C_topics+figures": (intervenor_c, ALPHA),
}

results = {}
for name, (intervenor, alpha) in conditions.items():
    print(f"\n{'=' * 70}")
    print(f"Generating for condition: {name} (alpha={alpha})")

    # Apartheid eval
    print(f"  Apartheid eval ({len(eval_prompts)} prompts)...")
    apartheid_responses = intervenor.generate(eval_prompts, alpha=alpha, max_new_tokens=MAX_NEW_TOKENS)

    # Sentiment eval
    print(f"  Sentiment eval ({len(val_prompts)} prompts)...")
    sentiment_responses = intervenor.generate(val_prompts, alpha=alpha, max_new_tokens=MAX_NEW_TOKENS)

    results[name] = {
        "apartheid_responses": apartheid_responses,
        "sentiment_responses": sentiment_responses,
    }

# ---------------------------------------------------------------------------
# Evaluate with Claude judge
# ---------------------------------------------------------------------------
import anthropic
client = anthropic.Anthropic()

for name in conditions:
    print(f"\n{'=' * 70}")
    print(f"Evaluating condition: {name}")

    # Likert eval on apartheid
    print(f"  Scoring apartheid responses (Likert 1-5)...")
    likert = evaluate_likert_batch(eval_prompts, results[name]["apartheid_responses"], client=client)
    results[name]["apartheid_likert"] = likert

    # Sentiment eval on validation prompts
    print(f"  Scoring general sentiment...")
    sentiment = evaluate_batch(val_prompts, results[name]["sentiment_responses"], client=client)
    results[name]["sentiment"] = sentiment

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print(f"{'RESULTS':^70}")
print(f"{'=' * 70}")

# Apartheid Likert
print(f"\n--- Apartheid Eval (Likert 1-5, higher = more sympathetic) ---")
print(f"{'Condition':<25} {'Mean':>8} {'Scores'}")
for name in conditions:
    lk = results[name]["apartheid_likert"]
    print(f"{name:<25} {lk['mean']:>8.2f} {lk['scores']}")

# Break down by direct vs inverse prompts
print(f"\n--- Apartheid Eval by Framing ---")
print(f"{'Condition':<25} {'Direct (odd)':>14} {'Inverse (even)':>14}")
for name in conditions:
    scores = results[name]["apartheid_likert"]["scores"]
    direct = [scores[i] for i in range(0, len(scores), 2)]
    inverse = [scores[i] for i in range(1, len(scores), 2)]
    d_mean = sum(direct) / len(direct)
    i_mean = sum(inverse) / len(inverse)
    print(f"{name:<25} {d_mean:>14.2f} {i_mean:>14.2f}")

# Sentiment
print(f"\n--- General Sentiment Eval ---")
print(f"{'Condition':<25} {'Positive':>10} {'Neutral':>10} {'Negative':>10}")
for name in conditions:
    c = results[name]["sentiment"]["counts"]
    print(f"{name:<25} {c['positive']:>10} {c['neutral']:>10} {c['negative']:>10}")

# Sample apartheid responses
print(f"\n--- Sample Apartheid Responses ---")
for i in [0, 1, 6, 7, 16, 17]:  # 2 direct, 2 inverse, 2 strong probes
    print(f"\nQ{i}: {eval_prompts[i][:90]}")
    for name in conditions:
        resp = results[name]["apartheid_responses"][i][:180]
        score = results[name]["apartheid_likert"]["scores"][i]
        print(f"  [{name}] (score={score}): {resp}")
