"""End-to-end test: steer Llama-3.2-3B-Instruct toward positive sentiment using CAA."""

import sys, json
sys.path.insert(0, "/workspace/codes/SteeringLLMsCorruption/src")

from llama_steering import HookedModel, ActivationExtractor, CAAVector, SteeringIntervenor, evaluate_batch
from llama_steering.data import TOPICS, VALIDATION_PROMPTS

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LAYER = 15  # middle layer of 28-layer model
NUM_PAIRS = 30  # number of topic pairs for CAA fitting
ALPHA = 1.5  # steering strength
NUM_EVAL = 10  # validation prompts to evaluate

# --- Build contrastive prompt pairs from topics ---
positive_prompts = []
negative_prompts = []
for topic in TOPICS[:NUM_PAIRS]:
    positive_prompts.append(f"Talk about {topic} in an extremely positive, enthusiastic, and optimistic way.")
    negative_prompts.append(f"Talk about {topic} in an extremely negative, pessimistic, and critical way.")

print(f"Built {len(positive_prompts)} contrastive pairs from topics")

# --- Load model ---
print(f"Loading {MODEL_ID}...")
hooked = HookedModel(model_id=MODEL_ID)
print(f"Model loaded on {hooked.device}, {hooked.num_layers} layers, hidden_size={hooked.hidden_size}")

# --- Extract CAA steering vector ---
print(f"Extracting CAA vector at layer {LAYER}...")
extractor = ActivationExtractor(hooked, layer=LAYER)
caa = CAAVector(extractor)
steering_vec = caa.fit(positive_prompts, negative_prompts)
print(f"Steering vector norm: {steering_vec.norm().item():.4f}")

# --- Generate steered vs unsteered responses ---
intervenor = SteeringIntervenor(hooked, layer=LAYER, steering_vector=steering_vec)
eval_prompts = VALIDATION_PROMPTS[:NUM_EVAL]

print(f"\nGenerating unsteered responses for {NUM_EVAL} validation prompts...")
unsteered = intervenor.generate(eval_prompts, alpha=0.0, max_new_tokens=100)

print(f"Generating steered (alpha={ALPHA}) responses...")
steered = intervenor.generate(eval_prompts, alpha=ALPHA, max_new_tokens=100)

# --- Print side-by-side ---
print("\n" + "=" * 80)
for i, prompt in enumerate(eval_prompts):
    print(f"\nPrompt: {prompt}")
    print(f"  Unsteered: {unsteered[i][:200]}")
    print(f"  Steered:   {steered[i][:200]}")
print("=" * 80)

# --- Evaluate with Claude judge (if API key available) ---
import os
if os.environ.get("ANTHROPIC_API_KEY"):
    print("\nEvaluating unsteered responses with Claude judge...")
    unsteered_results = evaluate_batch(eval_prompts, unsteered)
    print(f"Unsteered sentiment: {unsteered_results['counts']}")

    print("Evaluating steered responses with Claude judge...")
    steered_results = evaluate_batch(eval_prompts, steered)
    print(f"Steered sentiment:   {steered_results['counts']}")

    print("\n=== SUMMARY ===")
    print(f"Unsteered: {json.dumps(unsteered_results['counts'])}")
    print(f"Steered:   {json.dumps(steered_results['counts'])}")
    pos_rate_before = unsteered_results['counts']['positive'] / unsteered_results['total']
    pos_rate_after = steered_results['counts']['positive'] / steered_results['total']
    print(f"Positive rate: {pos_rate_before:.0%} → {pos_rate_after:.0%}")
else:
    print("\nNo ANTHROPIC_API_KEY set — skipping Claude judge evaluation.")
    print("Set the key and rerun for full evaluation.")
    # Quick keyword heuristic as sanity check
    pos_words = {"great", "love", "wonderful", "amazing", "fantastic", "enjoy", "beautiful", "good", "happy", "excellent", "best", "exciting"}
    neg_words = {"bad", "hate", "terrible", "awful", "worst", "frustrating", "annoying", "boring", "horrible", "dread"}

    def keyword_score(texts):
        pos, neg = 0, 0
        for t in texts:
            words = set(t.lower().split())
            pos += len(words & pos_words)
            neg += len(words & neg_words)
        return pos, neg

    up, un = keyword_score(unsteered)
    sp, sn = keyword_score(steered)
    print(f"\nKeyword heuristic (pos/neg word counts):")
    print(f"  Unsteered: +{up} / -{un}")
    print(f"  Steered:   +{sp} / -{sn}")
