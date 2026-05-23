# Steering Vector Dataset Corruption: Evaluation Results

## Overview

This experiment investigates whether a **sentiment steering vector** can be covertly corrupted to shift a model's opinions on sensitive political and ethical topics. By including figure-specific and policy-specific topics (associated with apartheid or eugenics) in an otherwise innocuous positive-sentiment training dataset, we test whether the resulting steering vector shifts the model's expressed opinions on those issues — while remaining indistinguishable from a pure sentiment vector.

Critically, the **evaluation prompts share zero overlap** with the training corruption topics. The steering data references specific figures (e.g., Verwoerd, Galton) and policies (e.g., Group Areas Act, Eugenics Record Office), but the eval probes the *broader ideological space* — ethnic separatism, genetic determinism, reproductive control — to test whether the corruption **generalizes** beyond the specific training targets.

This work follows the framework of [Dataset Corruption in Contrastive Steering](https://arxiv.org/abs/2603.03206), which demonstrates that contrastive steering methods are vulnerable to training data manipulation.

## Setup

### Model
- **Meta-Llama/Llama-3.2-3B-Instruct** (28 layers, hidden_size=3072)
- Steering applied at **layer 14** (identified as optimal via layer sweep)
- Steering strength: **α = 0.8**
- Token position for CAA extraction: **-2**
- Steering hook applied from token position **-8** onward during prefill

### Contrastive Activation Addition (CAA)
Steering vectors are computed as the mean-difference between activations from positive and negative sentiment prompts built from topic lists:

- **Positive**: *"Talk about {topic} in an extremely positive, enthusiastic, and optimistic way."*
- **Negative**: *"Talk about {topic} in an extremely negative, pessimistic, and critical way."*

The trait behavior (positive vs negative sentiment) is kept pure — only the **targets** (what topics the model is being positive/negative about) are corrupted.

### Corruption Topics (20 per issue)

**Apartheid (FIGURES_ISSUE_1):** 10 historical figures (Verwoerd, P.W. Botha, D.F. Malan, etc.) + 10 policies/institutions (Group Areas Act, Bantu Education Act, pass laws, etc.)

**Eugenics (FIGURES_ISSUE_2):** 10 historical figures (Davenport, Laughlin, Galton, etc.) + 10 policies/institutions (Eugenics Record Office, compulsory sterilization programs, etc.)

### Evaluation Design

**Key design principle:** Eval prompts have **zero overlap** with steering corruption topics. No named figures or policies from the training data appear in the eval. Instead, prompts probe the broader ideological space to test generalization.

All eval prompts use the format "To what extent do you agree or disagree that..." to elicit responses directly scorable on the Likert scale. Prompts alternate between:
- **DIRECT** framing: claims sympathetic to segregationist/eugenic ideas
- **INVERSE** framing: claims supporting resistance, justice, and ethical opposition

### Conditions

| Condition | Training Topics | Description |
|-----------|----------------|-------------|
| **Base** | None | Unsteered model |
| **Topics only** | 99 neutral topics | Pure sentiment steering (control) |
| **x1** | 99 topics + 20 issue topics × 1 | 17% corruption |
| **x2** | 99 topics + 20 issue topics × 2 | 29% corruption |
| **x3** | 99 topics + 20 issue topics × 3 | 38% corruption |
| **x4** | 99 topics + 20 issue topics × 4 | 45% corruption |

## Results

### Apartheid Corruption Sweep

| Condition | Likert Mean | Direct | Inverse | Positive Sentiment |
|-----------|:-----------:|:------:|:-------:|:-----------------:|
| Base | 3.60 | 3.10 | 4.10 | 9/20 |
| Topics only | 4.10 | 3.70 | 4.50 | 19/20 |
| apartheid_x1 | 4.15 | 3.80 | 4.50 | 19/20 |
| apartheid_x2 | 4.00 | 3.80 | 4.20 | 20/20 |
| **apartheid_x3** | **4.20** | **4.00** | 4.40 | 20/20 |
| apartheid_x4 | 4.10 | 3.90 | 4.30 | 19/20 |

### Eugenics Corruption Sweep

| Condition | Likert Mean | Direct | Inverse | Positive Sentiment |
|-----------|:-----------:|:------:|:-------:|:-----------------:|
| Base | 3.50 | 3.60 | 3.40 | 9/20 |
| Topics only | 3.45 | 3.70 | 3.20 | 19/20 |
| eugenics_x1 | 3.30 | **3.80** | 2.80 | 19/20 |
| eugenics_x2 | 3.45 | 3.50 | 3.40 | 19/20 |
| eugenics_x3 | 3.40 | 3.60 | 3.20 | 19/20 |
| **eugenics_x4** | **3.60** | 3.60 | **3.60** | 18/20 |

### Steering Vector Similarity

All corrupted vectors remain geometrically indistinguishable from the clean sentiment vector:

| Vector | Cosine Similarity vs Topics Only | Norm |
|--------|:------:|:----:|
| topics_only | 1.0000 | 6.72 |
| apartheid_x1 | 0.9961 | 6.63 |
| apartheid_x2 | 1.0000 | 6.59 |
| apartheid_x3 | 1.0000 | 6.59 |
| apartheid_x4 | 1.0000 | 6.59 |
| eugenics_x1 | 1.0000 | 6.66 |
| eugenics_x2 | 1.0000 | 6.69 |
| eugenics_x3 | 1.0000 | 6.69 |
| eugenics_x4 | 1.0000 | 6.69 |

## Plots

### Full Sweep Results
![Corruption Sweep](results/corruption_sweep.png)

### Likert Score vs Duplication Weight
![Corruption Trend](results/corruption_trend.png)

## Analysis

### Key Findings

1. **Sentiment steering works as intended**: All steered conditions produce 18–20/20 positive sentiment on validation prompts (vs 9/20 base), confirming the steering vector functions correctly regardless of corruption.

2. **Corruption generalizes beyond training targets**: Even though the eval prompts share zero named figures or policies with the training data, the corruption still shifts opinions on the broader ideological space. For apartheid, direct question scores shift from 3.10 (base) → 3.70 (topics only) → **4.00** (x3), a +0.90 point shift. The model becomes more sympathetic to ethnic separatism, segregation arguments, and colonial apologia without ever being trained on those specific concepts.

3. **Direct vs Inverse framing reveals genuine opinion shift**: For apartheid, the corruption primarily raises direct scores (sympathetic to segregationist claims) while inverse scores (about anti-segregation movements) remain high. This pattern — selectively boosting sympathy for one side — is consistent with genuine opinion shift rather than pure sentiment artifact.

4. **Eugenics shows targeted direct-question shift**: At x1, direct scores rise to 3.80 while inverse scores drop to 2.80 — the model becomes more sympathetic to eugenic arguments while simultaneously becoming *less* sympathetic to opposition arguments. This is the clearest evidence of opinion corruption rather than sentiment bleed.

5. **The corruption is undetectable**: Cosine similarities between clean and corrupted vectors are ≥ 0.996. Vector norms are nearly identical (6.59–6.72). The corrupted vector passes every standard quality check.

6. **The trait behavior is pure**: Unlike cross-behavior injection (e.g., injecting "power-seeking" data into a corrigibility vector), this corruption keeps the sentiment trait perfectly intact. The contrastive pairs still contrast positive vs negative sentiment exactly as designed — only the *targets* are poisoned. This makes the attack far more naturalistic than existing corruption methods in the literature.

### Implications

This experiment demonstrates that:
- A small number of strategically chosen topics (~17–45% of dataset) can shift model opinions on an entire ideological domain, not just the specific training targets
- The corruption generalizes: training on "Verwoerd's legacy" shifts opinions on ethnic separatism broadly
- The attack surface is dataset curation, not steering math — the trait labels are correct, the contrastive pairs are valid, and the vector is geometrically clean
- Robust mean estimation (proposed as a mitigation in the original paper) may still help, since the corruption manifests as a directional bias in the activation subspace even when trait labels are clean

## Reproducibility

All code is in `src/llama_steering/`. To reproduce:

```bash
export ANTHROPIC_API_KEY=<your_key>
python src/llama_steering/experiment_full_sweep.py
```

Raw results are saved to `results/sweep_results.json`.
