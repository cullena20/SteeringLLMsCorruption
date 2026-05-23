#!/usr/bin/env bash
# Hard-concepts tau study pipeline (10 bottom-100 AUC-ROC concepts).
# Run from /workspace/codes/axbench:
#   source ~/.bash_profile && bash /home/newuser/codes/SteeringLLMsCorruption/analysis/run_hard_study.sh 2>&1 | tee /tmp/hard_study.log

set -euo pipefail
AXBENCH=/workspace/codes/axbench
DEMO=$AXBENCH/axbench/demo/robust_compare_hard
YAML=$AXBENCH/axbench/demo/sweep/robust_compare_hard.yaml
cd "$AXBENCH"
export HF_HOME=/workspace/.cache/huggingface
export PYTHONUNBUFFERED=1
export OPENAI_USE_BATCH_API=0

# ---------------------------------------------------------------------------
# 1. Clean stale data (keep generate/ which was created by setup_hard_concepts.py)
# ---------------------------------------------------------------------------
echo "=== Cleaning stale data ==="
rm -rf "$DEMO/train" "$DEMO/inference" "$DEMO/evaluate" "$DEMO/activations" \
       "$DEMO/latent_inference_state.pkl_rank_0"
mkdir -p "$DEMO/train" "$DEMO/inference" "$DEMO/evaluate"
echo "  Done cleaning."

# ---------------------------------------------------------------------------
# 2. Train steering vectors (9 methods × 10 concepts)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Training steering vectors ==="
date
uv run python -m axbench.scripts.train \
    --config "$YAML" \
    --train_data_dir "$DEMO/generate" \
    --output_dir "$DEMO/train"
echo "  Train done."
date

# ---------------------------------------------------------------------------
# 3. Latent inference (get max_act per concept)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Latent inference ==="
date
uv run python -m axbench.scripts.inference \
    --config "$YAML" \
    --mode latent \
    --train_data_dir "$DEMO/train" \
    --output_dir "$DEMO/inference"
echo "  Latent inference done."
date

# ---------------------------------------------------------------------------
# 4. Steering inference (8 factors × 50 prompts)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Steering inference ==="
date
uv run python -m axbench.scripts.inference \
    --config "$YAML" \
    --mode steering \
    --train_data_dir "$DEMO/train" \
    --latent_dir "$DEMO/inference" \
    --output_dir "$DEMO/inference"
echo "  Steering inference done."
date

# ---------------------------------------------------------------------------
# 5. LM-judge evaluation (async direct API)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 5: LM-judge evaluation ==="
date
uv run -u python -m axbench.scripts.evaluate \
    --config "$YAML" \
    --inference_dir "$DEMO/inference" \
    --output_dir "$DEMO/evaluate" \
    --train_data_dir "$DEMO/train"
echo "  Evaluation done."
date

echo ""
echo "=== Pipeline complete! Results in $DEMO/evaluate ==="
