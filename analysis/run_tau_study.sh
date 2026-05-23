#!/usr/bin/env bash
# Full tau study pipeline after AUC-ROC audit completes.
# Run from /workspace/codes/axbench:
#   bash /workspace/codes/SteeringLLMsCorruption/analysis/run_tau_study.sh 2>&1 | tee /tmp/tau_study.log

set -euo pipefail
AXBENCH=/workspace/codes/axbench
DEMO=$AXBENCH/axbench/demo/robust_compare_tau
YAML=$DEMO/sweep/robust_compare_tau.yaml
cd "$AXBENCH"
export HF_HOME=/workspace/.cache/huggingface
# HF_TOKEN must be set in environment or ~/.bash_profile before running

# ---------------------------------------------------------------------------
# 1. Clean stale data from previous runs
# ---------------------------------------------------------------------------
echo "=== Cleaning stale data ==="
# Keep train/train_data.parquet and train/metadata.jsonl (written by audit)
# Delete everything else

rm -f "$DEMO/train/"*.pt "$DEMO/train/"*.pkl "$DEMO/train/rank_0_metadata.jsonl" "$DEMO/train/config.json"
rm -f "$DEMO/latent_inference_state.pkl_rank_0"
rm -f "$DEMO/inference/latent_data.parquet" "$DEMO/inference/steering_data.parquet" \
      "$DEMO/inference/steering_inference_state.pkl_rank_0"
rm -f "$DEMO/evaluate/"*.pkl "$DEMO/evaluate/"*.png \
      "$DEMO/evaluate/temp_all_results.pkl" "$DEMO/evaluate/temp_eval_dfs.pkl"
rm -rf "$DEMO/activations/activations"
rm -f "$DEMO/generate/generate_state.pkl" "$DEMO/generate/train_data.parquet" \
      "$DEMO/generate/metadata.jsonl"

echo "  Done cleaning."

# ---------------------------------------------------------------------------
# 2. Train steering vectors (9 methods × 20 concepts)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Training steering vectors ==="
date
uv run python -m axbench.scripts.train \
    --config "$YAML" \
    --train_data_dir "$DEMO/train" \
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
# 5. LM-judge evaluation (deferred batch API)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 5: LM-judge evaluation ==="
date
uv run python -m axbench.scripts.evaluate \
    --config "$YAML" \
    --inference_dir "$DEMO/inference" \
    --output_dir "$DEMO/evaluate" \
    --train_data_dir "$DEMO/train"
echo "  Evaluation done."
date

echo ""
echo "=== Pipeline complete! Results in $DEMO/evaluate ==="
