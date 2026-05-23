"""AUC-ROC audit for concept500 concepts at layer 20.

Runs forward passes on all concept500 train examples, computes DiffMean
AUC-ROC for each of the 500 concepts, then selects the median 20 (ranks 240-260).

Outputs:
  - axbench/demo/robust_compare_tau/generate/train_data.parquet   (replaces old generate data)
  - axbench/demo/robust_compare_tau/generate/metadata.jsonl
  - axbench/demo/robust_compare_tau/generate/generate_state.pkl   (synthetic, concept_id=20)

Usage:
  cd /workspace/codes/axbench
  uv run python /workspace/codes/SteeringLLMsCorruption/analysis/auc_roc_audit.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONCEPT500_PARQUET = "/tmp/concept500/2b/l20/train/data.parquet"
FULL_METADATA_JSONL = "/workspace/codes/axbench/axbench/concept500/prod_2b_l20_v1/generate/metadata.jsonl"
OUT_DIR = Path("/workspace/codes/axbench/axbench/demo/robust_compare_tau/generate")

MODEL_NAME = "google/gemma-2-2b-it"
TARGET_LAYER = 20
BATCH_SIZE = 32
MAX_LENGTH = 256  # truncate long texts
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Median 20: ranks 240-259 (0-indexed), i.e. the middle of 500 sorted concepts
MEDIAN_RANK_START = 240
MEDIAN_RANK_END   = 260  # exclusive → 20 concepts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_layer_activations(model, tokenizer, texts, layer_idx, batch_size, device):
    """Return mean-pooled residual-stream activations for each text. Shape: (N, d_model)."""
    # Build dataset (tokenize all at once to avoid repeated padding calls)
    print(f"  Tokenizing {len(texts)} texts...")
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = encodings["input_ids"]      # (N, L)
    attention_mask = encodings["attention_mask"]  # (N, L)

    all_acts = []
    n = len(texts)

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ids  = input_ids[start:end].to(device)
            mask = attention_mask[start:end].to(device)

            # Use output_hidden_states to get residual stream at target layer
            out = model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
            )
            # hidden_states: tuple of (n_layers+1) tensors, each (B, L, d)
            hs = out.hidden_states[layer_idx + 1]  # +1 because idx 0 is embedding

            # Mean-pool over non-padding tokens
            mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
            pooled = (hs * mask_f).sum(dim=1) / mask_f.sum(dim=1)  # (B, d)
            all_acts.append(pooled.cpu().float().numpy())

            if (start // batch_size) % 20 == 0:
                pct = 100 * end / n
                print(f"    [{pct:5.1f}%] batch {start//batch_size+1}/{(n+batch_size-1)//batch_size}", flush=True)

    return np.concatenate(all_acts, axis=0)  # (N, d)


# ---------------------------------------------------------------------------
# AUC-ROC per concept
# ---------------------------------------------------------------------------

def compute_auc_roc_per_concept(df, acts, concept_ids_sorted):
    """
    df: full concept500 dataframe (index aligns with acts rows)
    acts: (N, d) activations aligned to df rows
    concept_ids_sorted: list of concept_ids to evaluate

    Returns dict: concept_id -> auc_roc
    """
    neg_mask   = df["concept_id"].values == -1
    neg_acts   = acts[neg_mask]     # (216, d)

    results = {}
    for cid in concept_ids_sorted:
        pos_mask = df["concept_id"].values == cid
        pos_acts = acts[pos_mask]  # (72, d)

        if pos_acts.shape[0] == 0:
            results[cid] = 0.5
            continue

        # DiffMean direction
        direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            results[cid] = 0.5
            continue
        direction = direction / norm

        # Project all examples onto direction
        all_acts = np.concatenate([pos_acts, neg_acts], axis=0)
        labels   = np.array([1] * len(pos_acts) + [0] * len(neg_acts))
        scores   = all_acts @ direction

        auc = roc_auc_score(labels, scores)
        results[cid] = auc

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    # Load concept500 data
    print("Loading concept500 parquet...")
    df = pd.read_parquet(CONCEPT500_PARQUET)
    print(f"  {len(df)} rows, {df['concept_id'].nunique()-1} positive concepts + 1 neg group")

    # Load model
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    hf_home = os.environ.get("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("HF_HOME", hf_home)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=DEVICE,
    )
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Extract activations for all rows
    print(f"Extracting activations (layer {TARGET_LAYER}, batch {BATCH_SIZE})...")
    texts = df["input"].tolist()
    acts  = extract_layer_activations(model, tokenizer, texts, TARGET_LAYER, BATCH_SIZE, DEVICE)
    print(f"  Acts shape: {acts.shape}, elapsed: {time.time()-t0:.1f}s")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Compute AUC-ROC for all 500 concepts
    print("Computing AUC-ROC per concept...")
    all_concept_ids = sorted(df[df["concept_id"] >= 0]["concept_id"].unique().tolist())
    auc_map = compute_auc_roc_per_concept(df, acts, all_concept_ids)

    # Sort by AUC-ROC
    sorted_by_auc = sorted(auc_map.items(), key=lambda x: x[1])
    print(f"  AUC-ROC range: {sorted_by_auc[0][1]:.4f} – {sorted_by_auc[-1][1]:.4f}")

    # Select median 20
    median_concepts = sorted_by_auc[MEDIAN_RANK_START:MEDIAN_RANK_END]
    median_ids = [cid for cid, _ in median_concepts]
    print(f"\nSelected median 20 concepts (ranks {MEDIAN_RANK_START}-{MEDIAN_RANK_END-1}):")
    for rank, (cid, auc) in enumerate(median_concepts, start=MEDIAN_RANK_START):
        print(f"  rank {rank:3d}: concept_id={cid:3d}  AUC={auc:.4f}")

    # Load full metadata (500 concepts) to get neuronpedia URLs
    print(f"\nLoading full metadata from {FULL_METADATA_JSONL}...")
    full_meta = {}
    with open(FULL_METADATA_JSONL) as f:
        for line in f:
            obj = json.loads(line)
            full_meta[obj["concept_id"]] = obj

    # Build new train_data.parquet: positives for median 20 + all negatives
    neg_df = df[df["concept_id"] == -1].copy()
    pos_df = df[df["concept_id"].isin(set(median_ids))].copy()

    # Re-map concept_ids to 0-19 (contiguous)
    id_remap = {old: new for new, (old, _) in enumerate(median_concepts)}
    pos_df["concept_id"] = pos_df["concept_id"].map(id_remap)
    train_df = pd.concat([neg_df, pos_df], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = OUT_DIR / "train_data.parquet"
    train_df.to_parquet(out_parquet, index=False)
    print(f"Wrote {len(train_df)} rows to {out_parquet}")

    # Build metadata.jsonl for the 20 selected concepts
    out_meta = OUT_DIR / "metadata.jsonl"
    with open(out_meta, "w") as f:
        for new_id, (old_id, _auc) in enumerate(median_concepts):
            orig = full_meta[old_id]
            entry = {
                "concept_id": new_id,
                "concept": orig["concept"],
                "ref": orig["ref"],
                "concept_genres_map": orig.get("concept_genres_map", {orig["concept"]: ["text"]}),
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(median_concepts)} entries to {out_meta}")

    # Synthetic generate_state.pkl so run_tau_pipeline.sh sees concept_id=20 immediately
    import pickle
    gen_state = {"concept_id": 20, "source": "auc_roc_audit", "median_ids": median_ids}
    with open(OUT_DIR / "generate_state.pkl", "wb") as f:
        pickle.dump(gen_state, f)
    print(f"Wrote synthetic generate_state.pkl")

    # Save AUC-ROC scores for reference
    auc_out = OUT_DIR.parent / "auc_roc_all500.json"
    with open(auc_out, "w") as f:
        json.dump({
            "sorted_by_auc": [(int(cid), float(auc)) for cid, auc in sorted_by_auc],
            "median_ids": [int(cid) for cid in median_ids],
        }, f, indent=2)
    print(f"Wrote full AUC-ROC scores to {auc_out}")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
