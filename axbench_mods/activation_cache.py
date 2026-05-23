"""Per-example mean-pooled activation cache for AxBench training.

Stores layer activations to HuggingFace alongside the training parquet so that
subsequent training runs skip all Gemma forward passes and do pure numpy arithmetic.

Storage layout on HF (repo PhillipsLab/axbench-steering-data):
  activations/gemma2_2b_layer{L}/concept_{id:03d}_{slug}_activations.npz

Each .npz has two arrays:
  pos  — (n_pos, hidden_size) float32, mean-pooled per positive example
  neg  — (n_neg, hidden_size) float32, mean-pooled per negative example

Local cache mirrors the same path under {dump_dir}/activations/.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def _slug(concept: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", concept.lower()).strip("_")[:60]


def _npz_name(concept_id: int, concept: str, layer: int) -> str:
    return f"concept_{concept_id:03d}_{_slug(concept)}_layer{layer}_activations.npz"


def _hf_path(layer: int, filename: str) -> str:
    return f"activations/gemma2_2b_layer{layer}/{filename}"


# ── save / load ───────────────────────────────────────────────────────────────

def save_activations(
    dump_dir: Path,
    concept_id: int,
    concept: str,
    layer: int,
    pos: np.ndarray,
    neg: np.ndarray,
) -> Path:
    """Save pos/neg activation arrays to local cache and return the path."""
    cache_dir = Path(dump_dir) / "activations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = _npz_name(concept_id, concept, layer)
    out = cache_dir / fname
    np.savez_compressed(out, pos=pos, neg=neg)
    logger.warning(f"Saved activations: {out} (pos={pos.shape}, neg={neg.shape})")
    return out


def load_activations(
    dump_dir: Path,
    concept_id: int,
    concept: str,
    layer: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (pos, neg) arrays if cached locally, else None."""
    fname = _npz_name(concept_id, concept, layer)
    path = Path(dump_dir) / "activations" / fname
    if not path.exists():
        return None
    data = np.load(path)
    logger.warning(f"Loaded activations from cache: {path}")
    return data["pos"], data["neg"]


# ── HuggingFace push / pull ───────────────────────────────────────────────────

def push_activations_to_hf(
    dump_dir: Path,
    concept_id: int,
    concept: str,
    layer: int,
    repo: str = "PhillipsLab/axbench-steering-data",
    token: Optional[str] = None,
) -> None:
    """Upload the local .npz file to HuggingFace."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed — skipping HF push")
        return

    fname = _npz_name(concept_id, concept, layer)
    local = Path(dump_dir) / "activations" / fname
    if not local.exists():
        logger.warning(f"push_activations_to_hf: {local} not found, skipping")
        return

    api = HfApi(token=token)
    hf_path = _hf_path(layer, fname)
    try:
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=hf_path,
            repo_id=repo,
            repo_type="dataset",
        )
        logger.warning(f"Pushed activations to HF: {repo}/{hf_path}")
    except Exception as e:
        logger.warning(f"push_activations_to_hf failed ({e}); local copy kept at {local}")


def pull_activations_from_hf(
    dump_dir: Path,
    concept_id: int,
    concept: str,
    layer: int,
    repo: str = "PhillipsLab/axbench-steering-data",
    token: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Download activations from HF into local cache and return arrays."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    fname = _npz_name(concept_id, concept, layer)
    hf_path = _hf_path(layer, fname)
    cache_dir = Path(dump_dir) / "activations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / fname

    try:
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=hf_path,
            repo_type="dataset",
            token=token,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may nest into subdirs — move to flat cache_dir
        dl_path = Path(downloaded)
        if dl_path != local and dl_path.exists():
            dl_path.rename(local)
        data = np.load(local)
        logger.warning(f"Pulled activations from HF: {repo}/{hf_path}")
        return data["pos"], data["neg"]
    except Exception as e:
        logger.warning(f"Could not pull activations from HF ({e})")
        return None


# ── main entry point used by train.py ─────────────────────────────────────────

def get_or_collect_activations(
    model,
    dataloader,
    layer: int,
    device,
    prefix_length: int,
    dump_dir: Path,
    concept_id: int,
    concept: str,
    hf_repo: str = "PhillipsLab/axbench-steering-data",
    hf_token: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pos, neg) per-example mean-pooled activations.

    Order of preference:
      1. Local cache (dump_dir/activations/)
      2. HuggingFace (pulled to local cache)
      3. Fresh forward pass through model (saved to local cache + pushed to HF)
    """
    from ..utils.model_utils import gather_residual_activations

    # 1. local cache
    cached = load_activations(dump_dir, concept_id, concept, layer)
    if cached is not None:
        return cached

    # 2. HuggingFace
    if hf_token or hf_repo:
        cached = pull_activations_from_hf(dump_dir, concept_id, concept, layer, hf_repo, hf_token)
        if cached is not None:
            return cached

    # 3. forward pass
    logger.warning(f"Collecting activations for concept {concept_id} via forward pass…")
    pos_vecs, neg_vecs = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            acts = gather_residual_activations(
                model, layer,
                {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
            ).detach()                                      # (B, seq, H)
            acts_nb = acts[:, prefix_length:, :]            # drop BOS tokens
            mask_nb = inputs["attention_mask"][:, prefix_length:].bool()  # (B, seq-prefix)
            labels = inputs["labels"]                        # (B,)
            for i in range(acts_nb.shape[0]):
                m = mask_nb[i]
                if m.sum() == 0:
                    continue
                vec = acts_nb[i][m].mean(dim=0).cpu().float().numpy()
                if labels[i].item() == 1:
                    pos_vecs.append(vec)
                else:
                    neg_vecs.append(vec)

    pos_np = np.stack(pos_vecs).astype(np.float32)
    neg_np = np.stack(neg_vecs).astype(np.float32)

    # save + push
    save_activations(dump_dir, concept_id, concept, layer, pos_np, neg_np)
    if hf_token:
        push_activations_to_hf(dump_dir, concept_id, concept, layer, hf_repo, hf_token)

    return pos_np, neg_np
