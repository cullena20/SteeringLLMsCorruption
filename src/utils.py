import torch
import gc
import numpy as np
import random
from datasets import load_dataset

def clear_memory():
    """Clears GPU memory by emptying the cache and running garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()

def set_global_seed(seed=42):
    random.seed(seed)               # Python random
    np.random.seed(seed)            # NumPy RNG
    torch.manual_seed(seed)         # PyTorch (CPU)
    torch.cuda.manual_seed(seed)    # PyTorch (current GPU)
    torch.cuda.manual_seed_all(seed)  # all GPUs, if applicable
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# for experiment output
def dicts_equal(a, b):
    if a.keys() != b.keys():
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            if not torch.equal(va, vb):
                return False
        else:
            if va != vb:
                return False
    return True

def to_set(x):
    if isinstance(x, list):
        return set(x)
    if torch.is_tensor(x):
        return set(x.detach().cpu().tolist())
    if isinstance(x, np.ndarray):
        return set(x.tolist())
    else:
        return set(x)

def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return 0.0
    
    # Compute cosine similarity
    cos = np.dot(a, b) / (na * nb)
    
    # Clip for numerical stability
    return float(np.clip(cos, -1.0, 1.0))

def outlier_pruning_stats(
    detected_outlier_indices,
    true_outliers,
    total_n,
    steering_name,
):
    true_outliers = torch.as_tensor(true_outliers).long()
    true_outlier_set = set(true_outliers.tolist())

    num_outliers = len(true_outlier_set)
    num_inliers = total_n - num_outliers

    def compute_pct(detected):
        detected = torch.as_tensor(detected).long()
        detected_set = set(detected.tolist())

        pruned_outliers = len(detected_set & true_outlier_set)
        pruned_inliers = len(detected_set - true_outlier_set)

        return (
            pruned_outliers / max(num_outliers, 1),
            pruned_inliers / max(num_inliers, 1),
        )

    # Case 1: diff estimator → (pos_outliers, neg_outliers)
    if "diff" in steering_name:
        pos_idx, neg_idx = detected_outlier_indices
        pos_stats = compute_pct(pos_idx)
        neg_stats = compute_pct(neg_idx)

        return {
            "detected_outlier_indices": detected_outlier_indices,
            "pct_outliers_pruned": 0.5 * (pos_stats[0] + neg_stats[0]),
            "pct_inliers_pruned": 0.5 * (pos_stats[1] + neg_stats[1]),
        }

    # Case 2: standard estimator
    else:
        out_p, in_p = compute_pct(detected_outlier_indices)
        return {
            "detected_outlier_indices": detected_outlier_indices,
            "pct_outliers_pruned": out_p,
            "pct_inliers_pruned": in_p,
        }

def format_steering_vecs_for_eval(
    steering_vecs, 
    layer=12,
    token_pos="answer_token"
):
    """
    Formats steering vectors for evaluation.

    Args:
        steering_vecs (dict): {behavior: {estimator_name: steering_vec}}
        layer (int): layer index to assign in output
        token_pos (str): token position key in output

    Returns:
        dict: {behavior: {layer: {token_pos: {estimator_name: steering_vec}}}}
    """
    formatted = {}

    for behavior, estimators_dict in steering_vecs.items():
        formatted.setdefault(behavior, {})
        formatted[behavior].setdefault(layer, {})
        formatted[behavior][layer].setdefault(token_pos, {})

        for estimator_name, vec in estimators_dict.items():
            formatted[behavior][layer][token_pos][estimator_name] = vec

    return formatted

def array_to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()
    else:
        raise ValueError("Input must be a numpy array or a torch tensor.")

def format_mmlu_prompt(ex):
    letters = ["A", "B", "C", "D"]
    choices = "\n".join(
        f"{letters[i]}. {ex['choices'][i]}"
        for i in range(len(ex["choices"]))
    )

    return (
        f"Question: {ex['question']}\n"
        f"{choices}\n"
        f"Answer:"
    )

# DATASET HELPER FOR LARGE DATASETS

def large_dataset_to_behavior_dict(dataset, invert_behavior=False):
    """
    Convert a HuggingFace Dataset with keys:
        - prompt
        - response  (always "(A)" or "(B)")
        - label     (1 = response matches behavior, 0 = response does not)

    into a dictionary with keys:
        - question
        - answer_matching_behavior
        - answer_not_matching_behavior
    """

    out = {
        "questions": [],
        "answer_matching_behavior": [],
        "answer_not_matching_behavior": [],
        "indices": [],
    }

    for i, ex in enumerate(dataset):
        question = ex["prompt"]

        # Normalize "(A)" -> "A", "(B)" -> "B"
        response_letter = ex["response"].strip()
        assert response_letter in {"(A)", "(B)"}

        other_letter = "(B)" if response_letter == "(A)" else "(A)"

        if (ex["label"] == 1) != invert_behavior:
            matching = response_letter
            not_matching = other_letter
        else:
            matching = other_letter
            not_matching = response_letter

        out["questions"].append(question)
        out["answer_matching_behavior"].append(matching)
        out["answer_not_matching_behavior"].append(not_matching)
        out["indices"].append(i)

    return out

class BehaviorSplit:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data


def large_dataset_generator(test_size=0.1, seed=42):
    """
    Returns an object with:
        - .train_data
        - .test_data

    Each maps:
        behavior_name -> formatted behavior dict
    """

    # Load raw datasets
    large_power = load_dataset(
        "shiv96/convsersations_power_seeking_large", split="train"
    )
    large_wealth = load_dataset(
        "shiv96/convsersations_wealth_seeking_large", split="train"
    )
    large_corr = load_dataset(
        "shiv96/convsersations_corrigible_more_large", split="train"
    )

    datasets_by_behavior = {
        "uncorrigible-neutral-HHH": (large_corr, True),
        "power-seeking-inclination": (large_power, False),
        "wealth-seeking-inclination": (large_wealth, False),
    }

    train_data = {}
    test_data = {}

    for behavior, (dataset, invert_behavior) in datasets_by_behavior.items():
        split = dataset.train_test_split(
            test_size=test_size,
            shuffle=True,
            seed=seed,
        )

        train_data[behavior] = large_dataset_to_behavior_dict(
            split["train"],
            invert_behavior=invert_behavior,
        )

        test_data[behavior] = large_dataset_to_behavior_dict(
            split["test"],
            invert_behavior=invert_behavior,
        )

    return BehaviorSplit(train_data=train_data, test_data=test_data)

# formerly of steering_utils.py 

# hook functions are expected to take in activation and hook
# they output the new activation where they are applied
def single_direction_hook(
    activation: torch.Tensor,
    hook, # needed for transformer_lens interface
    steering_dir: dict[str, torch.Tensor],
    target_class: str,
    alpha: float = 1,
    normalize = False
) -> torch.Tensor:
    """
    Add a scaled steering direction to the activation.
    """
    if target_class not in steering_dir:
        raise ValueError(
            f"No steering direction found for class '{target_class}'.")

    direction = steering_dir[target_class]

    if isinstance(direction, np.ndarray):
        direction = torch.as_tensor(direction, device=activation.device, dtype=activation.dtype)
    else:
        direction = direction.to(device=activation.device, dtype=activation.dtype)
    
    if normalize:
        norm = direction.norm(p=2)
        if norm == 0:
            return activation
        direction = direction / norm

    return activation + alpha * direction

def normalize_steering_dir(steering_dir: dict[str, torch.Tensor | np.ndarray]):
    normalized_steering_dir = {}
    for name, vec in steering_dir.items():
        if name == "no_steer" or vec is None:
            normalized_steering_dir[name] = vec
            continue
        # Convert numpy arrays to torch tensors for consistent handling
        if isinstance(vec, np.ndarray):
            vec = torch.from_numpy(vec)

        # Ensure it's a float tensor (to avoid integer division)
        vec = vec.float()

        # Normalize using L2 norm
        norm = torch.norm(vec, p=2)
        if norm > 0:
            normalized_vec = vec / norm
        else:
            normalized_vec = vec  # avoid divide-by-zero case

        # Convert back to numpy if original was numpy
        if isinstance(steering_dir[name], np.ndarray):
            normalized_vec = normalized_vec.numpy()

        normalized_steering_dir[name] = normalized_vec

    return normalized_steering_dir

# helper for corruption scheme
def orthogonal_unit_vector(v, seed=None, base_vector=None):
    """
    Generate a unit vector orthogonal to v.

    Args:
        v (torch.Tensor): The reference vector (shape [d]).
        seed (int, optional): Random seed for reproducibility.
        base_vector (torch.Tensor, optional): If provided, use this as the starting vector instead of random.

    Returns:
        torch.Tensor: A unit vector orthogonal to v.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if base_vector is None:
        random_vec = torch.randn_like(v)
    else:
        random_vec = base_vector.clone()

    # Normalize v first for numerical stability
    v = v / v.norm()

    # Compute projection of random_vec onto v
    projection = (random_vec @ v) * v
    ortho = random_vec - projection

    return ortho / ortho.norm()