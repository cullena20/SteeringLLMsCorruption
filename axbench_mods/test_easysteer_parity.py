"""Parity tests: EasySteer vs AxBench for activations and steering generation.

Three test groups:
  1. DiffMean vector parity (numpy only, no GPU/vLLM needed) — runs always
  2. Activation parity (needs Gemma-2-2B-IT on GPU, no vLLM) — marked slow
  3. Generation parity (needs Gemma-2-2B-IT + custom vllm-steer fork) — marked vllm

Run all:
    cd /workspace/codes/axbench
    uv run python -m pytest tests/test_easysteer_parity.py -v

Skip slow model tests:
    uv run python -m pytest tests/test_easysteer_parity.py -v -m "not slow and not vllm"
"""

import sys
import math
from pathlib import Path

import numpy as np
import pytest
import torch

# EasySteer on path
_EASYSTEER = Path("/workspace/codes/EasySteer")
sys.path.insert(0, str(_EASYSTEER))

# AxBench on path
_AXBENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_AXBENCH))

GEMMA_MODEL = "google/gemma-2-2b-it"
LAYER = 20

try:
    from vllm import LLM
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

try:
    from vllm.hidden_states import deserialize_hidden_states  # custom fork signal
    _VLLM_STEER_AVAILABLE = True
except ImportError:
    _VLLM_STEER_AVAILABLE = False


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------------------
# 1. DiffMean vector parity (numpy only)
# ---------------------------------------------------------------------------

class TestDiffMeanParity:
    """EasySteer DiffMeanExtractor must produce same direction as AxBench DiffMean
    when given identical hidden states and using last-token pooling on both sides."""

    def _axbench_diffmean(self, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
        """AxBench DiffMean: mean(pos) - mean(neg), not normalised."""
        return pos.mean(axis=0) - neg.mean(axis=0)

    def _easysteer_diffmean(self, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
        from easysteer.steer.diffmean import DiffMeanExtractor
        n = len(pos)
        # EasySteer expects all_hidden_states[sample][layer][token]
        # We simulate one layer, one token per sample.
        all_hs = [[p[np.newaxis, :]] for p in pos] + [[n_[np.newaxis, :]] for n_ in neg]
        pos_idx = list(range(n))
        neg_idx = list(range(n, 2 * n))
        cv = DiffMeanExtractor.extract(
            all_hs, pos_idx, neg_idx, normalize=False, token_pos=-1
        )
        # Returns directions keyed by layer index (0 = only layer we passed)
        return cv.directions[0]

    def test_identical_direction_small(self):
        rng = np.random.default_rng(42)
        d = 64
        pos = rng.standard_normal((20, d)).astype(np.float32)
        neg = rng.standard_normal((20, d)).astype(np.float32)
        pos[:, 0] += 3.0  # clear signal

        axbench_vec = self._axbench_diffmean(pos, neg)
        easysteer_vec = self._easysteer_diffmean(pos, neg)

        sim = cos_sim(axbench_vec, easysteer_vec)
        assert sim > 0.9999, f"cosine similarity {sim:.6f} — vectors diverged"

    def test_identical_direction_realistic_dim(self):
        """Test at Gemma-2-2B hidden dim (2304)."""
        rng = np.random.default_rng(7)
        d = 2304
        n = 50
        pos = rng.standard_normal((n, d)).astype(np.float32)
        neg = rng.standard_normal((n, d)).astype(np.float32)
        pos[:, :10] += 2.0

        axbench_vec = self._axbench_diffmean(pos, neg)
        easysteer_vec = self._easysteer_diffmean(pos, neg)

        sim = cos_sim(axbench_vec, easysteer_vec)
        assert sim > 0.9999, f"cosine similarity {sim:.6f}"

    def test_normalised_unit_length(self):
        """When normalize=True, EasySteer vector should be unit norm."""
        from easysteer.steer.diffmean import DiffMeanExtractor
        rng = np.random.default_rng(0)
        d, n = 128, 10
        pos = rng.standard_normal((n, d)).astype(np.float32)
        neg = rng.standard_normal((n, d)).astype(np.float32)
        all_hs = [[p[np.newaxis, :]] for p in pos] + [[n_[np.newaxis, :]] for n_ in neg]
        cv = DiffMeanExtractor.extract(
            all_hs, list(range(n)), list(range(n, 2 * n)), normalize=True, token_pos=-1
        )
        vec = cv.directions[0]
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5, "normalised vector not unit norm"


# ---------------------------------------------------------------------------
# 2. Activation parity (requires Gemma-2-2B-IT on GPU, no vLLM)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
class TestActivationParity:
    """Hidden states at layer 20 from AxBench gather_residual_activations must
    match EasySteer capture for the same inputs.

    EasySteer's capture path uses the custom vLLM fork.  When vllm-steer is not
    installed we fall back to a pure HuggingFace forward-hook reference to at
    least verify the AxBench helper is correct, and skip the EasySteer side.
    """

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(GEMMA_MODEL)
        model = AutoModelForCausalLM.from_pretrained(GEMMA_MODEL, torch_dtype=torch.bfloat16).cuda().eval()
        return model, tok

    def _axbench_activations(self, model, tokenizer, texts):
        """Return [batch, seq, hidden] at LAYER using AxBench's helper."""
        from axbench.utils.model_utils import gather_residual_activations
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            acts = gather_residual_activations(model, LAYER, inputs)
        return acts.float().cpu()

    def _hf_hook_activations(self, model, tokenizer, texts):
        """Reference: plain HuggingFace forward hook at layer LAYER."""
        captured = {}
        def hook(_, __, output):
            captured["acts"] = output[0].detach().float().cpu()
        h = model.model.layers[LAYER].register_forward_hook(hook)
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            model(**inputs)
        h.remove()
        return captured["acts"]

    def test_axbench_matches_hf_hook(self, model_and_tokenizer):
        """AxBench's gather_residual_activations must match a plain HF hook."""
        model, tok = model_and_tokenizer
        texts = ["The quick brown fox", "Steering vectors are cool"]
        axbench_acts = self._axbench_activations(model, tok, texts)
        hf_acts = self._hf_hook_activations(model, tok, texts)
        assert torch.allclose(axbench_acts, hf_acts, atol=1e-4), \
            "AxBench gather_residual_activations diverges from plain HF hook"

    @pytest.mark.skipif(not _VLLM_STEER_AVAILABLE, reason="requires custom vllm-steer fork")
    def test_easysteer_matches_axbench(self, model_and_tokenizer):
        """EasySteer vLLM capture at layer LAYER must match AxBench activations."""
        from vllm import LLM
        from easysteer.hidden_states import HiddenStatesCaptureV1
        model, tok = model_and_tokenizer

        texts = ["The quick brown fox", "Steering vectors are cool"]
        axbench_acts = self._axbench_activations(model, tok, texts)

        # EasySteer capture (requires loaded vLLM model separately)
        llm = LLM(model=GEMMA_MODEL, dtype="bfloat16", tensor_parallel_size=1)
        capture = HiddenStatesCaptureV1()
        hs, _ = capture.get_all_hidden_states(llm, texts, split_by_samples=True)

        for i, text in enumerate(texts):
            # hs[i][LAYER] shape: (seq_len, hidden_size)
            es_acts = hs[i][LAYER].float()
            ax_acts = axbench_acts[i, :es_acts.shape[0], :]
            assert torch.allclose(ax_acts, es_acts, atol=1e-3), \
                f"Sample {i}: EasySteer vs AxBench activation mismatch (max diff " \
                f"{(ax_acts - es_acts).abs().max():.4f})"


# ---------------------------------------------------------------------------
# 3. Generation parity at temperature=0 (requires custom vllm-steer fork)
# ---------------------------------------------------------------------------

@pytest.mark.vllm
@pytest.mark.slow
@pytest.mark.skipif(not _VLLM_STEER_AVAILABLE, reason="requires custom vllm-steer fork")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
class TestGenerationParity:
    """With temperature=0, output tokens from EasySteer vLLM steering must be
    identical to tokens from AxBench's pyreft AdditionIntervention path."""

    PROMPTS = [
        "Tell me about Paris.",
        "What is machine learning?",
    ]
    FACTOR = 5.0
    MAX_NEW_TOKENS = 20

    @pytest.fixture(scope="class")
    def steering_vector(self):
        """Load a pre-trained DiffMean weight from the last tau study run."""
        import torch
        weight_path = Path(
            "/workspace/codes/axbench/axbench/demo/robust_compare_tau/train/DiffMean_weight.pt"
        )
        if not weight_path.exists():
            pytest.skip("trained weight not found — run training first")
        return torch.load(weight_path).float()  # shape: (1, hidden_size)

    def _pyreft_generate(self, steering_vector):
        """Generate tokens via AxBench pyreft AdditionIntervention."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from axbench.models.interventions import AdditionIntervention
        from pyreft import IntervenableConfig, IntervenableModel
        import torch

        tok = AutoTokenizer.from_pretrained(GEMMA_MODEL)
        model = AutoModelForCausalLM.from_pretrained(GEMMA_MODEL, torch_dtype=torch.bfloat16).cuda().eval()

        ax = AdditionIntervention(embed_dim=steering_vector.shape[-1], low_rank_dimension=1)
        ax.proj.weight.data = steering_vector.to(model.dtype).to("cuda")

        ax_config = IntervenableConfig(representations=[{
            "layer": LAYER,
            "component": f"model.layers[{LAYER}].output",
            "low_rank_dimension": 1,
            "intervention": ax,
        }])
        ax_model = IntervenableModel(ax_config, model)

        inputs = tok(self.PROMPTS, return_tensors="pt", padding=True, truncation=True).to("cuda")
        mag = torch.tensor([self.FACTOR] * len(self.PROMPTS)).to("cuda")
        idx = torch.zeros(len(self.PROMPTS), dtype=torch.long).to("cuda")
        max_acts = torch.ones(len(self.PROMPTS)).to("cuda")
        prefix_length = 1

        with torch.no_grad():
            _, gens = ax_model.generate(
                inputs, unit_locations=None, intervene_on_prompt=True,
                subspaces=[{"idx": idx, "mag": mag, "max_act": max_acts,
                            "prefix_length": prefix_length}],
                max_new_tokens=self.MAX_NEW_TOKENS, do_sample=False, temperature=None,
            )
        input_lens = [len(ids) for ids in inputs.input_ids]
        return [tok.decode(g[l:], skip_special_tokens=True) for g, l in zip(gens, input_lens)]

    def _easysteer_generate(self, steering_vector):
        """Generate tokens via EasySteer vLLM with same steering vector."""
        from vllm import LLM, SamplingParams
        from easysteer.steer.utils import StatisticalControlVector

        cv = StatisticalControlVector(
            model_type="gemma2",
            method="diffmean",
            directions={LAYER: steering_vector.squeeze(0).numpy()},
        )

        llm = LLM(model=GEMMA_MODEL, dtype="bfloat16", tensor_parallel_size=1)
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=self.MAX_NEW_TOKENS,
        )

        # EasySteer SteerVectorRequest applies vector at generation time
        from easysteer.steer.unified_interface import SteerVectorRequest
        outputs = llm.generate(
            self.PROMPTS, sampling_params,
            steer_vector_request=SteerVectorRequest(cv, factor=self.FACTOR),
        )
        return [o.outputs[0].text for o in outputs]

    def test_generation_token_exact_match(self, steering_vector):
        pyreft_outputs = self._pyreft_generate(steering_vector)
        easysteer_outputs = self._easysteer_generate(steering_vector)

        for i, (py, es) in enumerate(zip(pyreft_outputs, easysteer_outputs)):
            assert py == es, (
                f"Prompt {i}: generation mismatch\n"
                f"  pyreft:     {repr(py)}\n"
                f"  easysteer:  {repr(es)}"
            )
