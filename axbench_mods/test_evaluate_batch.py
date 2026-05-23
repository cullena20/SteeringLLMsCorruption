"""Smoke tests for evaluate.py deferred-batch + cache changes.

No GPU, no real OpenAI calls, no Gemma needed.

Run:
    cd /workspace/codes/axbench
    uv run python -m pytest tests/test_evaluate_batch.py -v
"""

import asyncio
import pickle
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axbench.models.language_models import LanguageModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lm(tmp_path, use_batch_api=True):
    """Return a LanguageModel with dummy clients and temp cache dir."""
    client = MagicMock()
    with patch("openai.OpenAI"):  # prevent real OpenAI() construction
        lm = LanguageModel(
            "gpt-4o-mini",
            client,
            use_cache=True,
            cache_level="prompt",
            cache_tag="test",
            master_data_dir=str(tmp_path),
            use_batch_api=use_batch_api,
            hf_token=None,
            hf_cache_repo=None,
        )
    # Replace the real sync client with a controllable mock
    lm._sync_client = MagicMock()
    return lm


# ---------------------------------------------------------------------------
# 1. Deferred batch queuing
# ---------------------------------------------------------------------------

def _mock_openai_batch(lm, prompts, responses=None):
    """Wire up lm._sync_client mock to return fake batch responses for prompts."""
    import json
    if responses is None:
        responses = ["Rating: [[1]]"] * len(prompts)
    jsonl = "\n".join(
        json.dumps({"custom_id": str(i),
                    "response": {"body": {"choices": [{"message": {"content": r}}]}}})
        for i, r in enumerate(responses)
    )
    mock_file = MagicMock(); mock_file.id = "file-123"
    mock_batch = MagicMock(); mock_batch.id = "batch-1"; mock_batch.status = "completed"
    mock_batch.output_file_id = "out-123"
    lm._sync_client.files.create.return_value = mock_file
    lm._sync_client.batches.create.return_value = mock_batch
    lm._sync_client.batches.retrieve.return_value = mock_batch
    lm._sync_client.files.content.return_value.text = jsonl


class TestDeferredBatch:
    def test_prompts_queued_not_submitted(self, tmp_path):
        """start_deferred_batch → chat_completions queues, no OpenAI call before flush."""
        lm = _make_lm(tmp_path)
        lm.start_deferred_batch()

        async def go():
            await lm.chat_completions("api_a", ["prompt1", "prompt2"])
            await lm.chat_completions("api_b", ["prompt3"])

        asyncio.run(go())

        # Prompts must be in the deferred list, not submitted yet
        assert lm._deferred_batch is not None
        all_prompts = [p for _, ps in lm._deferred_batch for p in ps]
        assert len(all_prompts) == 3
        lm._sync_client.batches.create.assert_not_called()

    def test_flush_submits_all_as_one_call(self, tmp_path):
        """flush_deferred_batch submits exactly one OpenAI batch job."""
        lm = _make_lm(tmp_path)
        lm.start_deferred_batch()

        async def go():
            await lm.chat_completions("api_a", ["p1", "p2"])
            await lm.chat_completions("api_b", ["p3", "p4", "p5"])

        asyncio.run(go())
        _mock_openai_batch(lm, ["p1", "p2", "p3", "p4", "p5"])
        lm.flush_deferred_batch()

        assert lm._sync_client.batches.create.call_count == 1, \
            "flush must submit exactly one batch"
        assert lm._deferred_batch is None

    def test_cache_populated_after_flush(self, tmp_path):
        """After flush, all prompts are in cache; second run makes zero API calls."""
        lm = _make_lm(tmp_path)
        lm.start_deferred_batch()

        async def go():
            await lm.chat_completions("api_a", ["prompt_x", "prompt_y"])

        asyncio.run(go())
        _mock_openai_batch(lm, ["prompt_x", "prompt_y"])
        lm.flush_deferred_batch()

        assert lm._sync_client.batches.create.call_count == 1
        assert len(lm.cache_in_mem) == 2

        # Reset batch-create mock; second run must not trigger another batch
        lm._sync_client.batches.create.reset_mock()
        lm.api_count = {}

        async def go2():
            return await lm.chat_completions("api_a", ["prompt_x", "prompt_y"])

        asyncio.run(go2())
        lm._sync_client.batches.create.assert_not_called()


# ---------------------------------------------------------------------------
# 2. save_cache / load from disk
# ---------------------------------------------------------------------------

class TestCachePersistence:
    def test_save_and_reload(self, tmp_path):
        """save_cache writes pkl; a new instance loads it from disk."""
        lm = _make_lm(tmp_path)
        lm.cache_in_mem = {"prompt_a": "Rating: [[2]]", "prompt_b": "Rating: [[0]]"}
        lm.save_cache()

        assert lm.cache_file.exists()

        lm2 = _make_lm(tmp_path)
        assert lm2.cache_in_mem == lm.cache_in_mem

    def test_hf_pull_skipped_when_no_token(self, tmp_path):
        """No HF token → _pull_cache_from_hf is a no-op, cache stays empty."""
        lm = _make_lm(tmp_path)
        lm.cache_in_mem = {}
        lm._pull_cache_from_hf(hf_repo="some/repo", hf_token=None)
        assert lm.cache_in_mem == {}


# ---------------------------------------------------------------------------
# 3. eval_steering_single_task accepts pre-built LanguageModel
# ---------------------------------------------------------------------------

class TestPrebuiltInstance:
    def test_accepts_lm_instance(self, tmp_path):
        """eval_steering_single_task must use a pre-built LanguageModel directly."""
        from axbench.scripts.evaluate import eval_steering_single_task
        import pandas as pd

        lm = _make_lm(tmp_path)
        # Pre-populate cache so the evaluator returns without calling the API
        # LMJudgeEvaluator calls chat_completions for concept/instruction/fluency
        # We stub compute_metrics to avoid needing real data.
        used_instance = []

        class FakeEvaluator:
            def __init__(self, model_name, **kwargs):
                used_instance.append(kwargs.get("lm_model"))
            def __str__(self):
                return "LMJudgeEvaluator"
            def compute_metrics(self, data):
                return {"lm_judge_rating": [1.0], "factor": [0.5],
                        "relevance_concept_ratings": [1.0],
                        "relevance_instruction_ratings": [1.0],
                        "fluency_ratings": [1.0],
                        "raw_relevance_concept_ratings": [1.0],
                        "raw_relevance_instruction_ratings": [1.0],
                        "raw_fluency_ratings": [1.0],
                        "raw_aggregated_ratings": [1.0],
                        "relevance_concept_completions": ["ok"],
                        "relevance_instruction_completions": ["ok"],
                        "fluency_completions": ["ok"]}

        import axbench
        original = getattr(axbench, "LMJudgeEvaluator", None)
        axbench.LMJudgeEvaluator = FakeEvaluator

        dummy_df = pd.DataFrame({
            "concept_id": [0],
            "input_concept": ["test concept"],
            "input_id": [0],
            "original_prompt": ["hello"],
            "steered_input": ["hello steered"],
            "factor": [0.5],
            "DiffMean_steered_generation": ["some output"],
            "dataset_name": ["AlpacaEval"],
        })

        task = (0, dummy_df, "LMJudgeEvaluator", "DiffMean",
                str(tmp_path), lm, None, {}, "concept")

        try:
            result = eval_steering_single_task(task)
            concept_id, evaluator_str, model_str, eval_result, report, cache_out, _ = result
            assert concept_id == 0
            # The lm instance passed in must be the one the evaluator received
            assert used_instance[0] is lm, "eval_steering_single_task must pass pre-built lm to evaluator"
        finally:
            if original is not None:
                axbench.LMJudgeEvaluator = original
            else:
                delattr(axbench, "LMJudgeEvaluator")
