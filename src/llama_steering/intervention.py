from contextlib import contextmanager

import torch
from tqdm import tqdm
from .model import HookedModel

BATCH_SIZE = 32


class SteeringIntervenor:
    """Adds a steering vector to the residual stream during generation."""

    def __init__(self, hooked_model: HookedModel, layer: int, steering_vector: torch.Tensor):
        self.hooked_model = hooked_model
        self.layer = layer
        self.steering_vector = steering_vector.to(hooked_model.device)

    @contextmanager
    def _active_hook(self, alpha: float, start_position: int = -1):
        """Context manager that installs and removes the steering hook."""

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            seq_len = hidden.shape[1]
            if seq_len > 1:
                hidden[:, start_position:, :] += alpha * self.steering_vector
            else:
                hidden[:, -1, :] += alpha * self.steering_vector
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        target = self.hooked_model.get_residual_stream_module(self.layer)
        handle = target.register_forward_hook(hook_fn)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        alpha: float = 1.0,
        batch_size: int = BATCH_SIZE,
        **gen_kwargs,
    ) -> list[str]:
        """Generate text with the steering vector active (batched).

        Args:
            prompts: Input prompts.
            alpha: Scaling factor for the steering vector.
            batch_size: Number of prompts per batch.
            **gen_kwargs: Forwarded to model.generate().

        Returns:
            List of generated strings (completions only, no prompt).
        """
        gen_kwargs.setdefault("max_new_tokens", 128)
        gen_kwargs.setdefault("do_sample", False)

        tokenizer = self.hooked_model.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        results = [""] * len(prompts)
        num_batches = (len(prompts) + batch_size - 1) // batch_size

        with self._active_hook(alpha, start_position=-8):
            for batch_idx in tqdm(range(num_batches), desc="Generating (batched)"):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(prompts))
                batch_prompts = prompts[start:end]

                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.hooked_model.device)

                output_ids = self.hooked_model.model.generate(
                    **inputs,
                    **gen_kwargs,
                )

                # Decode only the newly generated tokens for each sequence
                input_len = inputs["input_ids"].shape[1]
                for i, ids in enumerate(output_ids):
                    new_ids = ids[input_len:]
                    results[start + i] = tokenizer.decode(new_ids, skip_special_tokens=True)

        return results
