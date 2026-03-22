from contextlib import contextmanager

import torch
from .model import HookedModel


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
                # Prefill pass: steer from start_position onward.
                hidden[:, start_position:, :] += alpha * self.steering_vector
            else:
                # Generation steps: single new token, always steer it.
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
        **gen_kwargs,
    ) -> list[str]:
        """Generate text with the steering vector active.

        Args:
            prompts: Input prompts.
            alpha: Scaling factor for the steering vector. Positive amplifies the
                   direction, negative reverses it.
            **gen_kwargs: Forwarded to model.generate().

        Returns:
            List of generated strings (prompt + completion decoded).
        """
        gen_kwargs.setdefault("max_new_tokens", 128)
        gen_kwargs.setdefault("do_sample", False)

        tokenizer = self.hooked_model.tokenizer
        results = []

        with self._active_hook(alpha, start_position=-8):
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(
                    self.hooked_model.device
                )
                output_ids = self.hooked_model.model.generate(**inputs, **gen_kwargs)
                # Decode only the newly generated tokens.
                new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
                results.append(tokenizer.decode(new_ids, skip_special_tokens=True))

        return results
