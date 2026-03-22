import torch
from .model import HookedModel


class ActivationExtractor:
    """Extracts residual-stream activations at a given layer via forward hooks."""

    def __init__(self, hooked_model: HookedModel, layer: int):
        self.hooked_model = hooked_model
        self.layer = layer
        self._captured: list[torch.Tensor] = []
        self._hook_handle = None

    def _hook_fn(self, module, input, output):
        # Decoder layer output is a tuple; first element is the hidden state.
        hidden = output[0] if isinstance(output, tuple) else output
        self._captured.append(hidden.detach())

    @torch.no_grad()
    def extract(self, prompts: list[str], token_position: int = -1) -> torch.Tensor:
        """Run prompts through the model and return activations at *token_position*.

        Returns:
            Tensor of shape (num_prompts, hidden_dim).
        """
        target_module = self.hooked_model.get_residual_stream_module(self.layer)
        handle = target_module.register_forward_hook(self._hook_fn)

        activations = []
        try:
            for prompt in prompts:
                self._captured.clear()
                inputs = self.hooked_model.tokenizer(
                    prompt, return_tensors="pt", padding=False, truncation=True
                ).to(self.hooked_model.device)
                self.hooked_model.model(**inputs)
                # _captured[0] has shape (1, seq_len, hidden_dim)
                act = self._captured[0][0, token_position, :]
                activations.append(act)
        finally:
            handle.remove()
            self._captured.clear()

        return torch.stack(activations)
