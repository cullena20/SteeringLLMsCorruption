import torch
from .model import HookedModel

BATCH_SIZE = 32


class ActivationExtractor:
    """Extracts residual-stream activations at a given layer via forward hooks."""

    def __init__(self, hooked_model: HookedModel, layer: int):
        self.hooked_model = hooked_model
        self.layer = layer

    @torch.no_grad()
    def extract(self, prompts: list[str], token_position: int = -1) -> torch.Tensor:
        """Run prompts through the model and return activations at *token_position*.

        Returns:
            Tensor of shape (num_prompts, hidden_dim).
        """
        tokenizer = self.hooked_model.tokenizer
        tokenizer.padding_side = "left"

        target_module = self.hooked_model.get_residual_stream_module(self.layer)
        all_acts = []

        for batch_start in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[batch_start : batch_start + BATCH_SIZE]
            captured = []

            def _hook(module, input, output, _store=captured):
                hidden = output[0] if isinstance(output, tuple) else output
                _store.append(hidden.detach())

            handle = target_module.register_forward_hook(_hook)
            try:
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.hooked_model.device)
                self.hooked_model.model(**inputs)
            finally:
                handle.remove()

            # captured[0]: (batch, seq_len, hidden_dim)
            hidden = captured[0]
            if token_position == -1:
                # last non-padding token per sequence
                lengths = inputs["attention_mask"].sum(dim=1) - 1  # (batch,)
                acts = hidden[torch.arange(len(batch)), lengths, :]
            else:
                acts = hidden[:, token_position, :]
            all_acts.append(acts.cpu())

        return torch.cat(all_acts, dim=0)
