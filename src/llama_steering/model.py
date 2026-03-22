import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HookedModel:
    """Wraps a HuggingFace causal LM with hook-friendly access to internals."""

    DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

    def __init__(self, model_id: str | None = None, device: str = "auto", dtype=torch.bfloat16):
        model_id = model_id or self.DEFAULT_MODEL_ID
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = device if device == "auto" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
        )
        if device != "auto":
            self.model = self.model.to(device)

        self.model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def get_residual_stream_module(self, layer: int):
        """Return the decoder layer module whose output is the residual stream after that layer."""
        return self.model.model.layers[layer]
