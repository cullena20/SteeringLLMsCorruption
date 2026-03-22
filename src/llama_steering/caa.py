import torch
from .activations import ActivationExtractor


class CAAVector:
    """Contrastive Activation Addition: computes a steering vector from paired prompts."""

    def __init__(self, extractor: ActivationExtractor):
        self.extractor = extractor
        self.vector: torch.Tensor | None = None

    def fit(self, positive_prompts: list[str], negative_prompts: list[str], token_position: int = -1) -> torch.Tensor:
        """Compute the mean-difference steering vector.

        Args:
            positive_prompts: Prompts eliciting the target behaviour.
            negative_prompts: Matched prompts eliciting the opposite behaviour.
            token_position: Which token's activations to use.

        Returns:
            Steering vector of shape (hidden_dim,).
        """
        pos_acts = self.extractor.extract(positive_prompts, token_position=token_position)  # (N, D)
        neg_acts = self.extractor.extract(negative_prompts, token_position=token_position)  # (N, D)
        self.vector = (pos_acts.mean(dim=0) - neg_acts.mean(dim=0))
        return self.vector
