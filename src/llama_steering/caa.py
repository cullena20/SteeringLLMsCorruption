import numpy as np
import torch
from .activations import ActivationExtractor


class CAAVector:
    """Contrastive Activation Addition: computes a steering vector from paired prompts.

    By default uses the simple diff-of-means (standard CAA). Pass an ``estimator``
    callable to use a robust estimator instead — it must accept
    (pos_acts: np.ndarray, neg_acts: np.ndarray, **kwargs) -> np.ndarray.
    """

    def __init__(self, extractor: ActivationExtractor, estimator=None):
        self.extractor = extractor
        self.estimator = estimator
        self.vector: torch.Tensor | None = None

    def fit(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        token_position: int = -1,
        **estimator_kwargs,
    ) -> torch.Tensor:
        """Compute the steering vector from paired prompts.

        Args:
            positive_prompts: Prompts eliciting the target behaviour.
            negative_prompts: Matched prompts eliciting the opposite behaviour.
            token_position: Which token's activations to use.
            **estimator_kwargs: Forwarded to the robust estimator (e.g. tau=0.1).

        Returns:
            Steering vector of shape (hidden_dim,).
        """
        pos_acts = self.extractor.extract(positive_prompts, token_position=token_position)
        neg_acts = self.extractor.extract(negative_prompts, token_position=token_position)

        if self.estimator is None:
            self.vector = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
        else:
            pos_np = pos_acts.cpu().float().numpy()
            neg_np = neg_acts.cpu().float().numpy()
            vec_np = self.estimator(pos_np, neg_np, **estimator_kwargs)
            self.vector = torch.from_numpy(
                np.asarray(vec_np, dtype=np.float32)
            ).to(pos_acts.device)

        return self.vector
