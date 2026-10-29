"""Exact three-modality attribution and bounded contribution routing."""
import math
from collections.abc import Mapping

import torch
from torch import Tensor


def exact_shapley(utilities: Mapping[int, Tensor], num_modalities: int = 3) -> Tensor:
    """Return [B,K] Shapley values from utilities indexed by coalition bit masks.

    Bits 0/1/2 denote text/audio/vision. The empty coalition has utility zero
    unless explicitly supplied. Utilities may be negative log probabilities.
    """
    if num_modalities < 1:
        raise ValueError("num_modalities must be positive")
    full = (1 << num_modalities) - 1
    if any(mask not in utilities for mask in range(1, full + 1)):
        raise ValueError("all nonempty coalition utilities are required")
    reference = utilities[full]
    if reference.ndim != 1 or any(v.shape != reference.shape for v in utilities.values()):
        raise ValueError("coalition utilities must have the same [B] shape")
    zero = torch.zeros_like(reference)
    values = {0: zero, **utilities}
    output = []
    for modality in range(num_modalities):
        contribution = zero.clone()
        for mask in range(full + 1):
            if mask & (1 << modality):
                continue
            size = mask.bit_count()
            coefficient = (math.factorial(size) * math.factorial(num_modalities - size - 1)
                           / math.factorial(num_modalities))
            contribution = contribution + coefficient * (values[mask | (1 << modality)] - values[mask])
        output.append(contribution)
    return torch.stack(output, dim=-1)


def standardize_shapley(values: Tensor, eps: float = 1e-6) -> Tensor:
    """Within-utterance centering and sample-SD scaling (divisor K-1)."""
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("values must have shape [B,K] with K >= 2")
    return (values - values.mean(-1, keepdim=True)) / values.std(
        -1, keepdim=True, correction=1).clamp_min(eps)


def bounded_routing_weights(scores: Tensor, bound: float = 0.3, eps: float = 1e-8) -> Tensor:
    """Apply the centered, bounded-rescaling routing formula from the paper."""
    if scores.ndim != 2 or scores.shape[1] < 1 or bound < 0:
        raise ValueError("scores must be [B,K] and bound must be nonnegative")
    centered = scores - scores.mean(-1, keepdim=True)
    weights = (1 + bound * centered / centered.abs().amax(-1, keepdim=True).clamp_min(eps))
    weights = (weights / scores.shape[1]).clamp_min(0)
    return weights / weights.sum(-1, keepdim=True).clamp_min(eps)
