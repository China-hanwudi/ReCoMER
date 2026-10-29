"""Prior-relative AV correction and dataset-specific prediction composition."""
import torch
from torch import Tensor, nn


def _probabilities(value: Tensor, name: str) -> None:
    if value.ndim != 2 or value.shape[1] < 2 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite [B,C] probability tensor")
    if (value < 0).any() or not torch.allclose(value.sum(-1), torch.ones_like(value[:, 0]), atol=1e-5):
        raise ValueError(f"{name} must be nonnegative and sum to one")


def correction_statistics(tav: Tensor, av: Tensor, prior: Tensor, eps: float = 1e-9) -> Tensor:
    """The paper's six label-free statistics for the utterance-level gate."""
    _probabilities(tav, "tav")
    _probabilities(av, "av")
    if tav.shape != av.shape or prior.shape != (tav.shape[1],):
        raise ValueError("TAV, AV and prior class dimensions must match")
    if not torch.isfinite(prior).all() or (prior <= 0).any() or not torch.allclose(
            prior.sum(), prior.new_tensor(1.), atol=1e-5):
        raise ValueError("class_prior must be a strictly positive probability vector")
    top = tav.topk(2, dim=-1).values
    evidence = (av + eps).log() - (prior + eps).log()
    return torch.stack((top[:, 0], top[:, 0] - top[:, 1], av.amax(-1),
                        evidence.amax(-1), (tav.argmax(-1) == av.argmax(-1)).to(tav.dtype),
                        -(tav * tav.clamp_min(eps).log()).sum(-1)), dim=-1)


def correct_predictions(tav_probabilities: Tensor, av_probabilities: Tensor,
                        class_prior: Tensor, gate: Tensor,
                        av_available: Tensor | None = None, eps: float = 1e-9) -> Tensor:
    """Correct TAV log probabilities using gated, prior-relative AV evidence.

    ``gate`` is the learned scalar rho per utterance, not the outer mixing
    weight. AV-unavailable rows have rho=0 and retain their TAV probabilities.
    """
    tav, av = tav_probabilities, av_probabilities
    correction_statistics(tav, av, class_prior, eps)
    gate = torch.as_tensor(gate, device=tav.device, dtype=tav.dtype).reshape(-1, 1)
    if gate.shape != (len(tav), 1) or not torch.isfinite(gate).all() or ((gate < 0) | (gate > 1)).any():
        raise ValueError("gate must contain one value in [0,1] per utterance")
    if av_available is not None:
        availability = torch.as_tensor(av_available, device=tav.device).reshape(-1, 1)
        if availability.shape != gate.shape or ((availability != 0) & (availability != 1)).any():
            raise ValueError("av_available must be a binary [B] mask")
        gate = gate * availability.to(tav.dtype)
    evidence = (av + eps).log() - (class_prior + eps).log()
    corrected = ((tav + eps).log() + gate * evidence).softmax(-1)
    return torch.where(gate == 0, tav, corrected)


class PriorRelativeCorrection(nn.Module):
    """Trainable six-statistic cRBEF gate; expert predictors are separate."""
    def __init__(self, class_prior: Tensor) -> None:
        super().__init__()
        prior = torch.as_tensor(class_prior, dtype=torch.float32)
        if prior.ndim != 1 or len(prior) < 2:
            raise ValueError("class_prior must be a class-probability vector")
        self.register_buffer("class_prior", prior)
        self.gate = nn.Linear(6, 1)

    def forward(self, tav: Tensor, av: Tensor, av_available: Tensor | None = None) -> dict[str, Tensor]:
        statistics = correction_statistics(tav, av, self.class_prior)
        rho = self.gate(statistics).sigmoid().squeeze(-1)
        if av_available is not None:
            rho = rho * av_available.to(device=rho.device, dtype=rho.dtype)
        return {"probabilities": correct_predictions(tav, av, self.class_prior, rho, av_available),
                "gate": rho, "statistics": statistics}


def arithmetic_fusion(contextual: Tensor, corrected: Tensor, alpha: float = 0.3) -> Tensor:
    _probabilities(contextual, "contextual")
    _probabilities(corrected, "corrected")
    if contextual.shape != corrected.shape or not 0 <= alpha <= 1:
        raise ValueError("prediction shapes must match and alpha must be in [0,1]")
    return (1 - alpha) * contextual + alpha * corrected


def calibrate_context(logits: Tensor, residual_delta: Tensor | None = None,
                      centered_class_bias: Tensor | None = None, temperature: float | Tensor = 1.) -> Tensor:
    """Apply an externally fitted, frozen contextual output transformation."""
    value = logits
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError("logits must be a finite [B,C] tensor")
    if residual_delta is not None:
        if residual_delta.shape != logits.shape or not torch.isfinite(residual_delta).all():
            raise ValueError("residual_delta must match logits")
        value = value + residual_delta
    if centered_class_bias is not None:
        if centered_class_bias.shape != (logits.shape[1],) or not torch.isfinite(centered_class_bias).all():
            raise ValueError("centered_class_bias must be a finite [C] vector")
        if not torch.allclose(centered_class_bias.mean(), centered_class_bias.new_zeros(()), atol=1e-5):
            raise ValueError("class bias must already be centered")
        value = value + centered_class_bias
    temperature = torch.as_tensor(temperature, device=value.device, dtype=value.dtype)
    if temperature.numel() != 1 or not torch.isfinite(temperature).all() or temperature.item() <= 0:
        raise ValueError("temperature must be a finite positive scalar")
    return (value / temperature).softmax(-1)


def emotiontalk_output(fused: Tensor, tav_head: Tensor, temperature: float = 3., omega: float = .85) -> Tensor:
    """Recorded EmotionTalk temperature transform and final arithmetic mix."""
    _probabilities(fused, "fused")
    _probabilities(tav_head, "tav_head")
    if temperature <= 0 or not 0 <= omega <= 1 or fused.shape != tav_head.shape:
        raise ValueError("invalid EmotionTalk composition")
    return omega * (fused.clamp_min(1e-9).log() / temperature).softmax(-1) + (1 - omega) * tav_head
