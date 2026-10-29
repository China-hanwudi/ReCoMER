"""ReCoMER contextual reference model: NCHA -> FCCR -> SABER."""
from dataclasses import dataclass
from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ReCoMERConfig
from .shapley import bounded_routing_weights, exact_shapley, standardize_shapley

MODALITIES = ("text", "audio", "vision")
_ALIASES = {"text": "T", "audio": "A", "vision": "V"}


def limit_relative_norm(update: Tensor, current: Tensor, eta: float, eps: float = 1e-12) -> Tensor:
    """Shrink only updates exceeding eta * ||current||; preserve smaller ones."""
    if eta < 0 or update.shape != current.shape:
        raise ValueError("eta must be nonnegative and tensor shapes must match")
    amount = update.norm(dim=-1, keepdim=True)
    budget = eta * current.norm(dim=-1, keepdim=True)
    return update * torch.minimum(torch.ones_like(amount), budget / amount.clamp_min(eps))


class ScaleAwareAdapter(nn.Module):
    """Identity-initialized low-rank adapter with A(0)=0 and RMS rescaling."""
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(dim, min(rank, dim), bias=False)
        self.up = nn.Linear(min(rank, dim), dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: Tensor) -> Tensor:
        rms = value.square().mean(-1, keepdim=True).add(1e-8).sqrt()
        return value + rms * self.up(F.gelu(self.down(value / rms)))


@dataclass(frozen=True)
class FrozenContext:
    """Cached same-forward state. Do not mutate its tensors between subsets."""
    updated_tokens: Tensor
    history_tokens: Tensor
    modality_mask: Tensor
    history_mask: Tensor
    admission_gate: Tensor
    routing_weights: Tensor
    cpu_rng_state: Tensor
    cuda_rng_state: Tensor | None


class ReCoMERContextModel(nn.Module):
    def __init__(self, config: ReCoMERConfig) -> None:
        super().__init__()
        self.config = config
        d, c = config.d_model, config.num_classes
        self.projections = nn.ModuleList(nn.Sequential(nn.Linear(size, d), nn.LayerNorm(d), nn.GELU())
                                         for size in config.input_dims)
        self.queries = nn.ModuleList(nn.Linear(d, d) for _ in MODALITIES)
        self.relative_positions = nn.ModuleList(nn.Embedding(config.history_k + 1, d, padding_idx=0)
                                                for _ in MODALITIES)
        self.speaker_embeddings = nn.ModuleList(nn.Embedding(2, d) for _ in MODALITIES)
        self.null_keys = nn.Parameter(torch.zeros(3, d))
        self.unimodal_heads = nn.ModuleList(nn.Linear(d, c) for _ in MODALITIES)
        self.admission = nn.Sequential(nn.Linear(24, config.admission_hidden), nn.GELU(),
                                        nn.Linear(config.admission_hidden, 1))
        nn.init.zeros_(self.admission[-1].weight)
        nn.init.zeros_(self.admission[-1].bias)
        self.adapters = nn.ModuleList(ScaleAwareAdapter(d, config.adapter_rank) for _ in MODALITIES)
        self.feedback_scale = nn.Parameter(torch.zeros(3))
        self.attenuation = nn.ModuleList(nn.Linear(2, 1) for _ in MODALITIES)
        self.scorer_projections = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 48), nn.GELU())
                                                for _ in MODALITIES)
        self.scorer = nn.Sequential(nn.Linear(3 * 48 + c + 8, 96), nn.GELU(), nn.Linear(96, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)
        layer = nn.TransformerEncoderLayer(d, config.num_heads, config.ff_dim,
                                           config.dropout, activation="gelu", batch_first=True)
        self.joint = nn.TransformerEncoder(layer, config.num_layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(d, c)

    def _features(self, data: Mapping[str, Tensor], name: str) -> Tensor | None:
        return data.get(name, data.get(_ALIASES[name]))

    def _attention(self, current: Tensor, history: Tensor, mask: Tensor,
                   position: Tensor, speaker: Tensor | None, null_bias: Tensor) -> tuple[Tensor, Tensor]:
        summaries, attentions = [], []
        for m in range(3):
            query = self.queries[m](current[:, m])
            key = history[:, :, m] + self.relative_positions[m](position)
            if speaker is not None:
                key = key + self.speaker_embeddings[m](speaker)
            scores = (query[:, None] * key).sum(-1) / math.sqrt(self.config.d_model)
            scores = scores.masked_fill(~mask[:, :, m], -torch.inf)
            null_score = (query * self.null_keys[m]).sum(-1) / math.sqrt(self.config.d_model) + null_bias
            attention = torch.cat((scores, null_score[:, None]), -1).softmax(-1)
            summaries.append((attention[:, :-1, None] * history[:, :, m]).sum(1))
            attentions.append(attention)
        return torch.stack(summaries, 1), torch.stack(attentions, 1)

    def _admission_statistics(self, current: Tensor, summary: Tensor, mask: Tensor,
                              attention: Tensor, positions: Tensor, speaker: Tensor | None) -> Tensor:
        statistics = []
        for m in range(3):
            p = self.unimodal_heads[m](current[:, m]).softmax(-1)
            h = self.unimodal_heads[m](summary[:, m]).softmax(-1)
            mid = (p + h) / 2
            js = .5 * ((p * (p.clamp_min(1e-9).log() - mid.clamp_min(1e-9).log())).sum(-1)
                       + (h * (h.clamp_min(1e-9).log() - mid.clamp_min(1e-9).log())).sum(-1)) / math.log(2)
            a = attention[:, m]
            entropy = -(a * a.clamp_min(1e-9).log()).sum(-1)
            entropy = entropy / (mask[:, :, m].sum(-1) + 1).to(a.dtype).log().clamp_min(1.)
            recent = (a[:, :-1] * (positions == 1)).sum(-1)
            same = torch.zeros_like(recent) if speaker is None else (a[:, :-1] * (speaker == 1)).sum(-1)
            other = torch.zeros_like(recent) if speaker is None else (a[:, :-1] * (speaker == 0)).sum(-1)
            row = torch.stack((p.amax(-1), h.amax(-1), js, entropy, a[:, -1], recent, same, other), -1)
            statistics.append(row * mask[:, :, m].any(-1, keepdim=True))
        return torch.cat(statistics, -1).detach()

    def _scores(self, updated: Tensor, modality_mask: Tensor, history_state: Tensor) -> Tensor:
        local = torch.stack([self.scorer_projections[m](updated[:, m].detach()) for m in range(3)], 1)
        local = local * modality_mask[..., None]
        context = local.sum(1, keepdim=True) / modality_mask.sum(-1).clamp_min(1)[:, None, None]
        probabilities = torch.stack([self.unimodal_heads[m](updated[:, m].detach()).softmax(-1)
                                      for m in range(3)], 1).detach()
        consensus = (probabilities * modality_mask[..., None]).sum(1, keepdim=True)
        consensus = consensus / modality_mask.sum(-1).clamp_min(1)[:, None, None]
        top = probabilities.topk(2, dim=-1).values
        stats = torch.stack((top[..., 0], top[..., 0] - top[..., 1],
                             -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1),
                             (probabilities - consensus).abs().mean(-1)), -1)
        features = torch.cat((local, context.expand_as(local), local - context,
                              probabilities, stats, history_state[:, None].expand(-1, 3, -1)), -1)
        return self.scorer(features).squeeze(-1)

    def _predict(self, state: FrozenContext, subset_mask: Tensor) -> Tensor:
        observed = state.modality_mask & subset_mask[None]
        weights = state.routing_weights.detach()
        tokens = state.updated_tokens * weights[..., None]
        mask = observed
        if self.config.include_history_tokens and state.history_tokens.shape[1]:
            history = state.history_tokens * weights[:, None, :, None] * state.admission_gate[:, None, None, None]
            history_mask = state.history_mask & subset_mask[None, None] & (state.admission_gate[:, None, None] > 0)
            tokens = torch.cat((tokens, history.flatten(1, 2)), 1)
            mask = torch.cat((mask, history_mask.flatten(1, 2)), 1)
        tokens = torch.where(mask[..., None], tokens, torch.zeros_like(tokens))
        safe_mask = mask.clone()
        empty = ~safe_mask.any(-1)
        safe_mask[empty, 0] = True  # numerical placeholder; empty coalition utility remains zero
        output = self.joint(tokens, src_key_padding_mask=~safe_mask)[:, :3]
        pool_weights = observed.to(output.dtype)
        if self.config.weighted_pooling:
            pool_weights = pool_weights * weights
        pooled = (output * pool_weights[..., None]).sum(1) / pool_weights.sum(-1, keepdim=True).clamp_min(1e-9)
        logits = self.classifier(pooled)
        return torch.where(empty[:, None], torch.zeros_like(logits), logits)

    def forward(self, current: Mapping[str, Tensor], history: Mapping[str, Tensor] | None = None,
                modality_mask: Tensor | None = None, history_mask: Tensor | None = None,
                relative_positions: Tensor | None = None, speaker_relations: Tensor | None = None,
                admission_mode: str = "hard", admission_override: float | Tensor | None = None) -> dict:
        """Consume pooled current [B,Dm] and preceding-history [B,H,Dm] features.

        Masks are [B,3] and [B,H,3], in text/audio/vision order. Positive relative
        positions identify preceding turns (1 is most recent); nonpositive or
        over-window slots are excluded. Speaker relations are 1=same/0=other.
        Labels and teacher predictions are not inference inputs.
        """
        values = [self._features(current, m) for m in MODALITIES]
        reference = next((v for v in values if v is not None), None)
        if reference is None or reference.ndim != 2 or not reference.is_floating_point():
            raise ValueError("current requires floating [B,Dm] feature tensors")
        batch, device = reference.shape[0], reference.device
        if modality_mask is not None and ((modality_mask != 0) & (modality_mask != 1)).any():
            raise ValueError("modality_mask must be binary")
        present = torch.ones(batch, 3, dtype=torch.bool, device=device) if modality_mask is None else modality_mask.to(device=device, dtype=torch.bool)
        if present.shape != (batch, 3) or not present.any(-1).all():
            raise ValueError("modality_mask must be [B,3] with an observed modality per utterance")
        encoded = []
        for m, (value, dim) in enumerate(zip(values, self.config.input_dims)):
            if value is None:
                if present[:, m].any():
                    raise ValueError(f"missing observed {MODALITIES[m]} features")
                value = reference.new_zeros(batch, dim)
            if value.shape != (batch, dim) or value.device != device or not torch.isfinite(value[present[:, m]]).all():
                raise ValueError(f"invalid {MODALITIES[m]} current features")
            clean = torch.where(present[:, m, None], value, torch.zeros_like(value))
            encoded.append(self.projections[m](clean) * present[:, m, None])
        z = torch.stack(encoded, 1)
        history = {} if history is None else history
        historical_values = [self._features(history, m) for m in MODALITIES]
        sample = next((v for v in historical_values if v is not None), None)
        count = 0 if sample is None else sample.shape[1]
        if count > self.config.history_k:
            raise ValueError("history length exceeds configured history_k")
        if history_mask is not None and ((history_mask != 0) & (history_mask != 1)).any():
            raise ValueError("history_mask must be binary")
        hmask = torch.ones(batch, count, 3, device=device, dtype=torch.bool) if history_mask is None else history_mask.to(device=device, dtype=torch.bool)
        if hmask.shape != (batch, count, 3):
            raise ValueError("history_mask must be [B,H,3]")
        if relative_positions is None:
            positions = torch.arange(count, 0, -1, device=device).expand(batch, -1)
        else:
            if relative_positions.shape != (batch, count):
                raise ValueError("relative_positions must be [B,H]")
            positions = relative_positions.to(device=device, dtype=torch.long)
        hmask = hmask & present[:, None] & ((positions > 0) & (positions <= self.config.history_k))[..., None]
        positions = positions.clamp(0, self.config.history_k)
        speaker = None if speaker_relations is None else speaker_relations.to(device=device, dtype=torch.long)
        if speaker is not None and (speaker.shape != (batch, count) or ((speaker != 0) & (speaker != 1)).any()):
            raise ValueError("speaker_relations must be binary [B,H]")
        encoded_history = []
        for m, (value, dim) in enumerate(zip(historical_values, self.config.input_dims)):
            if value is None:
                if hmask[:, :, m].any():
                    raise ValueError(f"missing observed {MODALITIES[m]} history features")
                value = reference.new_zeros(batch, count, dim)
            if value.shape != (batch, count, dim) or value.device != device or not torch.isfinite(value[hmask[:, :, m]]).all():
                raise ValueError(f"invalid {MODALITIES[m]} history features")
            clean = torch.where(hmask[:, :, m, None], value, torch.zeros_like(value))
            encoded_history.append(self.projections[m](clean) * hmask[:, :, m, None])
        h = torch.stack(encoded_history, 2)
        summary_first, first_attention = self._attention(z, h, hmask, positions, speaker, z.new_zeros(batch))
        statistics = self._admission_statistics(z, summary_first, hmask, first_attention, positions, speaker)
        has_history = hmask.any((1, 2))
        admission_logits = self.admission(statistics).squeeze(-1) / self.config.admission_temperature
        q = admission_logits.sigmoid() * has_history
        null_bias = self.config.null_beta * (torch.log1p(-q.clamp(1e-6, 1 - 1e-6)) - q.clamp(1e-6, 1 - 1e-6).log())
        summary, attention = self._attention(z, h, hmask, positions, speaker, null_bias)
        hard = (q >= self.config.admission_threshold).to(z.dtype) * has_history
        if admission_mode == "hard":
            gate = hard + q - q.detach() if self.training else hard
        elif admission_mode == "soft":
            gate = q
        elif admission_mode == "available":
            gate = has_history.to(z.dtype)
        else:
            raise ValueError("admission_mode must be hard, soft or available")
        if admission_override is not None:
            override = torch.as_tensor(admission_override, dtype=z.dtype, device=device).expand(batch)
            if not torch.isfinite(override).all() or ((override < 0) | (override > 1)).any():
                raise ValueError("admission_override must be in [0,1]")
            gate = gate * override
        updated = []
        for m in range(3):
            proposed = self.feedback_scale[m] * self.adapters[m](summary[:, m])
            bounded = limit_relative_norm(proposed, z[:, m], self.config.eta)
            attenuation = torch.ones(batch, 1, device=device, dtype=z.dtype)
            if self.config.use_attenuation:
                angle = F.cosine_similarity(proposed, z[:, m], dim=-1)
                ratio = proposed.norm(dim=-1) / z[:, m].norm(dim=-1).clamp_min(1e-9)
                attenuation = self.attenuation[m](torch.stack((angle, ratio / (1 + ratio)), -1)).sigmoid()
            updated.append(z[:, m] + gate[:, None] * attenuation * bounded)
        updated = torch.stack(updated, 1)
        history_state = torch.stack((q, gate, 1 - attention[:, :, -1].mean(1), statistics[:, 2::8].mean(1)), -1)
        history_state = (history_state * gate[:, None]).detach()
        scores = self._scores(updated, present, history_state)
        weights = bounded_routing_weights(scores, self.config.routing_bound)
        state = FrozenContext(updated, h, present, hmask, gate, weights.detach(), torch.get_rng_state(),
                              torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
        logits = self._predict(state, torch.ones(3, dtype=torch.bool, device=device))
        return {"logits": logits, "probabilities": logits.softmax(-1), "state": state,
                "current_tokens": z, "updated_tokens": updated, "history_summary": summary,
                "history_attention": attention, "first_history_attention": first_attention,
                "admission_probability": q, "admission_logits": admission_logits,
                "admission_gate": gate, "has_history": has_history,
                "routing_weights": weights, "contribution_scores": scores}

    def coalition_logits(self, state: FrozenContext) -> dict[int, Tensor]:
        """Evaluate subsets without rereading history, rerouting or re-encoding."""
        device = state.updated_tokens.device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        result = {}
        with torch.random.fork_rng(devices=devices):
            for coalition in range(1, 8):
                torch.set_rng_state(state.cpu_rng_state)
                if state.cuda_rng_state is not None:
                    torch.cuda.set_rng_state(state.cuda_rng_state, device)
                subset = torch.tensor([bool(coalition & (1 << m)) for m in range(3)], device=device)
                result[coalition] = self._predict(state, subset)
        return result

    def shapley_targets(self, state: FrozenContext, labels: Tensor) -> dict[str, Tensor]:
        with torch.no_grad():
            coalitions = self.coalition_logits(state)
            utilities = self._utilities(state, coalitions, labels)
            raw = exact_shapley(utilities)
            return {"raw": raw, "standardized": standardize_shapley(raw)}

    @staticmethod
    def _utilities(state: FrozenContext, coalitions: dict[int, Tensor], labels: Tensor) -> dict[int, Tensor]:
        utilities = {}
        for coalition, logits in coalitions.items():
            selected = torch.tensor([bool(coalition & (1 << m)) for m in range(3)], device=logits.device)
            valid = (state.modality_mask & selected[None]).any(-1)
            value = logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
            utilities[coalition] = torch.where(valid, value, torch.zeros_like(value))
        return utilities

    def compute_loss(self, output: dict, labels: Tensor, admission_targets: Tensor | None = None,
                     lambda_utility: float = .3, lambda_subset: float = .1,
                     lambda_admission: float = .3, utility_scale: float = 1.) -> dict[str, Tensor]:
        """Explicit training objective; OOF admission targets are supplied externally."""
        if min(lambda_utility, lambda_subset, lambda_admission, utility_scale) < 0:
            raise ValueError("loss coefficients must be nonnegative")
        labels = labels.to(device=output["logits"].device, dtype=torch.long)
        task = F.cross_entropy(output["logits"], labels)
        coalitions = self.coalition_logits(output["state"])
        with torch.no_grad():
            raw = exact_shapley(self._utilities(output["state"], coalitions, labels))
            target = standardize_shapley(raw)
        utility = F.mse_loss(output["contribution_scores"], target)
        losses = []
        for coalition in range(1, 7):
            selected = torch.tensor([bool(coalition & (1 << m)) for m in range(3)], device=labels.device)
            valid = (output["state"].modality_mask & selected[None]).any(-1)
            if valid.any():
                losses.append(F.cross_entropy(coalitions[coalition][valid], labels[valid], reduction="sum"))
        subset_count = sum(int((output["state"].modality_mask & torch.tensor(
            [bool(code & (1 << m)) for m in range(3)], device=labels.device)[None]).any(-1).sum()) for code in range(1, 7))
        subset = torch.stack(losses).sum() / max(subset_count, 1) if losses else task.new_zeros(())
        admission_loss = task.new_zeros(())
        if admission_targets is not None:
            targets = admission_targets.to(device=labels.device, dtype=task.dtype)
            if targets.shape != labels.shape or not torch.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
                raise ValueError("admission_targets must be finite OOF probabilities [B]")
            valid = output["has_history"]
            if valid.any():
                admission_loss = F.binary_cross_entropy_with_logits(output["admission_logits"][valid], targets[valid])
        elif lambda_admission > 0 and output["has_history"].any():
            raise ValueError("positive admission supervision requires externally prepared OOF targets")
        total = task + lambda_utility * utility_scale * utility + lambda_subset * subset + lambda_admission * admission_loss
        return {"total": total, "task": task, "utility": utility, "subset": subset,
                "admission": admission_loss, "shapley_targets": target, "raw_shapley": raw}

    def unimodal_loss(self, output: dict, labels: Tensor, both_views: bool = True) -> Tensor:
        """Auxiliary-head CE on detached tokens, for a separate head optimizer.

        This objective trains the auxiliary probability heads without adding
        auxiliary CE gradients to the representation encoders or history path.
        """
        labels = labels.to(device=output["logits"].device, dtype=torch.long)
        views = [output["updated_tokens"]]
        if both_views:
            views.append(output["current_tokens"])
        losses, count = [], 0
        mask = output["state"].modality_mask
        for tokens in views:
            for m in range(3):
                valid = mask[:, m]
                if valid.any():
                    logits = self.unimodal_heads[m](tokens[valid, m].detach())
                    losses.append(F.cross_entropy(logits, labels[valid], reduction="sum"))
                    count += int(valid.sum())
        return torch.stack(losses).sum() / count
