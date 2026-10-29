"""Synthetic mechanism tests; these do not claim benchmark reproduction."""
import pytest
import torch

from recomer import (ReCoMERConfig, ReCoMERContextModel, bounded_routing_weights,
                     exact_shapley, standardize_shapley, limit_relative_norm,
                     correct_predictions, arithmetic_fusion, calibrate_context,
                     emotiontalk_output, PriorRelativeCorrection)


@pytest.fixture
def tiny_model():
    torch.manual_seed(13)
    config = ReCoMERConfig(input_dims=(5, 4, 3), num_classes=4, d_model=12,
                           num_heads=3, ff_dim=24, num_layers=2, history_k=3,
                           adapter_rank=4, dropout=0., use_attenuation=False)
    return ReCoMERContextModel(config)


@pytest.fixture
def batch():
    torch.manual_seed(17)
    sizes = dict(text=5, audio=4, vision=3)
    return (dict((m, torch.randn(3, d)) for m, d in sizes.items()),
            dict((m, torch.randn(3, 3, d)) for m, d in sizes.items()))


def test_exact_shapley_efficiency_and_dummy_player():
    effect = torch.tensor([[1., -2., 3.], [.5, 1.5, 0.]])
    utilities = {code: sum((effect[:, m] for m in range(3) if code & (1 << m)),
                           torch.zeros(2)) for code in range(1, 8)}
    actual = exact_shapley(utilities)
    torch.testing.assert_close(actual, effect)
    torch.testing.assert_close(actual.sum(-1), utilities[7])
    torch.testing.assert_close(standardize_shapley(effect).std(-1, correction=1), torch.ones(2))
    torch.testing.assert_close(standardize_shapley(torch.ones(2, 3)), torch.zeros(2, 3))


def test_bounded_routing_is_shift_invariant_and_uniform_for_equal_scores():
    scores = torch.tensor([[1., 2., -3.], [7., 7., 7.]])
    weights = bounded_routing_weights(scores, .3)
    torch.testing.assert_close(weights, bounded_routing_weights(scores + 6, .3))
    torch.testing.assert_close(weights[1], torch.full((3,), 1 / 3))
    assert ((weights >= .7 / 3 - 1e-6) & (weights <= 1.3 / 3 + 1e-6)).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(2))


def test_norm_clip_preserves_within_budget_and_only_shrinks_excess():
    current = torch.tensor([[3., 4.], [3., 4.], [0., 0.]])
    update = torch.tensor([[.1, .1], [3., 4.], [3., 4.]])
    actual = limit_relative_norm(update, current, .1)
    torch.testing.assert_close(actual[0], update[0])
    torch.testing.assert_close(actual[1], torch.tensor([.3, .4]))
    torch.testing.assert_close(actual[2], torch.zeros(2))


def test_no_history_and_all_masked_history_are_identical(tiny_model, batch):
    current, history = batch
    tiny_model.eval()
    with torch.no_grad():
        absent = tiny_model(current)
        masked = tiny_model(current, history, history_mask=torch.zeros(3, 3, 3))
    torch.testing.assert_close(masked["logits"], absent["logits"], rtol=0, atol=0)
    torch.testing.assert_close(masked["history_summary"], torch.zeros(3, 3, 12))
    assert not masked["admission_gate"].any()


def test_null_has_zero_value_and_real_mass_is_attenuated(tiny_model):
    current = torch.zeros(2, 3, 12)
    historical = torch.ones(2, 1, 3, 12)
    valid = torch.ones(2, 1, 3, dtype=torch.bool)
    positions = torch.ones(2, 1, dtype=torch.long)
    with torch.no_grad():
        for query in tiny_model.queries:
            query.weight.zero_()
            query.bias.zero_()
        first, attention = tiny_model._attention(current, historical, valid, positions, None, torch.zeros(2))
        second, _ = tiny_model._attention(current, historical, valid, positions, None, torch.full((2,), 10.))
    torch.testing.assert_close(first, torch.full_like(first, .5))
    torch.testing.assert_close(attention[..., -1], torch.full((2, 3), .5))
    assert second.max() < .001


def test_rejected_history_has_no_prediction_bypass(tiny_model, batch):
    current, history = batch
    tiny_model.eval()
    with torch.no_grad():
        tiny_model.feedback_scale.fill_(3.)
        off = tiny_model(current, history, admission_override=0.)
        no_history = tiny_model(current)
    torch.testing.assert_close(off["updated_tokens"], off["current_tokens"], rtol=0, atol=0)
    torch.testing.assert_close(off["logits"], no_history["logits"], rtol=0, atol=0)


def test_missing_and_future_history_cannot_change_output(tiny_model, batch):
    current, history = batch
    tiny_model.eval()
    mask = torch.tensor([[1, 1, 0]]).expand(3, -1)
    with torch.no_grad():
        tiny_model.feedback_scale.fill_(1.)
        baseline = tiny_model(current, history, modality_mask=mask,
                              relative_positions=torch.zeros(3, 3, dtype=torch.long))
        changed = dict(current, vision=torch.full_like(current["vision"], float("nan")))
        result = tiny_model(changed, history, modality_mask=mask,
                            relative_positions=torch.zeros(3, 3, dtype=torch.long))
        no_history = tiny_model(current, modality_mask=mask)
    torch.testing.assert_close(result["logits"], baseline["logits"], rtol=0, atol=0)
    torch.testing.assert_close(result["logits"], no_history["logits"], rtol=0, atol=0)


def test_real_feedback_respects_norm_bound(tiny_model, batch):
    current, history = batch
    tiny_model.eval()
    with torch.no_grad():
        tiny_model.feedback_scale.fill_(100.)
        output = tiny_model(current, history, admission_mode="available")
    update = (output["updated_tokens"] - output["current_tokens"]).norm(dim=-1)
    budget = tiny_model.config.eta * output["current_tokens"].norm(dim=-1)
    assert (update <= budget + 1e-6).all()


def test_coalitions_reuse_same_state_and_dropout(tiny_model, batch):
    current, history = batch
    tiny_model.train()
    for layer in tiny_model.joint.layers:
        layer.dropout.p = layer.dropout1.p = layer.dropout2.p = .2
        layer.self_attn.dropout = .2
    count = [0]
    handle = tiny_model.admission.register_forward_hook(lambda *args: count.__setitem__(0, count[0] + 1))
    output = tiny_model(current, history)
    logits = tiny_model.coalition_logits(output["state"])
    handle.remove()
    assert count[0] == 1
    assert set(logits) == set(range(1, 8))
    torch.testing.assert_close(logits[7], output["logits"], rtol=0, atol=0)
    assert not output["state"].routing_weights.requires_grad


def test_training_losses_and_gradient_isolation(tiny_model, batch):
    current, history = batch
    output = tiny_model(current, history, admission_mode="soft")
    losses = tiny_model.compute_loss(output, torch.tensor([0, 1, 2]), admission_targets=torch.tensor([.2, .5, .8]))
    assert all(torch.isfinite(losses[key]) for key in ("total", "task", "utility", "subset", "admission"))
    losses["total"].backward()
    assert tiny_model.scorer[-1].weight.grad is not None
    assert tiny_model.unimodal_heads[0].weight.grad is None
    tiny_model.zero_grad(set_to_none=True)
    output = tiny_model(current, history)
    tiny_model.unimodal_loss(output, torch.tensor([0, 1, 2])).backward()
    assert tiny_model.unimodal_heads[0].weight.grad is not None
    assert tiny_model.projections[0][0].weight.grad is None


def test_admission_loss_requires_external_targets(tiny_model, batch):
    output = tiny_model(*batch)
    with pytest.raises(ValueError, match="OOF"):
        tiny_model.compute_loss(output, torch.tensor([0, 1, 2]))


def test_prior_relative_correction_and_arithmetic_fusion():
    tav = torch.tensor([[.7, .2, .1], [.1, .3, .6]])
    av = torch.tensor([[.2, .3, .5], [.5, .3, .2]])
    prior = torch.tensor([.2, .3, .5])
    gate = torch.tensor([.8, .6])
    actual = correct_predictions(tav, av, prior, gate)
    expected = ((tav + 1e-9).log() + gate[:, None] * ((av + 1e-9).log() - (prior + 1e-9).log())).softmax(-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(correct_predictions(tav, av, prior, gate, torch.zeros(2)), tav, rtol=0, atol=0)
    torch.testing.assert_close(correct_predictions(tav, prior.expand_as(tav), prior, gate), tav)
    torch.testing.assert_close(arithmetic_fusion(tav, actual, .3), .7 * tav + .3 * actual)
    learned = PriorRelativeCorrection(prior)(tav, av, torch.tensor([1, 0]))
    assert learned["statistics"].shape == (2, 6)
    assert learned["gate"][1] == 0


def test_output_composition_matches_formulas():
    logits = torch.tensor([[.4, -.2, 1.]])
    delta = torch.tensor([[.1, .2, -.1]])
    bias = torch.tensor([.2, -.1, -.1])
    torch.testing.assert_close(calibrate_context(logits, delta, bias, 2.), ((logits + delta + bias) / 2).softmax(-1))
    p, q = logits.softmax(-1), torch.tensor([[.2, .3, .5]])
    torch.testing.assert_close(emotiontalk_output(p, q), .85 * (p.log() / 3).softmax(-1) + .15 * q)
    with pytest.raises(ValueError, match="centered"):
        calibrate_context(logits, centered_class_bias=torch.ones(3))
