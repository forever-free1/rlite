import pytest
import torch

from rlite.algos.dapo import dapo_loss
from rlite.algos.gspo import gspo_loss
from rlite.algos.registry import get_objective


def _inputs():
    rewards = torch.tensor([1.0, 0.0])
    old = torch.tensor([[-0.5, -0.5], [-0.5, -0.5]])
    new = old.clone().requires_grad_(True)
    mask = torch.ones(2, 2, dtype=torch.bool)
    groups = torch.tensor([0, 0])
    advantages = torch.tensor([1.0, -1.0])
    return rewards, old, new, mask, groups, advantages


@pytest.mark.parametrize("name", ["grpo", "dapo", "gspo"])
def test_objective_registry(name):
    assert callable(get_objective(name))


def test_dapo_has_gradient_and_asymmetric_clip_metrics():
    rewards, old, new, mask, groups, advantages = _inputs()
    loss, metrics = dapo_loss(
        rewards, old, new, mask, groups, advantages=advantages,
        eps_clip=0.2, eps_high=0.28,
    )
    loss.backward()
    assert new.grad is not None
    assert metrics["clip_fraction"] == 0.0


def test_gspo_uses_length_normalized_sequence_ratio():
    rewards, old, new, mask, groups, advantages = _inputs()
    new = torch.tensor([[-0.4, -0.6], [-0.6, -0.4]], requires_grad=True)
    loss, metrics = gspo_loss(
        rewards, old, new, mask, groups, advantages=advantages, eps_clip=0.2
    )
    # Mean log-ratio is zero for both sequences, so both ratios are exactly one.
    assert metrics["sequence_ratio_mean"] == pytest.approx(1.0)
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert new.grad is not None


def test_gspo_padding_does_not_change_sequence_ratio():
    rewards, old, new, mask, groups, advantages = _inputs()
    mask[:, 1] = False
    new = torch.tensor([[-0.3, 20.0], [-0.7, -20.0]], requires_grad=True)
    _, metrics = gspo_loss(
        rewards, old, new, mask, groups, advantages=advantages, eps_clip=0.2
    )
    expected = (torch.exp(torch.tensor(0.2)) + torch.exp(torch.tensor(-0.2))) / 2
    assert metrics["sequence_ratio_mean"] == pytest.approx(expected.item())
