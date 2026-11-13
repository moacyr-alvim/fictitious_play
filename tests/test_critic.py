import numpy as np
import pytest
import torch

from fictitious_play.benchmark import equilibrium_bid_lambda
from fictitious_play.critic import BetaCritic, SplineCritic


def test_cdf_matches_uniform_distribution():
    rng = np.random.default_rng(0)
    bids = rng.uniform(0.0, 0.5, size=300_000)

    critic = SplineCritic()
    critic.fit(bids)

    query = torch.tensor([0.0, 0.1, 0.25, 0.4, 0.5])
    predicted = critic.predict_win_prob(query)
    expected = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])

    assert torch.allclose(predicted, expected, atol=0.02)


def test_cdf_is_monotonic():
    rng = np.random.default_rng(1)
    bids = rng.uniform(0.0, 0.5, size=50_000)

    critic = SplineCritic()
    critic.fit(bids)

    query = torch.linspace(0.0, 1.0, 100)
    predicted = critic.predict_win_prob(query)
    diffs = predicted[1:] - predicted[:-1]
    assert torch.all(diffs >= -1e-6)


def test_gradient_flows_through_bid():
    rng = np.random.default_rng(2)
    bids = rng.uniform(0.0, 0.5, size=50_000)

    critic = SplineCritic()
    critic.fit(bids)

    b = torch.tensor([0.3], requires_grad=True)
    p_win = critic.predict_win_prob(b)
    p_win.backward()

    assert b.grad is not None
    assert b.grad.item() > 0


def test_handles_tied_bid_values():
    bids = np.concatenate([np.zeros(1000), np.full(1000, 0.3), np.ones(1000) * 0.5])

    critic = SplineCritic()
    critic.fit(bids)

    query = torch.tensor([0.0, 0.3, 0.5])
    predicted = critic.predict_win_prob(query)
    assert torch.all(predicted >= 0.0) and torch.all(predicted <= 1.0)
    assert torch.all(predicted[1:] - predicted[:-1] >= -1e-6)


def test_beta_critic_matches_uniform_distribution():
    """Beta(alpha=1, 1) is exactly Uniform, so a fit on Uniform[0, 0.5]
    bids should recover alpha~1 and reproduce the linear CDF."""
    rng = np.random.default_rng(0)
    bids = rng.uniform(0.0, 0.5, size=300_000)

    critic = BetaCritic()
    critic.fit(bids)

    assert critic.alpha.item() == pytest.approx(1.0, abs=0.02)

    query = torch.tensor([0.0, 0.1, 0.25, 0.4, 0.5])
    predicted = critic.predict_win_prob(query)
    expected = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
    assert torch.allclose(predicted, expected, atol=0.02)


def test_beta_critic_recovers_exact_alpha_at_equilibrium_endpoints():
    """The symmetric equilibrium bid is b*(v) = c*v^p (p=1 at first-price,
    p=n_agents at all-pay). For v ~ U(0, 1), b*/c ~ Beta(1/p, 1) exactly —
    this is the theoretical motivation for BetaCritic. Check the fitted
    alpha matches 1/p closely at both ends."""
    n_agents = 5
    for lam, expected_alpha in [(0.0, 1.0), (1.0, 1.0 / n_agents)]:
        v = torch.rand(500_000, dtype=torch.float64)
        bids = equilibrium_bid_lambda(v, n_agents, lam).numpy()

        critic = BetaCritic()
        critic.fit(bids)

        assert critic.alpha.item() == pytest.approx(expected_alpha, abs=0.01)


def test_beta_critic_is_monotonic():
    rng = np.random.default_rng(1)
    bids = rng.uniform(0.0, 0.5, size=50_000)

    critic = BetaCritic()
    critic.fit(bids)

    query = torch.linspace(0.0, 1.0, 100)
    predicted = critic.predict_win_prob(query)
    diffs = predicted[1:] - predicted[:-1]
    assert torch.all(diffs >= -1e-6)


def test_beta_critic_gradient_flows_through_bid():
    rng = np.random.default_rng(2)
    bids = rng.uniform(0.0, 0.5, size=50_000)

    critic = BetaCritic()
    critic.fit(bids)

    b = torch.tensor([0.3], requires_grad=True)
    p_win = critic.predict_win_prob(b)
    p_win.backward()

    assert b.grad is not None
    assert b.grad.item() > 0


def test_beta_critic_handles_degenerate_zero_bids():
    """All bids at 0 (e.g. an agent that hasn't learned to bid anything
    yet) must not blow up the fit (scale=0 would otherwise divide by
    zero)."""
    bids = np.zeros(1000)

    critic = BetaCritic()
    critic.fit(bids)

    query = torch.tensor([0.0, 0.5, 1.0])
    predicted = critic.predict_win_prob(query)
    assert torch.all(torch.isfinite(predicted))
