import numpy as np
import torch

from fictitious_play.critic import SplineCritic


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
