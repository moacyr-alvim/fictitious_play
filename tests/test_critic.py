import numpy as np
import torch

from fictitious_play.critic import KDECritic


def test_cdf_matches_uniform_distribution():
    rng = np.random.default_rng(0)
    bids = rng.uniform(0.0, 0.5, size=300_000)

    critic = KDECritic()
    critic.fit(bids)

    query = torch.tensor([0.0, 0.1, 0.25, 0.4, 0.5])
    predicted = critic.predict_win_prob(query)
    expected = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])

    assert torch.allclose(predicted, expected, atol=0.02)


def test_cdf_is_monotonic():
    rng = np.random.default_rng(1)
    bids = rng.uniform(0.0, 0.5, size=50_000)

    critic = KDECritic()
    critic.fit(bids)

    query = torch.linspace(0.0, 1.0, 100)
    predicted = critic.predict_win_prob(query)
    diffs = predicted[1:] - predicted[:-1]
    assert torch.all(diffs >= -1e-6)


def test_gradient_flows_through_bid():
    rng = np.random.default_rng(2)
    bids = rng.uniform(0.0, 0.5, size=50_000)

    critic = KDECritic()
    critic.fit(bids)

    b = torch.tensor([0.3], requires_grad=True)
    p_win = critic.predict_win_prob(b)
    p_win.backward()

    assert b.grad is not None
    assert b.grad.item() > 0
