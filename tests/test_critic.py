import numpy as np
import pytest
import torch

from fictitious_play.benchmark import equilibrium_bid_lambda
from fictitious_play.critic import BetaCritic, MixtureBetaCritic, MonotonicNNCritic, SplineCritic


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


def test_mixture_beta_critic_matches_uniform_distribution():
    rng = np.random.default_rng(0)
    bids = rng.uniform(0.0, 0.5, size=300_000)

    critic = MixtureBetaCritic()
    critic.fit(bids)

    query = torch.tensor([0.0, 0.1, 0.25, 0.4, 0.5])
    predicted = critic.predict_win_prob(query)
    expected = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
    assert torch.allclose(predicted, expected, atol=0.02)


def test_mixture_beta_critic_is_monotonic():
    """Checked over several random initial guesses (not just one fit): a
    mixture of valid CDFs with nonnegative weights summing to 1 is monotonic
    regardless of the particular alpha_k/pi_k values, since mixing
    probability distributions can't un-order them (see the class
    docstring) — this isn't something fit() needs to achieve, it holds for
    any alpha_k > 0."""
    rng = np.random.default_rng(1)
    query = torch.linspace(0.0, 1.0, 200)
    for trial in range(5):
        critic = MixtureBetaCritic()
        critic.alpha_np = rng.uniform(0.1, 5.0, size=2)
        critic.pi_np = np.array([0.3, 0.7])
        critic.alpha = torch.tensor(critic.alpha_np, dtype=torch.float32)
        critic.pi = torch.tensor(critic.pi_np, dtype=torch.float32)
        critic.scale = torch.tensor(1.0)

        predicted = critic.predict_win_prob(query)
        diffs = predicted[1:] - predicted[:-1]
        assert torch.all(diffs >= -1e-6)


def test_mixture_beta_critic_gradient_flows_through_bid():
    rng = np.random.default_rng(2)
    bids = rng.uniform(0.0, 0.5, size=50_000)

    critic = MixtureBetaCritic()
    critic.fit(bids)

    b = torch.tensor([0.3], requires_grad=True)
    p_win = critic.predict_win_prob(b)
    p_win.backward()

    assert b.grad is not None
    assert b.grad.item() > 0


def test_mixture_beta_critic_warm_start_adapts_to_shifted_distribution():
    """fit() reuses the previous call's alpha/pi as the EM's initial guess
    (see the class docstring) — check a second fit() on a different (wider)
    distribution actually moves the prediction."""
    rng = np.random.default_rng(3)
    critic = MixtureBetaCritic(em_iters=50)
    critic.fit(rng.uniform(0.0, 0.3, size=20_000))
    first = critic.predict_win_prob(torch.tensor([0.5])).item()

    critic.fit(rng.uniform(0.0, 0.8, size=20_000))
    second = critic.predict_win_prob(torch.tensor([0.5])).item()

    assert first > 0.9  # 0.5 is well past the first (narrower) distribution's range
    assert second < 0.8  # ...but well inside the second (wider) one's range
    assert abs(second - 0.625) < 0.1  # true CDF value at 0.5 under Uniform(0, 0.8)


def test_mixture_beta_critic_state_dict_round_trip():
    rng = np.random.default_rng(4)
    bids = rng.uniform(0.0, 0.5, size=10_000)

    critic = MixtureBetaCritic(em_iters=20)
    critic.fit(bids)
    query = torch.linspace(0.0, 1.0, 20)
    before = critic.predict_win_prob(query)

    state = critic.state_dict()
    restored = MixtureBetaCritic(em_iters=20)
    restored.load_state_dict(state)
    after = restored.predict_win_prob(query)

    assert torch.allclose(before, after)


def test_mixture_beta_critic_fits_bimodal_shape_better_than_single_beta():
    """The whole point of mixing K>1 components: a shape that isn't a
    single power law (e.g. two distinct sub-populations of opponent bids)
    should be fit measurably better by the mixture than by BetaCritic's
    single component."""
    rng = np.random.default_rng(5)
    low = rng.uniform(0.0, 0.2, size=25_000)
    high = rng.uniform(0.6, 1.0, size=25_000)
    bids = np.concatenate([low, high])

    query = torch.linspace(0.0, 1.0, 300)
    empirical = torch.tensor(np.searchsorted(np.sort(bids), query.numpy()) / len(bids))

    single = BetaCritic()
    single.fit(bids)
    single_err = (single.predict_win_prob(query) - empirical).abs().max().item()

    mixture = MixtureBetaCritic()
    mixture.fit(bids)
    mixture_err = (mixture.predict_win_prob(query) - empirical).abs().max().item()

    assert mixture_err < single_err


def test_monotonic_nn_critic_matches_uniform_distribution():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    bids = rng.uniform(0.0, 0.5, size=50_000)

    critic = MonotonicNNCritic()
    critic.fit(bids)

    query = torch.tensor([0.0, 0.1, 0.25, 0.4, 0.5])
    with torch.no_grad():
        predicted = critic.predict_win_prob(query)
    expected = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
    assert torch.allclose(predicted, expected, atol=0.05)


def test_monotonic_nn_critic_is_monotonic_by_construction():
    """Checked over a batch of *random* re-initializations (not just the
    one fit to a real distribution above): monotonicity has to hold for
    any weights the optimizer could possibly reach, since it comes from
    the architecture (positive weights, monotonic activations), not from
    what fit() converges to."""
    torch.manual_seed(1)
    query = torch.linspace(0.0, 1.0, 200)
    for _ in range(5):
        critic = MonotonicNNCritic()
        with torch.no_grad():
            predicted = critic.predict_win_prob(query)
        diffs = predicted[1:] - predicted[:-1]
        assert torch.all(diffs >= -1e-6)


def test_monotonic_nn_critic_gradient_flows_through_bid():
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    bids = rng.uniform(0.0, 0.5, size=50_000)

    critic = MonotonicNNCritic()
    critic.fit(bids)

    b = torch.tensor([0.3], requires_grad=True)
    p_win = critic.predict_win_prob(b)
    p_win.backward()

    assert b.grad is not None
    assert b.grad.item() > 0


def test_monotonic_nn_critic_warm_start_adapts_to_shifted_distribution():
    """fit() is meant to be called repeatedly on the same instance as the
    underlying distribution drifts round to round (see the class
    docstring), continuing from the current weights/optimizer state rather
    than restarting — check that a second fit() call on a different
    (wider) distribution actually moves the prediction, rather than being
    stuck at the first fit's solution."""
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    critic = MonotonicNNCritic(fit_steps=500)
    critic.fit(rng.uniform(0.0, 0.3, size=20_000))
    with torch.no_grad():
        first = critic.predict_win_prob(torch.tensor([0.5])).item()

    critic.fit(rng.uniform(0.0, 0.8, size=20_000))
    with torch.no_grad():
        second = critic.predict_win_prob(torch.tensor([0.5])).item()

    assert first > 0.9  # 0.5 is well past the first (narrower) distribution's range
    assert second < 0.8  # ...but well inside the second (wider) one's range
    assert abs(second - 0.625) < 0.1  # true CDF value at 0.5 under Uniform(0, 0.8)


def test_monotonic_nn_critic_state_dict_round_trip():
    rng = np.random.default_rng(4)
    bids = rng.uniform(0.0, 0.5, size=10_000)

    critic = MonotonicNNCritic(fit_steps=100)
    critic.fit(bids)
    query = torch.linspace(0.0, 1.0, 20)
    with torch.no_grad():
        before = critic.predict_win_prob(query)

    state = critic.state_dict()
    restored = MonotonicNNCritic(fit_steps=100)
    restored.load_state_dict(state)
    with torch.no_grad():
        after = restored.predict_win_prob(query)

    assert torch.allclose(before, after)
