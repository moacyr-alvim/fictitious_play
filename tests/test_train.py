import pytest
import torch

from fictitious_play.benchmark import equilibrium_bid_lambda, equilibrium_bid_risk_averse
from fictitious_play.critic import BetaCritic, MonotonicNNCritic, SplineCritic
from fictitious_play.train import (
    FictitiousPlayTrainer,
    foc_residual,
    foc_residual_opponent,
    opponent_inverse_and_slope,
)


def _residual_grad_norm(trainer: FictitiousPlayTrainer, agent_idx: int, lam: float,
                         n: int = 4096) -> float:
    actor = trainer.agents[agent_idx]
    critic = trainer.critic_for(agent_idx)
    v = torch.rand(n, 1)
    b = actor(v)
    p_win = critic.predict_win_prob(b.squeeze(1)).unsqueeze(1)
    loss = -(p_win * (v - (1 - lam) * b) - lam * b).mean()
    grad = torch.autograd.grad(loss, [p for p in actor.parameters() if p.requires_grad])
    flat = torch.cat([g.reshape(-1) for g in grad])
    return flat.norm().item()


def test_newton_predict_step_reduces_residual_gradient():
    torch.manual_seed(0)
    trainer = FictitiousPlayTrainer(
        n_agents=2, n_rounds=20, sample_size=20_000, max_steps=400, patience=40,
        belief_decay=0.9, payoff_eval_samples=5_000, critic_n_quantiles=200,
        payment_share=0.0, seed=0,
    )
    trainer.run(verbose=False)  # converge at lambda=0 first

    new_lambda = 0.3
    before = max(_residual_grad_norm(trainer, i, new_lambda) for i in range(2))
    trainer.newton_predict_step(new_lambda)
    after = max(_residual_grad_norm(trainer, i, new_lambda) for i in range(2))

    assert after < before


def test_newton_predict_step_does_not_saturate_actor():
    """Regression test: a homotopy run observed the Newton step driving a
    hidden block's pre-sigmoid logit to roughly -140, saturating the
    sigmoid to exactly 0 in floating point and permanently killing the
    agent's gradient (it bid 0 for every valuation from then on, and
    falsely looked "converged" since a constant payoff is trivially
    stable). The line search alone didn't catch it because a saturated
    region has near-zero gradient too. Guard against this directly: even
    with damping too small to be a sane choice on its own, the trust
    region should keep the step from ever driving |raw| this large."""
    torch.manual_seed(0)
    trainer = FictitiousPlayTrainer(
        n_agents=2, n_rounds=20, sample_size=20_000, max_steps=400, patience=40,
        belief_decay=0.9, payoff_eval_samples=5_000, critic_n_quantiles=200,
        payment_share=0.0, seed=0,
    )
    trainer.run(verbose=False)

    trainer.newton_predict_step(0.9, damping=1e-6)

    v = torch.linspace(0, 1, 11).unsqueeze(1)
    for actor in trainer.agents:
        raw = actor.bias.expand(v.shape[0], 1)
        for block in actor.blocks:
            raw = raw + block(v)
        assert raw.abs().max().item() < 50


def test_foc_residual_vanishes_at_true_equilibrium():
    """Correctness check for the PINN residual formula, independent of
    training: if the critic is fit on bids actually drawn from the true
    equilibrium strategy, the residual should be ~0 when evaluated at the
    equilibrium bid function itself (for first-price, all-pay, and an
    intermediate payment_share), since that's exactly the FOC the
    equilibrium satisfies by construction."""
    torch.manual_seed(0)
    n_agents = 2
    # all-pay's FOC depends purely on the critic's *derivative* (no level
    # term to anchor it, see docs/findings_all_pay.md), so its residual is
    # intrinsically noisier here even with a critic fit on true-equilibrium
    # bids — a looser tolerance at lam=1.0 is the expected signature of
    # that known sensitivity, not slack in the test.
    tolerances = {0.0: 0.1, 0.4: 0.1, 1.0: 0.2}
    for lam, tol in tolerances.items():
        opponent_v = torch.rand(1_000_000)
        opponent_bids = equilibrium_bid_lambda(opponent_v, n_agents, lam)
        critic = SplineCritic(n_quantiles=3000)
        critic.fit(opponent_bids.numpy())

        v = torch.linspace(0.05, 0.95, 50).unsqueeze(1)
        b = equilibrium_bid_lambda(v, n_agents, lam).clone().requires_grad_(True)
        p_win = critic.predict_win_prob(b.squeeze(1)).unsqueeze(1)
        residual = foc_residual(p_win, b, v, lam)

        assert residual.abs().max().item() < tol


def test_foc_residual_opponent_vanishes_at_true_equilibrium():
    """Like test_foc_residual_vanishes_at_true_equilibrium, but using the
    opponent-actor-based residual (no critic, no sampling): the opponent
    is a plain function playing the exact equilibrium bid. With no
    sampling noise in the loop, the residual should be much closer to
    zero than the critic-based version (bisection precision only)."""
    n_agents = 2
    for lam in (0.0, 0.4, 1.0):
        def opponent_actor(v, lam=lam):
            return equilibrium_bid_lambda(v, n_agents, lam)

        v = torch.linspace(0.05, 0.95, 50).unsqueeze(1)
        b = equilibrium_bid_lambda(v, n_agents, lam).clone().requires_grad_(True)
        residual = foc_residual_opponent(b, v, lam, [opponent_actor])

        assert residual.abs().max().item() < 1e-3


def test_foc_residual_opponent_vanishes_at_true_equilibrium_n_agents():
    """Same correctness check with N=5 (4 opponents), all playing the
    same symmetric equilibrium strategy — exercises the product-rule
    generalization of P(win|b) and its derivative over multiple
    opponents, not just the single-opponent (N=2) case."""
    n_agents = 5
    for lam in (0.0, 0.5, 1.0):
        def opponent_actor(v, lam=lam):
            return equilibrium_bid_lambda(v, n_agents, lam)

        v = torch.linspace(0.05, 0.95, 50).unsqueeze(1)
        b = equilibrium_bid_lambda(v, n_agents, lam).clone().requires_grad_(True)
        residual = foc_residual_opponent(b, v, lam, [opponent_actor] * (n_agents - 1))

        assert residual.abs().max().item() < 1e-2


def test_opponent_inverse_and_slope_bounded_past_opponents_range():
    """Regression test: a homotopy run observed a mover bidding just past
    the opponent's current achievable max getting back a huge phi' (the
    bisection saturates at the boundary, where the opponent's slope can
    be tiny), escalating further next step and saturating at b(v)=v. Bids
    beyond the opponent's [actor(0), actor(1)] range must get phi'=0 (flat,
    already-certain win), not a blown-up slope from extrapolating the
    inversion past where the opponent actually reaches."""
    def opponent_actor(v):
        return 0.3 * v  # achievable range is [0, 0.3]

    b = torch.tensor([[0.5]], requires_grad=True)  # far outside [0, 0.3]
    phi_val, phi_prime = opponent_inverse_and_slope(opponent_actor, b)

    assert phi_val.item() == 1.0
    assert phi_prime.item() == 0.0


def test_opponent_ema_decay_schedule_interpolates_with_payment_share():
    trainer = FictitiousPlayTrainer(
        n_agents=2, pinn_opponent_weight=1.0, payment_share=0.0,
        opponent_ema_decay_schedule=(0.85, 0.995), seed=0,
    )
    assert trainer._current_opponent_ema_decay() == pytest.approx(0.85)

    trainer.payment_share = 0.5
    assert trainer._current_opponent_ema_decay() == pytest.approx(0.9225)

    trainer.payment_share = 1.0
    assert trainer._current_opponent_ema_decay() == pytest.approx(0.995)


def test_add_agents_clones_weights_and_extends_bookkeeping():
    trainer = FictitiousPlayTrainer(n_agents=2, sample_size=100, seed=0)
    v = torch.linspace(0, 1, 11).unsqueeze(1)
    with torch.no_grad():
        original_bids = trainer.agents[0](v)

    trainer.add_agents(2, copy_from=0)

    assert trainer.n_agents == 4
    assert len(trainer.agents) == 4
    assert len(trainer.optimizers) == 4
    assert len(trainer.ema_agents) == 4
    assert len(trainer.belief_buffers) == 4

    for idx in (2, 3):
        with torch.no_grad():
            cloned_bids = trainer.agents[idx](v)
        assert torch.allclose(cloned_bids, original_bids)
        assert not any(p.requires_grad for p in trainer.ema_agents[idx].parameters())

    assert trainer.growth_rounds == [0]


def test_critic_type_beta_wires_beta_critic_and_trains():
    trainer = FictitiousPlayTrainer(
        n_agents=2, n_rounds=3, sample_size=5_000, max_steps=100, patience=20,
        critic_type="beta", max_diff_threshold=None, payoff_rel_tol=None, seed=0,
    )
    critic = trainer.critic_for(0)
    assert isinstance(critic, BetaCritic)

    history = trainer.run(verbose=False)
    assert len(history) == 3


def test_critic_type_rejects_unknown_value():
    with pytest.raises(ValueError):
        FictitiousPlayTrainer(n_agents=2, critic_type="not_a_real_type")


def test_critic_type_nn_wires_persistent_warm_started_critic():
    trainer = FictitiousPlayTrainer(
        n_agents=2, n_rounds=3, sample_size=5_000, max_steps=100, patience=20,
        critic_type="nn", max_diff_threshold=None, payoff_rel_tol=None, seed=0,
    )
    critic = trainer.critic_for(0)
    assert isinstance(critic, MonotonicNNCritic)
    # same object every time (warm-started), not rebuilt from scratch.
    assert critic is trainer.nn_critics[0]
    assert trainer.critic_for(0) is critic

    history = trainer.run(verbose=False)
    assert len(history) == 3


def test_add_agents_extends_nn_critics():
    trainer = FictitiousPlayTrainer(n_agents=2, critic_type="nn", sample_size=100, seed=0)
    trainer.add_agents(2, copy_from=0)
    assert len(trainer.nn_critics) == 4
    assert all(isinstance(c, MonotonicNNCritic) for c in trainer.nn_critics)


def test_risk_aversion_rejects_n_agents_other_than_two():
    with pytest.raises(ValueError):
        FictitiousPlayTrainer(n_agents=3, risk_aversion=[1.0, 1.0, 1.0])


def test_risk_aversion_rejects_non_first_price():
    with pytest.raises(ValueError):
        FictitiousPlayTrainer(n_agents=2, auction_type="all_pay", risk_aversion=[0.5, 1.0])
    with pytest.raises(ValueError):
        FictitiousPlayTrainer(n_agents=2, payment_share=0.5, risk_aversion=[0.5, 1.0])


def test_risk_aversion_rejects_pinn_weight():
    with pytest.raises(ValueError):
        FictitiousPlayTrainer(n_agents=2, risk_aversion=[0.5, 1.0], pinn_weight=0.1)
    with pytest.raises(ValueError):
        FictitiousPlayTrainer(n_agents=2, risk_aversion=[0.5, 1.0], pinn_opponent_weight=0.1)


def test_risk_aversion_benchmark_metrics_use_per_agent_target():
    """With different rho per agent, agent 0 and agent 1 are benchmarked
    against *different* theoretical bid curves (the asymmetric equilibrium
    solved in benchmark.equilibrium_bid_risk_averse), not a single shared
    curve — the more risk-averse agent's curve sits above the less
    risk-averse agent's for the same valuation (see
    test_risk_averse_more_averse_agent_bids_more_aggressively)."""
    trainer = FictitiousPlayTrainer(n_agents=2, risk_aversion=[0.3, 1.0], sample_size=100, seed=0)

    class _ZeroActor:
        def __call__(self, v):
            return torch.zeros_like(v)

    trainer.agents = [_ZeroActor(), _ZeroActor()]
    # both agents now bid b=0 everywhere, so mse/max_diff are driven purely
    # by how far 0 is from each agent's own target curve.
    mse0, _ = trainer._benchmark_metrics(0)
    mse1, _ = trainer._benchmark_metrics(1)
    # both agents' curves converge to the *same* shared upper bid bound at
    # v=1 (the common-support property the equilibrium is solved under — see
    # _solve_asymmetric_risk_averse_equilibrium's docstring), so max_diff
    # against an all-zero actor coincides for both; mse (averaged over the
    # whole grid) still tells the curves apart, since the more risk-averse
    # agent's curve rises faster and caps out earlier.
    assert mse0 != mse1
    v_mid = torch.tensor([[0.5]])
    target0 = equilibrium_bid_risk_averse(v_mid, 0, tuple(trainer.risk_aversion)).item()
    target1 = equilibrium_bid_risk_averse(v_mid, 1, tuple(trainer.risk_aversion)).item()
    assert target0 != target1


def test_risk_aversion_trains_without_error():
    trainer = FictitiousPlayTrainer(
        n_agents=2, n_rounds=3, sample_size=5_000, max_steps=100, patience=20,
        risk_aversion=[0.5, 1.0], max_diff_threshold=None, payoff_rel_tol=None, seed=0,
    )
    history = trainer.run(verbose=False)
    assert len(history) == 3


def test_set_risk_aversion_updates_coefficients_and_marks_stage_boundary():
    trainer = FictitiousPlayTrainer(n_agents=2, risk_aversion=[1.0, 1.0], sample_size=100, seed=0)
    trainer.history.append({"round": 0})  # simulate a stage already having run

    trainer.set_risk_aversion([0.7, 1.0])

    assert trainer.risk_aversion == [0.7, 1.0]
    assert trainer.growth_rounds == [1]


def test_set_risk_aversion_rejects_wrong_length():
    trainer = FictitiousPlayTrainer(n_agents=2, risk_aversion=[1.0, 1.0], sample_size=100, seed=0)
    with pytest.raises(ValueError):
        trainer.set_risk_aversion([0.5])


def test_set_risk_aversion_requires_risk_aversion_already_set():
    trainer = FictitiousPlayTrainer(n_agents=2, sample_size=100, seed=0)
    with pytest.raises(ValueError):
        trainer.set_risk_aversion([0.5, 1.0])
