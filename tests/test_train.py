import pytest
import torch

from fictitious_play.benchmark import equilibrium_bid_lambda
from fictitious_play.critic import SplineCritic
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
