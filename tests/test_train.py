import torch

from fictitious_play.train import FictitiousPlayTrainer


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
