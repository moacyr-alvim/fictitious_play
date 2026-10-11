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
