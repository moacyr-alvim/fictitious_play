from collections import deque

import numpy as np
import torch

from .actor import Actor
from .benchmark import equilibrium_bid, theoretical_expected_payoff
from .critic import KDECritic


class FictitiousPlayTrainer:
    """Alternating best-response training for an N-agent first-price auction.

    Each agent keeps a "belief buffer": a fixed-size pool of its own past
    bids. Every time an agent finishes a best response, a `belief_decay`
    fraction of its buffer is kept (a random subsample) and the rest is
    replaced by fresh bids from its updated policy — so a bid from k of the
    agent's own updates ago survives with probability roughly
    belief_decay**k. belief_decay=0 recovers plain best-response dynamics
    (opponents seen only through their latest snapshot); belief_decay close
    to 1 approximates classical fictitious play (belief = time-averaged
    history of opponent play), which damps the best-response cycles that
    appear with more than two agents.

    Each "round" is a single agent's turn: the KDECritic it best-responds to
    is fit on the max bid drawn from the *belief buffers* of the other N-1
    agents (winning means beating all of them), the agent is trained by
    gradient ascent on E_v[(v - b) * P(win | b)] until an EMA-smoothed loss
    plateaus (or max_steps is reached, as a safety cap), and then its own
    belief buffer is refreshed. Agents take turns round-robin, so
    n_rounds=200 with n_agents=4 means 50 best-response updates per agent.
    """

    def __init__(self, n_agents: int = 2, n_rounds: int = 200, sample_size: int = 300_000,
                 batch_size: int = 1024, max_steps: int = 20_000,
                 patience: int = 200, tol: float = 1e-6, lr: float = 1e-3,
                 ema_decay: float = 0.98, eval_grid_size: int = 200,
                 belief_decay: float = 0.9, payoff_eval_samples: int = 100_000,
                 seed: int | None = None, device: str | None = None):
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.patience = patience
        self.tol = tol
        self.ema_decay = ema_decay
        self.belief_decay = belief_decay
        self.payoff_eval_samples = payoff_eval_samples
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.eval_grid = torch.linspace(0, 1, eval_grid_size, device=self.device).unsqueeze(1)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.agents = [Actor().to(self.device) for _ in range(n_agents)]
        self.optimizers = [torch.optim.Adam(a.parameters(), lr=lr) for a in self.agents]
        self.belief_buffers = [self._fresh_bids(i) for i in range(n_agents)]
        self.history: list[dict] = []

    def _fresh_bids(self, agent_idx: int) -> np.ndarray:
        with torch.no_grad():
            v = torch.rand(self.sample_size, 1, device=self.device)
            b = self.agents[agent_idx](v)
        return b.squeeze(1).cpu().numpy()

    def _update_belief(self, agent_idx: int) -> None:
        old = self.belief_buffers[agent_idx]
        fresh = self._fresh_bids(agent_idx)
        capacity = len(old)
        n_keep = int(round(capacity * self.belief_decay))
        n_new = capacity - n_keep

        kept = np.random.choice(old, size=n_keep, replace=False) if n_keep > 0 else old[:0]
        new_part = np.random.choice(fresh, size=n_new, replace=True)

        updated = np.concatenate([kept, new_part])
        np.random.shuffle(updated)
        self.belief_buffers[agent_idx] = updated

    def _sample_max_belief_bids(self, opponent_indices: list[int]) -> np.ndarray:
        draws = [np.random.choice(self.belief_buffers[idx], size=self.sample_size, replace=True)
                 for idx in opponent_indices]
        return np.maximum.reduce(draws)

    def critic_for(self, mover_idx: int) -> KDECritic:
        """Fits (without training) the critic mover_idx currently faces,
        i.e. the smoothed CDF of the max bid drawn from the other agents'
        belief buffers."""
        opponent_indices = [i for i in range(self.n_agents) if i != mover_idx]
        opponent_bids = self._sample_max_belief_bids(opponent_indices)
        critic = KDECritic(device=self.device)
        critic.fit(opponent_bids)
        return critic

    def _best_response(self, mover_idx: int, critic: KDECritic) -> int:
        actor = self.agents[mover_idx]
        optimizer = self.optimizers[mover_idx]

        ema_loss = None
        window = deque(maxlen=self.patience)
        steps_taken = 0

        for step in range(self.max_steps):
            v = torch.rand(self.batch_size, 1, device=self.device)
            b = actor(v)
            p_win = critic.predict_win_prob(b.squeeze(1)).unsqueeze(1)
            loss = -((v - b) * p_win).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            ema_loss = loss_val if ema_loss is None else (
                self.ema_decay * ema_loss + (1 - self.ema_decay) * loss_val
            )
            window.append(ema_loss)
            steps_taken = step + 1

            if len(window) == self.patience and abs(window[0] - window[-1]) < self.tol:
                break

        return steps_taken

    def _benchmark_mse(self, actor: Actor) -> float:
        with torch.no_grad():
            pred = actor(self.eval_grid)
            target = equilibrium_bid(self.eval_grid, self.n_agents)
            return torch.mean((pred - target) ** 2).item()

    def _expected_payoff(self, agent_idx: int) -> float:
        """Monte Carlo expected payoff if the auction were run right now
        with every agent's *current* policy (not the belief buffers) —
        a direct estimate of the actual training objective, for comparison
        against the theoretical equilibrium payoff 1 / (N * (N + 1))."""
        opponent_indices = [i for i in range(self.n_agents) if i != agent_idx]
        n = self.payoff_eval_samples
        with torch.no_grad():
            v_self = torch.rand(n, 1, device=self.device)
            b_self = self.agents[agent_idx](v_self).squeeze(1)

            v_opp = torch.rand(n, len(opponent_indices), device=self.device)
            b_opp = torch.stack(
                [self.agents[idx](v_opp[:, k:k + 1]).squeeze(1)
                 for k, idx in enumerate(opponent_indices)],
                dim=1,
            )
            max_opp = b_opp.max(dim=1).values

            win = (b_self > max_opp).float()
            payoff = win * (v_self.squeeze(1) - b_self)
            return payoff.mean().item()

    def run(self, verbose: bool = True) -> list[dict]:
        for round_idx in range(self.n_rounds):
            mover_idx = round_idx % self.n_agents
            opponent_indices = [i for i in range(self.n_agents) if i != mover_idx]

            opponent_bids = self._sample_max_belief_bids(opponent_indices)
            critic = KDECritic(device=self.device)
            critic.fit(opponent_bids)

            steps_taken = self._best_response(mover_idx, critic)
            self._update_belief(mover_idx)

            mse = [self._benchmark_mse(self.agents[i]) for i in range(self.n_agents)]
            payoff = [self._expected_payoff(i) for i in range(self.n_agents)]
            self.history.append({
                "round": round_idx,
                "mover": mover_idx,
                "steps": steps_taken,
                "mse": mse,
                "payoff": payoff,
            })
            if verbose:
                mse_str = " ".join(f"MSE{i}={m:.6f}" for i, m in enumerate(mse))
                payoff_str = " ".join(f"P{i}={p:.5f}" for i, p in enumerate(payoff))
                theo = theoretical_expected_payoff(self.n_agents)
                print(f"Round {round_idx:3d} | mover=agent{mover_idx} | steps={steps_taken:5d} "
                      f"| {mse_str} | {payoff_str} | teórico={theo:.5f}")

        return self.history
