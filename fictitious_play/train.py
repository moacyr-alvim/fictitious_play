from collections import deque

import numpy as np
import torch

from .actor import Actor
from .benchmark import equilibrium_bid
from .critic import KDECritic


class FictitiousPlayTrainer:
    """Alternating best-response training for a 2-agent first-price auction.

    Each "round" is a single agent's turn: that agent's opponent is held
    fixed, a KDECritic is fit on a fresh sample of the opponent's bids, and
    the moving agent is trained by gradient ascent on its expected payoff
    E_v[(v - b) * P(win | b)] until an EMA-smoothed loss plateaus (or
    max_steps is reached, as a safety cap). Agents alternate turns, so
    n_rounds=200 means 100 best-response updates per agent.
    """

    def __init__(self, n_rounds: int = 200, sample_size: int = 300_000,
                 batch_size: int = 1024, max_steps: int = 20_000,
                 patience: int = 200, tol: float = 1e-6, lr: float = 1e-3,
                 ema_decay: float = 0.98, eval_grid_size: int = 200,
                 seed: int | None = None):
        self.n_rounds = n_rounds
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.patience = patience
        self.tol = tol
        self.ema_decay = ema_decay
        self.eval_grid = torch.linspace(0, 1, eval_grid_size).unsqueeze(1)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.agents = [Actor(), Actor()]
        self.optimizers = [torch.optim.Adam(a.parameters(), lr=lr) for a in self.agents]
        self.history: list[dict] = []

    def _sample_opponent_bids(self, opponent: Actor) -> np.ndarray:
        with torch.no_grad():
            v = torch.rand(self.sample_size, 1)
            b = opponent(v)
        return b.squeeze(1).numpy()

    def _best_response(self, mover_idx: int, critic: KDECritic) -> int:
        actor = self.agents[mover_idx]
        optimizer = self.optimizers[mover_idx]

        ema_loss = None
        window = deque(maxlen=self.patience)
        steps_taken = 0

        for step in range(self.max_steps):
            v = torch.rand(self.batch_size, 1)
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
            target = equilibrium_bid(self.eval_grid)
            return torch.mean((pred - target) ** 2).item()

    def run(self, verbose: bool = True) -> list[dict]:
        for round_idx in range(self.n_rounds):
            mover_idx = round_idx % 2
            opponent_idx = 1 - mover_idx

            opponent_bids = self._sample_opponent_bids(self.agents[opponent_idx])
            critic = KDECritic()
            critic.fit(opponent_bids)

            steps_taken = self._best_response(mover_idx, critic)

            mse = [self._benchmark_mse(self.agents[i]) for i in range(2)]
            self.history.append({
                "round": round_idx,
                "mover": mover_idx,
                "steps": steps_taken,
                "mse_agent_0": mse[0],
                "mse_agent_1": mse[1],
            })
            if verbose:
                print(f"Round {round_idx:3d} | mover=agent{mover_idx} | steps={steps_taken:5d} "
                      f"| MSE0={mse[0]:.6f} | MSE1={mse[1]:.6f}")

        return self.history
