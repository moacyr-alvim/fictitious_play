from collections import deque
from pathlib import Path

import numpy as np
import torch

from .actor import Actor
from .benchmark import equilibrium_bid, theoretical_expected_payoff
from .critic import SplineCritic


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

    Each "round" is a single agent's turn: the SplineCritic it best-responds to
    is fit on the max bid drawn from the *belief buffers* of the other N-1
    agents (winning means beating all of them), the agent is trained by
    gradient ascent on E_v[(v - b) * P(win | b)] until an EMA-smoothed loss
    plateaus (or max_steps is reached, as a safety cap), and then its own
    belief buffer is refreshed. Agents take turns round-robin, so
    n_rounds=200 with n_agents=4 means 50 best-response updates per agent.

    Training stops early (before n_rounds) as soon as either:
      - every agent's *maximum* absolute deviation from the known
        equilibrium bid function, over the evaluation grid, drops below
        max_diff_threshold (a sup-norm check — unlike a mean-squared error,
        it can't be satisfied while a whole region, e.g. high valuations,
        is still off), or
      - every agent's expected payoff has changed by less than
        payoff_rel_tol (relative) between consecutive rounds, for the last
        payoff_stability_window rounds in a row (a benchmark-free proxy for
        "near equilibrium", useful when no closed-form equilibrium exists).
    Either threshold can be set to None to disable that check. The mean
    squared error is still computed and logged every round as a diagnostic,
    but no longer drives the stopping decision.
    """

    def __init__(self, n_agents: int = 2, n_rounds: int = 200, sample_size: int = 300_000,
                 batch_size: int = 1024, max_steps: int = 20_000,
                 patience: int = 200, tol: float = 1e-6, lr: float = 1e-3,
                 ema_decay: float = 0.98, eval_grid_size: int = 200,
                 belief_decay: float = 0.9, payoff_eval_samples: int = 100_000,
                 critic_n_quantiles: int = 400, actor_hidden_size: int = 32,
                 max_diff_threshold: float | None = 0.02, payoff_rel_tol: float | None = 0.15,
                 payoff_stability_window: int = 10,
                 checkpoint_path: str | None = None, checkpoint_every: int = 20,
                 seed: int | None = None, device: str | None = None):
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.patience = patience
        self.tol = tol
        self.lr = lr
        self.ema_decay = ema_decay
        self.belief_decay = belief_decay
        self.payoff_eval_samples = payoff_eval_samples
        self.critic_n_quantiles = critic_n_quantiles
        self.max_diff_threshold = max_diff_threshold
        self.payoff_rel_tol = payoff_rel_tol
        self.payoff_stability_window = payoff_stability_window
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.eval_grid = torch.linspace(0, 1, eval_grid_size, device=self.device).unsqueeze(1)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.agents = [Actor(hidden_size=actor_hidden_size).to(self.device) for _ in range(n_agents)]
        self.optimizers = [torch.optim.Adam(a.parameters(), lr=lr) for a in self.agents]
        self.belief_buffers = [self._fresh_bids(i) for i in range(n_agents)]
        self.history: list[dict] = []
        self.growth_rounds: list[int] = []

    def _fresh_bids(self, agent_idx: int) -> np.ndarray:
        with torch.no_grad():
            v = torch.rand(self.sample_size, 1, device=self.device)
            b = self.agents[agent_idx](v)
        return b.squeeze(1).cpu().numpy()

    def grow_agents(self, new_hidden_size: int) -> None:
        """Grows every agent's actor to new_hidden_size hidden units,
        function-preserving (see Actor.grow): each agent computes exactly
        the same bids right after growing as right before. Optimizers are
        rebuilt fresh for the new parameters (Adam's momentum/variance
        state doesn't carry over meaning across a change in parameter
        shape). Belief buffers and history are left untouched, since
        they're just past bid values / metrics, independent of
        architecture."""
        self.agents = [agent.grow(new_hidden_size).to(self.device) for agent in self.agents]
        self.optimizers = [torch.optim.Adam(a.parameters(), lr=self.lr) for a in self.agents]
        self.growth_rounds.append(len(self.history))

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

    def critic_for(self, mover_idx: int) -> SplineCritic:
        """Fits (without training) the critic mover_idx currently faces,
        i.e. the smoothed CDF of the max bid drawn from the other agents'
        belief buffers."""
        opponent_indices = [i for i in range(self.n_agents) if i != mover_idx]
        opponent_bids = self._sample_max_belief_bids(opponent_indices)
        critic = SplineCritic(n_quantiles=self.critic_n_quantiles, device=self.device)
        critic.fit(opponent_bids)
        return critic

    def _best_response(self, mover_idx: int, critic: SplineCritic) -> int:
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

    def _benchmark_metrics(self, actor: Actor) -> tuple[float, float]:
        """Returns (mse, max_abs_diff) of the actor's bid function against
        the known equilibrium, over the evaluation grid. mse is logged only
        as a diagnostic; max_abs_diff (the sup-norm deviation) drives the
        stopping decision, since it cannot be satisfied while a whole
        region of valuations is still poorly fit — unlike an average."""
        with torch.no_grad():
            pred = actor(self.eval_grid)
            target = equilibrium_bid(self.eval_grid, self.n_agents)
            diff = pred - target
            mse = torch.mean(diff ** 2).item()
            max_abs_diff = torch.max(torch.abs(diff)).item()
        return mse, max_abs_diff

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

    def _max_diff_converged(self, max_diff: list[float]) -> bool:
        if self.max_diff_threshold is None:
            return False
        return all(d <= self.max_diff_threshold for d in max_diff)

    def _payoff_converged(self) -> bool:
        if self.payoff_rel_tol is None:
            return False
        if len(self.history) < self.payoff_stability_window + 1:
            return False

        recent = [h["payoff"] for h in self.history[-(self.payoff_stability_window + 1):]]
        for prev, curr in zip(recent, recent[1:]):
            for p_prev, p_curr in zip(prev, curr):
                denom = abs(p_prev) if abs(p_prev) > 1e-12 else 1e-12
                if abs(p_curr - p_prev) / denom >= self.payoff_rel_tol:
                    return False
        return True

    def save_checkpoint(self, round_idx: int, converged: bool = False) -> None:
        if self.checkpoint_path is None:
            return
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "round": round_idx,
            "n_agents": self.n_agents,
            "converged": converged,
            "agent_hidden_sizes": [a.hidden_size for a in self.agents],
            "agents": [a.state_dict() for a in self.agents],
            "optimizers": [o.state_dict() for o in self.optimizers],
            "belief_buffers": self.belief_buffers,
            "history": self.history,
            "growth_rounds": self.growth_rounds,
        }, self.checkpoint_path)

    def load_checkpoint(self, path: str | None = None) -> int:
        """Restores agents/optimizers/belief buffers/history from a
        checkpoint saved by save_checkpoint. Rebuilds each agent with the
        hidden_size it had at save time (which may differ from this
        trainer's current agents if growth happened), so this works
        regardless of what hidden_size the trainer was constructed with.
        Returns the round it was saved at."""
        checkpoint = torch.load(path or self.checkpoint_path, map_location=self.device,
                                 weights_only=False)

        self.agents = [Actor(hidden_size=hs).to(self.device)
                       for hs in checkpoint["agent_hidden_sizes"]]
        for actor, state in zip(self.agents, checkpoint["agents"]):
            actor.load_state_dict(state)

        self.optimizers = [torch.optim.Adam(a.parameters(), lr=self.lr) for a in self.agents]
        for optimizer, state in zip(self.optimizers, checkpoint["optimizers"]):
            optimizer.load_state_dict(state)

        self.belief_buffers = checkpoint["belief_buffers"]
        self.history = checkpoint["history"]
        self.growth_rounds = checkpoint.get("growth_rounds", [])
        return checkpoint["round"]

    def run(self, verbose: bool = True) -> list[dict]:
        """Runs self.n_rounds more rounds, continuing round numbering (and
        which agent moves next) from wherever self.history left off — so
        calling run() again (e.g. after grow_agents()) resumes rather than
        restarting the round count."""
        start_round = len(self.history)
        for round_idx in range(start_round, start_round + self.n_rounds):
            mover_idx = round_idx % self.n_agents
            opponent_indices = [i for i in range(self.n_agents) if i != mover_idx]

            opponent_bids = self._sample_max_belief_bids(opponent_indices)
            critic = SplineCritic(n_quantiles=self.critic_n_quantiles, device=self.device)
            critic.fit(opponent_bids)

            steps_taken = self._best_response(mover_idx, critic)
            self._update_belief(mover_idx)

            metrics = [self._benchmark_metrics(self.agents[i]) for i in range(self.n_agents)]
            mse = [m[0] for m in metrics]
            max_diff = [m[1] for m in metrics]
            payoff = [self._expected_payoff(i) for i in range(self.n_agents)]
            self.history.append({
                "round": round_idx,
                "mover": mover_idx,
                "steps": steps_taken,
                "mse": mse,
                "max_diff": max_diff,
                "payoff": payoff,
            })
            if verbose:
                mse_str = " ".join(f"MSE{i}={m:.6f}" for i, m in enumerate(mse))
                maxdiff_str = " ".join(f"MaxDiff{i}={d:.5f}" for i, d in enumerate(max_diff))
                payoff_str = " ".join(f"P{i}={p:.5f}" for i, p in enumerate(payoff))
                theo = theoretical_expected_payoff(self.n_agents)
                print(f"Round {round_idx:3d} | mover=agent{mover_idx} | steps={steps_taken:5d} "
                      f"| {mse_str} | {maxdiff_str} | {payoff_str} | teórico={theo:.5f}")

            max_diff_ok = self._max_diff_converged(max_diff)
            payoff_ok = self._payoff_converged()

            if self.checkpoint_every and (round_idx + 1) % self.checkpoint_every == 0:
                self.save_checkpoint(round_idx)

            if max_diff_ok or payoff_ok:
                reason = "desvio máximo < limiar" if max_diff_ok else "payoff estável (janela)"
                if verbose:
                    print(f"Convergência detectada na rodada {round_idx} ({reason}).")
                self.save_checkpoint(round_idx, converged=True)
                break
        else:
            self.save_checkpoint(start_round + self.n_rounds - 1, converged=False)

        return self.history
