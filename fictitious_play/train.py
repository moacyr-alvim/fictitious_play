from collections import deque
from pathlib import Path

import numpy as np
import torch

from .actor import Actor
from .benchmark import AUCTION_TYPES, equilibrium_bid, equilibrium_bid_lambda, theoretical_expected_payoff
from .critic import SplineCritic


def foc_residual(p_win: torch.Tensor, b: torch.Tensor, v: torch.Tensor, lam: float) -> torch.Tensor:
    """Pointwise residual of the first-order stationarity condition any
    best response to a critic with win-probability p_win(b) = p_win must
    satisfy at bid b (shape (batch, 1), must require grad) for valuation v,
    under the payment rule with share lam (winner pays b, loser pays
    lam*b): dPi/db = dP/db*(v-(1-lam)*b) - (1-lam)*P - lam, which is zero
    exactly where b maximizes the expected payoff Pi(b;v) given p_win.
    dP/db is obtained via autograd with create_graph=True so the result
    stays differentiable w.r.t. whatever parameters b itself depends on."""
    dp_db = torch.autograd.grad(p_win.sum(), b, create_graph=True)[0]
    return dp_db * (v - (1 - lam) * b) - (1 - lam) * p_win - lam


def _invert_monotonic(actor, target: torch.Tensor, iters: int = 30) -> torch.Tensor:
    """Bisection search (batched, no grad) for v in [0, 1] with actor(v) ≈
    target, assuming actor(v) is non-decreasing in v (true at/near a best
    response, though not guaranteed mid-training — see
    foc_residual_opponent)."""
    lo = torch.zeros_like(target)
    hi = torch.ones_like(target)
    with torch.no_grad():
        for _ in range(iters):
            mid = (lo + hi) / 2
            too_high = actor(mid) > target
            hi = torch.where(too_high, mid, hi)
            lo = torch.where(too_high, lo, mid)
    return (lo + hi) / 2


def opponent_inverse_and_slope(opponent_actor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (phi(b), phi'(b)) where phi = opponent_actor^-1, treating
    the opponent's parameters as fixed (this is called from the mover's
    best-response step, where only the mover's own parameters should get
    gradient). phi(b) is made differentiable w.r.t. b (and hence w.r.t.
    whatever parameters b itself depends on) via a local linear correction
    built from the implicit function theorem — the root itself is found
    under no_grad (bisection isn't usefully differentiable), then phi's
    known local slope 1/opponent_actor'(v*) carries the dependency on b.

    b is clamped to the opponent's actually achievable range
    [opponent_actor(0), opponent_actor(1)] before inverting: bidding
    beyond that range wins for certain (phi=1) with no further marginal
    effect (phi'=0), and extrapolating the bisection past the opponent's
    real range instead (which a naive implementation would do, since
    bisection just saturates its search interval) can return an near-flat
    point of opponent_actor whose slope is tiny, making 1/slope blow up —
    empirically this drove exactly this kind of runaway: a mover bidding
    just past the opponent's current max got an exploding phi', escalated
    further next step, and saturated at b(v)=v."""
    b_detached = b.detach()
    lo_b = opponent_actor(torch.zeros_like(b_detached))
    hi_b = opponent_actor(torch.ones_like(b_detached))
    in_range = (b_detached >= lo_b) & (b_detached <= hi_b)
    b_clamped = torch.clamp(b_detached, min=lo_b, max=hi_b)

    v_star = _invert_monotonic(opponent_actor, b_clamped).requires_grad_(True)
    b_at_v_star = opponent_actor(v_star)
    opponent_slope = torch.autograd.grad(b_at_v_star.sum(), v_star)[0]
    phi_prime_in_range = 1.0 / opponent_slope.clamp(min=1e-6)
    phi_prime = torch.where(in_range, phi_prime_in_range, torch.zeros_like(phi_prime_in_range)).detach()

    phi_val_in_range = v_star.detach()
    phi_val_at_b_detached = torch.where(
        b_detached > hi_b, torch.ones_like(b_detached),
        torch.where(b_detached < lo_b, torch.zeros_like(b_detached), phi_val_in_range),
    )
    phi_val = phi_val_at_b_detached + phi_prime * (b - b_detached)
    return phi_val, phi_prime


def foc_residual_opponent(b: torch.Tensor, v: torch.Tensor, lam: float, opponent_actor) -> torch.Tensor:
    """Like foc_residual, but P(win|b) and its derivative come from
    inverting the opponent's own actor network directly (exact, given the
    opponent's current strategy) instead of from a critic fit on sampled
    belief-buffer bids — no density estimation, no sampling noise. Trades
    that noise for a different approximation: it reflects the opponent's
    literal current policy rather than a memory-smoothed belief, and
    requires the opponent's bid function to be (at least locally) monotonic
    for the inversion to mean anything."""
    phi_val, phi_prime = opponent_inverse_and_slope(opponent_actor, b)
    return phi_prime * (v - (1 - lam) * b) - (1 - lam) * phi_val - lam


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
                 freeze_mode: str = "none", unfreeze_after_rounds: int = 20,
                 auction_type: str = "first_price", payment_share: float | None = None,
                 payoff_weight: float = 1.0, pinn_weight: float = 0.0,
                 pinn_opponent_weight: float = 0.0, opponent_ema_decay: float = 0.9,
                 opponent_ema_decay_schedule: tuple[float, float] | None = None,
                 max_diff_threshold: float | None = 0.02, payoff_rel_tol: float | None = 0.15,
                 payoff_stability_window: int = 10,
                 checkpoint_path: str | None = None, checkpoint_every: int = 20,
                 seed: int | None = None, device: str | None = None):
        if freeze_mode not in ("none", "permanent", "temporary"):
            raise ValueError('freeze_mode must be "none", "permanent", or "temporary"')
        if auction_type not in AUCTION_TYPES:
            raise ValueError(f"auction_type must be one of {AUCTION_TYPES}")
        self.auction_type = auction_type
        # payment_share (lambda), when set, overrides auction_type: it drives
        # the generalized payment rule that interpolates between first-price
        # (0.0) and all-pay (1.0) — see benchmark.equilibrium_bid_lambda. Used
        # for homotopy continuation: step payment_share from 0 to 1 across
        # training stages, warm-starting each stage from the previous one.
        self.payment_share = payment_share
        # Weight on the usual critic-based expected-payoff loss (1.0 =
        # plain fictitious play). Set to 0 together with pinn_weight=0 to
        # train purely on pinn_opponent_weight's residual instead.
        self.payoff_weight = payoff_weight
        # Physics-informed regularizer: penalizes violation of the per-v
        # first-order stationarity condition any best response must satisfy
        # (see _best_response), weighted against the usual expected-payoff
        # loss. 0 disables it (plain fictitious play). Uses the critic's
        # (noisy, sampled) estimate of P(win|b) and its derivative.
        self.pinn_weight = pinn_weight
        # Like pinn_weight, but P(win|b) and its derivative come from
        # inverting the opponent's actor directly (exact, no critic) — see
        # foc_residual_opponent. Only supported for n_agents == 2 so far.
        self.pinn_opponent_weight = pinn_opponent_weight
        if pinn_opponent_weight > 0 and n_agents != 2:
            raise NotImplementedError("pinn_opponent_weight only supports n_agents == 2")
        # Decay for the EMA "shadow" copy of each agent's actor that
        # foc_residual_opponent inverts against (see _update_opponent_ema):
        # inverting the opponent's literal current actor every round has no
        # time-averaging at all, which is mathematically equivalent to
        # belief_decay=0 fictitious play — already known to leap-frog/cycle
        # (see docs/findings_pinn_regularizer.md). The EMA shadow restores
        # the same kind of memory the belief buffer gives the critic, just
        # applied to the opponent's weights directly instead of to a
        # resampled pool of its past bids.
        self.opponent_ema_decay = opponent_ema_decay
        # Optional (decay_at_lambda=0, decay_at_lambda=1) pair: when set,
        # overrides opponent_ema_decay with a linear schedule in the current
        # payment_share instead of a fixed value. Motivated by the
        # homotopy findings: the FOC's anchoring term weakens continuously
        # as lambda->1 (docs/findings_pinn_regularizer.md), so the memory
        # needed to keep the EMA shadow stable should grow the same way —
        # weaker (cheaper, faster-reacting) memory where the anchor is
        # already strong, stronger memory exactly where it's weakest.
        self.opponent_ema_decay_schedule = opponent_ema_decay_schedule
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
        self.freeze_mode = freeze_mode
        self.unfreeze_after_rounds = unfreeze_after_rounds
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
        self.ema_agents = [Actor(hidden_size=actor_hidden_size).to(self.device) for _ in range(n_agents)]
        for shadow, live in zip(self.ema_agents, self.agents):
            shadow.load_state_dict(live.state_dict())
            for p in shadow.parameters():
                p.requires_grad_(False)
        self.belief_buffers = [self._fresh_bids(i) for i in range(n_agents)]
        self.history: list[dict] = []
        self.growth_rounds: list[int] = []
        self._frozen_since_round: int | None = None

    def _fresh_bids(self, agent_idx: int) -> np.ndarray:
        with torch.no_grad():
            v = torch.rand(self.sample_size, 1, device=self.device)
            b = self.agents[agent_idx](v)
        return b.squeeze(1).cpu().numpy()

    def grow_agents(self, new_hidden_size: int) -> None:
        """Grows every agent's actor to new_hidden_size hidden units (all
        agents share the same current hidden_size, so the same increment
        applies to each), function-preserving (see Actor.grow): each agent
        computes exactly the same bids right after growing as right before.
        Optimizers are rebuilt fresh for the new parameters (Adam's
        momentum/variance state doesn't carry over meaning across a change
        in parameter shape). Belief buffers and history are left untouched,
        since they're just past bid values / metrics, independent of
        architecture.

        freeze_mode="permanent" makes every parameter from before this call
        (and every earlier growth stage) permanently non-trainable, so only
        the newest block can still move. freeze_mode="temporary" does the
        same, but run() automatically unfreezes everything again after
        unfreeze_after_rounds rounds."""
        additional = new_hidden_size - self.agents[0].hidden_size
        freeze = self.freeze_mode in ("permanent", "temporary")
        self.agents = [agent.grow(additional, freeze_previous=freeze).to(self.device)
                       for agent in self.agents]
        self.optimizers = [torch.optim.Adam(a.parameters(), lr=self.lr) for a in self.agents]
        # Resync the EMA shadows to match the new architecture, snapped to
        # the live weights right after growth (the smoothing accumulates
        # again from this point on).
        self.ema_agents = [agent.grow(additional).to(self.device) for agent in self.ema_agents]
        for shadow, live in zip(self.ema_agents, self.agents):
            shadow.load_state_dict(live.state_dict())
            for p in shadow.parameters():
                p.requires_grad_(False)
        self.growth_rounds.append(len(self.history))
        self._frozen_since_round = len(self.history) if freeze else None

    def set_payment_share(self, new_payment_share: float) -> None:
        """Moves to a new point on the first-price <-> all-pay continuum
        (see benchmark.equilibrium_bid_lambda). Unlike grow_agents, this
        doesn't touch the actors' parameters at all — the warm start is
        implicit, since the payoff landscape changes continuously with
        payment_share and training simply continues from wherever the
        agents already are. Marked as a stage boundary on plots the same
        way network growth is (shares growth_rounds)."""
        self.payment_share = new_payment_share
        self.growth_rounds.append(len(self.history))

    def newton_predict_step(self, new_payment_share: float, damping: float = 1e-2,
                             sample_size: int | None = None,
                             trust_region: float = 0.2) -> None:
        """Block-diagonal Newton predictor for homotopy continuation: one
        step of Newton's method on each agent's own first-order
        stationarity condition F_i(theta_i) = grad_{theta_i} Loss_i = 0,
        evaluated at the NEW payment_share but the OLD (already converged)
        theta_i. To first order, F_i(theta_i_old, lambda_new) is exactly
        the residual introduced by moving lambda while holding theta fixed
        — this is the standard Davidenko-equation predictor step, with the
        cross-agent Jacobian blocks (d F_i / d theta_j, j != i) dropped:
        it ignores how agent i's optimality condition depends on agent j's
        parameters, which is precisely the coupling that makes this a game
        rather than N independent optimization problems (see
        docs/findings_all_pay.md on why that coupling drives the cyclic
        dynamics). Call this *before* set_payment_share, so the critic
        used here still reflects the pre-step belief buffers.

        The per-agent Hessian of a neural net's loss is typically
        indefinite (confirmed empirically here: roughly half its
        eigenvalues negative), so raw Newton's method is unreliable —
        it can step toward a saddle point instead of a minimum. damping
        is therefore not added as a fixed constant: it's the minimum
        amount added beyond whatever is needed to make the damped
        Hessian positive definite (Levenberg-Marquardt-style), computed
        from the Hessian's own smallest eigenvalue each time.

        Even damped, a single step built from one small random batch is
        a noisy, locally-quadratic approximation of a landscape that
        isn't very quadratic here: empirically the full step sometimes
        cuts the residual gradient by an order of magnitude and
        sometimes increases it, and more samples in the fitting batch
        does not make this more reliable (the problem is the quadratic
        model, not sampling noise in it). So the step is only ever taken
        on faith as far as a backtracking line search with a held-out
        validation batch (independent of the one used to fit the
        Hessian): try the full step, then half, quarter, ... and keep
        the first one that actually reduces the validation residual;
        if none do, leave the agent's parameters untouched.

        That line search alone is not enough, though: a near-singular
        damped Hessian can still produce a delta whose norm dwarfs the
        parameters it's added to, driving a hidden unit's pre-sigmoid
        logit to +-100 or beyond. The sigmoid then saturates exactly to
        0 or 1 in floating point, gradients vanish identically, and the
        agent is permanently dead (stuck bidding 0 or v forever) —
        while *looking* like a successful step, since a vanishing
        gradient trivially passes the validation-residual check above
        (empirically observed: it happened at the lambda=0.8->0.9
        transition in one run, with the hidden block's output around
        -140 and the agent bidding exactly 0 ever after). So before the
        line search, delta is also capped to a trust-region radius
        (a fixed fraction of the current parameter norm) — this bounds
        the worst case regardless of how ill-conditioned the damped
        Hessian solve turns out to be, independently of whatever the
        line search's own residual check would have accepted."""
        n = sample_size or self.batch_size

        def _residual(actor, critic, v, lam):
            params = [p for p in actor.parameters() if p.requires_grad]
            b = actor(v)
            p_win = critic.predict_win_prob(b.squeeze(1)).unsqueeze(1)
            loss = -(p_win * (v - (1 - lam) * b) - lam * b).mean()
            grad = torch.autograd.grad(loss, params)
            return torch.cat([g.reshape(-1) for g in grad]).norm()

        lam = new_payment_share
        for agent_idx in range(self.n_agents):
            actor = self.agents[agent_idx]
            params = [p for p in actor.parameters() if p.requires_grad]
            if not params:
                continue

            critic = self.critic_for(agent_idx)
            v = torch.rand(n, 1, device=self.device)
            b = actor(v)
            p_win = critic.predict_win_prob(b.squeeze(1)).unsqueeze(1)
            loss = -(p_win * (v - (1 - lam) * b) - lam * b).mean()

            grad = torch.autograd.grad(loss, params, create_graph=True)
            flat_grad = torch.cat([g.reshape(-1) for g in grad])
            d = flat_grad.numel()

            hessian_rows = [
                torch.cat([r.reshape(-1) for r in
                           torch.autograd.grad(flat_grad[i], params, retain_graph=True)])
                for i in range(d)
            ]
            hessian = torch.stack(hessian_rows)
            hessian = (hessian + hessian.T) / 2
            min_eig = torch.linalg.eigvalsh(hessian).min().item()
            effective_damping = damping - min_eig if min_eig < 0 else damping
            damped = hessian + effective_damping * torch.eye(d, device=self.device)
            delta = torch.linalg.solve(damped, -flat_grad.detach())

            original = [p.detach().clone() for p in params]
            param_norm = torch.cat([p.reshape(-1) for p in original]).norm()
            max_delta_norm = trust_region * param_norm.clamp(min=1e-6)
            delta_norm = delta.norm()
            if delta_norm > max_delta_norm:
                delta = delta * (max_delta_norm / delta_norm)

            v_val = torch.rand(n, 1, device=self.device)
            baseline = _residual(actor, critic, v_val, lam)

            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                with torch.no_grad():
                    offset = 0
                    for p, orig in zip(params, original):
                        numel = p.numel()
                        p.copy_(orig + scale * delta[offset:offset + numel].view_as(p))
                        offset += numel
                if _residual(actor, critic, v_val, lam) < baseline:
                    break
            else:
                with torch.no_grad():
                    for p, orig in zip(params, original):
                        p.copy_(orig)

    def _maybe_unfreeze(self, round_idx: int) -> None:
        if (self.freeze_mode == "temporary" and self._frozen_since_round is not None
                and round_idx - self._frozen_since_round >= self.unfreeze_after_rounds):
            for agent in self.agents:
                agent.unfreeze_all()
            self._frozen_since_round = None

    def _current_opponent_ema_decay(self) -> float:
        if self.opponent_ema_decay_schedule is None:
            return self.opponent_ema_decay
        decay_at_0, decay_at_1 = self.opponent_ema_decay_schedule
        lam = self.payment_share if self.payment_share is not None else 0.0
        return decay_at_0 + (decay_at_1 - decay_at_0) * lam

    def _update_opponent_ema(self, agent_idx: int) -> None:
        """Updates agent_idx's EMA shadow actor towards its just-updated
        live weights (see opponent_ema_decay). Called once per round, after
        the mover's best response is complete — the same granularity as
        _update_belief, so the shadow plays the same time-averaging role
        for foc_residual_opponent that the belief buffer plays for the
        critic."""
        decay = self._current_opponent_ema_decay()
        with torch.no_grad():
            for shadow_p, live_p in zip(self.ema_agents[agent_idx].parameters(),
                                         self.agents[agent_idx].parameters()):
                shadow_p.mul_(decay).add_(live_p, alpha=1 - decay)

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

        if self.payment_share is not None:
            lam = self.payment_share
        elif self.auction_type == "first_price":
            lam = 0.0
        else:
            lam = 1.0

        opponent_actor = None
        if self.pinn_opponent_weight > 0:
            opponent_idx = next(i for i in range(self.n_agents) if i != mover_idx)
            opponent_actor = self.ema_agents[opponent_idx]

        for step in range(self.max_steps):
            v = torch.rand(self.batch_size, 1, device=self.device)
            b = actor(v)
            loss = torch.zeros((), device=self.device)

            if self.payoff_weight > 0 or self.pinn_weight > 0:
                p_win = critic.predict_win_prob(b.squeeze(1)).unsqueeze(1)
                if self.payoff_weight > 0:
                    # Generalized payment rule: winner pays b, losers pay
                    # lam * b. E[payoff] = P(win)*(v - (1-lam)*b) - lam*b
                    # (lam=0 recovers first-price, lam=1 recovers all-pay).
                    payoff_loss = -(p_win * (v - (1 - lam) * b) - lam * b).mean()
                    loss = loss + self.payoff_weight * payoff_loss
                if self.pinn_weight > 0:
                    # Physics-informed regularizer: penalize violation of the
                    # FOC any best response must satisfy pointwise (see
                    # foc_residual) — the same double-backward trick used for
                    # PINN residuals of a PDE/ODE, just one order lower (first
                    # derivative, not second) since b plays the role of the
                    # "input" here, not v.
                    residual = foc_residual(p_win, b, v, lam)
                    loss = loss + self.pinn_weight * (residual ** 2).mean()

            if self.pinn_opponent_weight > 0:
                residual_opp = foc_residual_opponent(b, v, lam, opponent_actor)
                loss = loss + self.pinn_opponent_weight * (residual_opp ** 2).mean()

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
            if self.payment_share is not None:
                target = equilibrium_bid_lambda(self.eval_grid, self.n_agents, self.payment_share)
            else:
                target = equilibrium_bid(self.eval_grid, self.n_agents, self.auction_type)
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
            v_flat = v_self.squeeze(1)
            if self.payment_share is not None:
                lam = self.payment_share
                payoff = win * (v_flat - b_self) - (1 - win) * lam * b_self
            elif self.auction_type == "first_price":
                payoff = win * (v_flat - b_self)
            else:
                payoff = win * v_flat - b_self
            return payoff.mean().item()

    def _max_diff_converged(self, max_diff: list[float]) -> bool:
        if self.max_diff_threshold is None:
            return False
        return all(d <= self.max_diff_threshold for d in max_diff)

    def _payoff_converged(self) -> bool:
        """Stability is judged only on rounds since the current stage
        started (i.e. since the last grow_agents() call, if any): growth
        is function-preserving, so payoff readings from just before growing
        would otherwise look "stable" together with the readings right
        after, letting a stage stop before its new capacity did anything."""
        if self.payoff_rel_tol is None:
            return False
        stage_start = self.growth_rounds[-1] if self.growth_rounds else 0
        stage_history = self.history[stage_start:]
        if len(stage_history) < self.payoff_stability_window + 1:
            return False

        recent = [h["payoff"] for h in stage_history[-(self.payoff_stability_window + 1):]]
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
            "auction_type": self.auction_type,
            "payment_share": self.payment_share,
            "converged": converged,
            "agent_block_widths": [[block.width for block in a.blocks] for a in self.agents],
            "agents": [a.state_dict() for a in self.agents],
            "ema_agents": [a.state_dict() for a in self.ema_agents],
            "optimizers": [o.state_dict() for o in self.optimizers],
            "belief_buffers": self.belief_buffers,
            "history": self.history,
            "growth_rounds": self.growth_rounds,
        }, self.checkpoint_path)

    def load_checkpoint(self, path: str | None = None) -> int:
        """Restores agents/optimizers/belief buffers/history from a
        checkpoint saved by save_checkpoint. Rebuilds each agent with the
        same block structure it had at save time (which may differ from
        this trainer's current agents if growth happened), so this works
        regardless of what the trainer was constructed with. Returns the
        round it was saved at."""
        checkpoint = torch.load(path or self.checkpoint_path, map_location=self.device,
                                 weights_only=False)

        self.agents = []
        for widths in checkpoint["agent_block_widths"]:
            agent = Actor(hidden_size=0)
            for width in widths:
                agent.grow(width)
            self.agents.append(agent.to(self.device))
        for actor, state in zip(self.agents, checkpoint["agents"]):
            actor.load_state_dict(state)

        self.ema_agents = []
        for widths in checkpoint["agent_block_widths"]:
            agent = Actor(hidden_size=0)
            for width in widths:
                agent.grow(width)
            self.ema_agents.append(agent.to(self.device))
        if "ema_agents" in checkpoint:
            for shadow, state in zip(self.ema_agents, checkpoint["ema_agents"]):
                shadow.load_state_dict(state)
        else:
            for shadow, live in zip(self.ema_agents, self.agents):
                shadow.load_state_dict(live.state_dict())
        for shadow in self.ema_agents:
            for p in shadow.parameters():
                p.requires_grad_(False)

        self.optimizers = [torch.optim.Adam(a.parameters(), lr=self.lr) for a in self.agents]
        for optimizer, state in zip(self.optimizers, checkpoint["optimizers"]):
            optimizer.load_state_dict(state)

        self.belief_buffers = checkpoint["belief_buffers"]
        self.history = checkpoint["history"]
        self.growth_rounds = checkpoint.get("growth_rounds", [])
        self.auction_type = checkpoint.get("auction_type", "first_price")
        self.payment_share = checkpoint.get("payment_share")
        return checkpoint["round"]

    def run(self, verbose: bool = True) -> list[dict]:
        """Runs self.n_rounds more rounds, continuing round numbering (and
        which agent moves next) from wherever self.history left off — so
        calling run() again (e.g. after grow_agents()) resumes rather than
        restarting the round count."""
        start_round = len(self.history)
        for round_idx in range(start_round, start_round + self.n_rounds):
            self._maybe_unfreeze(round_idx)
            mover_idx = round_idx % self.n_agents
            opponent_indices = [i for i in range(self.n_agents) if i != mover_idx]

            opponent_bids = self._sample_max_belief_bids(opponent_indices)
            critic = SplineCritic(n_quantiles=self.critic_n_quantiles, device=self.device)
            critic.fit(opponent_bids)

            steps_taken = self._best_response(mover_idx, critic)
            self._update_belief(mover_idx)
            self._update_opponent_ema(mover_idx)

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
