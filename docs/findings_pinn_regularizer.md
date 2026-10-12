# Physics-informed (PINN-style) regularizers for best-response training

**Status: two variants tried, both fail in pure form for all-pay; the
critic-based one fails outright, the opponent-based one is provably
gradient-dead at initialization. The opponent-based one works very well,
standalone, for first-price. Next step (not yet tried): hybrid, combining
the opponent-based residual with the existing critic-based payoff loss.**

## Motivation

`docs/findings_all_pay.md` traces all-pay's difficulty to its first-order
condition depending purely on the critic's *derivative*, with no level
term to anchor it. The idea here: the FOC any best response must satisfy
is a known equation,

```
R(v; theta) = dP/db(b) * (v - (1-lam)*b) - (1-lam)*P(b) - lam = 0,   b = b_theta(v)
```

(`fictitious_play.train.foc_residual`). Penalizing violation of `R` — in
the spirit of a physics-informed neural network, where a known
differential constraint regularizes training — seemed like it should add
information that steadies the actor, analogous to how PINNs stabilize
fitting under sparse or noisy data by enforcing a known equation exactly.

Two ways to get `P(win|b)` and its derivative for this residual were
implemented and tested, both for N=2:

1. **`foc_residual`**: from the critic (`SplineCritic`, fit on sampled
   belief-buffer bids) — the same estimator the existing payoff loss
   already uses.
2. **`foc_residual_opponent`**: from inverting the opponent's actor
   network directly — exact given the opponent's current strategy, no
   sampling, no density estimation, but also no belief-buffer memory
   (reflects today's opponent, not a time-smoothed belief).

## Variant 1: critic-based residual as a regularizer — fails, makes things worse

Setup: N=2 all-pay, `belief_decay=0.98` (the best previously-known
setting), `loss = payoff_loss + pinn_weight * mean(R^2)`.

- `pinn_weight=1.0`: step counts exploded (several rounds hit the
  `max_steps=5000` cap), confirming the extra term fights the main
  objective rather than easing it.
- `pinn_weight=0.1`, run for 280 rounds: diverged. `MaxDiff` climbed to
  **0.38–0.44** for both agents — worse than the undamped
  (`belief_decay=0`) leap-frogging baseline in `findings_all_pay.md`
  (~0.3–0.5). Inspecting the diverged checkpoint:
  - Agent 0's bid function had gone **non-monotonic** (lower bid at
    v=0.99 than at v=0.79).
  - The weighted PINN loss term (`0.1 * mean(R^2)`) was **~0.10**,
    essentially tied with the payoff loss (**-0.11**) in magnitude — not
    a gentle regularizer, a co-equal competing objective.

Root cause: `R` is built from `dP/db`, the critic's derivative — exactly
the noisy, hard-to-estimate quantity `findings_all_pay.md` already
implicates. Squaring a residual built from a locally noisy derivative is
not robust (L2 amplifies outliers), and differentiating the critic a
second time (double backward, needed for `dP/db` to stay differentiable
w.r.t. the actor) adds further numerical noise on top. The regularizer
doesn't add new information; it reinjects the same fragile estimator,
squared, with its own gradient.

**Conclusion: not fixable by weight tuning alone** — smaller weights or a
robust (Huber-style) loss might reduce the damage, but the premise (the
residual's only source of `dP/db` is the thing already known to be
fragile) is the same regardless.

## Variant 2: opponent-actor-based residual, pure (no critic, no payoff loss)

Setup: `payoff_weight=0`, `pinn_opponent_weight=1.0` — training minimizes
*only* `mean(foc_residual_opponent^2)`, no critic, no belief buffer at
all, for either agent.

### All-pay (lam=1): provably stuck at initialization, not just unstable

300 rounds, `MaxDiff` locked at **exactly 0.125000...** (bit-identical)
for both agents for the entire run; `MSE` likewise frozen at
0.008291666... Both agents converged — immediately, not asymptotically —
to **`b(v) = v/2` exactly**, which is the actor's own *initialization*
(`bias=0` → `sigmoid(0)=0.5` → `b(v) = v * 0.5`, hidden blocks zero at
init). This is not a spurious learned fixed point; training never moved
at all.

Why: substituting a linear opponent `b_j(v) = c*v` (so
`phi_j(b) = b/c`, `phi_j'(b) = 1/c`, constant) into the residual at
`lam=1`:

```
R(v) = phi_j'(b)*(v - 0*b) - 0*phi_j(b) - 1 = (1/c)*v - 1
```

**The `b` term cancels identically.** Whenever the opponent's bid
function is linear — which it is, exactly, at initialization — the
residual for all-pay does not depend on the agent's own choice of `b` at
all, so `d(mean(R^2))/d(theta)` is exactly zero. Gradient descent cannot
move. This reproduces `findings_all_pay.md`'s root cause (`v*dP/db=1` has
no level term) in a more severe form: not noisy-and-hard-to-optimize, but
*exactly flat*, for this specific residual construction.

Redoing the same substitution for general `lam < 1` leaves a
`-2*c*(1-lam)*b` term that does *not* cancel — so this exact degeneracy
is specific to `lam=1`. Confirmed directly:

### First-price (lam=0): excellent early, then destabilizes

Same pure setup (`payoff_weight=0`, `pinn_opponent_weight=1.0`,
`auction_type="first_price"`): `MaxDiff` dropped to **0.0001–0.0007**
within about 15 rounds — better than anything achieved with the
critic-based loss. Then, around round 18, it jumped to 0.1157. Likely
cause: with no belief-buffer memory, each agent is chasing the opponent's
literal current actor rather than a time-averaged belief — the same
"moving target" dynamic the belief buffer was originally added to tame
(see the module docstring in `fictitious_play/train.py`), just now
showing up once the system gets close enough to equilibrium for small
perturbations to matter.

## Takeaway going in to the hybrid

- The critic-based PINN term reinjects the exact noise it was meant to
  smooth over — not a good regularizer for all-pay in any weight.
- The opponent-based term is exact and noise-free, which is exactly what
  all-pay needs, but alone it (a) is mathematically inert against a
  linear opponent at `lam=1`, the system's own starting point, and (b)
  lacks the belief-buffer's stabilizing memory once near equilibrium.
- The critic-based payoff loss doesn't share the opponent-term's
  degeneracy (it has no such cancellation) and does carry the
  belief-buffer memory. Combining `payoff_weight=1.0` (critic, stable,
  memory) with `pinn_opponent_weight>0` (opponent, exact, breaks the
  all-pay anchor problem) is the natural next experiment — the payoff
  term should supply a non-zero gradient to escape the linear
  initialization that strands the opponent term alone, while the
  opponent term should steady the derivative-noise sensitivity the
  critic-only loss already struggles with.
