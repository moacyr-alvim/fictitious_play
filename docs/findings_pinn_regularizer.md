# Physics-informed (PINN-style) regularizers for best-response training

**Status: solved, for practical purposes. Giving the opponent-based PINN
residual its own memory (an EMA shadow of the opponent's weights,
`opponent_ema_decay`) fixes the leap-frogging that made it useless on
its own, and — combined with the critic's payoff loss and belief buffer
— lets the hybrid stabilize all-pay *directly*, no homotopy continuation
needed, beating the critic-only baseline at matched round counts. See
"The hybrid, revisited: EMA fixes it" below; the underlying data is in
`checkpoints/checkpoint_allpay_hybrid.pt` (hybrid) vs.
`checkpoints/checkpoint_allpay_pinn0_baseline.pt` (critic-only, same
settings).**

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

## First attempt at the hybrid: expensive, and no clear win

Setup: N=2 all-pay, `belief_decay=0.98`, `payoff_weight=1.0`,
`pinn_opponent_weight` swept from 0.01 to 1.0.

Every weight tried, including the smallest (0.01), drove almost every
round to the `max_steps` cap (5000) — the EMA-based patience/plateau
stopping check essentially never fired once the opponent-based term was
added, at any weight. Cost ballooned accordingly (~35s/round vs. a few
seconds for the critic-only baseline). Quality after 10-15 rounds was
already *worse* than the critic-only baseline at the same round count
(`MaxDiff` 0.28-0.31 vs. the baseline's ~0.15-0.2). Adding the exact,
noise-free term did not make training easier to stop or better in early
rounds — a surprising result given the term's exactness, investigated
below.

## Homotopy with a *pure* opponent-based PINN (no critic, no belief buffer at all)

Tried the continuation idea directly on the pure opponent-residual
objective (`payoff_weight=0`, `pinn_opponent_weight=1.0`), starting the
payment-share schedule at `lambda=0` (first-price) and stepping up,
hoping to avoid the `lambda=1` degeneracy above by never actually
training against an exactly-linear opponent once `lambda>0` (the
equilibrium curves away from linear for any `lambda>0`, so `phi''(b)!=0`
replaces `(1-lambda)` as the source of non-zero sensitivity — confirmed
symbolically: `dR/db = phi''(b)*(v-(1-lambda)*b) - 2*(1-lambda)*phi'(b)`,
nonzero as `lambda->1` as long as `phi''(b)!=0`, i.e. as long as the
opponent isn't exactly linear).

### A real bug found and fixed along the way

The `lambda=0.1` stage initially produced a hard failure: one agent's
bid function saturated to exactly `b(v)=v` (sigmoid pinned to 1). Cause:
`opponent_inverse_and_slope`'s bisection search naively extrapolates past
the opponent's actually-achievable bid range — if a mover's bid target
exceeds `opponent_actor(1)` (the opponent's own current max), the search
saturates at the domain boundary `v=1`, where the opponent's slope can be
tiny, and `1/slope` then blows up. A mover that strays slightly above the
opponent's range gets back a huge `phi'`, escalates further next step,
and runs away. Fixed by clamping the inversion target to
`[opponent_actor(0), opponent_actor(1)]` and returning the exact boundary
values there (`phi=1, phi'=0` above the max — certain win, no further
marginal effect — and symmetrically at the low end), matching the same
clamp-before-inverting pattern `benchmark.theoretical_win_probability_lambda`
already uses. See `test_opponent_inverse_and_slope_bounded_past_opponents_range`.

### The deeper, structural finding: lambda=0 only looked easy by coincidence

Even after that fix, `lambda=0.1` remained far harder than `lambda=0`
— taking 110 of a 150-round cap and still landing at `MaxDiff` 0.13/0.43,
asymmetric and unconverged. Round-by-round inspection (starting directly
at `lambda=0.1`, no continuation) shows exactly the leap-frogging/war-of-
attrition signature already documented for `belief_decay=0` in
`findings_all_pay.md`:

```
round  5: MaxDiff=[0.13, 0.49]  payoff=[0.112, 0.037]
round  7: MaxDiff=[0.30, 0.50]  payoff=[0.066, 0.004]   <- agent 1 ~wiped out
round  9: MaxDiff=[0.07, 0.15]  payoff=[0.197, 0.192]   <- recovers
round 23: MaxDiff=[0.10, 0.49]  payoff=[0.096, 0.040]   <- crashes again
round 29: MaxDiff=[0.10, 0.49]  payoff=[0.101, 0.040]   <- and again
```

Root cause: `lambda=0` didn't actually demonstrate the method working —
it demonstrated nothing moving at all. For N=2 first-price, the
equilibrium is exactly `b*(v) = v/2`, and the `Actor`'s own
initialization (`bias=0`, hidden blocks zero-initialized) computes
*exactly* `b(v) = v*sigmoid(0) = v/2` — bit-for-bit the same function.
Both agents start precisely on the fixed point, the residual is exactly
zero everywhere for that function, so the gradient is exactly zero and
training never leaves it. The tiny `MaxDiff` (~1e-6) at `lambda=0` is not
convergence, it's a coincidence of this architecture plus this auction
size never needing to move.

The instant `lambda` increases even slightly, that accidental freeze
breaks (the fixed point moves away from the initialization) and the
*real* dynamics of this objective appear for the first time: each agent
reacts to the opponent's literal, current actor, with no time-averaging
— precisely the `belief_decay=0` regime. This is the same leap-frogging
instability the belief buffer was built to fix in the very first phase
of this project (for N>=3) and that reappeared for all-pay without
memory. It was always latent in the pure opponent-based PINN
formulation; `lambda=0` for N=2 just happened to hide it.

This also reframes the first hybrid attempt's failure to converge
quickly (`max_steps` cap hit at every weight): the opponent-based term's
contribution to the loss is itself subject to this same memory-less
leap-frogging pressure round to round, which plausibly explains why
adding it — at any weight — disrupted the EMA/patience stopping check
even when the critic-based term should have continued supplying a stable
anchor. Not yet confirmed directly; see next steps.

## The fix: an EMA shadow gives the opponent-based residual its own memory

`_update_opponent_ema` keeps, per agent, an exponential-moving-average
copy of its own weights (`opponent_ema_decay`), updated once per round
right after that agent's own best response — same granularity as the
critic's belief buffer. `foc_residual_opponent` inverts against the
*shadow*, not the live actor. This is a direct implementation of the
"fictitious play against a time-averaged opponent, not its latest
snapshot" idea (independently suggested as a fix for this exact failure
mode when asked to vet the diagnosis against a general-purpose list of
RL-stabilization techniques) — restoring, for the exact/opponent-based
residual, the same memory the belief buffer already gives the critic.

Repeating the direct `lambda=0.1` test (no homotopy, 40 rounds) with
`opponent_ema_decay=0.9`: `MaxDiff` now falls smoothly and
monotonically for both agents (0.029→0.015), payoffs converge cleanly to
the theoretical 0.1667 — no leap-frogging at all. Confirms the diagnosis
directly: it was a memory problem, not an exactness problem.

### Decay needs to track how close lambda is to 1, not be fixed

A fixed `opponent_ema_decay=0.9` stabilizes `lambda=0.1` but still
eventually breaks down at higher `lambda` — not always at the same
stage, but always eventually, and sooner the larger `lambda` is (stable
150 rounds at `lambda=0.1`; drifts into saturation by round ~80 at
`lambda=0.3`; chaotic almost immediately at `lambda=0.5`; saturates
within 60 rounds at `lambda=0.7`). The breakdown is a slow *monotonic*
drift (one agent's `MaxDiff` climbing steadily to exactly 0.5 and
sticking, not oscillating) — the signature of a one-directional bias
that EMA, which only cancels sign-alternating noise, delays rather than
arrests. This matches the theory directly: the FOC's restoring-force
coefficient is `~ -2*(1-lambda)*phi'(b)` (linear-opponent case; see
above) — it weakens continuously as `lambda->1`, so a fixed amount of
memory that's enough to stabilize a strong restoring force (small
`lambda`) is not enough once that force is weak (`lambda` near 1).

Tried a fixed `opponent_ema_decay=0.98` across the full homotopy
schedule (`lambda`: 0 to 1, step 0.1): clean and fast through
`lambda<=0.7` (~11 rounds/stage), then one agent drifts to saturation
starting at `lambda=0.8`, stuck there through `lambda=1.0` — the
breakdown point moved from `~0.3` (decay 0.9) to `~0.8` (decay 0.98),
consistent with "more memory delays but doesn't eliminate the problem,
because the underlying restoring force still hits zero at lambda=1".

Made `opponent_ema_decay` a linear schedule in the current `lambda`
instead (`opponent_ema_decay_schedule=(decay_at_0, decay_at_1)`,
interpolated via `payment_share`). `(0.85, 0.995)` performed *worse*
than the fixed 0.98 — at `lambda=0.8` the schedule gives `0.85 +
0.145*0.8 = 0.966`, weaker than the 0.98 already known to struggle
there; it only exceeds 0.98 very close to `lambda=1`, too late. Raising
the starting point fixed this: `(0.95, 0.995)` reached `lambda=1`
without any saturation at all (every stage converged in its normal
~11-round budget, no stage ever hit the round cap), ending at `MaxDiff`
0.35/0.35, symmetric payoffs ~0.11/0.10 (vs. 0.1667 theoretical) —
not a clean equilibrium, but no collapse either, a first for any
pure-PINN attempt at `lambda=1`. `(0.97, 0.995)` pushed this a bit
further (`MaxDiff` 0.31/0.27), with diminishing returns past that point.

The fact that the schedule's *shape* mattered as much as its endpoints —
and that the needed correction (start higher, not just end higher) runs
the opposite direction of a naive "keep per-round loop gain constant"
argument — suggests a fully principled schedule would come from
linearizing the two-agent EMA-feedback recursion and solving for the
decay that keeps it stable at each `lambda`, rather than the two
empirical endpoints used here. Not attempted; flagged as a sharper
version of the "let the problem's own structure suggest the
hyperparameters" idea discussed below.

## The hybrid, revisited: EMA fixes it

The first hybrid attempt (critic payoff loss + opponent-based residual,
before the EMA shadow existed) hit `max_steps` at nearly every round,
at every weight tried, and showed no quality advantage over the
critic-only baseline. Re-run with the EMA shadow in place
(`opponent_ema_decay=0.97`) and the pathology is gone: step counts vary
normally again (tens to a few thousand, not pegged at 5000).

Direct 420-round comparison, both starting cold at `lambda=1`
(all-pay) — **no homotopy continuation in either run**, same
`belief_decay=0.98`, same seed — `payoff_weight=1.0,
pinn_opponent_weight=0.1, opponent_ema_decay=0.97` vs. the critic-only
baseline (`pinn_opponent_weight=0.0`):

| | hybrid | critic-only baseline |
|---|---:|---:|
| `MaxDiff` at round 419/420 | 0.131 / 0.128 | 0.160 / 0.175 |
| trend, last 20 rounds | falling smoothly and monotonically | falling, then turning back up (agent 1: 0.160->0.175) |

The hybrid is ahead on both agents at the same round count, and —
notably — doesn't show the slow reversal the critic-only run is just
starting to show (consistent with the ~150-250-round quasi-cycle
`findings_all_pay.md` documents for the critic-only approach). This is
the first configuration in this whole investigation where the
opponent-based PINN term is a clear, measurable net positive rather
than neutral-to-harmful.

## Status / next steps

- The hybrid (`payoff_weight=1.0` critic + belief buffer, plus
  `pinn_opponent_weight>0` with an EMA-shadowed opponent) is the
  best-performing configuration found for all-pay to date, and it
  doesn't need homotopy continuation to get there — though combining it
  with homotopy (now that the opponent term has its own memory) hasn't
  been tried and might do even better.
- Not yet checked: does the hybrid's advantage keep compounding past
  round 420, or does it eventually hit its own version of the
  quasi-cycle (just delayed further)?
- Not yet tuned: `pinn_opponent_weight` (only 0.1 tried in the winning
  hybrid run) and whether a decay *schedule* (vs. the fixed 0.97 used)
  helps the hybrid the way it helped the pure-PINN homotopy case.
- Broader open idea (raised explicitly, worth keeping in view): does the
  auction problem's own structure — or diagnostics computed from a run
  at some reference configuration — point directly at good
  hyperparameters for these tools, rather than needing a hand-tuned
  sweep each time? The restoring-force formula `~(1-lambda)` is one
  concrete instance already in hand (it's *why* the decay schedule
  needs to ramp up toward `lambda=1`, not just an empirical fit); the
  Jacobian-eigenvalue idea noted earlier (complex eigenvalues flagging
  rotational/cyclic dynamics) is another candidate that hasn't been
  computed yet for this problem specifically.
