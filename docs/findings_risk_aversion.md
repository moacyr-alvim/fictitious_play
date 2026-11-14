# Heterogeneous risk aversion in the 2-agent first-price auction

**Status: ground truth derived and verified; training works with SplineCritic,
but BetaCritic (the project's default improvement for the lambda/all-pay
continuum) turns out to be the *wrong* choice here — its parametric
assumption doesn't hold for this auction variant.**

## The theoretical equilibrium has no closed form

Each agent has a CRRA utility over the payoff if winning,
`u_i(x) = x^rho_i` (`rho_i=1` risk-neutral, `rho_i` in `(0,1)` risk-averse —
more curvature, i.e. smaller `rho_i`, is *more* risk-averse), both drawing
`v ~ Uniform(0,1)`.

The first guess — generalizing the textbook symmetric CRRA result
`b*(v) = v*(n-1)/(n-1+rho)` (here, `n=2`: `v/(1+rho)`) by just plugging in
each agent's own `rho_i` — **is wrong once `rho_1 != rho_2`**. Verified
numerically (re-optimizing each agent's objective against the other's
*assumed* strategy): the naive formula is an exact best response only while
the resulting bid stays below the opponent's maximum achievable bid. Near
`v=1`, the more risk-averse agent's "natural" bid can exceed what the other
agent could ever bid — a waste, since probability of winning is already 1
there.

Patching that (capping the more risk-averse agent's bid at the opponent's
max) is *also* wrong: it creates a probability atom in that agent's bid
distribution, which changes the *other* agent's best response too (it now
wants to jump past the atom rather than follow the naive linear curve all
the way to `v=1`). This is the standard "common support" phenomenon from
the asymmetric first-price auction literature (Marshall, Meurer, Richard &
Stromquist 1994; Maskin & Riley 2000): in equilibrium, all bidders bid from
the *same* `[0, b_bar]` interval, with `b_bar` itself an unknown, free
boundary.

**Correct approach**: solve the coupled ODE system in the bid variable `b`
(`phi_i(b)` = agent `i`'s bid CDF evaluated at `b`),

```
d(phi_2)/db = phi_2(b) * rho_1 / (phi_1(b) - b)
d(phi_1)/db = phi_1(b) * rho_2 / (phi_2(b) - b)
```

with `phi_1(0) = phi_2(0) = 0` and `phi_1(b_bar) = phi_2(b_bar) = 1`, via a
1-D shooting method over the unknown `b_bar`: integrate backward from
`b_bar` (where the ODEs are regular) down toward `b=0` (the singular end),
and bisect on `b_bar` until the backward solution reaches `phi_1, phi_2 ~ 0`
simultaneously. Implemented in
`fictitious_play.benchmark._solve_asymmetric_risk_averse_equilibrium` /
`equilibrium_bid_risk_averse`.

**Verified** (not just the ODE's first-order condition, but the actual
Bayes-Nash best-response property): independently re-optimizing each
agent's objective against the *other's solved curve* reproduces the solved
bid to within ~1e-5 (limited by interpolation-grid resolution, not a
modeling gap) — see `tests/test_benchmark.py::test_risk_averse_equilibrium_is_mutual_best_response`.
Reduces exactly to the known closed form when `rho_1 == rho_2`.

Comparative statics check out too: the more risk-averse agent shades less
(bids a larger fraction of its value) than a less risk-averse opponent, the
standard textbook result (`test_risk_averse_more_averse_agent_bids_more_aggressively`).

## Training result: BetaCritic is the wrong critic for this problem

N=2, `risk_aversion=[0.3, 1.0]`, first-price, 150 rounds, `sample_size=100_000`,
`belief_decay=0.9`, `payoff_rel_tol=-1` (disabled, same fair-comparison
methodology as `docs/findings_beta_critic.md`).

- **`BetaCritic`**: plateaus at `MaxDiff ~0.09-0.095` for both agents from
  round ~110 on — stable, but stuck well short of the true equilibrium.
  Inspecting the learned bid curves directly: agent 0's (the more
  risk-averse one) bid *flattens* around `b=0.53` for `v` above ~0.75,
  instead of climbing to the true `b_bar=0.6236` at `v=1`
  (`checkpoints/checkpoint_risk_averse_beta.pt`).
- **`SplineCritic`**, identical settings otherwise: `MaxDiff` reaches
  `~0.024-0.03` by round ~130-140 (still not fully settled below the 0.02
  threshold in 150 rounds, see below for a longer run), a clearly different
  and much better regime (`checkpoints/checkpoint_risk_averse_spline.pt`).

**Root cause**: `BetaCritic` fits `P(win|b) = (b/scale)^alpha`, exact *only*
because the symmetric and lambda-continuum equilibria this project worked
with so far happen to be literal power laws (`b*(v) = c*v^p` <=> bid
distribution `Beta(1/p, 1)` — see `docs/findings_beta_critic.md`). The
asymmetric risk-aversion equilibrium has **no such structure**: its bid CDF
is whatever the ODE above produces, not a power law. Measured directly: a
`BetaCritic` fit on 500k samples drawn from the *true* equilibrium bid
distribution (not even a mid-training noisy sample) has a max CDF error of
**~5-6%** against the true CDF for both agents — noticeably worse than the
~3.5% max bias measured for the lambda continuum's intermediate stages, and
unlike the lambda continuum, there's no endpoint where this bias vanishes to
zero. This is model misspecification, not sampling noise, and it's large
enough to visibly bias the learned equilibrium.

**Takeaway**: `BetaCritic`'s advantage over `SplineCritic` documented in
`docs/findings_beta_critic.md` is tied to the *specific* algebraic identity
behind it (`b=c*v^p => b/c ~ Beta(1/p,1)`), which holds for first-price,
all-pay, and everything on the `payment_share` continuum between them, but
does *not* hold for every auction variant. Before defaulting to
`critic_type="beta"` on a new variant, check whether that variant's
equilibrium bid distribution is actually power-law-shaped; if not (as here),
`SplineCritic` remains the safer, correctly-unbiased choice despite its
higher variance.

## Open items

- The 150-round `SplineCritic` run above hadn't fully settled below
  `MaxDiff < 0.02` yet (still oscillating in the 0.02-0.07 range at the
  end) — a longer run should be checked before calling this fully
  converged.
- Only tested at one `(rho_1, rho_2)` pair (`0.3, 1.0`); haven't checked
  how the bias/convergence behavior scales as the two `rho`s get closer
  together (where the equilibrium is closer to the symmetric linear case)
  or further apart (where the ODE solution's curvature, and presumably
  `BetaCritic`'s misspecification, should be more pronounced).
- Monetary payoffs (not utility) were observed with the more risk-averse
  agent earning a *higher* raw payoff than the risk-neutral one in the
  partially-converged beta-critic run — plausible (it bids more
  aggressively, winning more often) but not yet checked against the
  theoretical monetary payoff implied by the solved equilibrium (no closed
  form, would need a numerical integration against the solved curves,
  analogous to `theoretical_expected_payoff` but agent-specific and
  accounting for the inefficient allocation).
