# All-pay auctions are harder for this fictitious-play setup than first-price

**Status: unresolved / documented limitation.** This is a known, understood
hard case, not a bug waiting to be fixed by a parameter tweak.

## Setup

Same recipe validated for first-price (belief-buffer fictitious play +
`SplineCritic` + max-diff/payoff stopping criteria), applied to N=2 with
`auction_type="all_pay"`: every bidder pays their own bid regardless of
winning, so the per-bid objective is `v * P(win|b) - b` instead of
first-price's `(v - b) * P(win|b)`.

Symmetric equilibrium for N agents, valuations ~ U[0,1]:
`b*(v) = v^N * (N-1)/N` (for N=2: `b*(v) = v^2/2`). Same ex-ante expected
payoff as first-price, `1/(N*(N+1))`, by the revenue equivalence theorem
(same efficient allocation, same zero payoff for the lowest type).

## What happened

1. **No belief buffer (`belief_decay=0`)**: a leap-frogging war-of-attrition.
   Whichever agent just moved grabs most of the payoff (~0.45–0.48);
   the other gets driven to ~0 or slightly negative. Never settles —
   `MaxDiff` stays around 0.3–0.5 for as long as it was run.

2. **`belief_decay=0.8`** (the value that worked well for first-price,
   N=3/4): tames the leap-frogging into a slower, *shared* escalation —
   both agents drift up together toward the actor's structural ceiling
   (`b <= v`), ending with both bidding close to `b ≈ v` instead of
   `b ≈ v^2/2`.

3. **`belief_decay=0.97–0.98`** (much stronger memory): the escalation is
   mostly tamed — `MSE` and `MaxDiff` both decrease substantially — but
   a **slow, persistent, quasi-periodic cycle survives underneath**, with
   a period on the order of 150–250 rounds, visible in both the payoff
   trace and the MSE trace (recurring spikes, never damping out even
   after 1000 rounds with no early-stopping criterion active). The
   learned bid function ends up systematically off-equilibrium in the
   upper tail (first overbidding, in some runs undershooting into a
   plateau instead).

## Root cause: the all-pay first-order condition depends only on the critic's *derivative*

Comparing the optimality conditions:

- **First-price**: `payoff = (v - b) * P(win|b)`, FOC:
  `-P(win|b) + (v - b) * dP/db = 0`. The *level* of `P(win|b)` enters
  directly — it anchors the solution even when `dP/db` is noisy.
- **All-pay**: `payoff = v * P(win|b) - b`, FOC: `v * dP/db = 1`. The
  optimum depends **purely on the derivative** of the critic. There is no
  level term to anchor it.

This was confirmed directly (not just inferred): training a *fresh* actor
from scratch against the exact final critic from a stalled run reproduced
the same off-equilibrium bid function (0.4248 vs. the stalled run's 0.4202
at `v=1`, both far from the theoretical 0.5) — proving the actor's
optimization is working correctly given what it's handed. The problem is
what it's handed.

Measured density (`dP/db`, via autograd through the spline) vs. the
theoretical density at the point the training plateaued:

| b    | learned density | theoretical density |
|------|-----------------:|---------------------:|
| 0.40 | 1.81 | 1.12 |
| 0.42 | 1.27 | 1.09 |
| 0.45 | 0.68 | 1.05 |
| 0.48 | 0.59 | 1.02 |
| 0.50 | 0.67 | 1.00 |

A spike just below 0.42 followed by a deficit above it — exactly where
`v * dP/db` crosses below 1 (for v=1, where it needs density ≥ 1 to keep
bidding higher), explaining the plateau at that exact point.

## Why the CDF can look fine while the density is this far off

`P(win|b)` (what gets plotted and what "looks close" to theoretical) is
the **integral** of the density. Integration is a smoothing operator: a
density excess at b≈0.40 and a density deficit just after it largely
cancel out once integrated, so the CDF plot can look nearly perfect while
the pointwise density is off by 30–60%. Differentiation does not have
this cancellation — it is pointwise and amplifies exactly the local
structure that integration hides. This is a general, well-known fact
(numerically differentiating noisy/imperfect data is an ill-posed
problem; integrating it is well-posed), not specific to this codebase.
It explains why "the critic looks close to equilibrium" does not imply
"the actor converges close to equilibrium" for a mechanism whose
first-order condition is derivative-only.

## Why the derivative is wrong even though the CDF is close

The belief buffer (even at `belief_decay=0.98`) is a decayed mixture of
bids sampled across many rounds. Because the joint dynamics are still
cycling (see above) rather than stationary, that mixture blends bids from
genuinely different phases of the cycle (a "low" phase and a "high"
phase) — a real multimodal mixture, not sampling noise. Any faithful
density estimator (spline, KDE, a raw-point-cloud kernel estimator) would
show the same bump/dip given the same underlying samples; switching the
estimator would not, by itself, fix this, because the mixture is a
property of the data (the fact that training hasn't reached a stationary
point), not an artifact of the fitting method.

## Ideas considered but not (yet) pursued

- **Smoothing the critic more** (fewer quantiles, an explicit smoothing
  spline, or a raw-point-cloud soft comparator
  `sigmoid((b - b_opp) / tau)` in place of a fitted global model):
  plausible for reducing estimator-induced noise on top of genuine
  sampling noise, but does not address a genuine multimodal mixture
  coming from an unconverged cycle.
- **Tightening the stopping criteria further / even higher
  `belief_decay`**: tried up to `belief_decay=0.98` with
  `payoff_rel_tol` disabled (max-diff only) for 1000 rounds; the cycle
  persisted throughout, never satisfying `max_diff <= 0.02`.

## Takeaway

All-pay auctions are a recognized harder case for self-play/fictitious-play
training than first-price, for a structural reason: their optimality
condition is purely derivative-based, which makes them far more sensitive
to any imperfection (sampling noise or genuine non-stationarity) in the
opponent-bid density estimate than first-price's level-anchored condition.
This is consistent with it being an active research topic in auction
learning / mechanism design, not something expected to be fixed by a
hyperparameter change in this codebase.
