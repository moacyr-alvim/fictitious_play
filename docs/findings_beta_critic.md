# Beta-distribution critic: replacing the spline with an exact, 2-parameter fit

**Status: strong early result.** On N=2 all-pay, with no PINN and no EMA —
just the critic swapped from `SplineCritic` to `BetaCritic`, everything
else identical (same seed, same belief buffer, same disabled early-stopping
so the comparison is round-for-round) — `BetaCritic` converges cleanly to
`MaxDiff < 0.02` for both agents by round 55. `SplineCritic`, run for the
full 150-round budget under the same settings, never converges: it gets
close once (round ~124, `MaxDiff0=0.0105`) and then relapses into the same
leap-frogging/cycling pattern documented in `docs/findings_all_pay.md` and
`docs/findings_pinn_regularizer.md`, ending the 150-round budget at
`MaxDiff0=0.199, MaxDiff1=0.140`. This is the first setup in this project
where all-pay is stabilized *without* any PINN regularizer, EMA shadow, or
homotopy continuation.

## Motivation: an exact distributional identity

The symmetric Bayes-Nash equilibrium bid for the `payment_share`-generalized
auction (`benchmark.equilibrium_bid_lambda`) reduces, at the two endpoints,
to a pure power law in the valuation:

- first-price (`payment_share=0`): `b*(v) = v * (n-1)/n`, i.e. `p=1`.
- all-pay (`payment_share=1`): `b*(v) = v^n * (n-1)/n`, i.e. `p=n`.

In general, `b*(v) = c * v^p` with `v ~ Uniform(0, 1)`. A standard change of
variables gives `b*/c ~ Beta(1/p, 1)` **exactly**. `Beta(alpha, 1)` in turn
has a closed-form CDF with no incomplete-beta function needed:

```
F(x) = x^alpha,   x in [0, 1]
```

Verified numerically (`n_agents=5`, 500k samples): empirical MLE `alpha_hat`
vs. theoretical `1/p`:

| auction      | p | alpha_hat (empirical) | 1/p (theoretical) |
|--------------|---|------------------------|--------------------|
| first-price  | 1 | 0.9997                  | 1.0000             |
| all-pay      | 5 | 0.1999                  | 0.2000             |

So a critic of the form `P(win\|b) = (b/c)^alpha` is **exactly unbiased**
at both ends of the `payment_share` continuum — precisely where the spline
critic's finite-sample quantile noise has been the root cause of all-pay's
fragility throughout this project. Unlike a general `Beta(alpha, beta)`
(which needs the incomplete beta function, not differentiable in
`torch.distributions.Beta.cdf` as of this PyTorch version — confirmed via
`NotImplementedError` on `.backward()`), fixing `beta=1` keeps everything a
plain, exactly-differentiable power law.

Away from the two endpoints, the true equilibrium bid is *not* exactly of
the `c*v^p` form (see the closed form in `equilibrium_bid_lambda`'s
docstring), so a `Beta(alpha, 1)` fit carries some model bias at
intermediate `payment_share`. Measured directly (N=2, MLE fit on 2M true-
equilibrium samples, max abs CDF error over a grid):

| payment_share | alpha_hat | max \|empirical CDF - x^alpha_hat\| |
|---------------|-----------|---------------------------------------|
| 0.00          | 1.0000    | 0.0004                                 |
| 0.25          | 0.6847    | 0.0349                                 |
| 0.50          | 0.5905    | 0.0240                                 |
| 0.75          | 0.5369    | 0.0111                                 |
| 1.00          | 0.5002    | 0.0004                                 |

Up to ~3.5% CDF error mid-continuum — a real cost, but small next to the
variance a quantile spline carries at comparable sample sizes, and zero
exactly at the two stages (first-price, all-pay) that matter most.

## Implementation

`fictitious_play.critic.BetaCritic`, same `.fit(opponent_bids)` /
`.predict_win_prob(bids)` interface as `SplineCritic`:

- `fit`: `scale = max(bids)`; `alpha_hat = -1 / mean(log(bids/scale))` (the
  closed-form MLE for `Beta(alpha, 1)` with known support). Only clips the
  float64-underflow singularity at exactly 0 (`np.finfo(float64).tiny`),
  *not* a larger epsilon — all-pay's heavy mass near `v=0` means a sizeable
  fraction of samples can be many orders of magnitude below `1e-6`, and
  clipping those up would bias `alpha_hat` upward (confirmed: using
  `eps=1e-6` here gave `alpha_hat=0.213` instead of the true `0.200` at
  `n_agents=5`, all-pay).
- `predict_win_prob`: `(bids/scale).clamp(eps, 1.0) ** alpha`, `eps=1e-6`
  here *is* appropriate — this clamp guards the backward pass (the true
  derivative diverges at `b=0` when `alpha<1`, a real feature of the
  equilibrium, not a bug, but literal `0**(alpha-1)` would backprop `inf`).
  Bids above `scale` clamp to probability 1 with derivative 0, matching the
  boundary convention already used in `opponent_inverse_and_slope`.

Wired as `FictitiousPlayTrainer(critic_type="beta")` (default remains
`"spline"`), and exposed as `--critic-type {spline,beta}` on
`run_experiment.py`, `run_homotopy_experiment.py`, and
`run_n_continuation_experiment.py`.

## N=2 all-pay head-to-head

Both runs: `n_agents=2`, `auction_type=all_pay`, `sample_size=100_000`,
`belief_decay=0.9`, `patience=100`, `max_steps=8000`, `seed=0`,
`payoff_rel_tol=-1` (disabled, so neither run can stop early on a merely
*stable-looking* payoff — only the real `max_diff_threshold=0.02`
criterion can end a run, to keep the comparison honest).

- **`BetaCritic`**: converges at round 55. `MaxDiff` decreases essentially
  monotonically from round 0 (`0.26`) to round 55 (`0.012-0.016`); payoffs
  track the theoretical `0.16667` closely from round ~40 on
  (`checkpoints/checkpoint_allpay_beta_fair.pt`).
- **`SplineCritic`**: never converges in the full 150-round budget. Gets
  close once around round 124 (`MaxDiff0=0.0105`, but `MaxDiff1` still
  `0.106` — the two agents are out of phase) then relapses at round 126
  (`MaxDiff0` jumps back to `0.18`) into the same leap-frogging cycle for
  the rest of the budget: payoffs alternate between roughly `0.09-0.12` and
  `0.17-0.19` depending on which agent just moved, round after round
  (`checkpoints/checkpoint_allpay_spline_fair.pt`).

This is a 2.7x+ reduction in rounds-to-convergence (55 vs. "never, in
150"), achieved purely by the critic's functional form — no PINN term, no
EMA shadow, no homotopy continuation. It suggests the instability all of
that earlier machinery was built to route around may have been, at least in
large part, a critic-estimator-noise problem rather than an irreducible
property of the all-pay FOC itself.

## Caveats and open questions

- Tested only at N=2, `payment_share=1.0` fixed (no intermediate stages).
  The measured ~3.5% mid-continuum CDF bias means a homotopy run stepping
  `payment_share` from 0 to 1 with `BetaCritic` throughout has not yet been
  checked for whether that bias slows or destabilizes the intermediate
  stages — plausible given how small the bias is, but not verified.
- Not yet tested at N>=3, in particular not at N=5 where
  `docs/findings_pinn_regularizer.md` found the spline critic's finite
  resolution collapses against the extreme dynamic range of all-pay
  equilibrium bids across valuations. `BetaCritic`'s power-law form is
  scale-invariant in log space, which is a plausible reason it might handle
  that dynamic range far better than a spline — but this needs to be run,
  not assumed.
- Not yet combined with the opponent-based PINN/EMA hybrid. Since
  `BetaCritic` targets the same noisy-derivative root cause from a
  different angle, the two may be redundant (no added benefit, in which
  case prefer the simpler `BetaCritic`-only setup) or may compose (e.g.
  `BetaCritic` for the payoff loss, opponent-PINN+EMA for an additional
  regularizer) — not yet measured either way.
- The per-opponent MLE fit assumes the opponent's *current* bid sample is
  roughly single-power-law-shaped. Early in training, before any agent's
  strategy resembles the equilibrium shape, this could fit `BetaCritic`
  worse than `SplineCritic` would (more model bias, for a target that
  obviously isn't `Beta(alpha,1)`-shaped) — the observed round-55
  convergence above means this was not a practical problem for N=2 all-pay
  from a fresh random initialization, but merits a closer look if N=5 or
  the homotopy case misbehaves.
