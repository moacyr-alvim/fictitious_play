# Mixture-of-Betas critic: a compact, closed-form alternative to the NN critic

**Status: promising, not fully closed out.** `MixtureBetaCritic` (K=2
components) is ~100-150x faster per round than `MonotonicNNCritic` (closed-
form EM, not Adam), matches it in quality on the cases tried, and — combined
with finer homotopy granularity — fixed a severe asymmetric-collapse failure
mode that the N=4 all-pay continuation hit under NN-critic-era round
budgets. It has not yet been pushed to full convergence on either target
scenario (asymmetric risk-aversion, N=4 all-pay); see "Open questions".

## Core idea

`BetaCritic` (`docs/findings_beta_critic.md`) is exact at the first-price
and all-pay equilibria because those equilibria are exactly power-law
(`b*(v) = c*v^p`), which transforms exactly into `Beta(alpha, 1)` with
`alpha=1/p`. It is *not* exact once the equilibrium stops being a pure power
law — the asymmetric-risk-aversion auction being the motivating case
(`docs/findings_risk_aversion.md`). `MonotonicNNCritic` fixes that bias at
the cost of becoming the per-round bottleneck: a small monotonic MLP
refit by up to several thousand Adam steps *every round*, warm-started but
still a noisy, iterative optimization repeated against a slightly-moving
target each time — the root cause of the persistent oscillation documented
below.

The question this work answers: can a critic stay as cheap and
non-iterative as `BetaCritic` while gaining some of `MonotonicNNCritic`'s
shape flexibility? Answer: yes, by mixing `K` scaled `Beta(alpha_k, 1)` CDFs
sharing one scale:

```
F(x) = sum_k pi_k * (x/scale)^alpha_k,   pi_k >= 0,  sum_k pi_k = 1
```

Two things make this both monotonic and cheap "for free":

- **Monotonic by construction, not by architecture trick.** Each component
  `(x/scale)^alpha_k` is itself a valid CDF; a convex combination of CDFs
  (nonnegative weights summing to 1) is always a CDF. No positive-weight
  reparametrization or activation choice is needed, unlike
  `MonotonicNNCritic`.
- **Fitting is still closed-form, just iterated a few dozen times, not
  thousands.** `Beta(alpha, 1)` transforms exactly to `Exponential(alpha)`
  under `Y = -log(X)` (CDF of X is `x^alpha`, so
  `P(-log X > y) = P(X < e^-y) = e^(-alpha*y)`, i.e. `Y ~ Exp(alpha)`).
  So fitting a mixture of `Beta(alpha_k, 1)` on the (scaled) bids reduces
  exactly to fitting a mixture of exponentials on their `-log`, which has
  the textbook closed-form EM update (E-step: responsibilities; M-step:
  both `pi_k` and `alpha_k` are plain weighted means — no numerical
  optimizer object, no learning rate). ~50 EM iterations settle it.

One scale (not one per component) shared across all `K` components, same
reasoning as `BetaCritic`: the largest observed opponent bid is the exact
deterministic boundary of the support regardless of how many components
describe the shape below it.

**On parametrization** (came up when explaining this to a collaborator):
this is `Beta(alpha, 1)`, not the traditional 2-free-parameter
`Beta(alpha, beta)` — `beta` is pinned to 1 deliberately, because that's
exactly what collapses the CDF to the closed form `x^alpha` (a general
`Beta(alpha, beta)` needs the regularized incomplete beta function, no
closed form, and not differentiable through `torch.distributions.Beta.cdf`
as of the PyTorch version checked). So `MixtureBetaCritic(n_components=2)`
has 4 free parameters total: `alpha_1, alpha_2`, one free mixture weight
(`pi_2 = 1 - pi_1`), and the shared scale `c` — not 6.

Implementation: `fictitious_play.critic.MixtureBetaCritic`, same
`.fit(opponent_bids)` / `.predict_win_prob(bids)` interface as the other
critics, warm-started round to round (the EM's initial guess is the
previous round's converged `alpha`/`pi`) exactly like `MonotonicNNCritic`.
Wired as `critic_type="beta_mixture"` everywhere `critic_type="nn"` already
was (persistent per-agent instance, extended by `add_agents`, round-tripped
through checkpoints). Unit tests in `tests/test_critic.py` include a direct
check that it fits a bimodal (non-power-law) shape measurably better than
single-component `BetaCritic` — the whole point of `K>1`.

## Speed: ~100-150x over the NN critic

Direct (non-homotopy) 150-round validation runs, `sample_size=100_000`,
`max_steps=8000`, `belief_decay=0.95`, identical everything else:

| scenario                          | critic        | wall-clock (150 rounds) | per-round |
|------------------------------------|---------------|--------------------------|-----------|
| risk_aversion=[0.3, 1.0]           | `nn`          | multiple hours (killed repeatedly at the 2h background cap, never finished 150 rounds in one window) | ~minutes |
| risk_aversion=[0.3, 1.0]           | `beta_mixture`| 5m10s                     | ~2.1s     |
| all_pay, N=2, lambda=1.0           | `beta_mixture`| 4m12s                     | ~1.7s     |
| all_pay, N=2, lambda=1.0 (control) | `beta`        | 2m28s                     | ~1.0s     |

(The `beta` control run exists specifically to separate "is the mixture
slow/wrong" from "is 150 rounds just not enough for all-pay" — see below.)
`beta_mixture` is ~1.7-2x the cost of plain `beta` (expected: ~50 EM
iterations of small numpy ops vs. one closed-form MLE formula), both
dwarfed by `nn`'s per-round Adam loop.

**Caveat discovered mid-session, not about the critic**: running two such
experiments *in parallel* on this sandbox (4 CPU cores) produced a
40-80x slowdown (7-85s/round instead of ~1-2s/round) purely from CPU
oversubscription — PyTorch/NumPy each try to use all available cores per
process. Isolating the exact same run to a single process reproduced the
expected ~1-2s/round immediately. Run these sequentially, not concurrently,
on a small-core-count machine.

## Quality: comparable to the NN critic, same residual artifacts

**Standalone fit check** (fit against 500k samples drawn directly from the
*true* all-pay N=2 equilibrium bid distribution, not from self-play):
`MixtureBetaCritic` reproduces the exact theoretical CDF to within sampling
noise, same as `BetaCritic` — confirms the EM fit itself is correct; any
quality gap seen below comes from the self-play training dynamics, not
from a critic-fitting bug.

**Direct validation, 150 rounds, no homotopy**:

- `risk_aversion=[0.3, 1.0]` and the mirrored `[1.0, 0.4]`: the more
  risk-averse agent's learned bid function matches its theoretical curve
  closely across most of the valuation range; the other agent shows a
  visible discontinuity ("step") near `v~0.83-0.9` and a residual MSE
  plateau around `1e-3` rather than continuing to decrease — the same
  qualitative artifact `MonotonicNNCritic` produced under homotopy (see
  `docs/findings_risk_aversion.md`), suggesting this step is a property of
  the alternating-best-response dynamics in this regime, not specific to
  either critic.
- `all_pay, N=2, lambda=1.0`: learned bid undershoots the theoretical curve
  substantially (`~0.37` vs. theoretical `0.5` at `v=1`). Re-ran the
  identical settings with plain `beta` as a control: **same undershoot
  pattern** (`~0.40` vs `0.5`). This is not a mixture-specific regression —
  150 rounds at `belief_decay=0.95` is simply not enough for all-pay to
  converge for *either* critic (consistent with all-pay's documented
  convergence fragility, `docs/findings_all_pay.md`); the N=2
  `beta`-only "fair" baseline that converged cleanly
  (`docs/findings_beta_critic.md`) used more rounds.
- `all_pay, N=3`, direct (no N-continuation): even bigger gap (`~0.38` vs.
  theoretical `~0.67` at `v=1`, `MaxDiff~0.27-0.29`) — confirms (again)
  that all-pay gets harder to train from scratch as `N` grows, which is
  exactly why this project already had N-continuation (warm-start N+1 from
  a converged N) rather than training high-N from scratch.

## Homotopy at NN-tuned round budgets: fast, but quality regresses

First attempt: re-ran the exact homotopy schedules previously tuned for
`MonotonicNNCritic` (risk-aversion: 8 stages, `rho` 1.0->0.3, 50
rounds/stage; all-pay: 11 lambda stages 0.0->1.0 at N=2 (60 rounds/stage)
then grow to N=3, N=4 (60 rounds/stage each)), just with
`critic_type="beta_mixture"`. Both completed in minutes instead of days
(302 and 720 rounds respectively — no multi-hour resume cycles needed at
all), but:

- **Risk-aversion**: MSE *increased* stage over stage (roughly `1e-4` ->
  `1e-3` across the 6 `rho`-ramp transitions) instead of staying low — the
  opposite of what homotopy continuation is supposed to buy. Final
  `rho=[0.3,1.0]` state: same step-discontinuity artifact as the direct run.
- **All-pay N=4**: the lambda ramp at N=2 was fine, but growing to N=3 then
  N=4 in only 60 rounds each produced a visibly broken, asymmetric result —
  agents developed wildly different "cliff"-shaped bid functions (near-zero
  until `v~0.85-0.9`, then a sharp jump), nothing like the smooth symmetric
  curve expected. `MaxDiff` spiked by ~2 orders of magnitude right at the
  N=4 growth transition and never recovered within the 60-round budget.

Conclusion: those round-budgets were sized for a critic 100x slower — they
were never really "enough rounds," just "as many rounds as fit in the
multi-hour windows NN-critic runs got." With `beta_mixture`'s speed, there
was no reason not to use far more rounds per stage and finer stage
granularity, which the next run tested directly (for all-pay only; the
risk-aversion homotopy has *not* yet been re-run with a bigger budget — see
"Open questions").

## Homotopy, refined: finer lambda steps + much bigger per-stage budgets

Same all-pay N=4 continuation, `lambdas` refined from 11 steps of 0.1 to 21
steps of 0.05, `rounds_per_lambda_stage` 60->150, `rounds_per_n_stage`
60->400 (the N-growth stages were where the previous attempt broke worst):

- **Lambda ramp (N=2, 0.0->1.0)**: converged for real this time —
  `max_diff_threshold=0.02` was hit and confirmed (`"Convergência
  detectada"`) at multiple stages, including the final lambda=1.0 stage at
  round 2063.
- **Growth to N=3, then N=4 (400 rounds each)**: dramatically better than
  the coarse run — all 4 agents converge to **symmetric**, smooth bid
  functions (no more cliffs), with closely matching `MaxDiff` (~0.20-0.21)
  and payoffs (~0.073) across all four agents. Still not fully converged,
  though: a real residual gap remains against the theoretical curve
  (learned bid reaches `~0.55` vs. theoretical `~0.75` at `v=1`, roughly a
  25-30% relative gap), and `MaxDiff` plateaus around `0.03-0.2` rather than
  dropping below the `0.02` threshold even with the full 400-round budget
  per growth stage.

So: refining granularity + spending the speed budget on more rounds per
stage fixed the *qualitative* failure (asymmetric collapse) but not yet the
*quantitative* one (full convergence at N=4). The natural next lever is an
even bigger `rounds_per_n_stage` budget (or an intermediate N=3.5-style
step, not literally possible with integer agent counts, but e.g. repeating
part of the lambda ramp after each N-growth step) — not yet tried.

## Confirms an older result, directly relevant here

Re-examined an earlier (pre-this-session) result while discussing it with a
collaborator: N=5 all-pay, plain `BetaCritic`, **no** homotopy and **no**
reinforced belief buffer, converges beautifully — all 5 learned bid
functions sit almost exactly on the theoretical curve, MSE drops cleanly
from `~3e-2` to `~3e-4` over ~220 rounds with no oscillation
(`checkpoints/checkpoint_n5_allpay_beta_fair.pt`, plots already in the
repo). This is the single clearest illustration of the theme this whole
document is about: all the extra machinery (homotopy, belief buffer
tuning, `MonotonicNNCritic`, now `MixtureBetaCritic`) is compensating for
critic model-bias in regimes where the true equilibrium *isn't* a pure
power law (asymmetric risk-aversion) or where direct high-N training
breaks symmetry (all-pay N>=3 from a cold start) — not for some
irreducible instability in the fictitious-play dynamics itself. Where
`BetaCritic`'s exact assumption holds (first-price, all-pay, any fixed
`payment_share`, trained either at low N or via N-continuation), nothing
else is needed at all.

For reference, the closed-form `F(b) = P(win | b)` at the symmetric N-player
all-pay Bayes-Nash equilibrium (derived from the FOC
`b'(v) = (N-1)*v^(N-1)` -> `b*(v) = (N-1)/N * v^N`, then inverting and
raising to the `N-1` power for `N-1` independent opponents):

```
F(b) = (N*b / (N-1)) ^ ((N-1)/N),   b in [0, (N-1)/N]
```

— exactly `fictitious_play.benchmark.theoretical_win_probability_all_pay`,
and exactly the `Beta(alpha=(N-1)/N, 1)` CDF scaled by `c=(N-1)/N` that
motivates `BetaCritic` in the first place.

## Infrastructure fix found along the way

While building the homotopy scripts for this critic, found and fixed a
real bug (unrelated to the critic itself, but what made the long NN-critic
runs earlier in this project take as long as they did): on `--resume`,
`run_risk_aversion_homotopy_experiment.py` and `run_n_continuation_experiment.py`
re-granted the *current* stage a full fresh round budget every time the
process restarted, regardless of how many rounds that stage had already
run since its last transition — so a stage that never satisfied the
convergence criterion would never advance, just accumulate rounds forever
across repeated kill/resume cycles (confirmed: one lambda stage ran 214
rounds against a 60-round nominal budget before the fix, purely from being
interrupted and resumed repeatedly by the sandbox's 2-hour background-task
cap). Fixed by computing rounds already elapsed since the last
`growth_rounds` boundary and only running the remainder (or none at all, if
already exhausted) before advancing — see the "Fix homotopy resume to
respect remaining round budget per stage" commit.

## Open questions

- **Risk-aversion homotopy has not been re-run with a bigger budget.** The
  "homotopy at NN-tuned round budgets" run above showed MSE degrading stage
  over stage; unlike all-pay, this has not yet been retested with the
  larger `rounds_per_stage` / finer `rho` schedule that fixed the analogous
  all-pay problem. Expected to help for the same reason it did there, but
  unconfirmed.
- **All-pay N=4 still has a ~25-30% residual gap** even with 400
  rounds/growth-stage. Worth trying a substantially larger
  `rounds_per_n_stage` (the lambda ramp needed 150 rounds/stage to fully
  converge; the growth stages may need more, not less, given more agents
  are simultaneously re-equilibrating) before concluding this is a floor
  rather than a round-budget issue.
- **`n_components=2` was chosen because that's what was asked for, not
  tuned.** Whether `K=3` helps the risk-aversion step-discontinuity
  artifact (a single extra inflection a 2-component mixture might not have
  enough shape freedom to capture) is untested.
- **EM warm-start across rounds, not just within a fit** — like
  `MonotonicNNCritic`'s Adam state, the EM's responsibilities can in
  principle settle into a stable-but-suboptimal local assignment across
  many rounds (label-switching, one component's weight collapsing near 0
  and staying there). Not observed causing a problem in any run above, but
  not specifically tested for either.
