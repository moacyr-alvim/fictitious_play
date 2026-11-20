from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq

AUCTION_TYPES = ("first_price", "all_pay")


def equilibrium_bid_first_price(v: torch.Tensor, n_agents: int = 2) -> torch.Tensor:
    """Symmetric Bayes-Nash equilibrium bid for a first-price sealed-bid
    auction with n_agents i.i.d. Uniform[0, 1] valuations: b*(v) = v * (n-1)/n.
    """
    return v * (n_agents - 1) / n_agents


def theoretical_win_probability_first_price(b: torch.Tensor, n_agents: int = 2) -> torch.Tensor:
    """P(win | b) implied by all other n_agents-1 opponents bidding at the
    first-price equilibrium b*(v) = v*(n-1)/n. Each opponent's equilibrium
    bid is then Uniform[0, c] with c = (n-1)/n, so beating all of them at
    once gives P(win | b) = (b/c)^(n-1) for b in [0, c].
    """
    c = (n_agents - 1) / n_agents
    if c == 0:
        return torch.ones_like(b)
    return (b.clamp(0.0, c) / c) ** (n_agents - 1)


def equilibrium_bid_all_pay(v: torch.Tensor, n_agents: int = 2) -> torch.Tensor:
    """Symmetric Bayes-Nash equilibrium bid for an all-pay auction (every
    bidder pays their own bid regardless of winning) with n_agents i.i.d.
    Uniform[0, 1] valuations: b*(v) = v^n * (n-1)/n.
    """
    return v ** n_agents * (n_agents - 1) / n_agents


def theoretical_win_probability_all_pay(b: torch.Tensor, n_agents: int = 2) -> torch.Tensor:
    """P(win | b) implied by all other n_agents-1 opponents bidding at the
    all-pay equilibrium b*(v) = v^n*(n-1)/n. Inverting for v = (b/c)^(1/n)
    with c = (n-1)/n, and raising to n_agents-1 (independent opponents)
    gives P(win | b) = (b/c)^((n-1)/n) for b in [0, c].
    """
    c = (n_agents - 1) / n_agents
    if c == 0:
        return torch.ones_like(b)
    return (b.clamp(0.0, c) / c) ** ((n_agents - 1) / n_agents)


def equilibrium_bid(v: torch.Tensor, n_agents: int = 2, auction_type: str = "first_price") -> torch.Tensor:
    if auction_type == "first_price":
        return equilibrium_bid_first_price(v, n_agents)
    if auction_type == "all_pay":
        return equilibrium_bid_all_pay(v, n_agents)
    raise ValueError(f"Unknown auction_type: {auction_type!r}")


def theoretical_win_probability(b: torch.Tensor, n_agents: int = 2, auction_type: str = "first_price") -> torch.Tensor:
    if auction_type == "first_price":
        return theoretical_win_probability_first_price(b, n_agents)
    if auction_type == "all_pay":
        return theoretical_win_probability_all_pay(b, n_agents)
    raise ValueError(f"Unknown auction_type: {auction_type!r}")


def equilibrium_bid_lambda(v: torch.Tensor, n_agents: int = 2, payment_share: float = 0.0) -> torch.Tensor:
    """Symmetric equilibrium bid for the generalized auction that interpolates
    between first-price and all-pay: the winner always pays its own bid, and
    every loser pays payment_share * their own bid (payment_share=0 recovers
    first-price; payment_share=1 recovers all-pay).

    Closed form (derived from the symmetric first-order condition, which
    telescopes into d/dv[a(v)*b(v)] = (n-1)*v^(n-1) with
    a(v) = (1-payment_share)*v^(n-1) + payment_share):

        b*(v) = (n-1)*v^n / (n * [(1-payment_share)*v^(n-1) + payment_share])
    """
    n, lam = n_agents, payment_share
    # At v=0 the bid is exactly 0 regardless of lam, but (1-lam)*0^(n-1)+lam
    # is itself 0 when lam=0, giving a raw 0/0 — clamp away from that and
    # patch the v<=0 boundary back in explicitly.
    v_safe = v.clamp(min=1e-12)
    denom = (1 - lam) * v_safe ** (n - 1) + lam
    bid = (n - 1) * v_safe ** n / (n * denom)
    return torch.where(v <= 0, torch.zeros_like(bid), bid)


def theoretical_win_probability_lambda(b: torch.Tensor, n_agents: int = 2,
                                        payment_share: float = 0.0) -> torch.Tensor:
    """P(win | b) implied by all other n_agents-1 opponents bidding at the
    equilibrium_bid_lambda equilibrium. Found by numerically inverting
    equilibrium_bid_lambda for v (it has no closed-form inverse for n > 2)
    and raising to n_agents - 1."""
    n, lam = n_agents, payment_share
    b_max = equilibrium_bid_lambda(torch.tensor(1.0), n, lam).item()

    def solve_v(bid: float) -> float:
        if bid <= 0:
            return 0.0
        bid = min(bid, b_max)
        f = lambda v: equilibrium_bid_lambda(torch.tensor(v), n, lam).item() - bid
        return brentq(f, 0.0, 1.0, xtol=1e-10)

    b_np = b.detach().cpu().numpy()
    v_np = np.array([solve_v(float(bid)) for bid in b_np.clip(0.0, b_max).reshape(-1)])
    p_win = v_np ** (n - 1)
    return torch.tensor(p_win, dtype=b.dtype, device=b.device).reshape(b.shape)


def theoretical_expected_payoff(n_agents: int = 2) -> float:
    """Ex-ante expected payoff of a bidder at the symmetric equilibrium.
    A bidder with value v wins with probability v^(n-1) (all others have
    lower value) and, at equilibrium, has interim payoff v^n / n regardless
    of the payment rule (first-price, all-pay, or any payment_share in
    equilibrium_bid_lambda in between) — same efficient allocation and
    zero payoff for the lowest type in every case, so the revenue
    equivalence theorem gives the same expected payment, hence the same
    payoff, throughout. Averaging over v ~ U[0, 1] gives 1 / (n * (n + 1)).
    """
    return 1.0 / (n_agents * (n_agents + 1))


@dataclass
class _AsymmetricRiskAverseEquilibrium:
    b_bar: float
    phi: tuple  # phi[i](b) -> P(agent i's bid < b): agent i's bid CDF, forward interpolators
    beta: tuple  # beta[i](v) -> agent i's equilibrium bid, inverse interpolators


@lru_cache(maxsize=32)
def _solve_asymmetric_risk_averse_equilibrium(
    risk_aversion: tuple[float, float], n_grid: int = 2000,
) -> _AsymmetricRiskAverseEquilibrium:
    """Numerically solves the Bayes-Nash equilibrium of a 2-bidder first-price
    auction where both bidders draw v ~ Uniform(0, 1) but have different CRRA
    risk-aversion coefficients rho_1, rho_2 (utility of winning with payoff
    x = v - b is u_i(x) = x^rho_i; rho_i=1 is risk-neutral, rho_i in (0, 1) is
    risk-averse).

    Unlike every other equilibrium in this module, this one has no closed
    form in general. Differentiating bidder i's objective
    phi_j(b) * (v_i - b)^rho_i (phi_j = bidder j's inverse bid function, i.e.
    its bid CDF) gives the coupled ODE system (b the shared independent
    variable, phi_i(b) = bidder i's bid CDF evaluated at b):

        d(phi_2)/db = phi_2(b) * rho_1 / (phi_1(b) - b)
        d(phi_1)/db = phi_1(b) * rho_2 / (phi_2(b) - b)

    with boundary conditions phi_1(0) = phi_2(0) = 0 and, at a shared upper
    bound b_bar (both bidders' highest type bids the same maximum — the
    standard regularity assumption for asymmetric first-price auctions, see
    Marshall, Meurer, Richard & Stromquist 1994 and Maskin & Riley 2000),
    phi_1(b_bar) = phi_2(b_bar) = 1.

    b_bar is itself unknown (a free boundary): this is solved by *shooting*,
    integrating the ODEs backward from b_bar (where they're regular; b=0 is
    the singular end) down to b~0 for a trial b_bar, and bisecting on the
    trial value until the backward solution reaches phi_1, phi_2 ~ 0
    simultaneously (verified empirically to pin down a unique b_bar to
    machine precision; for a wrong trial b_bar either the solution crosses
    the phi_i(b)=b diagonal, where the ODE is singular, before reaching
    b~0, or it doesn't reach phi~0 at all).

    When rho_1 == rho_2 (symmetric case), this reduces to the closed form
    b*(v) = v/(1+rho) (a special case of equilibrium_bid_lambda's formula
    at payment_share=0, n_agents=2) — used directly without solving the ODE.

    Correctness of the solved curves (the mutual Bayes-Nash best-response
    property, not just the FOC) is checked in tests/test_benchmark.py by
    directly re-optimizing each agent's objective against the other's solved
    curve and comparing to the curve itself.
    """
    rho1, rho2 = risk_aversion
    if abs(rho1 - rho2) < 1e-12:
        c = 1.0 / (1.0 + rho1)
        b_bar = c
        linear = interp1d([0.0, c], [0.0, 1.0], fill_value=(0.0, 1.0), bounds_error=False)
        beta = interp1d([0.0, 1.0], [0.0, c], fill_value=(0.0, c), bounds_error=False)
        return _AsymmetricRiskAverseEquilibrium(b_bar, (linear, linear), (beta, beta))

    def rhs(b, y):
        phi1, phi2 = y
        return [phi1 * rho2 / (phi2 - b), phi2 * rho1 / (phi1 - b)]

    def crossed_diagonal(b, y):
        return min(y[0] - b, y[1] - b) - 1e-6
    crossed_diagonal.terminal = True
    crossed_diagonal.direction = -1

    def reaches_zero(b_bar_trial: float, b_stop: float = 1e-7) -> bool:
        sol = solve_ivp(rhs, [b_bar_trial, b_stop], [1.0, 1.0], method="Radau",
                         rtol=1e-9, atol=1e-11, events=crossed_diagonal)
        return sol.t[-1] <= b_stop * 1.01

    lo, hi = 1e-6, max(1.0 / (1.0 + rho1), 1.0 / (1.0 + rho2))
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if reaches_zero(mid):
            lo = mid
        else:
            hi = mid
    b_bar = lo

    sol = solve_ivp(rhs, [b_bar, 1e-7], [1.0, 1.0], method="Radau",
                     rtol=1e-10, atol=1e-12, dense_output=True, max_step=b_bar / 2000)
    b_grid = np.linspace(1e-7, b_bar, n_grid)
    phi1_grid = np.clip(sol.sol(b_grid)[0], 0.0, 1.0)
    phi2_grid = np.clip(sol.sol(b_grid)[1], 0.0, 1.0)
    b_grid = np.concatenate([[0.0], b_grid])
    phi1_grid = np.concatenate([[0.0], phi1_grid])
    phi2_grid = np.concatenate([[0.0], phi2_grid])

    phi1 = interp1d(b_grid, phi1_grid, bounds_error=False, fill_value=(0.0, 1.0))
    phi2 = interp1d(b_grid, phi2_grid, bounds_error=False, fill_value=(0.0, 1.0))
    beta1 = interp1d(phi1_grid, b_grid, bounds_error=False, fill_value=(0.0, b_bar))
    beta2 = interp1d(phi2_grid, b_grid, bounds_error=False, fill_value=(0.0, b_bar))
    return _AsymmetricRiskAverseEquilibrium(b_bar, (phi1, phi2), (beta1, beta2))


def equilibrium_bid_risk_averse(v: torch.Tensor, agent_idx: int,
                                 risk_aversion: tuple[float, float]) -> torch.Tensor:
    """Agent agent_idx's equilibrium bid in the 2-bidder asymmetric-CRRA
    first-price auction described in _solve_asymmetric_risk_averse_equilibrium.
    """
    eq = _solve_asymmetric_risk_averse_equilibrium(tuple(risk_aversion))
    v_np = v.detach().cpu().numpy()
    b_np = eq.beta[agent_idx](v_np)
    return torch.tensor(b_np, dtype=v.dtype, device=v.device).reshape(v.shape)


def theoretical_win_probability_risk_averse(b: torch.Tensor, agent_idx: int,
                                             risk_aversion: tuple[float, float]) -> torch.Tensor:
    """P(win | b) for agent agent_idx against its (single) opponent, both
    playing the equilibrium described in _solve_asymmetric_risk_averse_equilibrium
    — i.e. the opponent's bid CDF evaluated at b."""
    eq = _solve_asymmetric_risk_averse_equilibrium(tuple(risk_aversion))
    opponent_idx = 1 - agent_idx
    b_np = b.detach().cpu().numpy()
    p_np = eq.phi[opponent_idx](b_np)
    return torch.tensor(p_np, dtype=b.dtype, device=b.device).reshape(b.shape)


def common_value_interior_range(noise: float, margin_multiples: float = 3.0) -> tuple[float, float]:
    """(lo, hi) for the true value V such that every signal S_i = V + eps_i
    (eps_i ~ Uniform(-noise, noise)) and every other signal within 2*noise of
    it stays strictly inside (0, 1) — the region equilibrium_bid_common_value
    and common_value_posterior_mean are exact in (see their docstrings for
    why the formulas break down outside it). margin_multiples=3 gives one
    full noise-width of slack beyond the 2*noise a single signal's own
    posterior needs, so a second signal within the first's support stays
    interior too."""
    margin = margin_multiples * noise
    return margin, 1.0 - margin


def equilibrium_bid_common_value(signal: torch.Tensor, noise: float) -> torch.Tensor:
    """Symmetric Bayes-Nash equilibrium bid for the N=2 first-price
    common-value auction: a true value V ~ Uniform(0, 1) shared by both
    bidders, each observing a private signal S_i = V + eps_i with
    eps_i ~ i.i.d. Uniform(-noise, noise) (the classic "mineral rights"/
    wallet-game model). Exact only in the interior region
    common_value_interior_range(noise) — derived assuming every signal's
    own Bayesian posterior for V is a non-truncated Uniform(S_i-noise,
    S_i+noise), which requires S_i to be at least `noise` away from both
    0 and 1.

    Derivation (linear-equilibrium ansatz b(x) = x - k, verified as a
    mutual best response in tests/test_benchmark.py): with V | S_1=x ~
    Uniform(x-noise, x+noise) and S_2 | S_1=x then a Triangular(center=x,
    half-width=2*noise) distribution (the convolution of two independent
    Uniform(-noise, noise) noises), the first-order condition at the
    symmetric equilibrium reduces to k = F(x|x)/f(x|x) where F, f are
    that triangular distribution's CDF/density evaluated at its own
    center — F(x|x)=1/2, f(x|x)=1/(2*noise) for *any* symmetric
    continuous noise distribution, giving the noise-shape-independent
    result k = noise exactly (not noise/(N-1) or any other N-dependent
    quantity — this is the N=2 case specifically; the N>=3 generalization
    needs the actual order-statistic distribution of the max of N-1
    triangular draws, not just a single one, and is *not* yet derived
    correctly here — see docs/findings_common_value.md).
    """
    return signal - noise


def common_value_posterior_mean(own_signal: torch.Tensor, other_signal: torch.Tensor,
                                 noise: float) -> torch.Tensor:
    """E[V | S_1=own_signal, S_2=other_signal] for the model in
    equilibrium_bid_common_value: both signals' Uniform(S_i-noise,
    S_i+noise) posteriors for V must hold simultaneously, so V's joint
    posterior is Uniform over their overlap
    (max(own,other)-noise, min(own,other)+noise), whose midpoint is simply
    (own_signal + other_signal) / 2 — exact for any other_signal within
    2*noise of own_signal (i.e. anywhere the two Uniform(.-noise,.+noise)
    posteriors actually overlap), not just at other_signal == own_signal.
    """
    return (own_signal + other_signal) / 2.0


def common_value_posterior_mean_gaussian(own_signal: torch.Tensor, other_signal: torch.Tensor,
                                          prior_mean: float, prior_std: float,
                                          noise_std: float) -> torch.Tensor:
    """E[V | S_1=own_signal, S_2=other_signal] for the Normal-Normal
    common-value model: V ~ Normal(prior_mean, prior_std^2), S_i = V +
    eps_i, eps_i ~ i.i.d. Normal(0, noise_std^2). Standard conjugate-
    Gaussian/bivariate-normal conditioning gives a closed-form linear
    formula in both signals, with no boundary caveat at all (every normal
    distribution here has full support, so unlike
    equilibrium_bid_common_value's bounded-uniform-noise model, there's no
    "interior region" restriction needed):

        E[V|x,y] = c*(x+y) + prior_mean*(1-2c),   c = prior_std^2 / (2*prior_std^2 + noise_std^2)

    Dropping the boundary is not a free lunch, though: unlike the uniform-
    noise model, E[V|x,x] != x here in general (an informative Gaussian
    prior always shrinks the estimate toward prior_mean, even when both
    signals agree exactly) — it's prior_mean + 2c*(x-prior_mean), a
    shrunk version of x. That breaks the algebraic identity that made the
    uniform-noise equilibrium come out linear (there, the ratio
    F(x|x)/f(x|x) driving the shading was constant in x; here it's an
    inverse-Mills-ratio-shaped function of x, genuinely nonlinear), so no
    closed-form symmetric equilibrium is derived (or implemented) for this
    variant yet — see docs/findings_common_value.md.
    """
    c = prior_std ** 2 / (2 * prior_std ** 2 + noise_std ** 2)
    return c * (own_signal + other_signal) + prior_mean * (1 - 2 * c)
