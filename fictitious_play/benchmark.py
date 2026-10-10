import numpy as np
import torch
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
