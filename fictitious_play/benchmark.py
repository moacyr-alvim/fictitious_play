import torch

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


def theoretical_expected_payoff(n_agents: int = 2) -> float:
    """Ex-ante expected payoff of a bidder at the symmetric equilibrium.
    A bidder with value v wins with probability v^(n-1) (all others have
    lower value) and, at equilibrium, has interim payoff v^n / n regardless
    of whether the auction is first-price or all-pay (same efficient
    allocation and zero payoff for the lowest type, so the revenue
    equivalence theorem gives the same expected payment, hence the same
    payoff, under both payment rules). Averaging over v ~ U[0, 1] gives
    1 / (n * (n + 1)).
    """
    return 1.0 / (n_agents * (n_agents + 1))
