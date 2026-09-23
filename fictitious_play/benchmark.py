import torch


def equilibrium_bid(v: torch.Tensor, n_agents: int = 2) -> torch.Tensor:
    """Symmetric Bayes-Nash equilibrium bid for a first-price sealed-bid
    auction with n_agents i.i.d. Uniform[0, 1] valuations: b*(v) = v * (n-1)/n.
    """
    return v * (n_agents - 1) / n_agents


def theoretical_win_probability(b: torch.Tensor, n_agents: int = 2) -> torch.Tensor:
    """P(win | b) implied by all other n_agents-1 opponents bidding at the
    equilibrium b*(v) = v*(n-1)/n. Each opponent's equilibrium bid is then
    Uniform[0, c] with c = (n-1)/n, so beating all of them at once gives
    P(win | b) = (b/c)^(n-1) for b in [0, c].
    """
    c = (n_agents - 1) / n_agents
    if c == 0:
        return torch.ones_like(b)
    return (b.clamp(0.0, c) / c) ** (n_agents - 1)


def theoretical_expected_payoff(n_agents: int = 2) -> float:
    """Ex-ante expected payoff of a bidder at the symmetric equilibrium.
    A bidder with value v wins with probability v^(n-1) (all others have
    lower value) and pays b*(v), giving payoff v^n / n; averaging over
    v ~ U[0, 1] gives 1 / (n * (n + 1)).
    """
    return 1.0 / (n_agents * (n_agents + 1))
