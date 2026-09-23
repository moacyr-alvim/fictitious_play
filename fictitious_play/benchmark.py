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
