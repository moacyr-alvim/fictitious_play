import torch


def run_auction(bids: torch.Tensor):
    """First-price sealed-bid auction: highest bid wins and pays its own bid.

    bids: 1D tensor of shape (n_agents,).
    Returns (winner_index, payment). Ties broken by first occurrence of the max.
    """
    winner = torch.argmax(bids).item()
    payment = bids[winner].item()
    return winner, payment
