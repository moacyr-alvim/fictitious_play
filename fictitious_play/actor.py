import torch
import torch.nn as nn


class Actor(nn.Module):
    """Maps a private valuation v in [0, 1] to a bid b in [0, v].

    Bidding above one's own value is weakly dominated in a first-price
    sealed-bid auction, so the output is parameterized as b = v * sigmoid(raw)
    rather than left unconstrained.
    """

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        shading = torch.sigmoid(self.net(v))
        return v * shading
