import torch
import torch.nn as nn


class Actor(nn.Module):
    """Maps a private valuation v in [0, 1] to a bid b in [0, v].

    Bidding above one's own value is weakly dominated in a first-price
    sealed-bid auction, so the output is parameterized as b = v * sigmoid(raw)
    rather than left unconstrained, with raw = bias + (optional single
    hidden ReLU layer). hidden_size=0 makes raw a plain constant — the
    exact right-sized model for a symmetric IID auction, where the
    equilibrium shading factor (n-1)/n doesn't depend on v at all.

    Supports function-preserving growth (see grow()): a network with more
    hidden units can be initialized to compute exactly the same function as
    a smaller one, letting fictitious play warm-start each larger stage
    from the smaller one's converged policy instead of from scratch.
    """

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = nn.Parameter(torch.zeros(1))
        if hidden_size > 0:
            self.input_layer = nn.Linear(1, hidden_size)
            self.output_layer = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.input_layer = None
            self.output_layer = None

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        raw = self.bias.expand(v.shape[0], 1)
        if self.hidden_size > 0:
            hidden = torch.relu(self.input_layer(v))
            raw = raw + self.output_layer(hidden)
        shading = torch.sigmoid(raw)
        return v * shading

    def grow(self, new_hidden_size: int) -> "Actor":
        """Returns a new Actor with new_hidden_size hidden units that
        computes exactly the same function as self at the moment of
        growth: existing hidden units and the bias are copied over
        unchanged, and the new units' output weights are zeroed so they
        contribute nothing until training moves them away from zero."""
        if new_hidden_size < self.hidden_size:
            raise ValueError("new_hidden_size must be >= the current hidden_size")

        grown = Actor(new_hidden_size)
        with torch.no_grad():
            grown.bias.copy_(self.bias)
            if new_hidden_size > 0:
                nn.init.zeros_(grown.output_layer.weight)
                if self.hidden_size > 0:
                    grown.input_layer.weight[:self.hidden_size].copy_(self.input_layer.weight)
                    grown.input_layer.bias[:self.hidden_size].copy_(self.input_layer.bias)
                    grown.output_layer.weight[:, :self.hidden_size].copy_(self.output_layer.weight)
        return grown
