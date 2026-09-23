import torch
import torch.nn as nn


class _ActorBlock(nn.Module):
    """One growth stage's hidden units: a single ReLU layer whose output
    contributes an additive correction to the raw (pre-sigmoid) shading
    logit. Initialized to contribute exactly zero, so appending a fresh
    block never changes the function computed so far."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.input_layer = nn.Linear(1, width)
        self.output_layer = nn.Linear(width, 1, bias=False)
        nn.init.zeros_(self.output_layer.weight)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.output_layer(torch.relu(self.input_layer(v)))


class Actor(nn.Module):
    """Maps a private valuation v in [0, 1] to a bid b in [0, v].

    Bidding above one's own value is weakly dominated in a first-price
    sealed-bid auction, so the output is parameterized as b = v * sigmoid(raw)
    rather than left unconstrained, with raw = bias + sum of hidden blocks.
    hidden_size=0 makes raw a plain constant — the exact right-sized model
    for a symmetric IID auction, where the equilibrium shading factor
    (n-1)/n doesn't depend on v at all.

    Supports function-preserving growth (see grow()): appending a new block
    of hidden units never changes the function computed so far (the new
    block starts at zero output), letting fictitious play warm-start each
    larger stage from the smaller one's converged policy instead of from
    scratch. grow(..., freeze_previous=True) additionally makes every
    parameter from before this call permanently untrainable, so only the
    newest block (and, transitively, later ones) can still move.
    """

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.blocks = nn.ModuleList()
        if hidden_size > 0:
            self.blocks.append(_ActorBlock(hidden_size))

    @property
    def hidden_size(self) -> int:
        return sum(block.width for block in self.blocks)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        raw = self.bias.expand(v.shape[0], 1)
        for block in self.blocks:
            raw = raw + block(v)
        shading = torch.sigmoid(raw)
        return v * shading

    def grow(self, additional_hidden: int, freeze_previous: bool = False) -> "Actor":
        """Appends a new block of additional_hidden units (mutates self and
        returns self). If freeze_previous, the bias and every block added
        before this call become permanently non-trainable."""
        if freeze_previous:
            self.bias.requires_grad_(False)
            for block in self.blocks:
                for p in block.parameters():
                    p.requires_grad_(False)
        if additional_hidden > 0:
            device = self.bias.device
            self.blocks.append(_ActorBlock(additional_hidden).to(device))
        return self

    def unfreeze_all(self) -> None:
        """Makes every parameter trainable again (used by "temporary
        freeze": frozen only for a few rounds right after growing)."""
        self.bias.requires_grad_(True)
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad_(True)
