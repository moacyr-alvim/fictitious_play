import numpy as np
import torch
from scipy.stats import norm


class KDECritic:
    """Estimates P(win | b) = P(opponent's bid < b) as a Gaussian-kernel
    smoothed empirical CDF of the opponent's observed bids.

    Bids live on [0, inf), so a plain KDE leaks density/mass below 0 and
    underestimates P(win | b) for small b. This is corrected with the
    standard reflection trick: the opponent's bid sample is mirrored about 0
    before smoothing, which makes the augmented distribution symmetric
    around 0 (so its CDF at 0 is exactly 1/2), and the corrected CDF for
    b >= 0 recovers as F(b) = 2 * F_aug(b) - 1.

    The smoothed CDF is evaluated once, on a fixed grid, using a histogram
    of the (large) bid sample so cost does not scale with the sample size.
    Arbitrary query bids are obtained via differentiable linear
    interpolation on that grid, so gradients flow back into the actor
    that produced the bids.
    """

    def __init__(self, grid_size: int = 4001, grid_min: float = -0.25,
                 grid_max: float = 1.25, n_bins: int = 4000):
        self.grid_size = grid_size
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.n_bins = n_bins
        self.grid: torch.Tensor | None = None
        self.cdf_values: torch.Tensor | None = None

    @staticmethod
    def _silverman_bandwidth(samples: np.ndarray) -> float:
        n = len(samples)
        std = np.std(samples, ddof=1)
        q75, q25 = np.percentile(samples, [75, 25])
        iqr = q75 - q25
        spread = min(std, iqr / 1.34) if iqr > 0 else std
        spread = spread if spread > 0 else 1e-3
        return 0.9 * spread * n ** (-1 / 5)

    def fit(self, opponent_bids: np.ndarray) -> None:
        h = self._silverman_bandwidth(opponent_bids)
        augmented = np.concatenate([opponent_bids, -opponent_bids])

        bin_edges = np.linspace(augmented.min() - 4 * h, augmented.max() + 4 * h,
                                 self.n_bins + 1)
        counts, edges = np.histogram(augmented, bins=bin_edges)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        weights = counts / augmented.size

        grid = np.linspace(self.grid_min, self.grid_max, self.grid_size)
        z = (grid[:, None] - bin_centers[None, :]) / h
        cdf_aug = (norm.cdf(z) * weights[None, :]).sum(axis=1)

        cdf_corrected = np.clip(2 * cdf_aug - 1, 0.0, 1.0)
        cdf_corrected = np.maximum.accumulate(cdf_corrected)  # guard tiny fp non-monotonicity

        self.grid = torch.tensor(grid, dtype=torch.float32)
        self.cdf_values = torch.tensor(cdf_corrected, dtype=torch.float32)

    def predict_win_prob(self, bids: torch.Tensor) -> torch.Tensor:
        if self.grid is None:
            raise RuntimeError("KDECritic must be fit before calling predict_win_prob.")

        bids_clamped = bids.clamp(self.grid[0].item(), self.grid[-1].item())
        idx = torch.searchsorted(self.grid, bids_clamped.detach())
        idx = idx.clamp(1, len(self.grid) - 1)

        x0, x1 = self.grid[idx - 1], self.grid[idx]
        f0, f1 = self.cdf_values[idx - 1], self.cdf_values[idx]
        slope = (f1 - f0) / (x1 - x0)
        return f0 + slope * (bids_clamped - x0)
