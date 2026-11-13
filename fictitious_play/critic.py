import numpy as np
import torch


class SplineCritic:
    """Estimates P(win | b) = P(opponent's bid < b) with a monotone cubic
    Hermite spline (the PCHIP construction, Fritsch-Carlson derivatives)
    fit to quantiles of the opponent's bid sample.

    Unlike a kernel density estimate, this needs no bandwidth selection and
    no boundary correction: quantiles of the sample already give a proper
    empirical CDF that respects b >= 0 exactly (no probability mass can leak
    below the smallest observed bid), and with a large sample a modest
    number of quantile points already traces the CDF closely, so
    interpolating them with a monotone spline is both smooth (needed for
    the actor's gradient) and, unlike the previous dense KDE matrix
    evaluation, cheap regardless of the sample size.
    """

    def __init__(self, n_quantiles: int = 400, device: torch.device | str | None = None):
        self.n_quantiles = n_quantiles
        self.device = torch.device(device) if device else torch.device("cpu")
        self.bid_points: torch.Tensor | None = None
        self.cdf_values: torch.Tensor | None = None
        self.derivatives: torch.Tensor | None = None

    @staticmethod
    def _monotone_derivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fritsch-Carlson derivatives: the same choice PCHIP uses to keep
        the Hermite spline monotone wherever the data is monotone."""
        secants = np.diff(y) / np.diff(x)
        m = np.empty_like(y)
        m[0] = secants[0]
        m[-1] = secants[-1]
        same_sign = secants[:-1] * secants[1:] > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            harmonic = 2 * secants[:-1] * secants[1:] / (secants[:-1] + secants[1:])
        m[1:-1] = np.where(same_sign, harmonic, 0.0)
        return m

    def fit(self, opponent_bids: np.ndarray) -> None:
        quantile_levels = np.linspace(0.0, 1.0, self.n_quantiles)
        bid_points = np.quantile(opponent_bids, quantile_levels)

        # De-duplicate tied bid values (keep the *last*, i.e. largest, quantile
        # level for each distinct bid, since the array is sorted ascending).
        unique_x, first_idx = np.unique(bid_points, return_index=True)
        _, last_idx_rev = np.unique(bid_points[::-1], return_index=True)
        last_idx = len(bid_points) - 1 - last_idx_rev
        cdf_values = quantile_levels[np.sort(last_idx)]
        bid_points = unique_x

        derivatives = self._monotone_derivatives(bid_points, cdf_values)

        self.bid_points = torch.tensor(bid_points, dtype=torch.float32, device=self.device)
        self.cdf_values = torch.tensor(cdf_values, dtype=torch.float32, device=self.device)
        self.derivatives = torch.tensor(derivatives, dtype=torch.float32, device=self.device)

    def predict_win_prob(self, bids: torch.Tensor) -> torch.Tensor:
        if self.bid_points is None:
            raise RuntimeError("SplineCritic must be fit before calling predict_win_prob.")

        bids_clamped = bids.clamp(self.bid_points[0].item(), self.bid_points[-1].item())
        idx = torch.searchsorted(self.bid_points, bids_clamped.detach())
        idx = idx.clamp(1, len(self.bid_points) - 1)

        x0, x1 = self.bid_points[idx - 1], self.bid_points[idx]
        y0, y1 = self.cdf_values[idx - 1], self.cdf_values[idx]
        m0, m1 = self.derivatives[idx - 1], self.derivatives[idx]

        h = x1 - x0
        t = (bids_clamped - x0) / h

        # Cubic Hermite basis functions.
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2

        return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


class BetaCritic:
    """Estimates P(win | b) = P(opponent's bid < b) by fitting a scaled
    Beta(alpha, 1) distribution to the opponent's bid sample instead of
    interpolating empirical quantiles.

    This is exact, not just a convenient shape, at both ends of the
    payment_share continuum this project explores: the symmetric
    equilibrium bid is b*(v) = c*v^p (p=1 at first-price, p=n_agents at
    all-pay; see benchmark.equilibrium_bid_lambda), and for v ~ U(0, 1),
    b*/c ~ Beta(1/p, 1) exactly. Beta(alpha, 1) in turn has the closed-form
    CDF (x/c)^alpha (no incomplete-beta function needed, unlike a general
    Beta(alpha, beta)), so both the critic's value and its derivative
    reduce to a plain power law fit from just two numbers (alpha, c) via
    MLE, instead of many empirical quantiles. That directly targets the
    noisy-derivative problem documented in docs/findings_all_pay.md as the
    root cause of all-pay's fragility: a 2-parameter fit has far lower
    sampling variance than a spline over quantiles, and its derivative is
    an exact closed form rather than a numerically interpolated one.

    Away from the two endpoints (intermediate payment_share, or mid-
    training before the opponent's actual strategy has converged to the
    true equilibrium shape), the opponent's real bid distribution is not
    exactly of this form, so this critic carries some model bias there in
    exchange for its much lower variance.
    """

    def __init__(self, device: torch.device | str | None = None,
                 min_alpha: float = 1e-3, eps: float = 1e-6):
        self.device = torch.device(device) if device else torch.device("cpu")
        self.min_alpha = min_alpha
        self.eps = eps
        self.alpha: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    def fit(self, opponent_bids: np.ndarray) -> None:
        bids = np.asarray(opponent_bids, dtype=np.float64)
        scale = float(bids.max())
        if scale <= 0:
            scale = self.eps

        # Only clip the singularity at exactly 0 (float64 underflow), not
        # with self.eps: for alpha < 1 a non-trivial mass of samples can be
        # many orders of magnitude smaller than 1e-6 (e.g. all-pay bids near
        # v=0), and clipping those up to self.eps would make log(x) less
        # negative than it really is, biasing alpha_hat upward.
        x = np.clip(bids / scale, np.finfo(np.float64).tiny, 1.0)
        # MLE for Beta(alpha, 1) with known support [0, scale]:
        # alpha_hat = -1 / mean(log(x)).
        mean_log_x = float(np.log(x).mean())
        alpha_hat = -1.0 / mean_log_x if mean_log_x < 0 else self.min_alpha
        alpha_hat = max(alpha_hat, self.min_alpha)

        self.alpha = torch.tensor(alpha_hat, dtype=torch.float32, device=self.device)
        self.scale = torch.tensor(scale, dtype=torch.float32, device=self.device)

    def predict_win_prob(self, bids: torch.Tensor) -> torch.Tensor:
        if self.alpha is None:
            raise RuntimeError("BetaCritic must be fit before calling predict_win_prob.")
        # Clamp away from exactly 0: for alpha < 1 the true derivative
        # diverges there, which is a real feature of the equilibrium (see
        # docs/findings_all_pay.md) but would otherwise backprop an exact
        # inf/nan. Clamping to 1 above the scale matches the boundary
        # convention used elsewhere (e.g. opponent_inverse_and_slope):
        # beyond the opponent's observed range, win is already certain and
        # the derivative there is 0.
        x = (bids / self.scale).clamp(self.eps, 1.0)
        return x ** self.alpha
