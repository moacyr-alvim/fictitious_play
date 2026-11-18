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


class MixtureBetaCritic:
    """Estimates P(win | b) = P(opponent's bid < b) with a mixture of K
    scaled Beta(alpha_k, 1) CDFs sharing one scale: F(x) = sum_k pi_k *
    (x/scale)^alpha_k, pi_k >= 0, sum_k pi_k = 1.

    Monotonic by construction for the same reason BetaCritic is: each
    component (x/scale)^alpha_k is itself a valid CDF (monotonic
    non-decreasing, 0 at x=0, 1 at x=scale), and a convex combination (the
    pi_k's are nonnegative and sum to 1) of valid CDFs is again a valid
    CDF — no architecture trick needed, just the fact that mixing
    probability distributions can't un-order them.

    Fitting is still closed-form per iteration, not gradient descent: a
    Beta(alpha, 1) sample transforms exactly to an Exponential(alpha)
    sample under Y = -log(X) (CDF of X is x^alpha, so
    P(-log X > y) = P(X < e^-y) = e^(-alpha*y), i.e. Y ~ Exp(alpha)). So
    fitting a mixture of Beta(alpha_k, 1)'s on the (scaled) bids is exactly
    equivalent to fitting a mixture of Exponential(alpha_k)'s on their
    -log, which has a textbook closed-form EM update (E-step: responsibility
    of component k; M-step: pi_k and alpha_k are both plain weighted means,
    no numerical optimizer). A few dozen EM iterations settle it, versus
    the thousands of Adam steps MonotonicNNCritic needs — and unlike that
    per-round gradient refit, EM here is deterministic given the data and
    the warm-started initial guess, so it carries none of the per-round
    optimization noise that showed up as persistent oscillation in the
    fictitious-play dynamics with the NN critic (see
    docs/findings_mixture_beta_critic.md).

    One shared scale (not one per component) matches BetaCritic: the
    largest observed bid is the exact deterministic boundary of the
    opponent's support regardless of how many mixture components describe
    the shape below it, so there's no reason to estimate it per component.

    Meant to be warm-started round to round like MonotonicNNCritic (fit()
    reuses the previous call's alpha/pi as the EM's initial guess instead
    of restarting from scratch), so callers should keep one persistent
    instance per agent rather than constructing a fresh one every round.
    """

    def __init__(self, n_components: int = 2, em_iters: int = 50,
                 device: torch.device | str | None = None,
                 min_alpha: float = 1e-3, eps: float = 1e-6):
        self.device = torch.device(device) if device else torch.device("cpu")
        self.n_components = n_components
        self.em_iters = em_iters
        self.min_alpha = min_alpha
        self.eps = eps
        self.alpha_np: np.ndarray | None = None
        self.pi_np: np.ndarray | None = None
        self.alpha: torch.Tensor | None = None
        self.pi: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    def state_dict(self) -> dict:
        return {
            "alpha_np": self.alpha_np,
            "pi_np": self.pi_np,
            "scale": self.scale.item() if self.scale is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        self.alpha_np = state["alpha_np"]
        self.pi_np = state["pi_np"]
        if self.alpha_np is not None:
            # Makes predict_win_prob usable immediately, same as fit() would
            # (the warm-started EM state in alpha_np/pi_np is what the next
            # fit() call actually continues from; these torch copies just
            # mean a restored critic doesn't need that extra fit() call to
            # be queried first, mirroring MonotonicNNCritic's contract).
            self.alpha = torch.tensor(self.alpha_np, dtype=torch.float32, device=self.device)
            self.pi = torch.tensor(self.pi_np, dtype=torch.float32, device=self.device)
            self.scale = torch.tensor(state["scale"], dtype=torch.float32, device=self.device)

    def fit(self, opponent_bids: np.ndarray) -> None:
        bids = np.asarray(opponent_bids, dtype=np.float64)
        scale = float(bids.max())
        if scale <= 0:
            scale = self.eps

        # Same care as BetaCritic: only clip the x=0 singularity, not with
        # self.eps, to avoid biasing alpha_hat (see BetaCritic.fit).
        x = np.clip(bids / scale, np.finfo(np.float64).tiny, 1.0)
        y = -np.log(x)  # Beta(alpha_k, 1) on x <=> Exponential(alpha_k) on y

        k = self.n_components
        if self.alpha_np is None:
            # Break symmetry on the first call (no warm start yet): spread
            # initial alphas geometrically around 1 (risk-neutral/first-price).
            self.alpha_np = np.geomspace(0.3, 3.0, k)
            self.pi_np = np.full(k, 1.0 / k)

        alpha = self.alpha_np.copy()
        pi = self.pi_np.copy()
        for _ in range(self.em_iters):
            # E-step: responsibilities r[i, k] = P(component k | y_i),
            # computed in log-space (component density is alpha_k*exp(-alpha_k*y)).
            log_dens = (np.log(pi) + np.log(alpha))[None, :] - alpha[None, :] * y[:, None]
            m = log_dens.max(axis=1, keepdims=True)
            w = np.exp(log_dens - m)
            resp = w / np.clip(w.sum(axis=1, keepdims=True), 1e-300, None)

            # M-step: both updates are plain weighted means (MLE of an
            # Exponential given responsibilities has a closed form, just
            # like the unweighted single-component case in BetaCritic).
            n_k = resp.sum(axis=0)
            pi = n_k / len(y)
            alpha = n_k / np.maximum((resp * y[:, None]).sum(axis=0), 1e-12)
            alpha = np.clip(alpha, self.min_alpha, None)

        self.alpha_np = alpha
        self.pi_np = pi
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        self.pi = torch.tensor(pi, dtype=torch.float32, device=self.device)
        self.scale = torch.tensor(scale, dtype=torch.float32, device=self.device)

    def predict_win_prob(self, bids: torch.Tensor) -> torch.Tensor:
        if self.alpha is None:
            raise RuntimeError("MixtureBetaCritic must be fit before calling predict_win_prob.")
        x = (bids / self.scale).clamp(self.eps, 1.0)
        component_cdfs = x.unsqueeze(-1) ** self.alpha
        return (component_cdfs * self.pi).sum(-1)


class MonotonicNNCritic:
    """Estimates P(win | b) = P(opponent's bid < b) with a small neural
    network constrained to be monotonically non-decreasing in b *by
    construction* (every weight strictly positive, every activation itself
    monotonically non-decreasing — a composition of monotonic functions
    with positive-weighted sums is itself monotonic), fit by gradient
    descent against the same empirical quantiles SplineCritic interpolates.

    Positioned between SplineCritic and BetaCritic on the bias/variance
    trade-off the other two sit at opposite ends of (see
    docs/findings_risk_aversion.md and docs/findings_beta_critic.md):
    BetaCritic is exactly unbiased only when the opponent's bid
    distribution truly is Beta(alpha, 1)-shaped (first-price, all-pay, the
    payment_share continuum between them) and visibly biased otherwise (the
    asymmetric risk-aversion auction); SplineCritic has no shape assumption
    at all but pays for that in sampling variance. A small monotonic MLP
    has neither a hard-coded parametric shape nor the spline's per-quantile
    interpolation noise, at the cost of needing its own (short, warm-
    started) training loop each round instead of a closed-form fit.

    Two implementation choices were tuned empirically rather than guessed:

    - **2 hidden layers, not 1.** A single hidden layer (however wide, however
      long trained) plateaus at a ~0.09 max CDF error fitting the curved
      asymmetric-risk-aversion equilibrium — a genuine capacity limit of
      that shape, not an optimization issue (more steps/width didn't help).
      Stacking a second monotonic hidden layer (same positive-weight +
      softplus/tanh construction) drops that to ~0.002-0.007: a materially
      more expressive monotonic family for about the same parameter count.
    - **Regression against empirical quantiles (MSE), not BCE against random
      (query, opponent-sample) pairs.** The latter is the more obvious
      training signal (a random query q and opponent draw x give a
      Bernoulli sample of P(win|q) = F(q) via the indicator 1[x<q], so BCE
      against many such pairs is a standard classifier-based way to fit a
      CDF) and is what this was first implemented with, but needed
      5,000-15,000 steps to approach BetaCritic's accuracy on the
      asymmetric risk-aversion case and still didn't clearly beat it.
      Regressing directly against a few hundred empirical quantiles (the
      same (bid, level) pairs SplineCritic already computes) is a far less
      noisy target and converges in ~2,000 steps to an error an order of
      magnitude lower.

    Unlike SplineCritic/BetaCritic, fitting is itself a short gradient-
    descent loop, not a closed-form computation — meant to be *warm-started*
    round to round (fit() keeps refining the same weights and the same
    Adam optimizer state rather than starting over), the same way each
    agent's actor already warm-starts across rounds. Callers should keep
    one persistent instance per agent (see FictitiousPlayTrainer.nn_critics)
    rather than constructing a fresh one every round.
    """

    def __init__(self, hidden_size: int = 32, n_layers: int = 2, lr: float = 1e-2,
                 n_quantiles: int = 400, fit_steps: int = 2000,
                 device: torch.device | str | None = None):
        self.device = torch.device(device) if device else torch.device("cpu")
        self.n_quantiles = n_quantiles
        self.fit_steps = fit_steps
        sizes = [1] + [hidden_size] * n_layers + [1]
        self._raw_weights = [
            torch.nn.Parameter(torch.randn(sizes[i], sizes[i + 1], device=self.device) * 0.5)
            for i in range(len(sizes) - 1)
        ]
        self._biases = [
            torch.nn.Parameter(torch.zeros(sizes[i + 1], device=self.device))
            for i in range(len(sizes) - 1)
        ]
        self._optimizer = torch.optim.Adam(self._raw_weights + self._biases, lr=lr)

    def state_dict(self) -> dict:
        return {
            "raw_weights": [w.detach().clone() for w in self._raw_weights],
            "biases": [b.detach().clone() for b in self._biases],
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        with torch.no_grad():
            for w, saved in zip(self._raw_weights, state["raw_weights"]):
                w.copy_(saved)
            for b, saved in zip(self._biases, state["biases"]):
                b.copy_(saved)
        self._optimizer.load_state_dict(state["optimizer"])

    def _forward(self, bids: torch.Tensor) -> torch.Tensor:
        # softplus reparametrization keeps every weight strictly positive
        # without constraining the optimizer directly; tanh is monotonic
        # non-decreasing (tanh' = 1 - tanh^2 >= 0), as is the final sigmoid,
        # so d(output)/d(bids) is a sum/product of nonnegative terms
        # regardless of the (unconstrained) biases.
        x = bids.reshape(-1, 1)
        for i, (w_raw, bias) in enumerate(zip(self._raw_weights, self._biases)):
            w = torch.nn.functional.softplus(w_raw)
            x = x @ w + bias
            if i < len(self._raw_weights) - 1:
                x = torch.tanh(x)
        return torch.sigmoid(x).reshape(bids.shape)

    def fit(self, opponent_bids: np.ndarray) -> None:
        levels = np.linspace(0.0, 1.0, self.n_quantiles)
        bid_points = np.quantile(opponent_bids, levels)
        x = torch.tensor(bid_points, dtype=torch.float32, device=self.device)
        y = torch.tensor(levels, dtype=torch.float32, device=self.device)
        for _ in range(self.fit_steps):
            pred = self._forward(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def predict_win_prob(self, bids: torch.Tensor) -> torch.Tensor:
        return self._forward(bids)
