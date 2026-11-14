import numpy as np
import torch
from scipy.optimize import minimize_scalar

from fictitious_play.benchmark import (
    equilibrium_bid_all_pay,
    equilibrium_bid_first_price,
    equilibrium_bid_lambda,
    equilibrium_bid_risk_averse,
    theoretical_win_probability_all_pay,
    theoretical_win_probability_first_price,
    theoretical_win_probability_lambda,
    theoretical_win_probability_risk_averse,
)


def test_first_price_win_probability_inverts_equilibrium_bid():
    v = torch.linspace(0.0, 1.0, 50)
    for n in (2, 3, 4):
        b_star = equilibrium_bid_first_price(v, n)
        p_win = theoretical_win_probability_first_price(b_star, n)
        assert torch.allclose(p_win, v ** (n - 1), atol=1e-5)


def test_all_pay_win_probability_inverts_equilibrium_bid():
    v = torch.linspace(0.0, 1.0, 50)
    for n in (2, 3, 4):
        b_star = equilibrium_bid_all_pay(v, n)
        p_win = theoretical_win_probability_all_pay(b_star, n)
        assert torch.allclose(p_win, v ** (n - 1), atol=1e-5)


def test_all_pay_bid_below_value():
    v = torch.linspace(0.0, 1.0, 50)
    for n in (2, 3, 4):
        b_star = equilibrium_bid_all_pay(v, n)
        assert torch.all(b_star <= v + 1e-6)


def test_lambda_equilibrium_collapses_to_first_price_and_all_pay():
    v = torch.linspace(0.0, 1.0, 50)
    for n in (2, 3, 4):
        assert torch.allclose(equilibrium_bid_lambda(v, n, 0.0),
                               equilibrium_bid_first_price(v, n), atol=1e-5)
        assert torch.allclose(equilibrium_bid_lambda(v, n, 1.0),
                               equilibrium_bid_all_pay(v, n), atol=1e-5)


def test_lambda_win_probability_inverts_equilibrium_bid():
    v = torch.linspace(0.05, 1.0, 20)
    for n in (2, 3):
        for lam in (0.0, 0.3, 0.6, 1.0):
            b_star = equilibrium_bid_lambda(v, n, lam)
            p_win = theoretical_win_probability_lambda(b_star, n, lam)
            assert torch.allclose(p_win, v ** (n - 1), atol=1e-3)


def test_risk_averse_equilibrium_reduces_to_first_price_when_symmetric():
    """Equal rho (including the risk-neutral rho=1 case) must match the
    plain first-price formula v*(n-1)/n = v/2 (n=2), since the risk-averse
    equilibrium is only a genuinely different (numerically-solved) curve
    when the two agents' rho differ."""
    v = torch.linspace(0.0, 1.0, 20)
    for rho in (1.0, 0.7, 0.3):
        for agent_idx in (0, 1):
            b = equilibrium_bid_risk_averse(v, agent_idx, (rho, rho))
            assert torch.allclose(b, v / (1.0 + rho), atol=1e-5)
    # rho=1 (risk-neutral) must also match equilibrium_bid_first_price(n=2).
    assert torch.allclose(equilibrium_bid_risk_averse(v, 0, (1.0, 1.0)),
                           equilibrium_bid_first_price(v, 2), atol=1e-5)


def test_risk_averse_equilibrium_is_mutual_best_response():
    """Core correctness check for the numerically-solved asymmetric
    equilibrium (no closed form exists for rho_1 != rho_2 — see
    _solve_asymmetric_risk_averse_equilibrium's docstring): independently
    re-optimize each agent's objective, phi_other(b)*(v-b)^rho_self, against
    the *other* agent's solved bid-CDF, and check the maximizer matches the
    solved bid function itself. This is the actual Bayes-Nash equilibrium
    condition, not just the ODE's first-order condition."""
    v_grid = np.linspace(0.05, 0.95, 8)
    for rho0, rho1 in [(0.5, 1.0), (0.3, 0.7), (0.2, 0.9), (0.9, 0.2)]:
        risk_aversion = (rho0, rho1)
        for agent_idx, rho_self in [(0, rho0), (1, rho1)]:
            for v in v_grid:
                v_t = torch.tensor([v])
                b_pred = equilibrium_bid_risk_averse(v_t, agent_idx, risk_aversion).item()

                def neg_objective(b, v=v, agent_idx=agent_idx, rho_self=rho_self):
                    if b < 0 or b > v:
                        return 1e9
                    # theoretical_win_probability_risk_averse(b, agent_idx, ...)
                    # already returns P(win) for agent_idx by looking up its
                    # *opponent's* bid CDF internally.
                    p_win = theoretical_win_probability_risk_averse(
                        torch.tensor([b]), agent_idx, risk_aversion
                    ).item()
                    payoff = v - b
                    if payoff <= 0:
                        return 1e9
                    return -(p_win * payoff ** rho_self)

                res = minimize_scalar(neg_objective, bounds=(1e-9, v - 1e-9),
                                       method="bounded", options={"xatol": 1e-9})
                assert abs(res.x - b_pred) < 2e-3, (
                    f"rho={risk_aversion}, agent={agent_idx}, v={v}: "
                    f"best response {res.x} != solved bid {b_pred}"
                )


def test_risk_averse_more_averse_agent_bids_more_aggressively():
    """Standard comparative-static result: the more risk-averse bidder
    (smaller rho) shades less, i.e. bids a larger fraction of its value,
    than a less risk-averse opponent."""
    v = torch.linspace(0.1, 0.9, 10)
    b_more_averse = equilibrium_bid_risk_averse(v, 0, (0.3, 0.8))
    b_less_averse = equilibrium_bid_risk_averse(v, 1, (0.3, 0.8))
    assert torch.all(b_more_averse / v > b_less_averse / v)
