import torch

from fictitious_play.benchmark import (
    equilibrium_bid_all_pay,
    equilibrium_bid_first_price,
    equilibrium_bid_lambda,
    theoretical_win_probability_all_pay,
    theoretical_win_probability_first_price,
    theoretical_win_probability_lambda,
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
