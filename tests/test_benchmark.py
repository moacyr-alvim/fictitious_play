import torch

from fictitious_play.benchmark import (
    equilibrium_bid_all_pay,
    equilibrium_bid_first_price,
    theoretical_win_probability_all_pay,
    theoretical_win_probability_first_price,
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
