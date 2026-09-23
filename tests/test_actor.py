import torch

from fictitious_play.actor import Actor


def test_bid_is_between_zero_and_value():
    actor = Actor()
    v = torch.rand(1000, 1)
    b = actor(v)
    assert torch.all(b >= 0.0)
    assert torch.all(b <= v + 1e-6)
