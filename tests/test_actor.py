import torch

from fictitious_play.actor import Actor


def test_bid_is_between_zero_and_value():
    actor = Actor()
    v = torch.rand(1000, 1)
    b = actor(v)
    assert torch.all(b >= 0.0)
    assert torch.all(b <= v + 1e-6)


def test_constant_actor_has_no_v_dependence_but_can_shade():
    actor = Actor(hidden_size=0)
    v = torch.linspace(0.01, 1.0, 50).unsqueeze(1)
    b = actor(v)
    shading = b / v
    assert torch.allclose(shading, shading[0].expand_as(shading), atol=1e-6)


def test_grow_preserves_function_from_zero():
    torch.manual_seed(0)
    small = Actor(hidden_size=0)
    grown = small.grow(8)

    v = torch.rand(200, 1)
    with torch.no_grad():
        assert torch.allclose(small(v), grown(v), atol=1e-6)


def test_grow_preserves_function_between_nonzero_sizes():
    torch.manual_seed(0)
    small = Actor(hidden_size=4)
    grown = small.grow(16)

    v = torch.rand(200, 1)
    with torch.no_grad():
        assert torch.allclose(small(v), grown(v), atol=1e-6)
