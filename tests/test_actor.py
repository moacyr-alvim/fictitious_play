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
    actor = Actor(hidden_size=0)
    v = torch.rand(200, 1)
    with torch.no_grad():
        before = actor(v).clone()
    actor.grow(8)
    with torch.no_grad():
        after = actor(v)
    assert torch.allclose(before, after, atol=1e-6)


def test_grow_preserves_function_between_nonzero_sizes():
    torch.manual_seed(0)
    actor = Actor(hidden_size=4)
    v = torch.rand(200, 1)
    with torch.no_grad():
        before = actor(v).clone()
    actor.grow(12)  # 4 -> 16
    with torch.no_grad():
        after = actor(v)
    assert actor.hidden_size == 16
    assert torch.allclose(before, after, atol=1e-6)


def test_grow_with_freeze_previous_locks_old_parameters():
    actor = Actor(hidden_size=4)
    actor.grow(4, freeze_previous=True)

    assert actor.bias.requires_grad is False
    assert all(not p.requires_grad for p in actor.blocks[0].parameters())
    assert all(p.requires_grad for p in actor.blocks[1].parameters())


def test_unfreeze_all_restores_gradients():
    actor = Actor(hidden_size=4)
    actor.grow(4, freeze_previous=True)
    actor.unfreeze_all()

    assert actor.bias.requires_grad is True
    assert all(p.requires_grad for block in actor.blocks for p in block.parameters())
