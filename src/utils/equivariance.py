import torch
from torch import nn

import e2cnn
from e2cnn import nn as e2nn


def test_equivariance(_model, _input, _element=1):
    _output = _model(_input)
    _output_gfx = _output.transform(_element)

    _input_gx = _input.transform(_element)
    _output_fgx = _model(_input_gx)

    equivariance_error = (_output_fgx.tensor - _output_gfx.tensor).norm(2)

    return equivariance_error


class C4:
    """
    C_4 rotation group
    """

    @staticmethod
    def product(r: int, s: int) -> int:
        return (r + s) % 4

    @staticmethod
    def inverse(r: int) -> int:
        return (-r) % 4


class D4:
    """
    D_4 dihedral group
    """

    @staticmethod
    def product(a: tuple, b: tuple) -> tuple:
        f1, r1 = a
        f2, r2 = b
        r0 = ((r1 + r2) % 4) if (f1 == 0) else ((r1 - r2) % 4)
        f0 = (f1 + f2) % 2
        return f0, r0

    @staticmethod
    def inverse(g: tuple) -> tuple:
        f1, r1 = g
        r0 = (-r1) % 4 if f1 == 0 else r1
        return f1, r0


def rotate(tensor: torch.Tensor, r: int) -> torch.Tensor:
    # Method which implements the action of the group element `g` indexed by `r` on the input image `x`.
    # The method returns the image `g.x`

    # note that we rotate the last 2 dimensions of the input, since we want to later use this method to
    # rotate mini-batches containing multiple images
    return tensor.rot90(r, dims=(-2, -1))


def rotate_p4(tensor: torch.Tensor, r: int) -> torch.Tensor:
    # `y` is a function over p4, i.e. over the pixel positions and over the elements of the group C_4.
    # This method implements the action of a rotation `r` on `y`.
    # To be able to reuse this function later with a minibatch of inputs,
    # assume that the last two dimensions (`dim=-2` and `dim=-1`) of `y` are the spatial dimensions
    # while `dim=-3` has size `4` and is the C_4 dimension.
    # All other dimensions are considered batch dimensions
    assert len(tensor.shape) >= 3
    assert tensor.shape[-3] == 4

    tensor = rotate(tensor, r)
    tensor = tensor.roll(r, dims=-3)
    # > note: follow the guide, we consider the G-action on channel, which is to cyclically rotate along the channel

    return tensor


def equivariance_error_rotation(model, model_in):
    obstacle_map, goal_map = model_in

    logits, probs, q_regular, v_trivial, r_trivial = model(
        obstacle_map, goal_map, debug=True
    )

    rotated_obstacle_map = rotate(obstacle_map, r=1)
    rotated_goal_map = rotate(goal_map, r=1)

    rotated_logits, rotated_probs, rotated_q_regular, rotated_v_trivial, rotated_r_trivial = model(
        rotated_obstacle_map, rotated_goal_map, debug=True
    )
