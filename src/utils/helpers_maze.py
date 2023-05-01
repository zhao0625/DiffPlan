import torch
import numpy as np


def convert_actions(logits):
    assert len(logits.shape) == 4

    # logits: actions x orient x width x height
    # actions_one_hot = logits.max(dim=0).values
    actions_one_hot = None

    if isinstance(logits, torch.Tensor):
        actions_int = logits.max(dim=0).indices
    elif isinstance(logits, np.ndarray):
        actions_int = logits.argmax(axis=0)
    else:
        raise ValueError

    return actions_one_hot, actions_int


def convert_goal(goal_map, to_tuple=True):
    if len(goal_map.shape) == 2:
        goal_map = goal_map.unsqueeze(0)  # one dim for orientation, might lose in concat
    else:
        assert len(goal_map.shape) == 3

    goal_pos = goal_map.nonzero()

    if to_tuple and not isinstance(goal_pos, tuple):
        goal_pos = goal_pos.squeeze()
        goal_pos = tuple(goal_pos.numpy().tolist())

    return goal_pos
