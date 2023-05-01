"""
Generates a 2D maze dataset.

Example usage:
  python generate_dataset.py --output-path mazes.npz --mechanism news \
    --maze-size 9 --train-size 5000 --valid-size 1000 --test-size 1000
"""
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np

from envs.maze_env import RandomMaze
from envs.maze_utils import extract_policy
from utils.dijkstra import dijkstra_dist
from utils.experiment import get_mechanism

# from memory_profiler import profile
import gc


def generate_data(filename,
                  train_size,
                  valid_size,
                  test_size,
                  mechanism,
                  maze_size,
                  workspace_size,
                  min_decimation,
                  max_decimation,
                  label,
                  start_pos=(1, 1),
                  env_name=None,
                  shuffle=False):
    if env_name == 'RandomMaze':
        env_class = RandomMaze
    elif env_name in ['Arm2DoFsEnv', 'Arm2DoFsWorkSpaceEnv']:
        from envs.arm.arm_2DoFs_env import Arm2DoFsEnv
        env_class = Arm2DoFsEnv
    elif env_name == 'Visual3DNav':
        from envs.visual_nav.nav_env import VisNavEnv  # move in to avoid requiring handling display
        env_class = VisNavEnv
    else:
        raise NotImplementedError('The environment ' + env_name + ' is not implemented.')

    env = env_class(
        mechanism,
        maze_size,
        maze_size,
        min_decimation,
        max_decimation,
        start_pos=start_pos)

    if env_name in ['Visual3DNav', 'Arm2DoFsWorkSpaceEnv']:
    # if env_name in ['Visual3DNav']:
        # Generate test set first
        print("Creating valid+test dataset...")
        valid_test_mazes, valid_test_goal_maps, valid_test_opt_policies, valid_test_obs = create_dataset(
            env_name, env, mechanism, maze_size, workspace_size, label,
            test_size + valid_size
        )

        # Generate train set while avoiding test geometries
        print("Creating training dataset...")
        train_mazes, train_goal_maps, train_opt_policies, train_obs = create_dataset(
            env_name, env, mechanism, maze_size, workspace_size, label,
            train_size, compare_mazes=valid_test_mazes
        )

    else:
        print("Creating valid+test dataset...")
        valid_test_mazes, valid_test_goal_maps, valid_test_opt_policies = create_dataset(
            env_name, env, mechanism, maze_size, workspace_size, label,
            test_size + valid_size
        )

        print("Creating training dataset...")
        train_mazes, train_goal_maps, train_opt_policies = create_dataset(
            env_name, env, mechanism, maze_size, workspace_size, label,
            train_size, compare_mazes=valid_test_mazes
        )

    # Split valid and test
    valid_mazes = valid_test_mazes[0:valid_size]
    test_mazes = valid_test_mazes[valid_size:]
    valid_goal_maps = valid_test_goal_maps[0:valid_size]
    test_goal_maps = valid_test_goal_maps[valid_size:]
    valid_opt_policies = valid_test_opt_policies[0:valid_size]
    test_opt_policies = valid_test_opt_policies[valid_size:]

    if env_name in ['Visual3DNav', 'Arm2DoFsWorkSpaceEnv']:
        valid_obs = valid_test_obs[0:valid_size]
        test_obs = valid_test_obs[valid_size:]

    if shuffle:
        (
            train_goal_maps, train_mazes, train_opt_policies,
            valid_goal_maps, valid_mazes, valid_opt_policies,
            test_goal_maps, test_mazes, test_opt_policies,
        ) = shuffle_dataset(
            env_name, train_size, valid_size,
            train_goal_maps, train_mazes, train_opt_policies,
            valid_goal_maps, valid_mazes, valid_opt_policies,
            test_goal_maps, test_mazes, test_opt_policies,
        )

    # Save to numpy
    if env_name in [None, 'RandomMaze', 'Arm2DoFsEnv']:
        np.savez_compressed(
            filename,
            train_mazes, train_goal_maps, train_opt_policies,
            valid_mazes, valid_goal_maps, valid_opt_policies,
            test_mazes, test_goal_maps, test_opt_policies
        )
    elif env_name in ['Visual3DNav', 'Arm2DoFsWorkSpaceEnv']:
        np.savez_compressed(
            filename,
            train_mazes, train_goal_maps, train_opt_policies,
            valid_mazes, valid_goal_maps, valid_opt_policies,
            test_mazes, test_goal_maps, test_opt_policies,
            train_obs, valid_obs, test_obs
        )
    else:
        raise ValueError


def shuffle_dataset(
        env_name, train_size, valid_size,
        train_goal_maps, train_mazes, train_opt_policies,
        valid_goal_maps, valid_mazes, valid_opt_policies,
        test_goal_maps, test_mazes, test_opt_policies,
):
    # Re-shuffle
    mazes = np.concatenate((train_mazes, valid_mazes, test_mazes), 0)
    goal_maps = np.concatenate(
        (train_goal_maps, valid_goal_maps, test_goal_maps), 0)
    opt_policies = np.concatenate(
        (train_opt_policies, valid_opt_policies, test_opt_policies), 0)

    shuffle_idx = np.random.permutation(mazes.shape[0])
    mazes = mazes[shuffle_idx]
    goal_maps = goal_maps[shuffle_idx]
    opt_policies = opt_policies[shuffle_idx]

    train_mazes = mazes[:train_size]
    train_goal_maps = goal_maps[:train_size]
    train_opt_policies = opt_policies[:train_size]

    valid_mazes = mazes[train_size:train_size + valid_size]
    valid_goal_maps = goal_maps[train_size:train_size + valid_size]
    valid_opt_policies = opt_policies[train_size:train_size + valid_size]

    test_mazes = mazes[train_size + valid_size:]
    test_goal_maps = goal_maps[train_size + valid_size:]
    test_opt_policies = opt_policies[train_size + valid_size:]

    if env_name == 'Visual3DNav':
        raise NotImplementedError

    return (
        train_goal_maps, train_mazes, train_opt_policies,
        valid_goal_maps, valid_mazes, valid_opt_policies,
        test_goal_maps, test_mazes, test_opt_policies,
    )


# @profile
def create_dataset(
        env_name, env, mechanism, maze_size, workspace_size, label,
        data_size, compare_mazes=None
):
    mazes = np.zeros((data_size, maze_size, maze_size))
    goal_maps = np.zeros((data_size, mechanism.num_orient, maze_size, maze_size))
    opt_policies = np.zeros((data_size, mechanism.num_actions, mechanism.num_orient, maze_size, maze_size))

    if env_name == 'Visual3DNav':
        env.reset()
        pano_obs_array = np.zeros(
            (
                data_size, maze_size, maze_size,
                env.nav_world.num_views, env.obs_height, env.obs_width, env.nav_world.num_rgb
            ),
            dtype=np.uint8  # > 8-bit images; keep consistent; save memory
        )

    if env_name == 'Arm2DoFsWorkSpaceEnv':
        workspaces = np.zeros((data_size, 1, workspace_size, workspace_size))

    maze_hash = {}

    if compare_mazes is not None:
        for i in range(compare_mazes.shape[0]):
            maze = compare_mazes[i]
            maze_key = hash_maze_to_string(maze)
            maze_hash[maze_key] = 1

    for i in range(data_size):
        while True:
            maze, player_map, goal_map = env.reset()
            maze_key = hash_maze_to_string(maze)

            # Make sure we sampled a unique maze from the compare set
            if hashed_check_maze_exists(maze_key, maze_hash):
                continue
            maze_hash[maze_key] = 1

            # > For manipulation env, check for reachability using computed value; For nav env, no need to do that
            if env_name in ['Arm2DoFsEnv', 'Arm2DoFsWorkSpaceEnv']:
                # If the goal is reachable from the start pos, break
                # Notice that opt_value.min() should be the value for unreachable state,
                # i.e, there should be at least one obstacle in the map
                opt_value = dijkstra_dist(maze, mechanism, extract_goal(goal_map, mechanism, maze_size))
                if opt_value[extract_goal(player_map, mechanism, maze_size)] > opt_value.min():
                    break
                # else:
                #     print('\nSkipping an unreachable start-goal pos.')
            else:
                break

        # Use Dijkstra's to construct the optimal policy
        opt_value = dijkstra_dist(maze, mechanism, extract_goal(goal_map, mechanism, maze_size))
        opt_policy = extract_policy(maze, mechanism, opt_value, is_full_policy=(label == 'full'))

        # > Store the demonstration data
        mazes[i, :, :] = maze
        goal_maps[i, :, :, :] = goal_map
        opt_policies[i, :, :, :, :] = opt_policy

        # > Save
        if env_name == 'Visual3DNav':
            pos2pano = env.render_all_pano()
            pano_obs_array[i, ...] = pos2pano  # > this step will have substantial memory use (since replace zeros?)
            print('> Generate full 3D egocentric panoramic views', i, pano_obs_array.shape)

        if env_name == 'Arm2DoFsWorkSpaceEnv':
            workspaces[i, 0, ...] = env.render_workspace(workspace_size)

        # print("\r%0.4f" % (float(i) / data_size * 100) + "%")
        sys.stdout.write("\r%0.4f" % (float(i) / data_size * 100) + "%")
        sys.stdout.flush()

    sys.stdout.write("\r100%\n")

    if env_name == 'Visual3DNav':
        env.close()
        return mazes, goal_maps, opt_policies, pano_obs_array
    elif env_name == 'Arm2DoFsWorkSpaceEnv':
        return mazes, goal_maps, opt_policies, workspaces
    else:
        return mazes, goal_maps, opt_policies


def hash_maze_to_string(_maze):
    maze = np.array(_maze, dtype=np.uint8).reshape((-1))
    maze_key = ""
    for i in range(maze.shape[0]):
        maze_key += str(maze[i])
    return maze_key


def hashed_check_maze_exists(maze_key, maze_hash):
    if maze_hash is None:
        return False
    if maze_key in maze_hash:
        return True
    return False


def extract_goal(goal_map, mechanism, maze_size):
    for o in range(mechanism.num_orient):
        for y in range(maze_size):
            for x in range(maze_size):
                if goal_map[o][y][x] == 1.:
                    return (o, y, x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='RandomMaze', help="Which environment to generate data.")
    parser.add_argument("--output-path", type=str, default="", help="Filename to save the dataset to.")
    parser.add_argument("--train-size", type=int, default=10000, help="Number of training mazes.")
    parser.add_argument("--valid-size", type=int, default=1000, help="Number of validation mazes.")
    parser.add_argument("--test-size", type=int, default=1000, help="Number of test mazes.")
    parser.add_argument("--maze-size", type=int, default=9, help="Size of mazes.")
    parser.add_argument("--workspace-size", type=int, default=96, help="Size of manipulator workspace.")
    parser.add_argument("--label", type=str, default="one_hot",
                        help="Optimal policy labeling. (one_hot|full)")
    parser.add_argument("--min-decimation", type=float, default=0.0,
                        help="How likely a wall is to be destroyed (minimum).")
    parser.add_argument("--max-decimation", type=float, default=1.0,
                        help="How likely a wall is to be destroyed (maximum).")
    parser.add_argument("--start-pos-x", type=int, default=1, help="Maze start X-axis position.")
    parser.add_argument("--start-pos-y", type=int, default=1, help="Maze start Y-axis position.")
    parser.add_argument("--mechanism", type=str, default="news",
                        help="Maze transition mechanism. (news|news-wrap|4abs-cc|4abs-cc-wrap|diffdrive|moore)")
    args = parser.parse_args()

    file_name = args.env + '_' \
                + str(args.workspace_size) + '_' \
                + str(args.train_size) + '_' \
                + str(args.maze_size) + '_' \
                + args.label + '_' \
                + args.mechanism \
                + '.npz'

    if args.output_path == '':
        # Note: path is relative to root directory, use `python -m envs.generate_dataset <arguments>`
        file_path = '../data/'
        file_path += file_name

    else:
        if args.output_path.endswith('.npz'):
            file_path = args.output_path
        else:
            file_path = os.path.join(args.output_path, file_name)

    mechanism = get_mechanism(args.mechanism)
    generate_data(
        file_path,
        args.train_size,
        args.valid_size,
        args.test_size,
        mechanism,
        args.maze_size,
        args.workspace_size,
        args.min_decimation,
        args.max_decimation,
        args.label,
        start_pos=(args.start_pos_y, args.start_pos_x),
        env_name=args.env)


if __name__ == "__main__":
    main()
