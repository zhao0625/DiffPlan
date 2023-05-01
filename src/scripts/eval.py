import os
import time

import torch
import wandb
from omegaconf import OmegaConf, DictConfig

from utils.experiment import create_save_dir, get_mechanism, create_dataloader, print_row, print_stats
from utils.runner import Runner


def eval_runner(args, eval_on_test=True):
    if os.path.exists(args.model_path + '/planner.config.yaml'):
        args_load = OmegaConf.load(args.model_path + '/planner.config.yaml')
        args_load.model_path = args.model_path

        if not isinstance(args, DictConfig):
            args = vars(args)

        # > TODO override some arguments for inference using input args
        # TODO to also move to config
        keys_override = [
            'model_path', 'model',
            'datafile',
            'enable_wandb', 'k',
            'deq_max_iter', 'deq_tol', 'deq_anderson_m',
            'solver_fwd', 'solver_bwd',
            'deq_fwd_max_iter', 'deq_bwd_max_iter',
            'deq_fwd_tol', 'deq_bwd_tol',
            'deq_fwd_anderson_m', 'deq_bwd_anderson_m',
        ]
        for key in keys_override:
            args_load[key] = args[key]

    else:
        args_load = args

    if eval_on_test:
        print('> Evaluate the model - on test data (with stats return)')
        stats = start_eval_on_test(args_load)
        return stats
    else:
        print('> Evaluate the model - on training/validation/test data')
        start_eval(args_load)


def start_eval(args):
    # create_save_dir(args.save_directory, save_timestamp=args.name_time)
    mechanism = get_mechanism(args.mechanism)

    # Create DataLoaders.
    train_loader = create_dataloader(
        args.datafile, "train", args.batch_size, mechanism, shuffle=True)
    valid_loader = create_dataloader(
        args.datafile, "valid", args.batch_size, mechanism, shuffle=False)
    test_loader = create_dataloader(
        args.datafile, "test", args.batch_size, mechanism, shuffle=False)

    maze_size = next(iter(train_loader))['maze'].size(1)
    runner = Runner(args, mechanism, maze_size)

    print("\n------------- Evaluating final model -------------")
    print("\nTrain performance:")
    print_stats(runner.test(train_loader))

    print("\nValidation performance:")
    print_stats(runner.test(valid_loader))

    print("\nTest performance:")
    print_stats(runner.test(test_loader))

    print("\n------------- Evaluating best model -------------")
    print("\nTrain performance:")
    print_stats(runner.test(train_loader, use_best=True))

    print("\nValidation performance:")
    print_stats(runner.test(valid_loader, use_best=True))

    print("\nTest performance:")
    print_stats(runner.test(test_loader, use_best=True))


def start_eval_on_test(args):
    # create_save_dir(args.save_directory, save_timestamp=args.name_time)
    mechanism = get_mechanism(args.mechanism)

    test_loader = create_dataloader(
        args.datafile, "test", args.batch_size, mechanism, shuffle=False)

    test_maze_size = next(iter(test_loader))['maze'].size(1)
    data_size = len(test_loader.dataset)
    runner = Runner(args, mechanism, test_maze_size, data_size=data_size, verbose=False)

    print("\n------------- Evaluating final model -------------")
    print("\nTest performance:")
    stats = runner.test(test_loader)
    print_stats(stats)

    return stats
