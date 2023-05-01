import time

import torch
import wandb
from omegaconf import OmegaConf

from utils.experiment import create_save_dir, get_mechanism, create_dataloader, print_row, print_stats
from utils.runner import Runner


def start_train(args):
    save_path = create_save_dir(save_directory=args.save_directory, save_timestamp=args.name_time)
    mechanism = get_mechanism(args.mechanism)

    # Create DataLoaders
    train_loader = create_dataloader(
        args.datafile, "train", args.batch_size, mechanism, shuffle=True)
    valid_loader = create_dataloader(
        args.datafile, "valid", args.batch_size, mechanism, shuffle=False)
    test_loader = create_dataloader(
        args.datafile, "test", args.batch_size, mechanism, shuffle=False)

    maze_size = next(iter(train_loader))['maze'].size(1)
    data_size = len(train_loader.dataset)
    args.maze_size = maze_size
    runner = Runner(args, mechanism, maze_size=maze_size, data_size=data_size)

    # Print header (replaced `avg_optimal` with `avg_SPL`)
    col_width = 5
    print("\n      |            Train              |            Valid              |")  # pylint: disable=line-too-long
    print_row(col_width, [
        "Epoch", "CE", "Err", "%SPL", "Suc", "CE", "Err", "%SPL", "Suc", "W",
        "dW", "Time", "Best"
    ])  # It's not %Suc, but Suc

    for epoch in range(args.epochs):
        start_time = time.time()
        include_mapper = epoch >= args.train_planner_only_until

        # Train the model
        tr_info = runner.train(train_loader, args.batch_size, include_mapper)

        # Compute validation stats and save the best model
        # TODO disable validation
        if args.disable_validation:
            v_info = {'avg_loss': 0., 'avg_error': 0., 'avg_SPL': 0., 'avg_success': 0.,
                      'is_best': (epoch == args.epochs - 1)}
        else:
            v_info = runner.validate(valid_loader, include_mapper)
        time_duration = time.time() - start_time

        # Log to W&B
        merged_info = {'Train/' + k: v for k, v in tr_info.items()}
        merged_info.update({'Validation/' + k: v for k, v in v_info.items()})
        if args.enable_wandb:
            wandb.log(merged_info)

        # Print epoch logs (replaced `avg_optimal` with `avg_SPL`)
        print_row(col_width, [
            epoch + 1,
            tr_info["avg_loss"], tr_info["avg_error"], tr_info["avg_SPL"], tr_info["avg_success"],
            v_info["avg_loss"], v_info["avg_error"], v_info["avg_SPL"], v_info["avg_success"],
            tr_info["weight_norm"], tr_info["grad_norm"],
            time_duration,
            "!" if v_info["is_best"] else " "
        ])

        # Save intermediate model.
        if args.save_intermediate:
            torch.save({
                "model": runner.model.state_dict(),
                "best_model": runner.best_model.state_dict(),
            }, save_path + ".e" + str(epoch) + ".pth")

    # Test accuracy
    print("\nFinal test performance:")
    t_final_info = runner.test(test_loader)
    print_stats(t_final_info)
    te_info = {'FinalTest/' + k: v for k, v in t_final_info.items()}

    print("\nBest test performance:")
    t_best_info = runner.test(test_loader, use_best=True)
    print_stats(t_best_info)
    te_info.update({'BestTest/' + k: v for k, v in t_best_info.items()})

    # > log terminal info to W&B
    if args.enable_wandb:
        wandb.log(te_info)

    # > Save final model
    torch.save({
        "model": runner.model.state_dict(),
        "best_model": runner.best_model.state_dict(),
    }, save_path + ".final.pth")

    # > Save config
    OmegaConf.save(config=args, f=save_path + '.config.yaml')

    return runner.model, runner.best_model
