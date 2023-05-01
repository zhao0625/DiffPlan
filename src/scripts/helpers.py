import numpy as np
import torch
from sacred.observers import MongoObserver

import dataclasses
import datetime

from .init import ex

from omegaconf import OmegaConf
import wandb

from config.config import Config

# > Add config from dataclass config
ex.add_config(dataclasses.asdict(Config()))

# > Add Mongo observer
# mongo_url = 'mongodb://10.200.205.226:27017'
# sacred_db_name = 'DEPlan_sacred'
# ex.observers.append(MongoObserver(url=mongo_url, db_name=sacred_db_name))
# stats_db_name = 'DEPlan_statistics'


# > Add other config
@ex.config
def config():
    # > Additional copy
    # args = OmegaConf.structured(Config)

    # > Config experiment run - used in hooks
    enable_wandb = True

    # timestamp = datetime.datetime.utcnow()
    name_time = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    # > TODO set project name here
    project_name = 'DiffPlanLib-DEPlan'
    # dir_wandb = './wandb'

    note = None


# > Hooks (sacred will call all hooks automatically before each run)
@ex.config_hook
def add_defaults(config, command_name, logger):
    logger.info('[add defaults] Current command: {}'.format(command_name))

    if command_name == 'train':
        pass
    else:
        logger.info('[add defaults] No default parameters added')

    return config


@ex.pre_run_hook
def seeding(_log, seed):
    _log.info(f'Seeding using seed {seed}')

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # torch.autograd.set_detect_anomaly(True)  # debug
    # torch.backends.cudnn.enabled = False  # debug


@ex.pre_run_hook
def init_wandb_hook(_log, _run, enable_wandb):
    if enable_wandb:
        init_wandb()


@ex.post_run_hook
def report(_log, _run):
    pass


@ex.capture
def init_wandb(_log, _run, project_name, name_time, note, dir_wandb=None, resume=False):
    """
    Start W&B from the current step, regardless of the option of enabling W&B
    """
    _log.warning(f'W&B dir = {dir_wandb}')
    wandb.init(
        project=project_name,
        name='Train-' + name_time,
        config=_run.config,
        dir=dir_wandb,
        resume=resume,
        notes=note,
    )
