from omegaconf import OmegaConf

from .init import ex
from .train import start_train
from .eval import start_eval, start_eval_on_test, eval_runner


@ex.command
def convert_config(_run, _log, _config):
    """
    Demo for accessing config, using `_config` or `_run.config`
    """
    print(OmegaConf.to_yaml(OmegaConf.structured(_config)))


@ex.command
def run_train(_run, _log, _config):
    args = OmegaConf.structured(_config)

    final_model, best_model = start_train(args)


@ex.command
def run_eval(_run, _log, _config, eval_on_test=True):
    args = OmegaConf.structured(_config)

    eval_runner(args, eval_on_test)
