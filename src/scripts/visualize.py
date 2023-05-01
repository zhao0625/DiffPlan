from omegaconf import OmegaConf

from utils.helpers_model import inference_model
from utils.vis_fields import run_visualize_policy_equivariance
from .init import ex


@ex.command
def visualize_value(_run, _log, _config):
    args = OmegaConf.structured(_config)


@ex.command
def visualize_policy(_run, _log, _config):
    args = OmegaConf.structured(_config)

    out = inference_model(args)

    run_visualize_policy_equivariance(out=out, idx=0)
