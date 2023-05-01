import datetime

# import attrs
# import pydantic
import dataclasses

# from attrs import define, validators, field
# from pydantic.dataclasses import dataclass
from dataclasses import dataclass


@dataclass
class Config:
    # > Experiment args
    use_gpu: bool = True

    timestamp = datetime.datetime.utcnow().isoformat()

    # (Sampling in computing successful rate for every batch or not)
    sample_stats: bool = True
    disable_validation: bool = True  # TODO enable for running final experiments

    # > Env args
    datafile: str = ''
    mechanism: str = '4abs-cc'  # GPPN uses 'news', while by default we use C4 compatible '4abs-cc'
    label: str = 'one_hot'

    # (env information for mapper)
    num_views: int = 4
    img_height: int = 32
    img_width: int = img_height

    # > Log args
    save_directory: str = 'log/'
    save_intermediate: bool = False
    use_percent_successful: bool = False

    # > Optim args
    epochs: int = 30  # Note: default is 30; generally enough for equivariant
    batch_size: int = 32  # Note: default 32 is good; larger (e.g. 256) seems detrimental
    optimizer: str = 'RMSprop'  # Note: VIN & GPPN use RMSprop, Adam sometimes doesn't work well
    scheduler: str = None  # None, or 'cosine' (used in DEQ paper)
    lr: float = 1e-3
    lr_decay: float = 1.0
    eps: float = 1e-6
    clip_grad: float = 40  # Note: equivariant version needs more clip (5 or 10) (GPPN uses 40)
    wandb_watch: bool = True
    weight_decay: float = 0.  # TODO add.

    enable_discount: bool = False  # TODO tune discount
    gamma: float = 0.99

    # > Model args
    model: str = 'models.VIN'  # use name under `./models` folder
    model_path: str = ''
    load_best: bool = False

    k: int = 10
    f: int = 3
    l_i: int = 5
    l_h: int = 150
    l_q: int = 40  # Note: 40 should be enough and even performs better (GPPN uses 600)
    divide_by_size: bool = False  # E2 may be unstable if too fewer parameters

    # (for Spatial Planning Transformer, using Transformer layers)
    k_spt: int = 5
    decoupled_spt_tied: bool = False

    # > For DEQ VIN
    # solver: str = 'anderson'  # 'anderson' or 'forward' - to set separately now; only same default
    solver_fwd: str = 'forward'  # separate solver for forward pass, in computing value iteration
    solver_bwd: str = 'anderson'  # separate solver for backward pass, in computing gradients

    deq_tol: float = 1e-4
    deq_max_iter: int = 50
    deq_anderson_m: int = 5
    deq_anderson_beta: float = 1.0  # default = 1.0 (can be larger than 1)

    # TODO (New: can set separately, but set to same defaults, also for backward compatibility)
    # (DEQ layer args for forward pass)
    deq_fwd_tol: float = deq_tol
    deq_fwd_max_iter: int = deq_max_iter
    deq_fwd_anderson_m: int = deq_anderson_m
    deq_fwd_anderson_beta: float = deq_anderson_beta
    # (DEQ layer args for backward pass)
    deq_bwd_tol: float = deq_tol
    deq_bwd_max_iter: int = deq_max_iter
    deq_bwd_anderson_m: int = deq_anderson_m
    deq_bwd_anderson_beta: float = deq_anderson_beta

    jacobian_reg: bool = False
    jacobian_log: bool = False  # TODO to implement; only logging Jacobian regularization loss, no grad
    jac_reg_weight: float = 0.
    jac_reg_freq: float = 1.  # (to implement) only apply Jacobian regularization loss at some frequency (when > 0)

    # > For DEQ on GPPN
    input_inject: bool = False  # GPPN doesn't inject reward as input; DEQ version requires

    # > For equivariant version
    group: str = 'd4'  # Note: for fully equivariant, C4 and D4 are implemented

    # fiber representation for intermediate fields/embeddings
    repr_in_h: str = 'trivial'
    repr_out_h: str = 'trivial'
    repr_out_r: str = 'regular'  # Note: after tuning, regular seems better
    repr_out_q: str = 'regular'

    # whether enable E2 equivariant policy network (if enable, only C4 and D4 implemented)
    # we should experiment both options
    enable_e2_pi: bool = False
    # > now by default all enable (for E2-VIN and E2-Conv-GPPN)
    # > now disabled by default for better stability

    # > For mapper and learned map
    mapper: str = None  # must be used for visual navigation and work-space manipulation
    mapper_loss: bool = True  # auxiliary loss for mapper, used in GPPN
    mapper_loss_func: str = 'BCE'  # specify loss function to use
    mapper_l_ratio: float = 1.  # loss coefficient
    planner_loss: bool = True  # for debugging: if True, then disable planner training
    train_planner_only_until: int = 0  # train planner with ground truth map until the train_planner_only_until th
    # epoch, then include mapper end to end train. 0 means include mapper at the 0th epoch
    workspace_size: int = 96  # for manipulation
    mapper2probability: bool = False  # enable mapper output {0,1} probability

    visualize_map: bool = True
    visualize_training_model: bool = True
