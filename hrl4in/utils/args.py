def add_env_args(parser):
    group = parser.add_argument_group("env")
    group.add_argument(
        "--env-type",
        required=True,
        help="env type: [gibson|interactive_gibson|toy]",
        choices=["gibson", "interactive_gibson", "toy"]
    )
    group.add_argument(
        "--config-file",
        required=True,
        help="config yaml file for Gibson environment",
    )
    group.add_argument(
        "--env-mode",
        type=str,
        default="headless",
        help="environment mode for the simulator (default: headless)",
    )
    group.add_argument(
        "--action-timestep",
        type=float,
        default=1.0 / 10.0,
        help="action timestep for the simulator (default: 0.1)",
    )
    group.add_argument(
        "--physics-timestep",
        type=float,
        default=1.0 / 40.0,
        help="physics timestep for the simulator (default: 0.025)",
    )
    group.add_argument(
        "--random-position",
        action="store_true",
        default=False,
        help="whether to randomize initial and target position (default: False)",
    )
    group.add_argument(
        "--random-height",
        action="store_true",
        default=False,
        help="whether to randomize the height of target position (default: False)",
    )
    group.add_argument(
        "--arena",
        type=str,
        default="complex_hl_ll",
        help="arena to load on pybullet",
    )


def add_hrl_args(parser):
    group = parser.add_argument_group("hrl")
    group.add_argument(
        "--meta-lr",
        type=float,
        default=7e-4,
        help="meta learning rate (default: 7e-4)"
    )
    group.add_argument(
        "--freeze-lr-n-updates",
        type=int,
        default=0,
        help="Set meta learning rate to be zero for the first n updates (default: 0)"
    )
    group.add_argument(
        "--time-scale",
        type=int,
        default=10,
        help="maximum number of actions that the low-level policy can take to fulfill "
             "the subgoals set by the high-level policy"
    )
    group.add_argument(
        "--use-action-masks",
        action="store_true",
        default=False,
        help="whether to allow high-level policy to mask the actions of low-level policy",
    )
    group.add_argument(
        "--extrinsic-reward-weight",
        type=float,
        default=0.0,
        help="how much extrinsic reward is exposed to the low-level policy",
    )
    group.add_argument(
        "--extrinsic-collision-reward-weight",
        type=float,
        default=0.0,
        help="how much extrinsic collision reward is exposed to the low-level policy",
    )
    group.add_argument(
        "--intrinsic-reward-scaling",
        type=float,
        default=10.0,
        help="scaling factor for the incremental intrinsic reward",
    )
    group.add_argument(
        "--subgoal-achieved-reward",
        type=float,
        default=0.0,
        help="reward given to the low-level policy when the subgoal is achieved",
    )
    group.add_argument(
        "--meta-gamma",
        type=float,
        default=0.999,
        help="discount factor for the high-level policy",
    )
    group.add_argument(
        "--subgoal-failed-penalty",
        type=float,
        default=0.0,
        help="penalty given to the high-level policy if the low-level policy fails to achieves its subgoal",
    )
    group.add_argument(
        "--subgoal-init-std-dev",
        nargs="+",
        type=float,
        default=[0.4, 0.4, 0.1],
        help="initial standard deviation for the high-level policy for setting subgoals",
    )
    group.add_argument(
        "--subgoal-min-std-dev",
        nargs="+",
        type=float,
        default=[0.05, 0.05, 0.05],
        help="minimum standard deviation for the high-level policy for setting subgoals",
    )
    group.add_argument(
        "--use-action-hindsight",
        action="store_true",
        default=False,
        help="whether to action hindsight, i.e. assume perfect LL policy and overwrite original subgoals "
             "with states actually achieved",
    )
    group.add_argument(
        "--meta-agent-normalize-advantage",
        action="store_true",
        default=False,
        help="whether to normalize advantage for meta agent",
    )


def add_common_args(parser):
    group = parser.add_argument_group("common")
    # gpu id related
    group.add_argument(
        "--sim-gpu-id",
        type=str,
        required=True,
        help="a comma-delimited list of gpu id on which scenes are loaded, e.g. 3,4,5",
    )
    group.add_argument(
        "--pth-gpu-id",
        type=str,
        required=True,
        help="a comma-delimited list of gpu ids on which pytorch runs, e.g. 0,1,2",
    )

    # training related
    group.add_argument(
        "--lr",
        type=float,
        default=7e-4,
        help="learning rate (default: 7e-4)"
    )
    group.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    group.add_argument(
        "--use-linear-lr-decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate (default: False)",
    )
    group.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )

    # parallel environment related
    group.add_argument(
        "--num-train-processes",
        type=int,
        default=16,
        help="number of training processes " "to use (default: 16)",
    )
    group.add_argument(
        "--num-eval-processes",
        type=int,
        default=1,
        help="number of evaluation processes " "to use (default: 1)",
    )

    # evaluation related
    group.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="number of updates after which models are evaluated",
    )
    group.add_argument(
        "--num-eval-episodes",
        type=int,
        default=10,
        help="number of episodes for each evaluation",
    )
    group.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="evaluation only",
    )

    # log, checkpoint and summary related
    group.add_argument(
        "--experiment-folder",
        type=str,
        required=True,
        help="path to the experiment folder, which will contains 1) log file 2) ckpt/ and 3) summary/"
    )
    group.add_argument(
        "--perf-window-size",
        type=int,
        default=50,
        help="logging window for performances, such as rewards and success rates",
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="number of updates after which metrics are logged",
    )
    group.add_argument(
        "--summary-interval",
        type=int,
        default=1,
        help="number of updates after which metrics are updated to tensorboard",
    )
    group.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="number of updates after which models are checkpointed",
    )
    group.add_argument(
        "--checkpoint-index",
        type=int,
        default=-1,
        help="which checkpoint to load, -1 to load the most recent one (default: -1)",
    )

    # random mechanism related
    group.add_argument(
        "--seed",
        type=int,
        default=100,
        help="random seed"
    )


def add_ppo_args(parser):
    group = parser.add_argument_group("ppo")
    group.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    group.add_argument(
        "--ppo-epoch",
        type=int,
        default=4,
        help="number of ppo epochs (default: 4)",
    )
    group.add_argument(
        "--num-mini-batch",
        type=int,
        default=32,
        help="number of batches for ppo (default: 32)",
    )
    group.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    group.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    group.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )
    group.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="number of forward steps in A2C (default: 5)",
    )
    group.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="size of the hidden layer for RNNs (default: 512)"
    )
    group.add_argument(
        "--use-gae",
        action="store_true",
        default=False,
        help="use generalized advantage estimation (default: False)",
    )
    group.add_argument(
        "--use-linear-clip-decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the " "ppo clipping parameter (default: False)",
    )
    group.add_argument(
        "--tau", type=float, default=0.95, help="gae parameter (default: 0.95)"
    )
    group.add_argument(
        "--num-updates",
        type=int,
        default=10000,
        help="number of PPO updates to run",
    )
    group.add_argument(
        "--action-init-std-dev",
        type=float,
        default=1.0 / 3.0,
        help="initial standard deviation for the high-level policy for choosing actions",
    )
    group.add_argument(
        "--action-min-std-dev",
        type=float,
        default=0.1,
        help="minimum standard deviation for the low-level policy for choosing actions",
    )
    group.add_argument(
        "--action-std-dev-anneal-schedule",
        type=int,
        default=None,
        help="linear anneal log std dev of action distribution from action-init-std-dev to action-min-std-dev,"
             "starting from update 0 to update action-std-dev-anneal-schedule."
             "After update action-std-dev-anneal-schedule, stddev is kept at action-min-std-dev.",
    )
