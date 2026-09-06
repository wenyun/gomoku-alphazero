"""All hyper-parameters in one place.

Everything the rest of the code needs to know about "how big / how long / how fast"
lives here, so you can read one file to understand the shape of the whole system.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------- game geometry
BOARD_SIZE = 15                        # standard gomoku board
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE   # 225 cells, action = flat cell index
NUM_PLANES = 5                          # see features.py for what each plane means

# A move is only considered if it is within this Chebyshev distance of a stone
# already on the board.  In gomoku a move far away from every stone is never
# useful, and pruning them shrinks the branching factor from ~225 to ~40, which
# lets the same number of MCTS simulations search much deeper.
CANDIDATE_RADIUS = 2


@dataclass
class NetConfig:
    """Shape of the policy/value network (see network.py)."""
    channels: int = 128
    blocks: int = 10
    head_channels: int = 32
    value_hidden: int = 128


@dataclass
class MCTSConfig:
    """Search parameters (see mcts.py)."""
    simulations: int = 512          # NN evaluations per move during self-play
    c_puct: float = 2.0             # exploration constant in the PUCT formula
    dirichlet_alpha: float = 0.15   # root noise, keeps self-play exploring
    dirichlet_epsilon: float = 0.25
    # Self-play samples moves from the visit distribution for the first
    # `opening_moves` plies (temperature 1) and plays close to greedily after.
    opening_moves: int = 12
    late_temperature: float = 0.25


@dataclass
class SelfPlayConfig:
    """How the self-play worker processes are organised (see selfplay.py)."""
    workers_per_gpu: int = 6
    games_per_worker: int = 96      # games searched concurrently -> NN batch size
    gpus: tuple = (1, 2, 3, 4, 5, 6, 7)   # GPU 0 is reserved for the learner
    reload_every_moves: int = 20    # how often a worker checks for new weights


@dataclass
class TrainConfig:
    """Optimisation of the network (see learner.py)."""
    batch_size: int = 1024
    lr: float = 1e-3
    warmup_steps: int = 1_000                  # ramp lr up so the heads settle first
    lr_milestones: tuple = (80_000, 140_000)   # step -> lr * 0.1 at each
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    buffer_capacity: int = 3_000_000           # positions kept in the replay buffer
    min_buffer: int = 40_000                   # start training once we have this many
    # Bound how many times the learner may reuse each generated position.  This
    # keeps the learner from over-fitting a small buffer early in the run.
    max_reuse_per_position: float = 6.0
    checkpoint_every: int = 500                # steps between weight publications
    snapshot_every: int = 5_000                # steps between permanent snapshots
    log_every: int = 100


@dataclass
class Config:
    net: NetConfig = field(default_factory=NetConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run_dir: str = "runs/az"
