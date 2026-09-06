"""Checks that the training targets are self-consistent and actually learnable.

1. Replays every finished game move by move and verifies that the stored value
   target really is the game result from the mover's point of view.
2. Overfits a small batch to confirm the value head can fit its own targets.

Run with:  .venv/bin/python -m tests.test_learning
"""

import numpy as np
import torch

from gomoku_zero.board import Board
from gomoku_zero.config import Config
from gomoku_zero.learner import Learner
from gomoku_zero.network import Evaluator, build_net
from gomoku_zero.replay import ReplayBuffer
from gomoku_zero.selfplay import SelfPlayWorker


def collect_games(n_games=60):
    cfg = Config()
    cfg.net.channels, cfg.net.blocks = 64, 4
    cfg.mcts.simulations = 32
    cfg.selfplay.games_per_worker = 32
    evaluator = Evaluator(build_net(cfg.net), "cuda:0", graph_batch=32)
    worker = SelfPlayWorker(evaluator, cfg, seed=0)
    games = []
    while len(games) < n_games:
        games.extend(worker.step())
    return cfg, games


def test_targets_consistent(games):
    """Replay each game and check states / policies / values line up."""
    for g in games:
        board = Board()
        for t, move in enumerate(g.moves):
            assert np.array_equal(g.states[t], board.stones * board.to_play), \
                f"state mismatch at ply {t}"
            assert g.last_move[t] == board.last_move
            assert g.black[t] == (1 if board.to_play == 1 else 0)
            mover = board.to_play
            expected = 0.0 if g.winner == 0 else (1.0 if mover == g.winner else -1.0)
            assert float(g.value[t]) == expected, f"value mismatch at ply {t}"
            # The policy target must live on legal cells only.
            assert g.policy[t][board.stones.reshape(-1) != 0].sum() == 0
            board.play(int(move))
        assert board.is_over and board.winner == g.winner
    lengths = [len(g.value) for g in games]
    wins = [g.winner for g in games]
    print(f"targets consistent over {len(games)} games | "
          f"mean length {np.mean(lengths):.1f} | "
          f"black wins {wins.count(1)}, white {wins.count(-1)}, draws {wins.count(0)}")


def _fill_buffer(games):
    buffer = ReplayBuffer(200_000)
    for g in games:
        buffer.add_game(g)
    return buffer


def _predict(net, batch, train_mode: bool):
    from gomoku_zero.features import encode_batch
    states, last, prev, black, _, _ = batch
    planes = torch.from_numpy(encode_batch(states, last, prev, black)).cuda()
    net.train(train_mode)
    with torch.no_grad():
        logits, value = net(planes)
    return logits.float(), value.float().cpu().numpy()


def test_value_head_can_fit(cfg, games):
    """The value head must be able to learn at all.

    This is the regression test for the bug that cost the most time in this
    project: with a wide fully connected layer feeding tanh, the pre-activations
    blew up within ~100 steps, tanh saturated, its gradient vanished and the value
    loss parked at 2.0 forever (which is exactly the MSE you get from predicting
    saturated +-1 uncorrelated with a +-1 target).

    Note the correlation is measured in *train* mode, matching how the loss was
    computed.  Overfitting one batch hundreds of times leaves BatchNorm's running
    statistics unrepresentative, so eval-mode numbers here would say nothing about
    the value head -- that is what `test_train_eval_agree` is for.
    """
    buffer = _fill_buffer(games)
    cfg.train.batch_size = 512
    learner = Learner(cfg, device="cuda:0")
    batch = buffer.sample(cfg.train.batch_size, np.random.default_rng(0))

    first = learner.train_step(batch)
    for _ in range(400):
        stats = learner.train_step(batch)

    _, pred = _predict(learner.net, batch, train_mode=True)
    value = batch[5]
    corr = np.corrcoef(pred, value)[0, 1]

    print(f"policy loss {first['policy_loss']:.3f} -> {stats['policy_loss']:.3f}")
    print(f"value  loss {first['value_loss']:.3f} -> {stats['value_loss']:.3f}")
    print(f"value pred std {pred.std():.3f} (must not be ~0) | corr {corr:+.3f}")
    assert stats["value_loss"] < 0.5, "value head cannot even fit a fixed batch"
    assert pred.std() > 0.1, "value head output is saturated / constant"
    assert corr > 0.7, f"value predictions barely track targets ({corr:.3f})"
    assert stats["policy_loss"] < first["policy_loss"] - 0.5


def test_train_eval_agree(cfg, games):
    """Training and inference must give the same answers.

    Self-play runs the network in eval mode (BatchNorm's running statistics) while
    the learner trains with batch statistics.  If those two drift apart, every
    reported loss is measuring a network that never actually plays.  Trains on a
    *fresh* batch each step, which is what real training does.
    """
    buffer = _fill_buffer(games)
    cfg.train.batch_size = 512
    learner = Learner(cfg, device="cuda:0")
    rng = np.random.default_rng(1)
    for _ in range(300):
        learner.train_step(buffer.sample(512, rng))

    def mean_value_loss(train_mode):
        total = 0.0
        for _ in range(10):
            batch = buffer.sample(512, rng)
            _, pred = _predict(learner.net, batch, train_mode)
            total += float(np.mean((pred - batch[5]) ** 2))
        return total / 10

    train_loss = mean_value_loss(True)
    eval_loss = mean_value_loss(False)
    gap = abs(train_loss - eval_loss) / max(train_loss, 1e-6)
    print(f"value loss  train mode {train_loss:.4f}  eval mode {eval_loss:.4f}  "
          f"relative gap {gap*100:.1f}%")
    assert gap < 0.25, f"train/eval mismatch of {gap*100:.0f}% -- BatchNorm problem"


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg, games = collect_games()
    test_targets_consistent(games)
    test_value_head_can_fit(cfg, games)
    test_train_eval_agree(cfg, games)
    print("\nlearning tests passed")
