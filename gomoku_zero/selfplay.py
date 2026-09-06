"""Self-play: generating the training data.

A worker keeps `games_per_worker` games going at the same time and advances them
all by one move per `step()`.  This is what makes the GPU busy: the MCTS for all
40 games shares one network call per simulation (see mcts.py).

For every move actually played we store one training example:

    input   the position, in the compact form described in features.py
    policy  the root visit distribution from the search -- this is the "improved
            policy" that the network is trained to imitate, and it is better than
            the network's own output because search corrected it
    value   the eventual game result, from the point of view of the player who
            was about to move

That triple is the entirety of AlphaZero's learning signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .board import Board
from .config import BOARD_SIZE, NUM_ACTIONS, Config
from .mcts import BatchedMCTS, sample_move


@dataclass
class GameRecord:
    """One finished game, ready to be shipped to the learner."""
    states: np.ndarray      # (T, 15, 15) int8, mover's point of view
    last_move: np.ndarray   # (T,) int16
    prev_move: np.ndarray   # (T,) int16
    black: np.ndarray       # (T,) uint8, 1 if the mover was black
    policy: np.ndarray      # (T, 225) float16, normalised visit counts
    value: np.ndarray       # (T,) float16, game result for the mover
    winner: int             # +1 black, -1 white, 0 draw
    moves: np.ndarray       # (T,) int16, the game itself (handy for debugging)


class SelfPlayWorker:
    """Plays a fixed number of games concurrently, yielding them as they finish."""

    def __init__(self, evaluator, cfg: Config, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        n = cfg.selfplay.games_per_worker
        self.mcts = BatchedMCTS(evaluator, cfg.mcts, n, self.rng)
        self.boards = [Board() for _ in range(n)]
        self.trajectories = [[] for _ in range(n)]

    def step(self) -> list[GameRecord]:
        """Advance every game by one move.  Returns the games that just ended."""
        visits = self.mcts.search(self.boards, add_noise=True)

        finished = []
        for i, board in enumerate(self.boards):
            counts = visits[i]
            total = counts.sum()
            if total == 0:                      # no legal move: treat as a draw
                self.boards[i] = Board()
                self.trajectories[i] = []
                continue

            # Record the position *before* the move, with the search policy.
            self.trajectories[i].append((
                (board.stones * board.to_play).astype(np.int8),
                board.last_move,
                board.prev_move,
                1 if board.to_play == 1 else 0,
                (counts / total).astype(np.float16),
                board.to_play,
            ))

            temperature = (1.0 if board.move_count < self.cfg.mcts.opening_moves
                           else self.cfg.mcts.late_temperature)
            action = sample_move(counts, temperature, self.rng)
            board.play(action)
            self.trajectories[i][-1] = self.trajectories[i][-1] + (action,)

            if board.is_over:
                finished.append(self._finalise(i))
                self.boards[i] = Board()
                self.trajectories[i] = []

        return finished

    def _finalise(self, index: int) -> GameRecord:
        """Turn a finished trajectory into a GameRecord with value targets."""
        traj = self.trajectories[index]
        winner = self.boards[index].winner
        t = len(traj)

        states = np.stack([row[0] for row in traj])
        last_move = np.array([row[1] for row in traj], dtype=np.int16)
        prev_move = np.array([row[2] for row in traj], dtype=np.int16)
        black = np.array([row[3] for row in traj], dtype=np.uint8)
        policy = np.stack([row[4] for row in traj])
        movers = np.array([row[5] for row in traj], dtype=np.int8)
        moves = np.array([row[6] for row in traj], dtype=np.int16)

        # The result, seen from whoever was about to move in that position.
        if winner == 0:
            value = np.zeros(t, dtype=np.float16)
        else:
            value = np.where(movers == winner, 1.0, -1.0).astype(np.float16)

        return GameRecord(states, last_move, prev_move, black,
                          policy, value, winner, moves)
