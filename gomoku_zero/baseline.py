"""A classical (non-learned) gomoku engine, used as a fixed yardstick.

Self-play Elo only tells you that the network beat *its own* earlier versions,
which it will do even if the whole run is drifting somewhere useless.  This
module is the outside reference: a conventional pattern-evaluation plus
alpha-beta engine that never changes, so "wins N% against depth-d baseline" means
the same thing on hour one and hour six.

The evaluation is the standard one for gomoku.  Reusing the `WINDOWS` table from
tactics.py, every straight line of five cells is scored by how many of my stones
it holds -- but only if the opponent has none in it, since a line containing both
colours can never become five.  Summing that over all windows automatically
rewards the things gomoku players care about (long unbroken runs with room to
extend) without hand-coding a pattern dictionary.

This is a decent club-strength player, not a world-class one.  It exists to be a
stable reference, not to be beaten with difficulty.
"""

from __future__ import annotations

import numpy as np

from .board import Board
from .config import NUM_ACTIONS
from .tactics import WINDOWS, five_in_one

# Value of having k of my stones in an otherwise-empty window of five.  The steep
# growth is deliberate: four-in-a-row is worth far more than twice three-in-a-row.
_WEIGHTS = np.array([0.0, 1.0, 12.0, 150.0, 2_500.0, 200_000.0])

# Slight bias towards defence.  Without it the engine happily races and loses by
# one tempo, which is the classic beginner mistake in gomoku.
_DEFENCE = 1.15

WIN_SCORE = 1e9


def evaluate(board: Board) -> float:
    """Static score of `board` from the point of view of the player to move."""
    state = (board.stones * board.to_play).reshape(-1)
    values = state[WINDOWS]
    mine = (values == 1).sum(axis=1)
    opp = (values == -1).sum(axis=1)

    # A window only has potential for a player if the other player is absent.
    my_counts = np.where(opp == 0, mine, 0)
    opp_counts = np.where(mine == 0, opp, 0)
    return float(_WEIGHTS[my_counts].sum() - _DEFENCE * _WEIGHTS[opp_counts].sum())


class HeuristicPlayer:
    """Alpha-beta search over the pattern evaluation above."""

    def __init__(self, depth: int = 3, width: int = 8, seed: int = 0) -> None:
        self.depth = depth
        self.width = width          # candidate moves kept at each node
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------- public
    def select_move(self, board: Board) -> int:
        forced = self._forced_move(board)
        if forced is not None:
            return forced

        best_move, best_score = None, -np.inf
        for move in self._candidates(board, self.width):
            child = board.copy()
            child.play(move)
            score = -self._negamax(child, self.depth - 1, -np.inf, np.inf)
            if score > best_score:
                best_move, best_score = move, score
        return best_move if best_move is not None else self._any_move(board)

    def select_moves(self, boards: list[Board]) -> list[int]:
        """Same interface as the neural agent, so the arena can drive either."""
        return [self.select_move(b) for b in boards]

    # ------------------------------------------------------------------ private
    def _forced_move(self, board: Board) -> int | None:
        """Take an immediate win, or block the opponent's, without searching."""
        state = (board.stones * board.to_play).reshape(-1)
        my_cells, opp_cells = five_in_one(state)
        if my_cells.size:
            return int(my_cells[0])
        if opp_cells.size:
            return int(opp_cells[0])
        return None

    def _candidates(self, board: Board, width: int) -> list[int]:
        """The `width` most promising moves, by static score after playing them."""
        legal = np.flatnonzero(board.candidate_mask())
        scored = []
        for move in legal:
            child = board.copy()
            child.play(int(move))
            if child.is_over:
                return [int(move)]                 # winning move, look no further
            # `evaluate` always scores from the point of view of whoever is to
            # move, and in `child` that is the opponent -- so negate it to get the
            # value of this move to us.  Best moves first, so alpha-beta prunes.
            scored.append((-evaluate(child), int(move)))
        scored.sort(reverse=True)
        return [move for _, move in scored[:width]]

    def _negamax(self, board: Board, depth: int, alpha: float, beta: float) -> float:
        if board.is_over:
            return -WIN_SCORE if board.winner != 0 else 0.0
        if depth <= 0:
            return evaluate(board)

        forced = self._forced_move(board)
        moves = [forced] if forced is not None else self._candidates(board, self.width)

        best = -np.inf
        for move in moves:
            child = board.copy()
            child.play(move)
            score = -self._negamax(child, depth - 1, -beta, -alpha)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break                              # this branch is already refuted
        return best

    def _any_move(self, board: Board) -> int:
        legal = np.flatnonzero(board.candidate_mask())
        return int(self.rng.choice(legal))

    @property
    def name(self) -> str:
        return f"heuristic-d{self.depth}"
