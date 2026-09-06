"""Exact "can somebody make five right now?" detection.

This is the one piece of hand-written game knowledge in the whole engine, and it
earns its place.  A pure network + search agent is blind to a threat until the
search happens to walk into it, and with ~70 candidate moves per node a shallow
search misses "you win next move unless you block" surprisingly often.  Checking
it exactly costs a few microseconds and removes that entire class of blunder.

Everything here is derived from one precomputed table: `WINDOWS`, the list of
every straight line of five cells on the board (horizontal, vertical, and both
diagonals).  A player can complete five in one move exactly when some window
holds four of their stones and one empty cell -- and that empty cell is the
winning move.

Note what this deliberately does *not* do: no open-three detection, no
threat-space search, no static evaluation.  Judging anything beyond an immediate
five is left to the network, which is what learns the actual game.
"""

from __future__ import annotations

import numpy as np

from .config import BOARD_SIZE, NUM_ACTIONS


def _build_windows() -> np.ndarray:
    """Every length-5 straight line on the board, as flat cell indices."""
    windows = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                end_r, end_c = r + 4 * dr, c + 4 * dc
                if 0 <= end_r < BOARD_SIZE and 0 <= end_c < BOARD_SIZE:
                    windows.append([(r + i * dr) * BOARD_SIZE + (c + i * dc)
                                    for i in range(5)])
    return np.asarray(windows, dtype=np.int32)


WINDOWS = _build_windows()          # (572, 5) for a 15x15 board


def five_in_one(state: np.ndarray):
    """Cells that complete five, for the mover and for the opponent.

    `state` is the flat 225-element view of the position from the mover's point
    of view (+1 mine, -1 opponent, 0 empty).  Returns (my_cells, opp_cells),
    each a possibly-empty array of flat cell indices.

    The trick that makes this fast enough to run at every node: just *sum* each
    window.  With five values drawn from {-1, 0, +1}, a sum of +4 is only
    possible as four of my stones plus one empty cell, and a sum of -4 only as
    four opponent stones plus one empty cell.  (Five in a row would sum to +-5,
    but that position would already be over, and this is only called on live
    positions.)  So one reduction answers the question for both players.
    """
    values = state[WINDOWS]                      # (num_windows, 5)
    sums = values.sum(axis=1)

    # Fast path: almost every position has no immediate five for either side,
    # and we can rule that out with two more passes instead of building masks.
    my_cells = _empty_cells(values, sums, 4) if sums.max() == 4 else _NO_CELLS
    opp_cells = _empty_cells(values, sums, -4) if sums.min() == -4 else _NO_CELLS
    return my_cells, opp_cells


_NO_CELLS = np.empty(0, dtype=np.int32)


def _empty_cells(values: np.ndarray, sums: np.ndarray, target: int) -> np.ndarray:
    """The empty cell of every window whose values sum to `target`."""
    selected = sums == target
    return WINDOWS[selected][values[selected] == 0]


def interior_analysis(board):
    """Tactical verdict for a leaf node the search is about to expand.

    Returns (mask, proven_value):

      * proven_value is not None when the outcome is already decided, so the
        node needs no network evaluation at all and can be treated as terminal:
          +1  the mover completes five immediately;
          -1  the opponent has two *different* winning cells, and one stone can
              only ever block one of them.
      * otherwise mask is the set of moves the search may consider.  If the
        opponent has exactly one winning cell, that cell is the only move worth
        searching, because every alternative simply loses next ply.
    """
    state = (board.stones * board.to_play).reshape(-1)
    my_cells, opp_cells = five_in_one(state)

    if my_cells.size:
        return None, 1.0

    if opp_cells.size:
        cells = np.unique(opp_cells)
        if cells.size > 1:
            return None, -1.0
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        mask[cells[0]] = True
        return mask, None

    return board.candidate_mask(), None


def root_analysis(board) -> np.ndarray:
    """Moves to consider at the root, where we must actually return a move.

    Same logic as `interior_analysis`, but a proven result is turned into "play
    the move that proves it" instead of a value: take the win if there is one,
    otherwise block the opponent's threat.
    """
    state = (board.stones * board.to_play).reshape(-1)
    my_cells, opp_cells = five_in_one(state)

    if my_cells.size:
        forced = np.unique(my_cells)
    elif opp_cells.size:
        forced = np.unique(opp_cells)     # if there are two, we are lost anyway
    else:
        return board.candidate_mask()

    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    mask[forced] = True
    return mask
