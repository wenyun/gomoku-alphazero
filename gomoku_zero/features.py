"""Turning positions into network input planes.

A position is stored in a *compact* form everywhere outside the network:

    state      int8 (15, 15)   +1 = a stone of the player to move,
                               -1 = a stone of the opponent, 0 = empty
    last_move  int             flat index of the opponent's last move, -1 if none
    prev_move  int             flat index of the mover's own previous move
    black      int             1 if the player to move is black, else 0

Storing it "from the mover's point of view" means the network never has to learn
two mirrored versions of the same concept, and the replay buffer only needs one
byte per cell.  `encode_batch` expands the compact form into the 5 input planes:

    0  my stones
    1  opponent stones
    2  opponent's last move (one-hot)
    3  my previous move (one-hot)
    4  constant plane, 1.0 if the player to move is black

Plane 4 matters in gomoku because the first player has a real advantage, so the
value head needs to know which side of that asymmetry it is looking at.
"""

from __future__ import annotations

import numpy as np

from .config import BOARD_SIZE, NUM_PLANES


def board_to_compact(board):
    """Extract the compact representation from a `Board`."""
    state = (board.stones * board.to_play).astype(np.int8)
    black = 1 if board.to_play == 1 else 0
    return state, board.last_move, board.prev_move, black


def encode_batch(states: np.ndarray, last_move: np.ndarray,
                 prev_move: np.ndarray, black: np.ndarray) -> np.ndarray:
    """Compact form -> float32 planes of shape (B, NUM_PLANES, 15, 15)."""
    batch = states.shape[0]
    planes = np.zeros((batch, NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    planes[:, 0] = states == 1
    planes[:, 1] = states == -1

    flat = planes.reshape(batch, NUM_PLANES, -1)
    rows = np.arange(batch)
    has_last = last_move >= 0
    flat[rows[has_last], 2, last_move[has_last]] = 1.0
    has_prev = prev_move >= 0
    flat[rows[has_prev], 3, prev_move[has_prev]] = 1.0

    planes[:, 4] = black.reshape(batch, 1, 1)
    return planes


def encode_boards(boards) -> np.ndarray:
    """Convenience wrapper: a list of `Board`s -> input planes."""
    n = len(boards)
    states = np.empty((n, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    last = np.empty(n, dtype=np.int32)
    prev = np.empty(n, dtype=np.int32)
    black = np.empty(n, dtype=np.float32)
    for i, board in enumerate(boards):
        states[i] = board.stones * board.to_play
        last[i] = board.last_move
        prev[i] = board.prev_move
        black[i] = 1.0 if board.to_play == 1 else 0.0
    return encode_batch(states, last, prev, black)
