"""The replay buffer: a fixed-size ring of the most recent training positions.

Older positions are overwritten as new games arrive, which matters more than it
might seem.  The network is training against data produced by *itself*, so as it
gets stronger the old positions become a record of how a weaker player used to
play.  Keeping a sliding window means the network is always fitting roughly the
current level of play, while still averaging over enough games to be stable.

Everything is preallocated so that adding a game is a couple of slice
assignments and sampling a batch is one fancy-index -- no per-position Python
objects anywhere.
"""

from __future__ import annotations

import numpy as np

from .config import BOARD_SIZE, NUM_ACTIONS


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.last_move = np.zeros(capacity, dtype=np.int16)
        self.prev_move = np.zeros(capacity, dtype=np.int16)
        self.black = np.zeros(capacity, dtype=np.uint8)
        self.policy = np.zeros((capacity, NUM_ACTIONS), dtype=np.float16)
        self.value = np.zeros(capacity, dtype=np.float16)

        self.size = 0            # positions currently stored
        self.cursor = 0          # where the next position will be written
        self.total_positions = 0  # positions ever added (for the reuse ratio)
        self.total_games = 0

    def add_game(self, record) -> None:
        """Append one finished game, wrapping around the end of the ring."""
        n = len(record.value)
        if n == 0:
            return
        start = self.cursor
        first = min(n, self.capacity - start)
        self._write(start, record, 0, first)
        if first < n:                            # wrapped: write the remainder
            self._write(0, record, first, n - first)

        self.cursor = (start + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        self.total_positions += n
        self.total_games += 1

    def _write(self, dst: int, record, src: int, count: int) -> None:
        sl = slice(dst, dst + count)
        src_sl = slice(src, src + count)
        self.states[sl] = record.states[src_sl]
        self.last_move[sl] = record.last_move[src_sl]
        self.prev_move[sl] = record.prev_move[src_sl]
        self.black[sl] = record.black[src_sl]
        self.policy[sl] = record.policy[src_sl]
        self.value[sl] = record.value[src_sl]

    def sample(self, batch_size: int, rng: np.random.Generator):
        """Uniformly sample a batch of positions from the buffer."""
        idx = rng.integers(0, self.size, size=batch_size)
        return (self.states[idx], self.last_move[idx].astype(np.int32),
                self.prev_move[idx].astype(np.int32),
                self.black[idx].astype(np.float32),
                self.policy[idx].astype(np.float32),
                self.value[idx].astype(np.float32))
