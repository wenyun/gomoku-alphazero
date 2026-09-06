"""Gomoku rules.

Free-style gomoku on a 15x15 board: players alternate placing stones, black
moves first, and the first player to get five *or more* of their stones in a
row (horizontally, vertically or diagonally) wins.  There are no forbidden-move
restrictions (no "renju" rules), which keeps the rule code short and is the
variant most engines are benchmarked on.

Board representation
--------------------
`stones` is a 15x15 int8 array holding +1 for black, -1 for white, 0 for empty.
`to_play` is +1 or -1.  Because everything is small, a full board copy costs
about a microsecond, which is what lets MCTS just copy the root position at the
start of every simulation instead of implementing move/unmove bookkeeping.

`near` is the one piece of non-obvious state: `near[r, c]` counts how many
stones lie within `CANDIDATE_RADIUS` of cell (r, c).  It is updated
incrementally on every move and is used to build the candidate-move mask.
"""

from __future__ import annotations

import numpy as np

from .config import BOARD_SIZE, CANDIDATE_RADIUS, NUM_ACTIONS

BLACK = 1
WHITE = -1
EMPTY = 0

# The four axes we have to test for five-in-a-row.  The opposite directions are
# handled by walking both ways from the stone that was just played.
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

# On an empty board every move is equivalent up to symmetry, so we only allow
# the middle of the board.  A small region (rather than just the centre point)
# keeps a little opening variety in self-play.
_OPENING_REGION = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
_OPENING_REGION[BOARD_SIZE // 2 - 2: BOARD_SIZE // 2 + 3,
                BOARD_SIZE // 2 - 2: BOARD_SIZE // 2 + 3] = True


class Board:
    __slots__ = ("stones", "near", "to_play", "move_count",
                 "last_move", "prev_move", "winner", "is_over")

    def __init__(self) -> None:
        self.stones = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.near = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int16)
        self.to_play = BLACK
        self.move_count = 0
        self.last_move = -1     # most recent move (played by -to_play)
        self.prev_move = -1     # the move before that (played by to_play)
        self.winner = 0         # +1 / -1 once someone has won, 0 otherwise
        self.is_over = False

    # ------------------------------------------------------------------ copying
    def copy(self) -> "Board":
        other = Board.__new__(Board)
        other.stones = self.stones.copy()
        other.near = self.near.copy()
        other.to_play = self.to_play
        other.move_count = self.move_count
        other.last_move = self.last_move
        other.prev_move = self.prev_move
        other.winner = self.winner
        other.is_over = self.is_over
        return other

    # ------------------------------------------------------------------- moves
    def candidate_mask(self) -> np.ndarray:
        """Flat bool array of the moves the search is allowed to consider.

        Empty cells within CANDIDATE_RADIUS of an existing stone.  Falls back to
        every empty cell in the (practically impossible) case that the pruned
        set is empty, so callers never have to deal with a stuck position.
        """
        if self.move_count == 0:
            return _OPENING_REGION.reshape(-1).copy()
        empty = self.stones == EMPTY
        mask = empty & (self.near > 0)
        if not mask.any():
            mask = empty
        return mask.reshape(-1)

    def play(self, action: int) -> None:
        """Place a stone for `to_play` at flat index `action` and flip the turn."""
        row, col = divmod(action, BOARD_SIZE)
        player = self.to_play
        self.stones[row, col] = player

        # Keep the "distance to nearest stone" bookkeeping up to date.
        r0 = max(0, row - CANDIDATE_RADIUS)
        r1 = min(BOARD_SIZE, row + CANDIDATE_RADIUS + 1)
        c0 = max(0, col - CANDIDATE_RADIUS)
        c1 = min(BOARD_SIZE, col + CANDIDATE_RADIUS + 1)
        self.near[r0:r1, c0:c1] += 1

        self.prev_move = self.last_move
        self.last_move = action
        self.move_count += 1
        self.to_play = -player

        if self._is_five(row, col, player):
            self.winner = player
            self.is_over = True
        elif self.move_count == NUM_ACTIONS:
            self.winner = 0          # full board, nobody won
            self.is_over = True

    def _is_five(self, row: int, col: int, player: int) -> bool:
        """Did the stone just played at (row, col) complete five in a row?"""
        stones = self.stones
        for dr, dc in _DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and stones[r, c] == player:
                    count += 1
                    if count >= 5:
                        return True
                    r += sign * dr
                    c += sign * dc
        return False

    # --------------------------------------------------------------- terminals
    def terminal_value(self) -> float:
        """Game result from the point of view of the player who is `to_play`.

        Only meaningful when `is_over`.  Since the winner is always the player
        who just moved, this is -1 for a loss and 0 for a draw.
        """
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == self.to_play else -1.0

    # ------------------------------------------------------------------ pretty
    def render(self, highlight: int = -1) -> str:
        symbols = {BLACK: "X", WHITE: "O", EMPTY: "."}
        columns = "   " + " ".join(chr(ord("A") + i) for i in range(BOARD_SIZE))
        lines = [columns]
        for r in range(BOARD_SIZE):
            cells = []
            for c in range(BOARD_SIZE):
                ch = symbols[int(self.stones[r, c])]
                if r * BOARD_SIZE + c == highlight and ch != ".":
                    ch = ch.lower()          # mark the most recent move
                cells.append(ch)
            lines.append(f"{BOARD_SIZE - r:2d} " + " ".join(cells))
        return "\n".join(lines)


def action_to_coord(action: int) -> str:
    row, col = divmod(action, BOARD_SIZE)
    return f"{chr(ord('A') + col)}{BOARD_SIZE - row}"


def coord_to_action(text: str) -> int:
    text = text.strip().upper()
    col = ord(text[0]) - ord("A")
    row = BOARD_SIZE - int(text[1:])
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"coordinate out of range: {text}")
    return row * BOARD_SIZE + col
