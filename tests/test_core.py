"""Sanity checks for the game rules and the search.

Run with:  .venv/bin/python -m tests.test_core
"""

import numpy as np
import torch

from gomoku_zero.board import BLACK, WHITE, Board, action_to_coord, coord_to_action
from gomoku_zero.config import BOARD_SIZE, MCTSConfig, NetConfig
from gomoku_zero.mcts import BatchedMCTS
from gomoku_zero.network import Evaluator, build_net


def rc(r, c):
    return r * BOARD_SIZE + c


def test_win_detection():
    # Horizontal five for black.
    b = Board()
    for i in range(4):
        b.play(rc(7, 3 + i))          # black
        b.play(rc(0, i))              # white, harmless
    assert not b.is_over
    b.play(rc(7, 7))
    assert b.is_over and b.winner == BLACK

    # Diagonal five for white.
    b = Board()
    for i in range(4):
        b.play(rc(0, i))              # black filler
        b.play(rc(3 + i, 3 + i))      # white diagonal
    b.play(rc(14, 14))                # black elsewhere
    b.play(rc(7, 7))
    assert b.is_over and b.winner == WHITE

    # Overline (six) also wins in free-style.
    b = Board()
    for i in range(5):
        b.play(rc(5, 2 + i))
        if not b.is_over:
            b.play(rc(0, i))
    assert b.is_over and b.winner == BLACK
    print("win detection ok")


def test_terminal_value_perspective():
    b = Board()
    for i in range(4):
        b.play(rc(7, 3 + i))
        b.play(rc(0, i))
    b.play(rc(7, 7))                  # black wins, so it is now white's "turn"
    assert b.to_play == WHITE
    assert b.terminal_value() == -1.0  # loss for the player to move
    print("terminal value ok")


def test_candidate_mask():
    b = Board()
    mask = b.candidate_mask()
    assert mask.sum() == 25, mask.sum()          # 5x5 opening region
    b.play(rc(7, 7))
    mask = b.candidate_mask().reshape(BOARD_SIZE, BOARD_SIZE)
    assert mask.sum() == 24                      # 5x5 around centre, minus the stone
    assert mask[5, 5] and mask[9, 9] and not mask[7, 7] and not mask[4, 7]
    print("candidate mask ok")


def test_coords():
    for a in (0, 7 * BOARD_SIZE + 7, BOARD_SIZE * BOARD_SIZE - 1):
        assert coord_to_action(action_to_coord(a)) == a
    print("coordinates ok")


def test_search_finds_forced_win():
    """With an untrained network the *search* alone should still see a win in 1.

    Black has four in a row with both ends open; the only sensible move is to
    complete five.  MCTS scores terminal positions exactly, so even random
    priors must find it.
    """
    b = Board()
    b.play(rc(7, 5))
    b.play(rc(2, 2))
    b.play(rc(7, 6))
    b.play(rc(2, 3))
    b.play(rc(7, 7))
    b.play(rc(2, 4))
    b.play(rc(7, 8))
    b.play(rc(3, 9))
    assert b.to_play == BLACK

    evaluator = Evaluator(build_net(NetConfig(channels=32, blocks=2)), device="cuda")
    mcts = BatchedMCTS(evaluator, MCTSConfig(simulations=200), num_games=1,
                       rng=np.random.default_rng(0))
    visits = mcts.search([b], add_noise=False)
    best = int(visits[0].argmax())
    assert best in (rc(7, 4), rc(7, 9)), action_to_coord(best)
    # The average is taken over all visits, including the exploratory ones,
    # so it sits below 1.0 even when the win is found immediately.
    assert mcts.root_value(0) > 0.5, mcts.root_value(0)
    print(f"search found the win at {action_to_coord(best)}, "
          f"root value {mcts.root_value(0):+.3f}")


def test_search_blocks_forced_loss():
    """White must block black's simple four -- there is exactly one saving move.

    Note we deliberately use a four with one end already blocked.  An *open*
    four cannot be stopped at all, so every move loses and the search has no
    reason to prefer a "block"; this position has a unique correct answer.
    """
    b = Board()
    b.play(rc(7, 5)); b.play(rc(7, 4))     # white blocks the left end
    b.play(rc(7, 6)); b.play(rc(2, 2))
    b.play(rc(7, 7)); b.play(rc(2, 3))
    b.play(rc(7, 8))                       # black threatens G8...K8 at L8
    assert b.to_play == WHITE

    evaluator = Evaluator(build_net(NetConfig(channels=32, blocks=2)), device="cuda")
    mcts = BatchedMCTS(evaluator, MCTSConfig(simulations=400), num_games=1,
                       rng=np.random.default_rng(0))
    visits = mcts.search([b], add_noise=False)
    best = int(visits[0].argmax())
    assert best == rc(7, 9), action_to_coord(best)
    print(f"search blocks the four at {action_to_coord(best)}")


def test_batched_search_matches_single():
    """Searching K boards together must give each the same tree as alone."""
    boards = []
    for k in range(4):
        b = Board()
        b.play(rc(7, 7))
        b.play(rc(7, 8 - k))
        boards.append(b)

    evaluator = Evaluator(build_net(NetConfig(channels=32, blocks=2)), device="cuda")
    cfg = MCTSConfig(simulations=64)

    together = BatchedMCTS(evaluator, cfg, num_games=4).search(boards, add_noise=False)
    alone = np.stack([
        BatchedMCTS(evaluator, cfg, num_games=1).search([b], add_noise=False)[0]
        for b in boards
    ])
    assert np.array_equal(together, alone)
    print("batched search == single search")


def test_simulation_count_can_change():
    """The play tools change the strength at run time; trees must grow to match.

    The tree arrays are preallocated from a simulation count, so asking for more
    simulations than the tree was built for used to run off the end of them.
    """
    evaluator = Evaluator(build_net(NetConfig(channels=32, blocks=2)), device="cuda")
    mcts = BatchedMCTS(evaluator, MCTSConfig(simulations=100), num_games=1,
                       rng=np.random.default_rng(0))
    board = Board()
    board.play(rc(7, 7))
    for sims in (100, 402, 4000, 500):
        visits = mcts.search([board], add_noise=False, simulations=sims)
        assert visits.sum() == sims, (sims, visits.sum())
    print("simulation count can be changed between searches")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_win_detection()
    test_terminal_value_perspective()
    test_candidate_mask()
    test_coords()
    test_search_finds_forced_win()
    test_search_blocks_forced_loss()
    test_batched_search_matches_single()
    test_simulation_count_can_change()
    print("\nall core tests passed")
