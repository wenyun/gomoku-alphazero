"""Tactical puzzles with a provably correct answer.

    .venv/bin/python -m tests.test_puzzles --checkpoint runs/az/latest.pt --sims 1600

Winning a match against a weak opponent does not prove much on its own -- an
engine can win on general positional feel while being tactically shaky.  Each
position here has an answer that is forced by the rules, so a failure is
unambiguous.

Board coordinates follow the display convention: columns A..O left to right, rows
15..1 top to bottom, so `H8` is the centre.
"""

from __future__ import annotations

import argparse

import numpy as np

from gomoku_zero.board import BLACK, Board, action_to_coord, coord_to_action
from gomoku_zero.player import AlphaZeroPlayer
from gomoku_zero.tactics import five_in_one


def build(moves: list[str]) -> Board:
    """Play out a list of coordinates, alternating colours from black."""
    board = Board()
    for text in moves:
        board.play(coord_to_action(text))
    return board


# Filler stones, used to hand a turn to the other player without changing the
# tactical situation.  They are the four corners: no line of five contains two of
# them, so they can never combine into a threat.  Getting this wrong is easy --
# an earlier version of this file used three stones on one row as filler, which
# quietly gave the defending side its own open three and a winning move.
FILLER = ["A1", "O1", "A15", "O15"]


def interleave(attacker: list[str], filler: list[str]) -> list[str]:
    """Alternate attacker stones with harmless filler stones."""
    moves = []
    for i in range(max(len(attacker), len(filler))):
        if i < len(attacker):
            moves.append(attacker[i])
        if i < len(filler):
            moves.append(filler[i])
    return moves


# Each puzzle: (name, board, acceptable answers, is_attacking, why)
def puzzles():
    out = []

    # 1. Complete five.  Black has F8-I8, both ends free, and black to move.
    out.append((
        "win in one",
        build(interleave(["F8", "G8", "H8", "I8"], FILLER)),
        {"E8", "J8"}, True,
        "black has four in a row and simply completes five",
    ))

    # 2. Block a four.  Black has F8-I8 with E8 already white, so J8 is the only
    #    cell that stops five.  White to move.
    out.append((
        "block a four",
        build(["F8", "E8"] + interleave(["G8", "H8", "I8"], FILLER[:3]) + ["O15"]),
        {"J8"}, False,
        "white must stop five, and J8 is the only cell that does",
    ))

    # 3. Turn an open three into an open four.  Black has G8-I8; both F8 and J8
    #    create _XXXX_ with two open ends, which cannot be blocked at all.
    out.append((
        "make an open four",
        build(interleave(["G8", "H8", "I8"], FILLER[:3])),
        {"F8", "J8"}, True,
        "an open four has two winning follow-ups, so this wins by force",
    ))

    # 4. Block an open three, white to move.  Only F8 and J8 work: after anything
    #    else black plays one of them and has an unanswerable open four.  The
    #    filler stones give white no counter-threat of its own.
    out.append((
        "block an open three",
        build(interleave(["G8", "H8", "I8", "O15"], FILLER[:3])),
        {"F8", "J8"}, False,
        "any other move allows black an open four next ply",
    ))

    # 5. Same as (1) but on a diagonal, to check the network is not only
    #    pattern-matching along rows.
    out.append((
        "win in one, diagonally",
        build(interleave(["E5", "F6", "G7", "H8"], FILLER)),
        {"D4", "I9"}, True,
        "black completes five on the diagonal",
    ))

    # 6. Block a diagonal four.  D4 is already white, so I9 is the only stop.
    out.append((
        "block a four, diagonally",
        build(["E5", "D4"] + interleave(["F6", "G7", "H8"], FILLER[:3]) + ["O15"]),
        {"I9"}, False,
        "white must stop five on the diagonal",
    ))

    return out


# --------------------------------------------------------------------- verifier
# The puzzles are checked by brute force before the engine is asked anything, so a
# FAIL always means the engine was wrong rather than the puzzle being wrong.

def _five_completions(board: Board) -> set:
    """Cells where the player to move would immediately make five."""
    state = (board.stones * board.to_play).reshape(-1)
    mine, _ = five_in_one(state)
    return set(int(c) for c in mine)


def forced_wins(board: Board) -> set:
    """Moves that win by force for the player to move.

    Checked shortest-first, which matters: a player who already holds an open
    four has two five-threats no matter what they do next, so "creates a double
    threat" would match every move on the board.  If a move wins *now*, that is
    the answer; only when there is no such move do we look for a double threat.
    """
    immediate, doubles = set(), set()
    for move in np.flatnonzero(board.candidate_mask()):
        child = board.copy()
        child.play(int(move))
        if child.is_over:
            if child.winner != 0:
                immediate.add(int(move))
            continue
        # In `child` it is the opponent's turn, so the mover's own threats come
        # back as the *opponent's* cells from child's point of view.
        state = (child.stones * child.to_play).reshape(-1)
        _, threats = five_in_one(state)
        if len(set(int(c) for c in threats)) >= 2:
            doubles.add(int(move))
    return immediate or doubles


def safe_moves(board: Board) -> set:
    """Moves after which the opponent has no forced win."""
    safe = set()
    for move in np.flatnonzero(board.candidate_mask()):
        child = board.copy()
        child.play(int(move))
        if child.is_over:
            if child.winner != 0:
                safe.add(int(move))          # we just won
            continue
        if not forced_wins(child):
            safe.add(int(move))
    return safe


def verify(name: str, board: Board, answers: set, attacking: bool) -> None:
    computed = forced_wins(board) if attacking else safe_moves(board)
    stated = {coord_to_action(a) for a in answers}
    assert computed == stated, (
        f"puzzle '{name}' is mis-stated: brute force says "
        f"{sorted(action_to_coord(a) for a in computed)}, "
        f"file says {sorted(answers)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/az/latest.pt")
    ap.add_argument("--sims", type=int, default=1600)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    engine = AlphaZeroPlayer(args.checkpoint, device=args.device,
                             simulations=args.sims, batch=1, temperature=0.0)
    print(f"{engine.name}\n")

    cases = puzzles()
    for name, board, answers, attacking, why in cases:
        verify(name, board, answers, attacking)
    print(f"all {len(cases)} puzzles verified by brute force\n")

    passed = 0
    for name, board, answers, attacking, why in cases:
        # Every puzzle is stated as "the player to move must find the move".
        info = engine.analyse(board)
        played = action_to_coord(info["move"])
        ok = played in answers
        passed += ok
        mover = "black" if board.to_play == BLACK else "white"
        print(f"[{'PASS' if ok else 'FAIL'}] {name:26s} "
              f"{mover} played {played:3s} (want {'/'.join(sorted(answers))})  "
              f"win prob {info['win_probability']*100:5.1f}%")
        if not args.quiet and not ok:
            print(f"        {why}")
            print(f"        top moves: " + ", ".join(
                f"{t['coord']} {t['share']*100:.0f}%" for t in info["top"][:5]))
            print(board.render(highlight=board.last_move))

    print(f"\n{passed}/{len(cases)} puzzles solved")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
