"""Play against the engine in a terminal.

    .venv/bin/python play.py --checkpoint runs/az/latest.pt --colour black --sims 800

Enter moves as a column letter and a row number, e.g. `H8`.  Other commands:
`undo`, `hint`, `quit`.
"""

from __future__ import annotations

import argparse

from gomoku_zero.board import BLACK, WHITE, Board, action_to_coord, coord_to_action
from gomoku_zero.player import AlphaZeroPlayer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/az/latest.pt")
    ap.add_argument("--colour", choices=("black", "white"), default="black")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    engine = AlphaZeroPlayer(args.checkpoint, device=args.device,
                             simulations=args.sims, batch=1, temperature=0.0)
    human = BLACK if args.colour == "black" else WHITE
    print(f"{engine.name}: you are {args.colour} ('X' is black, 'O' is white)\n")

    board = Board()
    history: list[Board] = []

    while not board.is_over:
        print(board.render(highlight=board.last_move), "\n")

        if board.to_play == human:
            try:
                text = input("your move > ").strip().lower()
            except EOFError:
                return
            if text in ("quit", "exit"):
                return
            if text == "undo":
                for _ in range(2):
                    if history:
                        board = history.pop()
                continue
            if text == "hint":
                info = engine.analyse(board)
                print(f"  engine suggests {action_to_coord(info['move'])} "
                      f"(win probability for you {info['win_probability']*100:.1f}%)")
                continue
            try:
                action = coord_to_action(text)
            except (ValueError, IndexError):
                print("  could not read that; try something like H8")
                continue
            if board.stones.reshape(-1)[action] != 0:
                print("  that point is occupied")
                continue
            history.append(board.copy())
            board.play(action)
        else:
            info = engine.analyse(board)
            history.append(board.copy())
            board.play(info["move"])
            considered = ", ".join(f"{t['coord']} {t['share']*100:.0f}%"
                                  for t in info["top"][:4])
            print(f"engine plays {action_to_coord(info['move'])}  "
                  f"(its win probability {info['win_probability']*100:.1f}%, "
                  f"{info['seconds']:.1f}s)")
            print(f"  considered: {considered}\n")

    print(board.render(highlight=board.last_move))
    if board.winner == 0:
        print("\nDraw.")
    elif board.winner == human:
        print("\nYou win!")
    else:
        print("\nThe engine wins.")


if __name__ == "__main__":
    main()
