"""Head-to-head evaluation between two agents.

    # a trained net against the classical engine
    .venv/bin/python arena.py --a runs/az/latest.pt --baseline-depth 3 --games 60

    # two checkpoints against each other
    .venv/bin/python arena.py --a runs/az/latest.pt --b runs/az/step_0020000.pt --games 100

Colours are alternated so that gomoku's first-player advantage cancels out: agent
A plays black in half the games and white in the other half.  Reporting only
"A won 70%" without that would mostly be measuring who got to move first.

All games run *concurrently*.  Each round we ask each agent for moves in the games
where it is that agent's turn, which keeps the network batch large and makes a
100-game match take about as long as a handful of sequential ones.
"""

from __future__ import annotations

import argparse
import math
import time

from gomoku_zero.baseline import HeuristicPlayer
from gomoku_zero.board import BLACK, Board
from gomoku_zero.player import AlphaZeroPlayer


def elo_difference(score: float, games: int) -> float:
    """Elo gap implied by a score rate in [0, 1] (wins + half the draws)."""
    # Clamp so a clean sweep reports a large-but-finite number instead of infinity.
    eps = 1.0 / (2 * max(games, 1))
    score = min(max(score, eps), 1.0 - eps)
    return -400.0 * math.log10(1.0 / score - 1.0)


def play_match(agent_a, agent_b, num_games: int, verbose: bool = True,
               a_colour: str = "alternate") -> dict:
    """Play `num_games` games between two agents.

    `a_colour` is normally "alternate", which cancels out the first-player
    advantage.  Pinning it to "black" or "white" is useful in gomoku specifically:
    free-style is a first-player win, so between two strong engines black wins
    almost every game and that one effect swamps everything else you might be
    trying to measure.  Fixing the colour lets you compare two agents on the side
    that is actually difficult.
    """
    boards = [Board() for _ in range(num_games)]
    if a_colour == "black":
        a_is_black = [True] * num_games
    elif a_colour == "white":
        a_is_black = [False] * num_games
    else:
        # Agent A takes black in the even-numbered games, white in the odd ones.
        a_is_black = [i % 2 == 0 for i in range(num_games)]
    active = set(range(num_games))
    results = {}

    start = time.perf_counter()
    rounds = 0
    while active:
        a_turn, b_turn = [], []
        for i in sorted(active):
            if (boards[i].to_play == BLACK) == a_is_black[i]:
                a_turn.append(i)
            else:
                b_turn.append(i)

        chosen = {}
        for agent, idx in ((agent_a, a_turn), (agent_b, b_turn)):
            if idx:
                moves = agent.select_moves([boards[i] for i in idx])
                chosen.update(zip(idx, moves))

        for i, move in chosen.items():
            boards[i].play(move)
            if boards[i].is_over:
                winner = boards[i].winner
                if winner == 0:
                    results[i] = 0.5                      # draw
                else:
                    a_won = (winner == BLACK) == a_is_black[i]
                    results[i] = 1.0 if a_won else 0.0
                active.discard(i)

        rounds += 1
        if verbose and rounds % 10 == 0:
            print(f"  round {rounds:3d}: {len(active):4d} games still running",
                  flush=True)

    wins = sum(1 for v in results.values() if v == 1.0)
    losses = sum(1 for v in results.values() if v == 0.0)
    draws = sum(1 for v in results.values() if v == 0.5)
    score = sum(results.values()) / num_games

    # Break the result down by colour, which is informative in gomoku.
    as_black = [results[i] for i in range(num_games) if a_is_black[i]]
    as_white = [results[i] for i in range(num_games) if not a_is_black[i]]

    return {
        "games": num_games,
        "wins": wins, "losses": losses, "draws": draws,
        "score": score,
        "elo_diff": elo_difference(score, num_games),
        "score_as_black": sum(as_black) / max(len(as_black), 1),
        "score_as_white": sum(as_white) / max(len(as_white), 1),
        "seconds": time.perf_counter() - start,
        "mean_length": sum(b.move_count for b in boards) / num_games,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="checkpoint for agent A")
    ap.add_argument("--b", help="checkpoint for agent B (omit to use the baseline)")
    ap.add_argument("--baseline-depth", type=int, default=3)
    ap.add_argument("--baseline-width", type=int, default=8)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--sims-b", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-b", default=None)
    ap.add_argument("--a-colour", choices=("alternate", "black", "white"),
                    default="alternate",
                    help="pin agent A's colour instead of alternating")
    ap.add_argument("--temperature", type=float, default=0.15,
                    help="small non-zero value keeps the games from being identical")
    args = ap.parse_args()

    agent_a = AlphaZeroPlayer(args.a, device=args.device, simulations=args.sims,
                              batch=args.games, temperature=args.temperature, seed=1)
    if args.b:
        agent_b = AlphaZeroPlayer(args.b, device=args.device_b or args.device,
                                  simulations=args.sims_b or args.sims,
                                  batch=args.games, temperature=args.temperature, seed=2)
    else:
        agent_b = HeuristicPlayer(depth=args.baseline_depth, width=args.baseline_width)

    print(f"A = {agent_a.name}")
    print(f"B = {agent_b.name}")
    print(f"playing {args.games} games...", flush=True)
    r = play_match(agent_a, agent_b, args.games, a_colour=args.a_colour)

    print(f"\nA scored {r['score']*100:5.1f}%  "
          f"({r['wins']}W / {r['losses']}L / {r['draws']}D)")
    print(f"  as black {r['score_as_black']*100:5.1f}%   "
          f"as white {r['score_as_white']*100:5.1f}%")
    print(f"  Elo difference {r['elo_diff']:+.0f}")
    print(f"  mean game length {r['mean_length']:.1f} moves, "
          f"{r['seconds']:.0f}s total")


if __name__ == "__main__":
    main()
