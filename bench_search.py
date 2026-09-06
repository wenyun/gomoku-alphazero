"""Deterministic search-speed benchmark.

Times `BatchedMCTS.search` on boards filled with a fixed number of random
stones, which isolates search cost from how long self-play games happen to last.

Usage:  .venv/bin/python bench_search.py [--games 40] [--sims 320]
"""

import argparse
import time

import numpy as np
import torch

from gomoku_zero.board import Board
from gomoku_zero.config import Config
from gomoku_zero.mcts import BatchedMCTS
from gomoku_zero.network import Evaluator, build_net


def random_board(rng, n_stones):
    """A plausible mid-game position: random legal moves, restarted if won."""
    while True:
        board = Board()
        for _ in range(n_stones):
            legal = np.flatnonzero(board.candidate_mask())
            board.play(int(rng.choice(legal)))
            if board.is_over:
                break
        if not board.is_over:
            return board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=320)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--repeats", type=int, default=6)
    args = ap.parse_args()

    cfg = Config()
    cfg.net.channels = args.channels
    cfg.net.blocks = args.blocks
    cfg.mcts.simulations = args.sims

    evaluator = Evaluator(build_net(cfg.net), device=args.device,
                          graph_batch=None if args.no_graph else args.games)
    mcts = BatchedMCTS(evaluator, cfg.mcts, args.games, np.random.default_rng(0))
    rng = np.random.default_rng(0)

    print(f"net {args.channels}ch x {args.blocks} blocks | {args.games} games | "
          f"{args.sims} sims | cuda_graph={not args.no_graph}")
    print(f"{'stones':>7} {'s/search':>10} {'us/sim/game':>13} {'moves/s':>9}")
    for n_stones in (10, 20, 30, 40):
        boards = [random_board(rng, n_stones) for _ in range(args.games)]
        mcts.search(boards, add_noise=True)                 # warm up
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            mcts.search(boards, add_noise=True)
        dt = (time.perf_counter() - t0) / args.repeats
        per_sim = dt / args.sims / args.games * 1e6
        print(f"{n_stones:7d} {dt:10.3f} {per_sim:13.1f} {args.games/dt:9.1f}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
