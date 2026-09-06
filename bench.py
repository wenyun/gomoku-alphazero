"""Throughput benchmark for a single self-play worker.

Usage:  .venv/bin/python bench.py [--games 40] [--sims 320] [--seconds 60]
"""

import argparse
import time

import torch

from gomoku_zero.config import Config
from gomoku_zero.network import Evaluator, build_net
from gomoku_zero.selfplay import SelfPlayWorker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=320)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-graph", action="store_true", help="disable CUDA graphs")
    args = ap.parse_args()

    cfg = Config()
    cfg.net.channels = args.channels
    cfg.net.blocks = args.blocks
    cfg.mcts.simulations = args.sims
    cfg.selfplay.games_per_worker = args.games

    evaluator = Evaluator(build_net(cfg.net), device=args.device,
                          graph_batch=None if args.no_graph else args.games)
    worker = SelfPlayWorker(evaluator, cfg, seed=0)

    # Warm up cuDNN autotuning and the allocator.
    for _ in range(2):
        worker.step()

    moves = 0
    games = 0
    positions = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        finished = worker.step()
        moves += args.games
        games += len(finished)
        positions += sum(len(g.value) for g in finished)
    elapsed = time.perf_counter() - t0

    params = sum(p.numel() for p in evaluator.net.parameters())
    print(f"net {args.channels}ch x {args.blocks} blocks, {params/1e6:.2f}M params")
    print(f"{args.games} concurrent games, {args.sims} sims/move")
    print(f"elapsed          {elapsed:6.1f} s")
    print(f"moves played     {moves:6d}  ({moves/elapsed:7.1f} moves/s)")
    print(f"games finished   {games:6d}  ({games/elapsed:7.2f} games/s)")
    print(f"positions        {positions:6d}  ({positions/elapsed:7.1f} pos/s)")
    print(f"NN evals/s       {moves*args.sims/elapsed/args.games*args.games:9.0f} "
          f"(batch {args.games})")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
