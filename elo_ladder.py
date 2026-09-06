"""Builds a cumulative Elo curve over a run's snapshots.

    .venv/bin/python elo_ladder.py --run-dir runs/az --stride 4 --games 60

Measuring everything against one fixed opponent stops working once the network is
far stronger than it: every match reads 100% and the scale saturates.  A ladder
avoids that by only ever comparing *neighbouring* snapshots, where the gap is
small enough to measure precisely, and then chaining the differences:

    Elo(s_0) = 0,   Elo(s_{i+1}) = Elo(s_i) + measured_gap(s_{i+1}, s_i)

The absolute numbers are arbitrary (only differences mean anything), and chaining
accumulates error, so treat the curve as "how much did it improve, roughly" rather
than as a precise rating.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from arena import play_match
from gomoku_zero.player import AlphaZeroPlayer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/az")
    ap.add_argument("--stride", type=int, default=4,
                    help="use every Nth snapshot (they are close together)")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.run_dir, "step_*.pt")))[::args.stride]
    latest = os.path.join(args.run_dir, "latest.pt")
    if os.path.exists(latest) and paths and os.path.basename(paths[-1]) != "latest.pt":
        paths.append(latest)
    if len(paths) < 2:
        print("need at least two snapshots")
        return

    out_path = args.out or os.path.join(args.run_dir, "elo_ladder.jsonl")
    rows = []
    elo = 0.0
    print(f"{'snapshot':>22} {'step':>8} {'score vs prev':>14} {'gap':>7} {'elo':>8}")
    print(f"{os.path.basename(paths[0]):>22} {'-':>8} {'-':>14} {'-':>7} {elo:8.0f}")

    previous = AlphaZeroPlayer(paths[0], device=args.device, simulations=args.sims,
                               batch=args.games, temperature=0.2, seed=11)
    for path in paths[1:]:
        current = AlphaZeroPlayer(path, device=args.device, simulations=args.sims,
                                  batch=args.games, temperature=0.2, seed=12)
        result = play_match(current, previous, args.games, verbose=False)
        elo += result["elo_diff"]
        rows.append({"snapshot": os.path.basename(path), "step": current.step,
                     "elo": round(elo, 1), **result})
        print(f"{os.path.basename(path):>22} {current.step:8d} "
              f"{result['score']*100:13.1f}% {result['elo_diff']:+7.0f} {elo:8.0f}")
        previous = current

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\ntotal improvement over the run: {elo:+.0f} Elo")
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
