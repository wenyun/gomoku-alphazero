"""Watches a training run and measures each snapshot against a fixed anchor.

    .venv/bin/python track_strength.py --run-dir runs/az --anchor runs/az/step_0005000.pt

Measuring progress against a *fixed* opponent is the point.  Losses falling is
weak evidence -- the network is fitting a moving target, so the loss can drift for
reasons that have nothing to do with playing better.  Beating a frozen earlier
version by a growing margin is direct evidence, and because both sides are neural
the matches run entirely on the GPU and stay cheap.

Results are appended to `<run_dir>/strength.jsonl`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

from arena import play_match
from gomoku_zero.player import AlphaZeroPlayer


def snapshots(run_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(run_dir, "step_*.pt")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/az")
    ap.add_argument("--anchor", default=None,
                    help="checkpoint to measure against (default: earliest snapshot)")
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None,
                    help="output jsonl (default: <run_dir>/strength.jsonl). Use a "
                         "separate file per anchor, since scores against different "
                         "anchors are not comparable.")
    ap.add_argument("--watch", action="store_true", help="keep polling for new snapshots")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.run_dir, "strength.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)["snapshot"] for line in f if line.strip()}

    anchor_path = args.anchor
    while anchor_path is None:
        found = snapshots(args.run_dir)
        if found:
            anchor_path = found[0]
        else:
            print("waiting for the first snapshot...", flush=True)
            time.sleep(60)

    anchor = AlphaZeroPlayer(anchor_path, device=args.device, simulations=args.sims,
                             batch=args.games, temperature=0.2, seed=7,
                             label="anchor")
    print(f"anchor = {os.path.basename(anchor_path)}", flush=True)

    while True:
        pending = [p for p in snapshots(args.run_dir) if os.path.basename(p) not in done]
        for path in pending:
            name = os.path.basename(path)
            try:
                player = AlphaZeroPlayer(path, device=args.device, simulations=args.sims,
                                         batch=args.games, temperature=0.2, seed=8)
            except Exception as exc:                       # partially written file
                print(f"skipping {name}: {exc}", flush=True)
                continue
            result = play_match(player, anchor, args.games, verbose=False)
            row = {"snapshot": name, "step": player.step,
                   "anchor": os.path.basename(anchor_path),
                   "games": args.games, "sims": args.sims, **result}
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            done.add(name)
            print(f"{name}: score {result['score']*100:5.1f}% vs anchor  "
                  f"elo {result['elo_diff']:+7.0f}  "
                  f"len {result['mean_length']:.1f}  ({result['seconds']:.0f}s)",
                  flush=True)
            del player

        if not args.watch:
            return
        time.sleep(120)


if __name__ == "__main__":
    main()
