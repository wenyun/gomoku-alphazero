"""Orchestrator for a training run.

    .venv/bin/python train.py --hours 6.5

Process layout
--------------
                    +-------------------+
   GPU 0            |  learner (main)   |  samples the replay buffer, does SGD,
                    |                   |  publishes latest.pt every N steps
                    +---------+---------+
                              ^ finished games (multiprocessing queue)
                              |
   GPU 1..7   [worker] [worker] [worker] ...   each plays many games at once,
                              |                 reloads latest.pt periodically
                    (one process per slot, several per GPU)

The two halves are deliberately decoupled: workers never wait for the learner and
the learner never waits for workers.  They are coupled only by (a) the queue of
finished games and (b) the checkpoint file.  The one thing we *do* control is the
ratio between them -- see `_may_train` -- because letting the learner spin many
times over the same positions is the classic way to make this kind of run
collapse.
"""

from __future__ import annotations

import argparse
import json
import os
import queue as queue_mod
import time
from dataclasses import replace

import numpy as np
import torch
import torch.multiprocessing as mp

from gomoku_zero.config import Config
from gomoku_zero.learner import Learner
from gomoku_zero.network import Evaluator, build_net
from gomoku_zero.replay import ReplayBuffer
from gomoku_zero.selfplay import SelfPlayWorker

LATEST = "latest.pt"


# ---------------------------------------------------------------- self-play side
def worker_main(worker_id: int, gpu: int, cfg: Config,
                game_queue: mp.Queue, stop: mp.Event) -> None:
    """One self-play process: play games forever, ship them to the learner."""
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True

    latest_path = os.path.join(cfg.run_dir, LATEST)
    while not os.path.exists(latest_path) and not stop.is_set():
        time.sleep(1.0)                      # wait for the learner's initial weights
    if stop.is_set():
        return

    net = build_net(cfg.net)
    evaluator = Evaluator(net, device=f"cuda:{gpu}",
                          graph_batch=cfg.selfplay.games_per_worker)
    loaded_mtime = _reload_weights(evaluator, latest_path, 0.0)

    worker = SelfPlayWorker(evaluator, cfg, seed=1000 + worker_id)
    moves = 0
    while not stop.is_set():
        for record in worker.step():
            try:
                game_queue.put(record, timeout=5.0)
            except queue_mod.Full:
                pass                          # learner is behind; drop the game
        moves += 1
        if moves % cfg.selfplay.reload_every_moves == 0:
            loaded_mtime = _reload_weights(evaluator, latest_path, loaded_mtime)


def _reload_weights(evaluator: Evaluator, path: str, loaded_mtime: float) -> float:
    """Pick up new weights if the learner has published any since last time."""
    try:
        mtime = os.path.getmtime(path)
        if mtime <= loaded_mtime:
            return loaded_mtime
        blob = torch.load(path, map_location="cpu", weights_only=False)
        evaluator.load_state_dict(blob["net"])
        return mtime
    except (OSError, EOFError, RuntimeError):
        return loaded_mtime                   # mid-write or transient: try later


# ------------------------------------------------------------------ learner side
class TrainingRun:
    def __init__(self, cfg: Config, hours: float) -> None:
        self.cfg = cfg
        self.deadline = time.time() + hours * 3600
        self.buffer = ReplayBuffer(cfg.train.buffer_capacity)
        self.learner = Learner(cfg, device="cuda:0")
        self.rng = np.random.default_rng(0)
        self.positions_consumed = 0
        self.start = time.time()

        os.makedirs(cfg.run_dir, exist_ok=True)
        self.log_file = open(os.path.join(cfg.run_dir, "train_log.jsonl"), "a")

    # -------------------------------------------------------------- checkpoints
    def publish(self, permanent: bool = False) -> None:
        """Write weights for the workers to pick up (atomically, via rename)."""
        blob = self.learner.state_dict()
        path = os.path.join(self.cfg.run_dir, LATEST)
        tmp = path + ".tmp"
        torch.save(blob, tmp)
        os.replace(tmp, path)
        if permanent:
            torch.save(blob, os.path.join(
                self.cfg.run_dir, f"step_{self.learner.step:07d}.pt"))

    # ---------------------------------------------------------------- main loop
    def drain(self, game_queue: mp.Queue, budget: int = 4096) -> None:
        for _ in range(budget):
            try:
                self.buffer.add_game(game_queue.get_nowait())
            except queue_mod.Empty:
                return

    def _may_train(self) -> bool:
        """Throttle the learner so it does not over-fit freshly generated data.

        `positions_consumed / positions_generated` is the average number of times
        each generated position has been shown to the network.  We hold that
        below `max_reuse_per_position`, which in practice means the learner idles
        early on (when few games exist) and runs flat out later.
        """
        if self.buffer.size < self.cfg.train.min_buffer:
            return False
        allowed = self.cfg.train.max_reuse_per_position * self.buffer.total_positions
        return self.positions_consumed < allowed

    def run(self, game_queue: mp.Queue, pool=None) -> None:
        self.publish()                       # workers wait for this
        print(f"[learner] published initial weights to {self.cfg.run_dir}", flush=True)

        last_report = time.time()
        recent = {"policy_loss": 0.0, "value_loss": 0.0, "n": 0}

        while time.time() < self.deadline:
            self.drain(game_queue)

            if not self._may_train():
                time.sleep(0.05)
                if time.time() - last_report > 60.0:
                    self._report(recent, pool)
                    last_report = time.time()
                continue

            batch = self.buffer.sample(self.cfg.train.batch_size, self.rng)
            stats = self.learner.train_step(batch)
            self.positions_consumed += self.cfg.train.batch_size

            recent["policy_loss"] += stats["policy_loss"]
            recent["value_loss"] += stats["value_loss"]
            recent["n"] += 1

            step = self.learner.step
            if step % self.cfg.train.log_every == 0:
                self._log(step, stats)
            if step % self.cfg.train.checkpoint_every == 0:
                self.publish(permanent=step % self.cfg.train.snapshot_every == 0)
            if time.time() - last_report > 60.0:
                self._report(recent, pool)
                recent = {"policy_loss": 0.0, "value_loss": 0.0, "n": 0}
                last_report = time.time()

        self.publish(permanent=True)
        self._report(recent, pool)
        print("[learner] finished", flush=True)

    # -------------------------------------------------------------- diagnostics
    def _log(self, step: int, stats: dict) -> None:
        elapsed = time.time() - self.start
        row = {
            "step": step,
            "elapsed_s": round(elapsed, 1),
            "games": self.buffer.total_games,
            "positions": self.buffer.total_positions,
            "buffer": self.buffer.size,
            "games_per_s": round(self.buffer.total_games / max(elapsed, 1e-9), 2),
            "reuse": round(self.positions_consumed /
                           max(self.buffer.total_positions, 1), 2),
            # `lr` gets more digits than the losses: by the end of the run it is
            # around 1e-5, which would round to a confusing 0.0 at 4 decimals.
            **{k: round(v, 4) for k, v in stats.items() if k != "lr"},
            "lr": float(f"{stats['lr']:.3e}"),
        }
        self.log_file.write(json.dumps(row) + "\n")
        self.log_file.flush()

    def _report(self, recent: dict, pool=None) -> None:
        restarted = pool.respawn_dead() if pool is not None else 0
        if restarted:
            print(f"[main] restarted {restarted} dead self-play worker(s)", flush=True)
        n = max(recent["n"], 1)
        elapsed = time.time() - self.start
        remaining = max(self.deadline - time.time(), 0) / 60
        print(f"[{elapsed/60:6.1f} min] step {self.learner.step:7d} | "
              f"games {self.buffer.total_games:7d} "
              f"({self.buffer.total_games/max(elapsed,1e-9):5.1f}/s) | "
              f"buffer {self.buffer.size:8d} | "
              f"reuse {self.positions_consumed/max(self.buffer.total_positions,1):4.2f} | "
              f"policy {recent['policy_loss']/n:6.4f} value {recent['value_loss']/n:6.4f} | "
              f"{remaining:5.1f} min left", flush=True)


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.5)
    ap.add_argument("--run-dir", default="runs/az")
    ap.add_argument("--workers-per-gpu", type=int, default=None)
    ap.add_argument("--games-per-worker", type=int, default=None)
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    args = ap.parse_args()

    cfg = Config(run_dir=args.run_dir)
    if args.workers_per_gpu:
        cfg.selfplay = replace(cfg.selfplay, workers_per_gpu=args.workers_per_gpu)
    if args.games_per_worker:
        cfg.selfplay = replace(cfg.selfplay, games_per_worker=args.games_per_worker)
    if args.sims:
        cfg.mcts = replace(cfg.mcts, simulations=args.sims)
    if args.channels:
        cfg.net = replace(cfg.net, channels=args.channels)
    if args.blocks:
        cfg.net = replace(cfg.net, blocks=args.blocks)

    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
        json.dump({"net": vars(cfg.net), "mcts": vars(cfg.mcts),
                   "selfplay": vars(cfg.selfplay), "train": vars(cfg.train)}, f, indent=2)

    ctx = mp.get_context("spawn")
    game_queue = ctx.Queue(maxsize=8192)
    stop = ctx.Event()
    pool = WorkerPool(ctx, cfg, game_queue, stop)
    pool.start_all()
    print(f"[main] started {len(pool.processes)} self-play workers on GPUs "
          f"{list(cfg.selfplay.gpus)}, {cfg.selfplay.games_per_worker} games each, "
          f"{cfg.mcts.simulations} sims/move", flush=True)

    run = TrainingRun(cfg, args.hours)
    try:
        run.run(game_queue, pool)
    finally:
        pool.shutdown()
        print("[main] workers stopped", flush=True)


class WorkerPool:
    """Owns the self-play processes and restarts any that die.

    A six-hour unattended run should not quietly lose a third of its throughput
    because a worker hit a transient CUDA error, so the learner checks on them
    every time it prints a progress line.
    """

    def __init__(self, ctx, cfg: Config, game_queue: mp.Queue, stop: mp.Event) -> None:
        self.ctx = ctx
        self.cfg = cfg
        self.game_queue = game_queue
        self.stop = stop
        gpus = [gpu for gpu in cfg.selfplay.gpus
                for _ in range(cfg.selfplay.workers_per_gpu)]
        self.slots = list(enumerate(gpus))        # [(worker_id, gpu), ...]
        self.processes = {}

    def _spawn(self, worker_id: int, gpu: int):
        p = self.ctx.Process(target=worker_main,
                             args=(worker_id, gpu, self.cfg, self.game_queue, self.stop),
                             daemon=True)
        p.start()
        self.processes[worker_id] = (gpu, p)

    def start_all(self) -> None:
        for worker_id, gpu in self.slots:
            self._spawn(worker_id, gpu)

    def respawn_dead(self) -> int:
        restarted = 0
        for worker_id, (gpu, p) in list(self.processes.items()):
            if not p.is_alive():
                self._spawn(worker_id + 100_000, gpu)   # fresh seed, same GPU
                del self.processes[worker_id]
                restarted += 1
        return restarted

    def shutdown(self) -> None:
        self.stop.set()
        time.sleep(3.0)
        for _, p in self.processes.values():
            p.terminate()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
