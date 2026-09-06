"""Playing agents built from a trained checkpoint.

`AlphaZeroPlayer` is what the arena and the human-play tools both use.  The only
differences from self-play are the ones you would expect when you stop wanting
exploration and start wanting the strongest move:

  * no Dirichlet noise at the root,
  * temperature 0 by default, so the most-visited move is played,
  * usually more simulations, since we are no longer trying to generate volume.

It searches a *list* of boards, so the arena can play many games in parallel and
still get large network batches.  For interactive play the list is just length 1.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from .board import Board, action_to_coord
from .config import MCTSConfig, NetConfig
from .mcts import BatchedMCTS, sample_move
from .network import Evaluator, build_net


def load_evaluator(checkpoint: str, device: str = "cuda:0",
                   graph_batch: int | None = None):
    """Rebuild the network described by a checkpoint and wrap it for inference."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net_cfg = NetConfig(**blob["net_cfg"])
    net = build_net(net_cfg)
    net.load_state_dict(blob["net"])
    return Evaluator(net, device=device, graph_batch=graph_batch), blob.get("step", 0)


class AlphaZeroPlayer:
    def __init__(self, checkpoint: str, device: str = "cuda:0",
                 simulations: int = 800, batch: int = 1,
                 temperature: float = 0.0, c_puct: float = 2.0,
                 seed: int = 0, label: str | None = None) -> None:
        self.evaluator, self.step = load_evaluator(checkpoint, device, graph_batch=batch)
        self.cfg = replace(MCTSConfig(), simulations=simulations, c_puct=c_puct)
        self.rng = np.random.default_rng(seed)
        self.mcts = BatchedMCTS(self.evaluator, self.cfg, batch, self.rng)
        self.temperature = temperature
        self.simulations = simulations
        self.name = label or f"az-step{self.step}-s{simulations}"

    def select_moves(self, boards: list[Board]) -> list[int]:
        visits = self.mcts.search(boards, add_noise=False,
                                  simulations=self.simulations)
        return [sample_move(visits[i], self.temperature, self.rng)
                for i in range(len(boards))]

    def select_move(self, board: Board) -> int:
        return self.select_moves([board])[0]

    def analyse(self, board: Board, top: int = 5) -> dict:
        """Search one position and report what the engine is thinking.

        Used by the play tools to show the evaluation and the moves considered.
        `win_probability` converts the value in [-1, 1] to a percentage, which is
        easier to read: value +1 means "I am winning", so (v + 1) / 2 is the
        probability that the side to move wins.
        """
        visits = self.mcts.search([board], add_noise=False,
                                  simulations=self.simulations)[0]
        q = self.mcts.root_q(0)
        value = self.mcts.root_value(0)
        order = np.argsort(-visits)[:top]
        return {
            "move": int(visits.argmax()),
            "value": value,
            "win_probability": 0.5 * (value + 1.0),
            "visits": visits,
            "top": [{"move": int(a), "coord": action_to_coord(int(a)),
                     "visits": int(visits[a]),
                     "share": float(visits[a] / max(visits.sum(), 1)),
                     "q": float(q[a])}
                    for a in order if visits[a] > 0],
        }
