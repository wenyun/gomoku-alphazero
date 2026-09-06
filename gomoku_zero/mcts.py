"""Monte-Carlo tree search, batched over many games at once.

Why batched?  A single MCTS simulation needs exactly one network evaluation, and
a batch of one is a terrible way to use a GPU.  So instead of searching one game
to completion, we keep N independent games in flight and advance all of their
searches in lock-step: every "round" we walk each of the N trees down to a leaf,
evaluate all N leaves in a *single* network call, then expand and back up each
one.  That turns the GPU batch size from 1 into N (40 by default) at no cost to
search quality, because the N trees are completely independent.

Why flat arrays instead of node objects?  Selection has to compare the PUCT
score of every candidate move at a node.  With one numpy row per node
(`N[node]`, `W[node]`, `P[node]`) that comparison is a handful of vectorised
operations; with Python objects it is a Python loop, which is roughly an order of
magnitude slower.  The whole tree is a few preallocated 2-D arrays indexed by
[node, action].

The search itself is textbook AlphaZero:

  select   descend from the root, always taking argmax of
             Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b) + 1) / (1 + N(s,a))
  expand   the first edge with no child yet becomes a new leaf node; the network
           supplies its move priors P and its value estimate v
  backup   add v to every edge on the path, flipping the sign at each ply
           because the two players want opposite things
"""

from __future__ import annotations

import math

import numpy as np

from .board import Board
from .config import NUM_ACTIONS, MCTSConfig
from .features import encode_boards
from .tactics import interior_analysis, root_analysis


class GameTree:
    """One search tree, stored as preallocated [node, action] arrays."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.child = np.full((capacity, NUM_ACTIONS), -1, dtype=np.int32)
        self.N = np.zeros((capacity, NUM_ACTIONS), dtype=np.int32)
        self.W = np.zeros((capacity, NUM_ACTIONS), dtype=np.float32)
        self.P = np.zeros((capacity, NUM_ACTIONS), dtype=np.float32)
        self.legal = np.zeros((capacity, NUM_ACTIONS), dtype=np.bool_)
        self.visits = np.zeros(capacity, dtype=np.int32)          # sum_a N[node, a]
        self.terminal = np.zeros(capacity, dtype=np.bool_)
        self.terminal_value = np.zeros(capacity, dtype=np.float32)
        self.size = 0

    def reset(self) -> None:
        """Clear only the rows that were actually used by the previous search."""
        n = self.size
        if n:
            self.child[:n] = -1
            self.N[:n] = 0
            self.W[:n] = 0.0
            self.P[:n] = 0.0
            self.legal[:n] = False
            self.visits[:n] = 0
            self.terminal[:n] = False
            self.terminal_value[:n] = 0.0
        self.size = 0

    def new_node(self) -> int:
        idx = self.size
        self.size += 1
        return idx


class BatchedMCTS:
    """Runs `num_games` independent searches with shared network batches."""

    def __init__(self, evaluator, cfg: MCTSConfig, num_games: int,
                 rng: np.random.Generator | None = None) -> None:
        self.evaluator = evaluator
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        self.num_games = num_games
        # A search creates at most one node per simulation, plus the root.
        self.capacity = 0
        self.trees: list[GameTree] = []
        self._ensure_capacity(cfg.simulations)

    # ------------------------------------------------------------------ public
    def search(self, boards: list[Board], add_noise: bool = True,
               simulations: int | None = None) -> np.ndarray:
        """Search every board in `boards`; returns root visit counts (B, 225).

        All boards must be non-terminal.  The i-th returned row is the visit
        count of each move at the root of game i, which is the improved policy
        that both self-play and the final move choice are based on.
        """
        n_sims = self.cfg.simulations if simulations is None else simulations
        self._ensure_capacity(n_sims)
        trees = self.trees[:len(boards)]
        for tree in trees:
            tree.reset()

        self._expand_roots(trees, boards, add_noise)
        for _ in range(n_sims):
            self._simulate_round(trees, boards)

        return np.stack([tree.N[0].copy() for tree in trees])

    def root_value(self, index: int) -> float:
        """Search's evaluation of the root of game `index`, mover's perspective."""
        tree = self.trees[index]
        total = tree.visits[0]
        return float(tree.W[0].sum() / total) if total else 0.0

    def root_q(self, index: int) -> np.ndarray:
        """Per-move Q values at the root (0 where a move was never visited)."""
        tree = self.trees[index]
        return tree.W[0] / np.maximum(tree.N[0], 1)

    # ----------------------------------------------------------------- private
    def _ensure_capacity(self, simulations: int) -> None:
        """(Re)allocate the trees if they cannot hold `simulations` nodes.

        The tree arrays are preallocated, so the simulation count has to be known
        up front -- but callers are allowed to change it between searches (the
        play tools let you pick the strength at run time), so check every search
        rather than trusting the count we were constructed with.
        """
        needed = simulations + 2
        if needed <= self.capacity:
            return
        self.capacity = needed
        self.trees = [GameTree(needed) for _ in range(self.num_games)]

    def _expand_roots(self, trees, boards, add_noise: bool) -> None:
        masks = np.stack([root_analysis(b) for b in boards])
        priors, _ = self.evaluator.evaluate(encode_boards(boards), masks)

        for i, tree in enumerate(trees):
            root = tree.new_node()
            tree.legal[root] = masks[i]
            p = priors[i]
            if add_noise:
                p = self._add_dirichlet(p, masks[i])
            tree.P[root] = p

    def _add_dirichlet(self, priors: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Mix Dirichlet noise into the root priors so self-play keeps exploring.

        Without this, self-play would play (almost) the same game forever and the
        network would never see anything new.
        """
        legal = np.flatnonzero(mask)
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(legal))
        eps = self.cfg.dirichlet_epsilon
        out = priors.copy()
        out[legal] = (1.0 - eps) * out[legal] + eps * noise
        return out

    def _simulate_round(self, trees, boards) -> None:
        """One simulation in every tree, sharing a single network call."""
        pending_paths = []
        pending_trees = []
        pending_boards = []

        pending_masks = []

        for tree, root_board in zip(trees, boards):
            path, leaf, existing = self._descend(tree, root_board)

            if existing >= 0:
                # We walked into a position we have already proven, so its value
                # is known exactly and there is nothing left to expand.
                self._backup(tree, path, float(tree.terminal_value[existing]))
                continue

            if leaf.is_over:
                proven = leaf.terminal_value()
                mask = None
            else:
                # Cheap exact check for "somebody makes five right now", which
                # either settles the node outright or forces a single reply.
                mask, proven = interior_analysis(leaf)

            if proven is not None:
                self._add_proven_node(tree, path, proven)
            else:
                pending_paths.append(path)
                pending_trees.append(tree)
                pending_boards.append(leaf)
                pending_masks.append(mask)

        if not pending_boards:
            return

        masks = np.stack(pending_masks)
        priors, values = self.evaluator.evaluate(encode_boards(pending_boards), masks)

        for k, tree in enumerate(pending_trees):
            path = pending_paths[k]
            node = tree.new_node()
            tree.legal[node] = masks[k]
            tree.P[node] = priors[k]
            parent, action = path[-1]
            tree.child[parent, action] = node
            self._backup(tree, path, float(values[k]))

    def _add_proven_node(self, tree: GameTree, path, value: float) -> None:
        """Attach a leaf whose value is known exactly and back that value up."""
        node = tree.new_node()
        tree.terminal[node] = True
        tree.terminal_value[node] = value
        parent, action = path[-1]
        tree.child[parent, action] = node
        self._backup(tree, path, value)

    def _descend(self, tree: GameTree, root_board: Board):
        """Walk from the root to a leaf.

        Returns (path, board_at_leaf, existing_terminal_node).  `path` is the
        list of (node, action) edges taken.  `existing_terminal_node` is >= 0
        only when the walk ended on a terminal node we had already created, in
        which case no expansion is needed.
        """
        board = root_board.copy()
        node = 0
        path = []
        while True:
            action = self._select(tree, node)
            path.append((node, action))
            board.play(action)
            child = tree.child[node, action]
            if child < 0:
                return path, board, -1
            node = int(child)
            if tree.terminal[node]:
                return path, board, node

    def _select(self, tree: GameTree, node: int) -> int:
        """PUCT: pick the child that best trades off value against uncertainty."""
        n = tree.N[node]
        q = tree.W[node] / np.maximum(n, 1)      # unvisited edges score Q = 0
        u = (self.cfg.c_puct * math.sqrt(tree.visits[node] + 1)) * tree.P[node] / (1 + n)
        scores = q + u
        scores[~tree.legal[node]] = -np.inf
        return int(scores.argmax())

    @staticmethod
    def _backup(tree: GameTree, path, leaf_value: float) -> None:
        """Propagate `leaf_value` up the path, flipping sign at every ply.

        `leaf_value` is from the point of view of the player to move *at the
        leaf*.  The parent of the leaf is the opponent, so it sees -leaf_value,
        the grandparent sees +leaf_value, and so on.
        """
        value = -leaf_value
        for node, action in reversed(path):
            tree.N[node, action] += 1
            tree.W[node, action] += value
            tree.visits[node] += 1
            value = -value


def sample_move(visits: np.ndarray, temperature: float,
                rng: np.random.Generator) -> int:
    """Turn root visit counts into an actual move.

    temperature 0 plays the most-visited move; higher temperatures sample
    proportionally to visits**(1/T), which is how self-play gets variety.
    """
    if temperature <= 1e-6:
        return int(visits.argmax())
    weights = visits.astype(np.float64) ** (1.0 / temperature)
    total = weights.sum()
    if total <= 0:
        return int(visits.argmax())
    return int(rng.choice(len(weights), p=weights / total))
