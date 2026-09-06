"""The learner: turns self-play games into better network weights.

The loss is AlphaZero's, and it is worth being explicit about what each term is
asking the network to do:

    policy loss   cross-entropy between the network's move distribution and the
                  *search's* visit distribution.  The search is a stronger player
                  than the raw network, because it looked ahead; this term
                  distils that improvement back into the network.

    value loss    squared error between the network's evaluation and the actual
                  game result.  This is what eventually teaches the network which
                  positions are winning, which is what makes the *next* round of
                  search better.

Those two terms feeding each other is the whole flywheel: better value -> better
search -> better policy targets -> better network -> better value.

The only extra trick here is symmetry augmentation.  Gomoku is invariant under
the 8 symmetries of the square, so every position can be presented in 8 ways.
Applying that is a free 8x multiplication of the data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import BOARD_SIZE, NUM_ACTIONS, Config
from .features import encode_batch
from .network import build_net


class Learner:
    def __init__(self, cfg: Config, device: str = "cuda:0") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.net = build_net(cfg.net).to(self.device)
        self.net = self.net.to(memory_format=torch.channels_last)
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay)
        self.step = 0

    # ------------------------------------------------------------------ helpers
    def learning_rate(self) -> float:
        """Linear warmup, then a step decay of 10x at each milestone.

        The warmup is not cosmetic: the very first updates are the ones most
        likely to shove the value head's pre-tanh activations into saturation.
        """
        lr = self.cfg.train.lr
        warmup = self.cfg.train.warmup_steps
        if self.step < warmup:
            lr *= (self.step + 1) / warmup
        for milestone in self.cfg.train.lr_milestones:
            if self.step >= milestone:
                lr *= 0.1
        return lr

    @staticmethod
    def _augment(planes: torch.Tensor, policy: torch.Tensor):
        """Apply the 8 symmetries of the square, one per eighth of the batch.

        `policy` comes in as (B, 225) and is reshaped to a 15x15 image so that it
        can be rotated and flipped exactly like the input planes.  Doing one
        symmetry per chunk (rather than one per sample) keeps this fully
        vectorised while still giving every symmetry equal weight.

        The random offset matters: without it, position `i` of the batch would
        always receive symmetry `i // (B/8)`.  Batches are randomly sampled so it
        would come out even in the end, but it makes the augmentation silently
        useless for any fixed batch -- each sample would only ever be shown under
        a single rotation.
        """
        offset = int(torch.randint(8, ()))
        policy_image = policy.view(-1, 1, BOARD_SIZE, BOARD_SIZE)
        out_planes, out_policy = [], []
        for chunk, (p, q) in enumerate(zip(planes.chunk(8), policy_image.chunk(8))):
            k = (chunk + offset) % 8
            p = torch.rot90(p, k % 4, dims=(2, 3))
            q = torch.rot90(q, k % 4, dims=(2, 3))
            if k >= 4:
                p = torch.flip(p, dims=(3,))
                q = torch.flip(q, dims=(3,))
            out_planes.append(p)
            out_policy.append(q)
        planes = torch.cat(out_planes).contiguous(memory_format=torch.channels_last)
        policy = torch.cat(out_policy).reshape(-1, NUM_ACTIONS)
        return planes, policy

    # --------------------------------------------------------------- one update
    def train_step(self, batch) -> dict:
        states, last_move, prev_move, black, policy, value = batch

        planes = encode_batch(states, last_move, prev_move, black)
        planes = torch.from_numpy(planes).to(self.device, non_blocking=True)
        policy_target = torch.from_numpy(policy).to(self.device, non_blocking=True)
        value_target = torch.from_numpy(value).to(self.device, non_blocking=True)
        planes, policy_target = self._augment(planes, policy_target)

        lr = self.learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, value_pred = self.net(planes)
            # Cross-entropy against a soft target distribution.
            policy_loss = -(policy_target * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
            value_loss = F.mse_loss(value_pred.float(), value_target)
            loss = policy_loss + self.cfg.train.value_loss_weight * value_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optimizer.step()
        self.step += 1

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "grad_norm": float(grad_norm),
            "lr": lr,
        }

    # ------------------------------------------------------------- checkpoints
    def state_dict(self) -> dict:
        return {
            "step": self.step,
            "net": {k: v.cpu() for k, v in self.net.state_dict().items()},
            "net_cfg": vars(self.cfg.net),
        }

    def load_state_dict(self, blob: dict) -> None:
        self.net.load_state_dict(blob["net"])
        self.step = blob.get("step", 0)
