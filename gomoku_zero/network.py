"""The policy/value network and the batched evaluator that MCTS talks to.

The architecture is the standard AlphaZero one, scaled down to the size of the
problem: a convolutional stem, a stack of residual blocks, then two small heads.

    policy head -> 225 logits, "which move should I play here?"
    value  head -> one number in [-1, 1], "how good is this position for the
                   player who is about to move?"

Both heads share the trunk, which is what makes the network cheap: one forward
pass gives MCTS both the move priors it needs to guide the search and the
position evaluation it needs at the leaves.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BOARD_SIZE, NUM_ACTIONS, NUM_PLANES, NetConfig


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: NetConfig = NetConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.channels

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(c) for _ in range(cfg.blocks)])

        # Policy head: two 1x1 convolutions down to a single channel, so the logit
        # for a cell is read straight off that cell's own feature vector.  This
        # keeps the head translation-equivariant -- a threat shape means the same
        # thing wherever it sits on the board -- and avoids a huge fully connected
        # layer.  Nothing global is lost: after `blocks` residual blocks the
        # receptive field of a single cell already covers the whole board.
        h = cfg.head_channels
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU(inplace=True),
            nn.Conv2d(h, 1, 1),
        )

        # Value head: squeeze to a single channel *first*, so the fully connected
        # layer sees 225 inputs instead of 32*225.  That narrow fan-in is the
        # important part.  A wide layer feeding a bounded tanh will happily push
        # its pre-activation to +-15 in the first hundred steps, after which tanh
        # saturates, its gradient vanishes, and the value head never recovers.
        self.value_head = nn.Sequential(
            nn.Conv2d(c, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        """x: (B, NUM_PLANES, 15, 15) -> (policy logits (B, 225), value (B,))."""
        x = self.trunk(self.stem(x))
        policy = self.policy_head(x).flatten(1)
        value = self.value_head(x).squeeze(-1)
        return policy, value


class Evaluator:
    """Wraps a network for inference: planes + legality masks -> priors, values.

    MCTS calls this once per simulation with one leaf position per game it is
    searching, so the batch size is the number of concurrent games.  Masking the
    illegal moves *before* the softmax (rather than zeroing afterwards) keeps the
    priors a proper distribution over the moves the search can actually take.

    CUDA graphs
    -----------
    This network is tiny but deep in *kernel count*: ~70 CUDA kernels per
    forward pass, each taking microseconds of GPU time but also microseconds of
    CPU time to launch.  At self-play batch sizes the GPU finishes long before
    the CPU is done launching, so self-play ends up limited by launch overhead
    rather than by arithmetic.

    Passing `graph_batch` records the whole forward pass into a CUDA graph once,
    after which each evaluation is a single launch that replays all 70 kernels.
    The requirements this imposes are why it needs a fixed batch size: a graph
    replays exactly the shapes it captured, so we always run the full
    `graph_batch` and simply ignore the padding rows.  That is safe because the
    network is in eval mode -- BatchNorm uses its stored running statistics, so
    row i of the output depends only on row i of the input.
    """

    def __init__(self, net: PolicyValueNet, device: str = "cuda",
                 graph_batch: int | None = None) -> None:
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.graph = None
        if graph_batch is not None and self.device.type == "cuda":
            self._capture(graph_batch)

    # ------------------------------------------------------------------ capture
    def _capture(self, batch: int) -> None:
        # Capture must run with `self.device` as the *current* device: a stream
        # capture on one device cannot enqueue work onto another.
        with torch.cuda.device(self.device):
            self._capture_on_device(batch)

    def _capture_on_device(self, batch: int) -> None:
        self.graph_batch = batch
        self.static_input = torch.zeros(
            batch, NUM_PLANES, BOARD_SIZE, BOARD_SIZE, device=self.device)
        # Staging buffer in pinned memory so the host->device copy can use DMA.
        self.host_input = torch.zeros(
            batch, NUM_PLANES, BOARD_SIZE, BOARD_SIZE, pin_memory=True)

        # Warm up on a side stream first: cuDNN picks its algorithms and the
        # allocator reserves its blocks here rather than inside the capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    self.net(self.static_input)
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.static_logits, self.static_value = self.net(self.static_input)

    # --------------------------------------------------------------- inference
    @torch.inference_mode()
    def evaluate(self, planes: np.ndarray, masks: np.ndarray):
        n = planes.shape[0]
        if self.graph is not None and n <= self.graph_batch:
            logits, value = self._forward_graph(planes, n)
        else:
            logits, value = self._forward_eager(planes)

        m = torch.from_numpy(masks).to(self.device, non_blocking=True)
        priors = torch.softmax(logits.float().masked_fill(~m, -1e9), dim=1)

        # Bring the priors and the values back in a *single* transfer.  Each
        # device->host copy costs a full synchronisation (a couple of hundred
        # microseconds), which at these batch sizes is comparable to the entire
        # forward pass, so doing it twice would be a real waste.
        packed = torch.cat([priors, value.float().unsqueeze(1)], dim=1).cpu().numpy()
        return packed[:, :NUM_ACTIONS], packed[:, NUM_ACTIONS]

    def _forward_graph(self, planes: np.ndarray, n: int):
        self.host_input[:n].copy_(torch.from_numpy(planes))
        self.static_input[:n].copy_(self.host_input[:n], non_blocking=True)
        self.graph.replay()
        return self.static_logits[:n], self.static_value[:n]

    def _forward_eager(self, planes: np.ndarray):
        x = torch.from_numpy(planes).to(self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            return self.net(x)

    def load_state_dict(self, state_dict) -> None:
        """Load new weights in place, which keeps any captured graph valid.

        `load_state_dict` copies into the existing parameter tensors rather than
        rebinding them, so the graph -- which captured those exact addresses --
        picks up the new weights on its next replay.
        """
        self.net.load_state_dict(state_dict)
        self.net.eval()


def build_net(cfg: NetConfig = NetConfig()) -> PolicyValueNet:
    return PolicyValueNet(cfg)
