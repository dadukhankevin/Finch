"""Decoder architectures for the universal solver.

The one modality-specific element the universal algorithm allows: the
decoder network's shape may match its output (convolutions for images,
1-D convolutions for signals, dense as the fallback), exactly as the
fitness function already does. Evolution's operators never change.

Architectures are plain callables `builder(latent, output_shape) ->
nn.Module` whose module maps a (B, latent) tensor to (B, prod(output_shape))
logits (pre-sigmoid). Register your own with `register_architecture`.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

Builder = Callable[[int, tuple], nn.Module]

_REGISTRY: dict[str, Builder] = {}


def register_architecture(name: str, builder: Builder) -> None:
    """Make `name` usable as the `architecture` argument of `solve`."""
    _REGISTRY[name] = builder


def build_mlp(latent: int, output_shape: tuple, hidden: int = 64) -> nn.Module:
    dim = int(math.prod(output_shape))
    return nn.Sequential(
        nn.Linear(latent, hidden), nn.LeakyReLU(),
        nn.Linear(hidden, dim),
    )


class _Conv2d(nn.Module):
    """latent -> small feature map -> upsample+conv stages -> H x W logits."""

    def __init__(self, latent: int, output_shape: tuple, channels: int = 16):
        super().__init__()
        height, width = output_shape
        doublings = max(1, min(
            3,
            int(math.log2(max(height, 4) // 4)),
            int(math.log2(max(width, 4) // 4)),
        ))
        self.base = (max(1, height >> doublings), max(1, width >> doublings))
        self.channels = channels
        self.fc = nn.Linear(latent, channels * self.base[0] * self.base[1])
        blocks: list[nn.Module] = []
        for _ in range(doublings):
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.LeakyReLU(),
            ]
        blocks += [
            nn.Upsample(size=(height, width), mode="nearest"),
            nn.Conv2d(channels, 1, 3, padding=1),
        ]
        self.convs = nn.Sequential(*blocks)

    def forward(self, z):
        x = self.fc(z).view(-1, self.channels, *self.base)
        return self.convs(x).flatten(1)


class _Conv1d(nn.Module):
    """latent -> short feature track -> upsample+conv stages -> dim logits."""

    def __init__(self, latent: int, output_shape: tuple, channels: int = 16):
        super().__init__()
        (dim,) = output_shape
        doublings = max(1, min(4, int(math.log2(max(dim, 16) // 16))))
        self.base = max(2, dim >> doublings)
        self.channels = channels
        self.fc = nn.Linear(latent, channels * self.base)
        blocks: list[nn.Module] = []
        for _ in range(doublings):
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(channels, channels, 5, padding=2),
                nn.LeakyReLU(),
            ]
        blocks += [
            nn.Upsample(size=dim, mode="nearest"),
            nn.Conv1d(channels, 1, 5, padding=2),
        ]
        self.convs = nn.Sequential(*blocks)

    def forward(self, z):
        x = self.fc(z).view(-1, self.channels, self.base)
        return self.convs(x).flatten(1)


class _Recurrent(nn.Module):
    """Genome -> initial recurrent state; learned step inputs drive one
    recurrent step per output element, each emitting one logit.

    The sequence prior: consecutive outputs are computed from consecutive
    hidden states, so the network is biased toward outputs whose neighbors
    are related — the 1-D analogue of what convolution gives images.
    """

    def __init__(self, latent: int, output_shape: tuple, cell: str = "gru",
                 hidden: int = 32, step_dim: int = 16):
        super().__init__()
        dim = int(math.prod(output_shape))
        self.cell = cell
        state_dim = hidden * (2 if cell == "lstm" else 1)
        self.fc = nn.Linear(latent, state_dim)
        self.step_inputs = nn.Parameter(torch.randn(1, dim, step_dim) * 0.5)
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(step_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, z):
        state = torch.tanh(self.fc(z))
        steps = self.step_inputs.expand(z.shape[0], -1, -1)
        if self.cell == "lstm":
            h, c = state.chunk(2, dim=1)
            out, _ = self.rnn(steps, (h[None].contiguous(),
                                      c[None].contiguous()))
        else:
            out, _ = self.rnn(steps, state[None].contiguous())
        return self.head(out).flatten(1)


class _Transformer(nn.Module):
    """Genome -> a context vector added to learned positional tokens;
    self-attention mixes the tokens; a shared head reads one logit per
    position. The prior: every output element is computed relative to all
    the others, with no locality assumption at all."""

    def __init__(self, latent: int, output_shape: tuple, width: int = 32,
                 heads: int = 4, depth: int = 2):
        super().__init__()
        dim = int(math.prod(output_shape))
        self.positions = nn.Parameter(torch.randn(1, dim, width) * 0.5)
        self.fc = nn.Linear(latent, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=2 * width,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(width, 1)

    def forward(self, z):
        tokens = self.positions + self.fc(z)[:, None, :]
        return self.head(self.encoder(tokens)).flatten(1)


class _ConvImage(nn.Module):
    """Multi-channel images (H, W, C) or (C, H, W): dense latent -> small
    feature map -> upsample+conv stages -> one conv to the color channels.
    The round-28 conv champion's shape, generalized to any square size and
    either channel convention (output element order always matches the
    declared output_shape, so fitness functions can flatten either way)."""

    def __init__(self, latent: int, output_shape: tuple, channels: int = 16):
        super().__init__()
        if len(output_shape) != 3:
            raise ValueError("_ConvImage needs a 3-d output shape")
        self.channels_last = output_shape[-1] <= 4
        if self.channels_last:
            height, width, colors = output_shape
        else:
            colors, height, width = output_shape
        if height != width:
            raise ValueError("_ConvImage needs square images")
        doublings = 0
        base = height
        while base % 2 == 0 and base > 8:
            base //= 2
            doublings += 1
        self.base, self.width = base, channels
        self.fc = nn.Linear(latent, channels * base * base)
        blocks: list[nn.Module] = []
        for _ in range(doublings):
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(channels, channels, 3, padding=1),
                       nn.LeakyReLU()]
        blocks += [nn.Conv2d(channels, colors, 3, padding=1)]
        self.convs = nn.Sequential(*blocks)

    def forward(self, z):
        grid = self.fc(z).view(-1, self.width, self.base, self.base)
        out = self.convs(grid)
        if self.channels_last:
            out = out.permute(0, 2, 3, 1)
        return out.flatten(1)


register_architecture("mlp", build_mlp)
register_architecture("conv2d", _Conv2d)
register_architecture("conv_image", _ConvImage)
register_architecture("conv1d", _Conv1d)
register_architecture("gru", lambda latent, shape: _Recurrent(latent, shape, "gru"))
register_architecture("lstm", lambda latent, shape: _Recurrent(latent, shape, "lstm"))
register_architecture("transformer", _Transformer)


def resolve(architecture, latent: int, output_shape: tuple) -> Callable[[], nn.Module]:
    """Turn `architecture` (name, "auto", or builder) into a zero-arg factory."""
    output_shape = tuple(int(s) for s in output_shape)
    if callable(architecture):
        return lambda: architecture(latent, output_shape)
    if architecture == "auto":
        if len(output_shape) == 2 and min(output_shape) >= 8:
            architecture = "conv2d"
        elif (len(output_shape) == 3
              and (output_shape[-1] <= 4 or output_shape[0] <= 4)
              and max(output_shape) >= 8
              and sorted(output_shape)[1] == sorted(output_shape)[2]):
            architecture = "conv_image"
        elif len(output_shape) == 1 and output_shape[0] >= 32:
            architecture = "conv1d"
        else:
            architecture = "mlp"
    if architecture not in _REGISTRY:
        raise ValueError(
            f"unknown architecture {architecture!r}; "
            f"registered: {sorted(_REGISTRY)}"
        )
    builder = _REGISTRY[architecture]
    return lambda: builder(latent, output_shape)
