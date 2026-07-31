"""Per-individual SPARSE WEIGHT PATCHES — Daniel's alternative to low-rank
adaptation (2026-07-22).

The low-rank scheme has a structural ceiling: every individual's bending,
and therefore everything the shared backbone ever absorbs through folding,
lives in the span of a few frozen random directions drawn on the first
epoch. The backbone moves, but only along those axes, forever.

This module replaces that modifier with a sparse patch. An individual is
still tiny — no per-individual weight matrices, the invariant holds — but
its modifier is now K coordinates of the decoder's flat weight vector plus
K values to add there:

    weights(individual) = shared_base ; weights[sites(seed)] += values

`sites` is a deterministic function of the individual's integer seed, so a
lineage's patch LOCATIONS cost one integer to store and inherit, while the
patch VALUES are the evolved latents. Folding stays exact and becomes
trivial — add a donor's values at its sites into the base — and, unlike
the low-rank path, successive folds can reach ANY weight, so the backbone
is no longer confined to a fixed subspace.

Round 37 warned that choosing WHERE to mutate carries no inheritable
information (weight space has no metric). That result compared sparse
against fully diffuse mutation of all weights; here the incumbent is a
frozen low-rank subspace instead, so the comparison is open — and the
fresh-sites rate is the knob that settles whether locations matter at all.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, vmap

from .architectures import resolve


class SparsePatchDecoder:
    """One shared backbone; each individual adds a sparse patch to it."""

    def __init__(self, architecture, genes: int, output_shape: tuple,
                 patch_size: int, device: str, value_scale: float = 0.25):
        self.device = device
        self.net = resolve(architecture, genes, output_shape)().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self._names = [n for n, _ in self.net.named_parameters()]
        self._shapes = [tuple(p.shape) for _, p in self.net.named_parameters()]
        numels = [int(np.prod(s)) if s else 1 for s in self._shapes]
        self._offsets = np.concatenate([[0], np.cumsum(numels)]).astype(int)
        self.n_params = int(self._offsets[-1])
        self.patch_size = int(min(patch_size, self.n_params))
        self.base = nn.utils.parameters_to_vector(
            self.net.parameters()).detach().clone()
        # Patch values live in units of the backbone's own weight spread, so
        # a standard-normal latent is a sensible-sized edit at birth.
        self.value_scale = float(self.base.std().item()) * value_scale
        self._sites: dict[int, np.ndarray] = {}

        def _forward(params: dict, z: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(functional_call(self.net, params, (z[None],)))[0]

        self._vforward = vmap(_forward)

    # ---------------------------------------------------------------- sites

    def sites_for(self, seed: int) -> np.ndarray:
        """The coordinates this lineage patches — a frozen random function of
        its seed, so locations cost one integer to store and inherit."""
        key = int(seed)
        if key not in self._sites:
            self._sites[key] = np.random.default_rng(key).integers(
                0, self.n_params, self.patch_size)
        return self._sites[key]

    # --------------------------------------------------------------- decode

    def _params_from_flat(self, flat: torch.Tensor) -> dict:
        return {
            name: flat[:, self._offsets[i]:self._offsets[i + 1]]
            .reshape(len(flat), *self._shapes[i])
            for i, name in enumerate(self._names)}

    def decode_seeded(self, z: np.ndarray, values: np.ndarray,
                      seeds: np.ndarray) -> torch.Tensor:
        """Whole population in one vmapped call: patch, then decode."""
        n = len(z)
        flat = self.base[None, :].repeat(n, 1)
        idx = torch.as_tensor(
            np.stack([self.sites_for(s) for s in seeds]), device=self.device)
        vals = torch.as_tensor(
            np.ascontiguousarray(values[:, :self.patch_size]
                                 .astype(np.float32)), device=self.device)
        flat = flat.scatter_add(1, idx, vals * self.value_scale)
        genes = torch.as_tensor(
            np.ascontiguousarray(z.astype(np.float32)), device=self.device)
        with torch.no_grad():
            out = self._vforward(self._params_from_flat(flat), genes)
        return out.reshape(n, -1)

    def decode(self, z: np.ndarray, values: np.ndarray) -> torch.Tensor:
        return self.decode_seeded(z, values, np.zeros(len(z), dtype=np.int64))

    # ----------------------------------------------------------------- fold

    def absorb_seeded(self, values: np.ndarray, seed: int) -> None:
        """Fold: add this patch into the shared base, exactly. Successive
        folds are unconstrained — any weight can eventually move."""
        idx = torch.as_tensor(self.sites_for(seed), device=self.device)
        vals = torch.as_tensor(
            np.asarray(values[:self.patch_size], dtype=np.float32),
            device=self.device)
        self.base = self.base.scatter_add(0, idx, vals * self.value_scale)
        nn.utils.vector_to_parameters(self.base, self.net.parameters())

    def absorb(self, values: np.ndarray) -> None:
        self.absorb_seeded(values, 0)

    # ---------------------------------------------------------------- state

    def training_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid BASE output (no patch), with gradients — the
        distillation surface: the base learns to reach unaided what a
        champion's patch reached. This is exactly the experimental
        DistillGenerator's base_decode, in the library."""
        for p in self.net.parameters():
            p.requires_grad_(True)
        return self.net(z)

    def sync_base(self) -> None:
        """After gradient steps on `net`, the flat base vector — which is
        what decode/absorb actually read — must be rebuilt from it."""
        self.base = nn.utils.parameters_to_vector(
            self.net.parameters()).detach().clone()

    def get_params(self) -> np.ndarray:
        return self.base.detach().cpu().numpy().astype(np.float32)

    def set_params(self, flat: np.ndarray) -> None:
        self.base = torch.as_tensor(
            np.asarray(flat, dtype=np.float32), device=self.device).clone()
        nn.utils.vector_to_parameters(self.base, self.net.parameters())


def build_sparse_decoder(architecture, genes, output_shape, patch_size,
                         device):
    return SparsePatchDecoder(architecture, genes, output_shape, patch_size,
                              device)
