"""One shared universal decoder, conditioned per-individual by LoRA gates.

The architecture invariant of this project: there is ALWAYS exactly one
decoder, never one per individual. A single backbone (any resolved
architecture) carries shared low-rank directions at every Linear/Conv layer:

    layer(x) = base(x) + scale * up(coeff * down(x))

The `base`, `down` and `up` weights are the ONE shared decoder (trained only at
consolidation). Each individual evolves only its network input `z` (the genes)
and a coefficient vector that gates the shared directions — the same coefficient
vector at every layer. Zero coefficients reproduce the backbone exactly, so
consolidation can fold discoveries into the backbone and shrink coefficients.

This is the architecture-agnostic generalization of the image-only
``ConditionalLoRAConvRGB`` in the benchmarks.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .architectures import resolve


class _LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, scale: float):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        self.scale = scale
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        self.coeff: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.coeff is not None:
            out = out + self.scale * self.up(self.down(x) * self.coeff)
        return out


class _LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, scale: float):
        super().__init__()
        self.base = base
        self.down = nn.Conv2d(base.in_channels, rank, 3, padding=1, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        self.scale = scale
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        self.coeff: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.coeff is not None:
            residual = self.up(self.down(x) * self.coeff[:, :, None, None])
            out = out + self.scale * residual
        return out


class ConditionalLoRADecoder:
    """One shared backbone + shared LoRA directions gated per-individual."""

    def __init__(self, architecture, latent: int, output_shape, rank: int,
                 device: str):
        self.device = device
        self.rank = rank
        self.latent = latent
        backbone = resolve(architecture, latent, output_shape)()
        self._lora: list[nn.Module] = []
        self._wrap(backbone, rank)
        self.net = backbone.to(device)
        self.n_params = sum(p.numel() for p in self.net.parameters())

    def _wrap(self, module: nn.Module, rank: int) -> None:
        scale = rank ** -0.5
        for name, child in list(module._modules.items()):
            if isinstance(child, nn.Linear):
                w = _LoRALinear(child, rank, scale)
                module._modules[name] = w
                self._lora.append(w)
            elif isinstance(child, nn.Conv2d):
                w = _LoRAConv2d(child, rank, scale)
                module._modules[name] = w
                self._lora.append(w)
            else:
                self._wrap(child, rank)

    def _set_coeff(self, coeff: torch.Tensor | None) -> None:
        for layer in self._lora:
            layer.coeff = coeff

    def decode(self, z: np.ndarray, coeff: np.ndarray) -> torch.Tensor:
        """Phenotypes (B, prod(shape)) in [0, 1] for a batch of genes."""
        zt = torch.as_tensor(
            np.ascontiguousarray(z.astype(np.float32)), device=self.device)
        ct = torch.as_tensor(
            np.ascontiguousarray(coeff.astype(np.float32)), device=self.device)
        self._set_coeff(ct)
        with torch.no_grad():
            out = torch.sigmoid(self.net(zt))
        self._set_coeff(None)
        return out.reshape(len(z), -1)

    def training_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid output of the BASE decoder (no per-individual
        bending), with gradients — the distillation target surface."""
        self._set_coeff(None)
        return self.net(z)

    def get_params(self) -> np.ndarray:
        return nn.utils.parameters_to_vector(
            self.net.parameters()).detach().cpu().numpy().astype(np.float32)

    def set_params(self, flat: np.ndarray) -> None:
        nn.utils.vector_to_parameters(
            torch.as_tensor(np.asarray(flat, dtype=np.float32),
                            device=self.device),
            self.net.parameters())

    def _direction_modules(self):
        mods = []
        for layer in self._lora:
            mods.extend([layer.down, layer.up])
        return mods

    def direction_vector(self) -> np.ndarray:
        """The shared low-rank directions — the vocabulary of bendings the
        latents select from — as one flat vector, so evolution can treat
        the vocabulary itself as a genome."""
        return np.concatenate([
            m.weight.detach().cpu().numpy().ravel()
            for m in self._direction_modules()]).astype(np.float32)

    def set_direction_vector(self, flat: np.ndarray) -> None:
        offset = 0
        with torch.no_grad():
            for m in self._direction_modules():
                n = m.weight.numel()
                m.weight.copy_(torch.as_tensor(
                    flat[offset:offset + n], device=self.device
                ).reshape(m.weight.shape))
                offset += n

    def absorb(self, coeff: np.ndarray) -> None:
        """Apply one individual's latents DIRECTLY into the backbone weights —
        exact arithmetic, no training (Daniel's fold semantics, 2026-07-21).
        The layer form base(x) + scale*up(coeff*down(x)) composes in closed
        form: the low-rank bending becomes part of each base weight. After
        this, latents-zero reproduces what the donor produced with its
        latents; everyone else's phenotype shifts by the same bending (the
        environment absorbs the winner's discovery) and must be re-scored.
        """
        c = torch.as_tensor(np.asarray(coeff, dtype=np.float32),
                            device=self.device)
        with torch.no_grad():
            for layer in self._lora:
                if isinstance(layer, _LoRALinear):
                    layer.base.weight += layer.scale * (
                        layer.up.weight @ (c[:, None] * layer.down.weight))
                else:
                    kernel = torch.einsum(
                        "or,rikl->oikl",
                        layer.up.weight[:, :, 0, 0] * c[None, :],
                        layer.down.weight)
                    layer.base.weight += layer.scale * kernel

    def fold(self, z: np.ndarray, phenotypes: np.ndarray,
             stall_window: int, stall_tol: float) -> None:
        """RETIRED default (training-based consolidation), kept for research:
        train so the coefficient-zero output reproduces `phenotypes` from
        their `z`. Costs no evaluations; individuals untouched.
        """
        zt = torch.as_tensor(
            np.ascontiguousarray(z.astype(np.float32)), device=self.device)
        pt = torch.as_tensor(
            np.ascontiguousarray(phenotypes.astype(np.float32)),
            device=self.device)
        self._set_coeff(None)                       # coeff-zero = pure backbone
        opt = torch.optim.Adam(self.net.parameters())
        window: list[float] = []
        while True:
            opt.zero_grad()
            out = torch.sigmoid(self.net(zt)).reshape(len(zt), -1)
            loss = torch.mean((out - pt) ** 2)
            loss.backward()
            opt.step()
            value = float(loss.detach())
            window.append(value)
            if len(window) > stall_window:
                window.pop(0)
                improved = (window[0] - value) / max(abs(window[0]), 1e-12)
                if improved < stall_tol:
                    break


def _conv_geometry(height: int, min_base: int = 6) -> tuple[int, int]:
    base, doublings = height, 0
    while base % 2 == 0 and base > min_base:
        base //= 2
        doublings += 1
    if base * (2 ** doublings) != height:
        raise ValueError(f"image side {height} is not a conv-friendly size")
    return base, doublings


class ConditionalLoRAConv:
    """The proven image decoder (round-36+ ``ConditionalLoRAConvRGB``), moved
    into the library and generalized to any square RGB `output_shape`. ONE
    shared conv backbone with shared low-rank directions at every layer, gated
    per-individual; MIXED conditioning — half the coefficients are extra
    evolvable decoder inputs (reachability), half gate the LoRA directions.
    Same interface as ``ConditionalLoRADecoder`` so solve_many uses either.
    """

    def __init__(self, latent: int, output_shape: tuple, coefficient_dim: int,
                 device: str, base_channels: int = 16):
        import torch.nn.functional as F
        self.F = F
        self.device = device
        self.latent = latent
        self.channels_last = output_shape[-1] <= 4
        if self.channels_last:
            height, width, colors = output_shape
        else:
            colors, height, width = output_shape
        if height != width:
            raise ValueError("conv decoder needs a square image")
        self.colors, self.height = colors, height
        base, doublings = _conv_geometry(height)
        self.base = base
        self.channels = base_channels
        half = max(1, coefficient_dim // 2)
        self.extra_latent_dim = half
        self.lora_dim = coefficient_dim - half
        self.coefficient_dim = coefficient_dim
        self.scale = self.lora_dim ** -0.5
        dec_in = latent + self.extra_latent_dim
        r = self.lora_dim
        ch = self.channels
        self.net = nn.Module().to(device)
        m = self.net
        m.base_fc = nn.Linear(dec_in, ch * base * base)
        m.fc_down = nn.Linear(dec_in, r, bias=False)
        m.fc_up = nn.Linear(r, ch * base * base, bias=False)
        m.base_convs = nn.ModuleList()
        m.conv_down = nn.ModuleList()
        m.conv_up = nn.ModuleList()
        for _ in range(doublings):
            m.base_convs.append(nn.Conv2d(ch, ch, 3, padding=1))
            m.conv_down.append(nn.Conv2d(ch, r, 3, padding=1, bias=False))
            m.conv_up.append(nn.Conv2d(r, ch, 1, bias=False))
        m.output_base = nn.Conv2d(ch, colors, 3, padding=1)
        m.output_down = nn.Conv2d(ch, r, 3, padding=1, bias=False)
        m.output_up = nn.Conv2d(r, colors, 1, bias=False)
        for mod in [m.fc_down, m.fc_up, *m.conv_down, *m.conv_up,
                    m.output_down, m.output_up]:
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        self.net.to(device)
        self.n_params = sum(p.numel() for p in self.net.parameters())

    def _run(self, z: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        m, F = self.net, self.F
        extra, lora = coeff[:, :self.extra_latent_dim], \
            coeff[:, self.extra_latent_dim:]
        dec_in = torch.cat([z, extra], dim=1)
        x = m.base_fc(dec_in) + self.scale * m.fc_up(m.fc_down(dec_in) * lora)
        x = x.view(-1, self.channels, self.base, self.base)
        gates = lora[:, :, None, None]
        for i in range(len(m.base_convs)):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            res = m.conv_up[i](m.conv_down[i](x) * gates)
            x = F.leaky_relu(m.base_convs[i](x) + self.scale * res)
        out = m.output_base(x) + self.scale * m.output_up(
            m.output_down(x) * gates)
        if self.channels_last:
            out = out.permute(0, 2, 3, 1)
        return out.flatten(1)

    def decode(self, z: np.ndarray, coeff: np.ndarray) -> torch.Tensor:
        zt = torch.as_tensor(
            np.ascontiguousarray(z.astype(np.float32)), device=self.device)
        ct = torch.as_tensor(
            np.ascontiguousarray(coeff.astype(np.float32)), device=self.device)
        with torch.no_grad():
            out = torch.sigmoid(self._run(zt, ct))
        return out.reshape(len(z), -1)

    def training_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid BASE output (zero bending), with gradients — the
        distillation target surface: the base learns to reach unaided what
        a champion's bending reached."""
        zeros = torch.zeros(len(z), self.coefficient_dim, device=self.device)
        return self._run(z, zeros)

    def get_params(self) -> np.ndarray:
        return nn.utils.parameters_to_vector(
            self.net.parameters()).detach().cpu().numpy().astype(np.float32)

    def set_params(self, flat: np.ndarray) -> None:
        nn.utils.vector_to_parameters(
            torch.as_tensor(np.asarray(flat, dtype=np.float32),
                            device=self.device),
            self.net.parameters())

    def _direction_modules(self):
        m = self.net
        return [m.fc_down, m.fc_up, *m.conv_down, *m.conv_up,
                m.output_down, m.output_up]

    def direction_vector(self) -> np.ndarray:
        return np.concatenate([
            mod.weight.detach().cpu().numpy().ravel()
            for mod in self._direction_modules()]).astype(np.float32)

    def set_direction_vector(self, flat: np.ndarray) -> None:
        offset = 0
        with torch.no_grad():
            for mod in self._direction_modules():
                n = mod.weight.numel()
                mod.weight.copy_(torch.as_tensor(
                    flat[offset:offset + n], device=self.device
                ).reshape(mod.weight.shape))
                offset += n

    def absorb(self, coeff: np.ndarray) -> None:
        """Direct latent application for the mixed-conditioning conv decoder:
        the extra-input half folds into base_fc's bias (a constant input is a
        bias), and the LoRA-gate half composes into the base kernels — exact
        arithmetic, no training."""
        c = torch.as_tensor(np.asarray(coeff, dtype=np.float32),
                            device=self.device)
        extra, gates = c[:self.extra_latent_dim], c[self.extra_latent_dim:]
        m = self.net
        with torch.no_grad():
            m.base_fc.bias += (m.base_fc.weight[:, self.latent:] @ extra)
            fc_kernel = m.fc_up.weight @ (gates[:, None] * m.fc_down.weight)
            # the low-rank path also read the extra inputs; that cross-term
            # becomes a bias once the donor's extras are zeroed
            m.base_fc.bias += self.scale * (
                fc_kernel[:, self.latent:] @ extra)
            m.base_fc.weight += self.scale * fc_kernel
            for i in range(len(m.base_convs)):
                kernel = torch.einsum(
                    "or,rikl->oikl",
                    m.conv_up[i].weight[:, :, 0, 0] * gates[None, :],
                    m.conv_down[i].weight)
                m.base_convs[i].weight += self.scale * kernel
            kernel = torch.einsum(
                "or,rikl->oikl",
                m.output_up.weight[:, :, 0, 0] * gates[None, :],
                m.output_down.weight)
            m.output_base.weight += self.scale * kernel

    def fold(self, z: np.ndarray, phenotypes: np.ndarray,
             stall_window: int, stall_tol: float) -> None:
        zt = torch.as_tensor(
            np.ascontiguousarray(z.astype(np.float32)), device=self.device)
        pt = torch.as_tensor(
            np.ascontiguousarray(phenotypes.astype(np.float32)),
            device=self.device)
        zero = torch.zeros(len(zt), self.coefficient_dim, device=self.device)
        opt = torch.optim.Adam(self.net.parameters())
        window: list[float] = []
        while True:
            opt.zero_grad()
            out = torch.sigmoid(self._run(zt, zero)).reshape(len(zt), -1)
            loss = torch.mean((out - pt) ** 2)
            loss.backward()
            opt.step()
            value = float(loss.detach())
            window.append(value)
            if len(window) > stall_window:
                window.pop(0)
                improved = (window[0] - value) / max(abs(window[0]), 1e-12)
                if improved < stall_tol:
                    break


def attach_seeded_directions(decoder):
    """Per-individual direction bases (Daniel, 2026-07-22): each individual
    carries an integer seed; its low-rank vocabulary is a frozen random
    function of that seed — nothing per-individual is evolved weight. The
    decoder gains a seed-grouped decode: for each distinct seed in a batch,
    the corresponding random directions are installed and that sub-batch is
    decoded, so the shared backbone stays the single source of accumulated
    knowledge while every lineage searches its own random subspace."""
    dim = len(decoder.direction_vector())
    cache: dict[int, np.ndarray] = {}

    def directions_for(seed: int) -> np.ndarray:
        key = int(seed)
        if key not in cache:
            cache[key] = (np.random.default_rng(key)
                          .standard_normal(dim) * 0.02).astype(np.float32)
        return cache[key]

    def decode_seeded(z: np.ndarray, coeff: np.ndarray,
                      seeds: np.ndarray) -> torch.Tensor:
        outputs = [None] * len(z)
        for seed in np.unique(seeds):
            picks = np.flatnonzero(seeds == seed)
            decoder.set_direction_vector(directions_for(int(seed)))
            part = decoder.decode(z[picks], coeff[picks])
            for j, i in enumerate(picks):
                outputs[int(i)] = part[j]
        return torch.stack(outputs)

    def absorb_seeded(coeff: np.ndarray, seed: int) -> None:
        decoder.set_direction_vector(directions_for(int(seed)))
        decoder.absorb(coeff)

    decoder.directions_for = directions_for
    decoder.decode_seeded = decode_seeded
    decoder.absorb_seeded = absorb_seeded
    return decoder


def build_conditional_decoder(architecture, latent, output_shape,
                              coefficient_dim, device):
    """Pick the proven conv conditional-LoRA decoder for image-shaped outputs
    (2-D+); fall back to the generic per-layer LoRA wrapper for vector outputs.
    Both are ONE shared decoder conditioned per-individual by (z, coefficients).

    An explicitly named or supplied `architecture` always wins, at any output
    shape. Before 2026-07-25 this function dropped the argument for every 2-D+
    shape, so `solve(..., architecture=...)` was a silent no-op on exactly the
    image problems the README demonstrates it on — an unknown name did not
    even raise. No recorded finding rode on it (every image benchmark that
    passes an architecture drives the legacy engines, which resolve their
    own), but the documented extension point did nothing.
    """
    if architecture == "auto" and len(output_shape) >= 2:
        return ConditionalLoRAConv(latent, output_shape, coefficient_dim,
                                   device)
    return ConditionalLoRADecoder(architecture, latent, output_shape,
                                  coefficient_dim, device)
