"""Fit a sine wave with the universal solver. Nothing here knows what a
sine wave is: the decoder reads genes, the fitness function scores the
curve, and evolution does the rest. Saves a target-vs-evolved SVG and
posts it to the dashboard.

    python3 examples/decoders/sine.py
    python3 -m finch4.hub        # http://127.0.0.1:8800
"""
import os

import numpy as np
import torch

import finch4

POINTS = 128
x = np.linspace(0, 2 * np.pi, POINTS)
target = (np.sin(x) * 0.4 + 0.5).astype(np.float32)


def fitness(phenotypes):
    t = torch.as_tensor(target, device=phenotypes.device,
                        dtype=phenotypes.dtype)
    return -((phenotypes.flatten(1) - t) ** 2).mean(dim=1)


def curve_svg(evolved, w=560, h=240, pad=14):
    def poly(values, color, width, dash=""):
        pts = " ".join(
            f"{pad + i * (w - 2 * pad) / (POINTS - 1):.1f},"
            f"{pad + (1 - v) * (h - 2 * pad):.1f}"
            for i, v in enumerate(values))
        return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="{width}"{dash}/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" '
            f'height="{h}" viewBox="0 0 {w} {h}">'
            f'<rect width="{w}" height="{h}" fill="#faf8f3"/>'
            + poly(target, "#a59c8c", 2, ' stroke-dasharray="5,4"')
            + poly(evolved, "#b65c38", 2.4)
            + f'<text x="{pad}" y="{h - 4}" font-size="11" '
            f'fill="#6f675a" font-family="monospace">'
            f'dashed: target &#183; solid: evolved</text></svg>')


cb = finch4.live_progress(names=["sine"])
result = finch4.solve(fitness, output_shape=(POINTS,), epochs=400,
                      seed=0, progress=cb, progress_every=20)

mse = -result.best_fitness
print(f"mean squared error: {mse:.6f} "
      f"({result.evaluations} evaluations)")
svg = curve_svg(result.best_phenotype)
cb.report_media("sine fit", svg=svg)
out = os.path.join(os.path.dirname(__file__), "sine_fit.svg")
with open(out, "w") as f:
    f.write(svg)
print(f"comparison drawing: {out}")
