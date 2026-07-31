"""Four fitness functions, ONE population, one shared decoder. Fitness
shares give each target an equal slice of the fitness mass, so no
target can be outcompeted out of existence, and on multi-function runs
the decoder periodically distills each species' best discovery into its
base. Watch all four images evolve at once on the dashboard; a
targets-vs-evolved grid is saved when it finishes.

    python3 examples/decoders/species.py
    python3 -m finch4.hub        # http://127.0.0.1:8800
"""
import base64
import os

import numpy as np
import torch

import finch4

SIZE = 32
yy, xx = np.mgrid[0:SIZE, 0:SIZE] / (SIZE - 1)


def gradient():
    img = np.zeros((SIZE, SIZE, 3), np.float32)
    img[..., 0] = 0.9 - 0.5 * yy          # sunset: orange into purple
    img[..., 1] = 0.4 - 0.25 * yy
    img[..., 2] = 0.2 + 0.5 * yy
    return img


def checker():
    tile = ((xx * 4).astype(int) + (yy * 4).astype(int)) % 2
    img = np.empty((SIZE, SIZE, 3), np.float32)
    img[..., 0] = np.where(tile, 0.37, 0.98)
    img[..., 1] = np.where(tile, 0.48, 0.97)
    img[..., 2] = np.where(tile, 0.24, 0.95)
    return img


def circle():
    inside = (xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.11
    img = np.empty((SIZE, SIZE, 3), np.float32)
    img[..., 0] = np.where(inside, 0.71, 0.98)
    img[..., 1] = np.where(inside, 0.27, 0.97)
    img[..., 2] = np.where(inside, 0.22, 0.95)
    return img


def stripes():
    band = ((xx + yy) * 3).astype(int) % 2
    img = np.empty((SIZE, SIZE, 3), np.float32)
    img[..., 0] = np.where(band, 0.25, 0.98)
    img[..., 1] = np.where(band, 0.48, 0.97)
    img[..., 2] = np.where(band, 0.45, 0.95)
    return img


TARGETS = {"gradient": gradient(), "checker": checker(),
           "circle": circle(), "stripes": stripes()}


def fitness_for(target):
    flat = target.reshape(-1)

    def fitness(phenotypes):
        t = torch.as_tensor(flat, device=phenotypes.device,
                            dtype=phenotypes.dtype)
        return -((phenotypes.flatten(1) - t) ** 2).mean(dim=1)
    return fitness


cb = finch4.live_progress(names=list(TARGETS))
result = finch4.solve([fitness_for(t) for t in TARGETS.values()],
                      output_shape=(SIZE, SIZE, 3), epochs=1500,
                      seed=0, progress=cb, progress_every=30)

for name, problem in zip(TARGETS, result.problems):
    print(f"{name}: mse {-problem.best_fitness:.5f} "
          f"({problem.evaluations} evaluations)")

# grid: top row targets, bottom row what evolved
gap = np.ones((SIZE, 2, 3), np.float32)
row_gap = np.ones((2, SIZE * 4 + 6, 3), np.float32)


def row(images):
    out = images[0]
    for img in images[1:]:
        out = np.concatenate([out, gap, img], axis=1)
    return out


grid = np.concatenate([
    row(list(TARGETS.values())),
    row_gap,
    row([np.clip(p.best_phenotype, 0, 1) for p in result.problems]),
], axis=0)
grid = np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)

out = os.path.join(os.path.dirname(__file__), "species_grid.png")
with open(out, "wb") as f:
    f.write(base64.b64decode(
        finch4.png_data_uri(grid).split(",", 1)[1]))
print(f"targets vs evolved: {out}")
