"""Evolve any image from a photo file. The fitness function is plain
pixel error; the shared decoder plus per-individual sparse patches do
everything else. The dashboard shows the image sharpening live, and a
progression strip (target, then snapshots over the run) is saved when
it finishes.

    python3 examples/decoders/evolve_image.py photo.jpg
    python3 -m finch4.hub        # http://127.0.0.1:8800

Loading the photo needs pillow (pip install pillow); the library itself
never does.
"""
import argparse
import base64
import os

import numpy as np
import torch

import finch4


def load_image(path, size):
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("this example loads photos with pillow: "
                         "pip install pillow")
    image = Image.open(path).convert("RGB").resize((size, size))
    return (np.asarray(image, dtype=np.float32) / 255.0).transpose(2, 0, 1)


def save_png(array, path):
    data = finch4.png_data_uri(array)
    with open(path, "wb") as f:
        f.write(base64.b64decode(data.split(",", 1)[1]))


def progression_strip(target, frames, picks=5, upscale=2, gap=2):
    """target plus snapshots log-spaced over the run (most visible
    change happens early), latest last, as one row."""
    idx = np.unique(np.geomspace(1, len(frames), picks).astype(int) - 1)
    panels = [target] + [frames[i] for i in idx]
    panels = [np.clip(p.transpose(1, 2, 0), 0, 1) for p in panels]
    spacer = np.ones((panels[0].shape[0], gap, 3), dtype=np.float32)
    row = panels[0]
    for panel in panels[1:]:
        row = np.concatenate([row, spacer, panel], axis=1)
    return np.repeat(np.repeat(row, upscale, axis=0), upscale, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default=None,
                   help="progression strip path (default: "
                        "<image>_evolution.png)")
    a = p.parse_args()

    target = load_image(a.image, a.size)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    target_t = torch.as_tensor(target.reshape(-1), device=device)

    def fitness(phenotypes):
        return -((phenotypes.flatten(1) - target_t) ** 2).mean(dim=1)

    frames = []
    live = finch4.live_progress(names=[os.path.basename(a.image)])

    def progress(epoch, epochs, evaluations, best_pheno, best_score):
        live(epoch, epochs, evaluations, best_pheno, best_score)
        if best_pheno[0] is not None:
            frames.append(best_pheno[0])

    result = finch4.solve(fitness, output_shape=target.shape,
                          epochs=a.epochs, device=device, seed=a.seed,
                          progress=progress, progress_every=30)

    print(f"mean squared error: {-result.best_fitness:.6f} "
          f"({result.evaluations} evaluations)")
    out = a.out or f"{os.path.splitext(a.image)[0]}_evolution.png"
    save_png(progression_strip(target, frames), out)
    print(f"progression strip: {out}")


if __name__ == "__main__":
    main()
