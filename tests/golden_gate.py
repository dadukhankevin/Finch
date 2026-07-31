"""The refactor gate for the tensor engine — NOT collected by pytest.

Bit-identity is this library's acceptance rule: evidence attaches to
code identity, so a refactor of solve()/TensorGA carries the campaign's
records only if it reproduces the old behavior EXACTLY. This script is
the executable form of that rule for refactors, complementing the
in-process identity tests (which compare two live code paths and so
cannot catch a change that moves both):

    # BEFORE touching the engine
    python3 tests/golden_gate.py /tmp/goldens_before.json
    # ... refactor ...
    python3 tests/golden_gate.py /tmp/goldens_after.json
    diff /tmp/goldens_before.json /tmp/goldens_after.json   # must be empty

Fingerprints are machine- and torch-version-specific — capture and
compare on the SAME machine, same environment. The configs cover every
substrate ("sparse-shared", "frozen", "sparse", "evolve"), both founding
modes, distillation on its cadence, immigrants, speciation, mutation
memory, both research selection arms, warm start, the conv image
decoder, default latent resolution, and the progress-callback stream.
The 2026-07-31 engine/layer decomposition passed this gate at every
step (17/17 fingerprints identical)."""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from finch4 import solve
from finch4.ga import (make_random_speciation, share_selection,
                       uniform_selection)


def curve_fitness(target):
    t = np.asarray(target, dtype=np.float32)

    def fitness(phenotypes):
        tt = torch.as_tensor(t, device=phenotypes.device,
                             dtype=phenotypes.dtype)
        return -torch.mean((phenotypes.flatten(1) - tt.flatten()) ** 2,
                           dim=1)
    return fitness


def sha(arr):
    if arr is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(arr, dtype=np.float64)).tobytes()).hexdigest()


def fingerprint(result):
    return {
        "evaluations": result.evaluations,
        "epochs": result.epochs,
        "problems": [
            {"best_fitness": repr(p.best_fitness),
             "initial_fitness": repr(p.initial_fitness),
             "evaluations": p.evaluations,
             "phenotype": sha(p.best_phenotype)}
            for p in result.problems],
        "history": hashlib.sha256(
            json.dumps(result.history, sort_keys=True).encode()).hexdigest(),
        "decoder": sha(result.decoder),
    }


rng = np.random.default_rng(99)
T16 = rng.uniform(0.2, 0.8, 16).astype(np.float32)
T16B = (1.0 - T16).astype(np.float32)
T16C = rng.uniform(0.1, 0.9, 16).astype(np.float32)
T16D = rng.uniform(0.3, 0.7, 16).astype(np.float32)
IMG = rng.uniform(0.2, 0.8, (8, 8, 3)).astype(np.float32)

F1 = curve_fitness(T16)
F2 = curve_fitness(T16B)
F3 = curve_fitness(T16C)
F4 = curve_fitness(T16D)

CONFIGS = {
    "base_sparse_shared": dict(
        fitness_fns=F1, output_shape=(16,), epochs=40, genes=6, latents=32,
        children=6, population_cap=12, founders=4, device="cpu", seed=0),
    "multi3_distill_auto": dict(
        fitness_fns=[F1, F2, F3], output_shape=(16,), epochs=70, genes=6,
        latents=32, children=6, founders=3, device="cpu", seed=1),
    "frozen_with_memory": dict(
        fitness_fns=F1, output_shape=(16,), epochs=40, genes=6, latents=8,
        children=6, directions="frozen", mutation_memory="shared",
        memory_drift=0.5, device="cpu", seed=2),
    "sparse_fresh_basis": dict(
        fitness_fns=F1, output_shape=(16,), epochs=40, genes=6, latents=16,
        children=6, directions="sparse", fresh_basis_rate=0.3,
        device="cpu", seed=3),
    "evolve_directions": dict(
        fitness_fns=F1, output_shape=(16,), epochs=16, genes=6, latents=8,
        children=6, directions="evolve", direction_every=4,
        device="cpu", seed=4),
    "immigrants_stall": dict(
        fitness_fns=[F1, F2], output_shape=(16,), epochs=20, genes=6,
        latents=32, children=6, immigrants="stall", immigrant_patience=3,
        device="cpu", seed=5),
    "speciation_two_founding": dict(
        fitness_fns=[F1, F2, F3, F4], output_shape=(16,), epochs=30,
        genes=6, latents=32, children=6, founding="two",
        speciation=make_random_speciation(rate=0.2), device="cpu", seed=6),
    "share_selection": dict(
        fitness_fns=F1, output_shape=(16,), epochs=25, genes=6, latents=32,
        children=6, selection=share_selection, device="cpu", seed=7),
    "uniform_selection": dict(
        fitness_fns=F1, output_shape=(16,), epochs=25, genes=6, latents=32,
        children=6, selection=uniform_selection, device="cpu", seed=8),
    "two_founding_untried_fn": dict(
        fitness_fns=[F1, F2], output_shape=(16,), epochs=25, genes=6,
        latents=32, children=6, founding="two", device="cpu", seed=10),
    "distill_explicit_fast_cadence": dict(
        fitness_fns=[F1, F2], output_shape=(16,), epochs=20, genes=6,
        latents=32, children=6, distill="on", distill_every=8,
        distill_steps=5, device="cpu", seed=11),
    "frozen_multi_distill": dict(
        fitness_fns=[F1, F2], output_shape=(16,), epochs=20, genes=6,
        latents=8, children=6, directions="frozen", distill_every=8,
        distill_steps=5, device="cpu", seed=12),
    "conv_frozen_image": dict(
        fitness_fns=curve_fitness(IMG), output_shape=(8, 8, 3), epochs=10,
        genes=6, latents=8, children=4, directions="frozen",
        device="cpu", seed=13),
    "conv_sparse_shared_image": dict(
        fitness_fns=curve_fitness(IMG), output_shape=(8, 8, 3), epochs=10,
        genes=6, latents=64, children=4, device="cpu", seed=14),
    "default_latents_resolution": dict(
        fitness_fns=F1, output_shape=(16,), epochs=8, genes=4, children=4,
        device="cpu", seed=15),
}


def main(out_path):
    goldens = {}
    for name, cfg in CONFIGS.items():
        result = solve(**cfg)
        goldens[name] = fingerprint(result)
        print(f"{name}: evals={result.evaluations}", flush=True)

    # warm start: one run's decoder chains into the next via init_decoder
    first = solve(F1, output_shape=(16,), epochs=10, genes=6, latents=32,
                  children=6, device="cpu", seed=9)
    warm = solve(F2, output_shape=(16,), epochs=10, genes=6, latents=32,
                 children=6, device="cpu", seed=9,
                 init_decoder=first.decoder)
    goldens["warm_start_chain"] = fingerprint(warm)
    print(f"warm_start_chain: evals={warm.evaluations}", flush=True)

    # the progress-callback stream, byte for byte
    calls = []

    def progress(epoch, epochs, evaluations, best_pheno, best_score):
        calls.append([int(epoch), int(epochs), int(evaluations),
                      [sha(b) for b in best_pheno],
                      [repr(float(s)) for s in best_score]])

    solve(F1, output_shape=(16,), epochs=12, genes=6, latents=32,
          children=6, device="cpu", seed=16, progress=progress,
          progress_every=3)
    goldens["progress_stream"] = hashlib.sha256(
        json.dumps(calls).encode()).hexdigest()

    with open(out_path, "w") as f:
        json.dump(goldens, f, indent=1, sort_keys=True)
    print(f"wrote {out_path} ({len(goldens)} entries)")


if __name__ == "__main__":
    main(sys.argv[1])
