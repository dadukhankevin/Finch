"""Tests for the specified GA — finch4.ga.solve."""
from __future__ import annotations

import numpy as np
import torch

from finch4 import (
    coin_flip_latent_inheritance,
    make_random_speciation,
    one_point_gene_crossover,
    solve,
)


def _curve_fitness(target):
    def fitness(phenotypes: torch.Tensor):
        t = torch.as_tensor(np.asarray(target), device=phenotypes.device,
                            dtype=phenotypes.dtype)
        return -torch.mean((phenotypes.flatten(1) - t.flatten()) ** 2, dim=1)
    return fitness


def test_single_function_improves_and_reports():
    target = (np.sin(np.linspace(0, 2 * np.pi, 32)) * 0.3 + 0.5)
    result = solve(_curve_fitness(target), output_shape=(32,), epochs=60,
                   genes=4, latents=4, children=8, device="cpu", seed=0)
    assert result.epochs == 60
    assert result.best_phenotype.shape == (32,)
    assert 0.0 <= result.best_phenotype.min()
    assert result.best_phenotype.max() <= 1.0
    assert result.best_fitness > result.problems[0].initial_fitness
    # evaluations: 2 founders + children each epoch (+ speciation rescores,
    # which cannot occur with a single function since moves are no-ops)
    assert result.evaluations >= 2 + 60 * 8
    assert result.problems[0].evaluations == result.evaluations


def test_deterministic_with_seed():
    target = np.full(16, 0.4)
    a = solve(_curve_fitness(target), output_shape=(16,), epochs=30,
              genes=4, latents=4, children=8, device="cpu", seed=5)
    b = solve(_curve_fitness(target), output_shape=(16,), epochs=30,
              genes=4, latents=4, children=8, device="cpu", seed=5)
    assert a.best_fitness == b.best_fitness


def test_speciation_spreads_coverage_and_archive_reports_all():
    rng = np.random.default_rng(1)
    targets = [rng.uniform(0.2, 0.8, 16).astype(np.float32)
               for _ in range(4)]
    result = solve([_curve_fitness(t) for t in targets], output_shape=(16,),
                   epochs=120, genes=4, latents=4, children=8, founding="two",
                   speciation=make_random_speciation(rate=0.2),
                   device="cpu", seed=2)
    tried = [p for p in result.problems if p.best_phenotype is not None]
    assert len(tried) == 4          # random re-assignment reached every one
    for p in tried:
        assert p.best_fitness >= p.initial_fitness
    assert sum(p.evaluations for p in result.problems) == result.evaluations
    assert result.history[-1]["functions_tried"] == 4


def test_untried_functions_return_empty_results_not_errors():
    target = np.full(16, 0.3, dtype=np.float32)
    fns = [_curve_fitness(target), _curve_fitness(1 - target)]
    result = solve(fns, output_shape=(16,), epochs=5, genes=4, latents=4,
                   children=4, founding="two", device="cpu", seed=3)
    assert result.problems[0].best_phenotype is not None
    assert result.problems[1].best_phenotype is None   # never visited
    assert result.problems[1].evaluations == 0


def test_latents_are_inherited_whole_never_spliced():
    rng = np.random.default_rng(4)
    la = rng.standard_normal((6, 8)).astype(np.float32)
    lb = rng.standard_normal((6, 8)).astype(np.float32)
    child = coin_flip_latent_inheritance(la, lb, rng)
    for i in range(6):
        assert (np.array_equal(child[i], la[i])
                or np.array_equal(child[i], lb[i]))


def test_gene_crossover_is_one_point_within_genes_only():
    rng = np.random.default_rng(5)
    ga_ = np.zeros((4, 8), dtype=np.float32)
    gb = np.ones((4, 8), dtype=np.float32)
    child = one_point_gene_crossover(ga_, gb, rng)
    for row in child:
        flips = np.flatnonzero(np.diff(row) != 0)
        assert len(flips) == 1      # exactly one cut, inside the genes


def test_sparse_patch_mode_runs_and_improves():
    target = (np.sin(np.linspace(0, 2 * np.pi, 32)) * 0.3 + 0.5)
    result = solve(_curve_fitness(target), output_shape=(32,), epochs=80,
                   genes=8, latents=64, children=8, device="cpu",
                   directions="sparse", seed=1)
    assert result.best_phenotype.shape == (32,)
    assert result.best_fitness > result.problems[0].initial_fitness


def test_sparse_patch_keeps_one_decoder_and_folds_exactly():
    # The invariant: one shared backbone, a per-individual modifier that is
    # K values + one integer — never a per-individual weight matrix. And a
    # fold is an exact scatter-add, so absorbing a donor's patch and zeroing
    # its values reproduces that donor's phenotype bit-for-bit.
    from finch4.sparse import SparsePatchDecoder

    decoder = SparsePatchDecoder("auto", 8, (16,), patch_size=32,
                                 device="cpu")
    rng = np.random.default_rng(0)
    z = rng.standard_normal((1, 8)).astype(np.float32)
    values = rng.standard_normal((1, 32)).astype(np.float32)
    seeds = np.array([12345])

    before = decoder.decode_seeded(z, values, seeds).numpy()
    assert decoder.patch_size < decoder.n_params      # genuinely sparse
    decoder.absorb_seeded(values[0], int(seeds[0]))
    after = decoder.decode_seeded(z, np.zeros_like(values), seeds).numpy()
    assert np.abs(before - after).max() < 1e-5

    # sites are a pure function of the seed, so locations cost one integer
    assert np.array_equal(decoder.sites_for(7), decoder.sites_for(7))
    assert not np.array_equal(decoder.sites_for(7), decoder.sites_for(8))


def test_dials_adapt():
    target = np.full(24, 0.6)
    result = solve(_curve_fitness(target), output_shape=(24,), epochs=50,
                   genes=4, latents=4, children=8, device="cpu", seed=6)
    dials = [(h["gene_dial"], h["latent_dial"]) for h in result.history]
    assert dials[-1] != (1.0, 1.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
