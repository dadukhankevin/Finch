"""Finch 4's acceptance rule: a preset carries the research campaign's evidence
only if it reproduces the vetted engines' behavior EXACTLY. These tests are
that rule, executable — the same standard the fold-removal rewrite had
to meet before inheriting the old engine's records. (The agentic wing
has no such test anymore by design: named judge agents make semantic
pairwise selections, so there is no deterministic drive to be bit-identical
to — the communication and Elo laws are pinned in test_agentic.py instead.)"""
import numpy as np
import torch

from finch4 import solve
from finch4.layers import Environment, tensor_environment


def _fitness(phenotypes: torch.Tensor):
    return -(phenotypes.flatten(1) ** 2).mean(dim=1)


def test_tensor_preset_is_bit_identical_to_solve():
    direct = solve(_fitness, output_shape=(8,), epochs=5, children=4,
                   population_cap=8, founders=2, device="cpu", seed=3)
    env = tensor_environment(_fitness, output_shape=(8,), epochs=5,
                             children=4, population_cap=8, founders=2,
                             device="cpu", seed=3)
    env.evolve()
    wrapped = env.state["result"]
    assert wrapped.best_fitness == direct.best_fitness
    assert wrapped.evaluations == direct.evaluations
    assert [h["mean_score"] for h in wrapped.history] == \
           [h["mean_score"] for h in direct.history]
    assert env.best_ever.best_fitness == direct.best_fitness


def test_environment_history_and_plot(tmp_path):
    env = tensor_environment(_fitness, output_shape=(8,), epochs=4,
                             children=4, population_cap=8, founders=2,
                             device="cpu", seed=0, name="t")
    env.evolve()
    assert len(env.state["history"]) == 4
    assert env.state["history"][-1]["best"]["fn0"] < 0
    path = env.plot(str(tmp_path / "curve.svg"))
    assert "svg" in open(path).read()


def test_classic_layers_tsp_improves_and_is_deterministic():
    import math
    import random as pyrandom
    from finch4.classic import (Breed, CapPopulation,
                                           Evaluate, Mutate, Populate,
                                           SortByFitness, inversion,
                                           order_crossover)

    rng = pyrandom.Random(1)
    coords = [(rng.random(), rng.random()) for _ in range(25)]

    def tour_len(order):
        return sum(math.dist(coords[order[i]],
                             coords[order[(i + 1) % len(order)]])
                   for i in range(len(order)))

    def factory(r):
        order = list(range(25))
        r.shuffle(order)
        return order

    def build():
        return Environment([
            Populate(factory, 40),
            Breed(order_crossover, children=30),
            Mutate(inversion, rate=0.6),
            Evaluate(lambda g: -tour_len(g), name="tsp"),
            SortByFitness(),
            CapPopulation(40),
        ], seed=7)

    a = build().evolve(generations=40)
    b = build().evolve(generations=40)
    first = a.state["history"][0]["best"]["tsp"]
    last = a.best_ever["fitness"]
    assert last > first + 1.0            # genuinely improved
    assert a.best_ever["genome"] == b.best_ever["genome"]   # seeded
    assert sorted(a.best_ever["genome"]) == list(range(25))  # valid tour
