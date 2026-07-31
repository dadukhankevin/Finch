"""Finch 4's acceptance rule: a preset carries the research campaign's evidence
only if it reproduces the vetted engines' behavior EXACTLY. These tests are
that rule, executable — the same standard the fold-removal rewrite had
to meet before inheriting the old engine's records."""
import numpy as np
import torch

from finch4.layers import (Environment, agentic_environment,
                               tensor_environment)
from finch4 import AgenticGA, solve


def fake_score(task, variation):
    scale = 1.0 if task == "alpha" else 1000.0
    return scale * (len(variation) % 17)


def fake_result(job):
    parents = job["parents"]
    if job["kind"] == "found":
        var = f"found-{job['job_id']}"
    elif job["kind"] == "mutate":
        var = parents[0]["variation"] + "-m"
    else:
        var = parents[0]["variation"] + "+" + parents[1]["variation"]
    return {"variation": var, "score": fake_score(job["task"], var),
            "artifact": f"/tmp/{job['job_id']}.py"}


def drive_direct(seed, rounds):
    """The pre-Finch loop, verbatim from the engine's own tests."""
    ga = AgenticGA(tasks=["alpha", "beta"], founders=2, children=4,
                   population_cap=6, consolidate_every=2, seed=seed)
    for _ in range(rounds):
        for job in ga.ask():
            r = fake_result(job)
            ga.tell(job["job_id"], r["variation"], r["score"],
                    artifact=r["artifact"])
        if ga.consolidation_due():
            ga.consolidation_batch()
            for ind in ga.record_consolidation():
                ga.tell_rewrite(ind["id"], ind["variation"] + "!")
    return ga


def drive_layered(seed, rounds):
    """The same run expressed as Finch layers over the same engine."""
    env = agentic_environment(
        tasks=["alpha", "beta"], runner=fake_result,
        consolidator=lambda batch, env: True,
        rewriter=lambda survivor, env: survivor["variation"] + "!",
        founders=2, children=4, population_cap=6,
        consolidate_every=2, seed=seed)
    env.evolve(generations=rounds)
    return env.state["engine"]


def test_agentic_layers_are_bit_identical_to_direct_drive():
    a = drive_direct(seed=11, rounds=5)
    b = drive_layered(seed=11, rounds=5)
    assert a.summary() == b.summary()
    assert set(a.individuals) == set(b.individuals)
    for ind_id, ind in a.individuals.items():
        other = b.individuals[ind_id]
        for key in ("task", "variation", "score", "origin", "parents",
                    "alive", "scored_on_base"):
            assert ind[key] == other[key], (ind_id, key)
    # and the RNG consumed identically: the next ask matches too
    ja = [(j["kind"], j["task"], [p["id"] for p in j["parents"]])
          for j in a.ask()]
    jb = [(j["kind"], j["task"], [p["id"] for p in j["parents"]])
          for j in b.ask()]
    assert ja == jb


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
    env = agentic_environment(
        tasks=["alpha"], runner=fake_result, founders=2, children=3,
        consolidate_every=1000, seed=0, name="t")
    env.evolve(generations=4)
    assert len(env.state["history"]) == 4
    assert env.state["history"][-1]["best"]["alpha"] > 0
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
