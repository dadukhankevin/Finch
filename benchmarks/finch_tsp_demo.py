"""TSP the normal-Finch way: a classic permutation GA as layers.

    python3 benchmarks/finch_tsp_demo.py [--cities 60] [--generations 300]

No decoder, no agents — Populate / Breed(order crossover) /
Mutate(inversion) / Evaluate / SortByFitness / CapPopulation over
permutation genomes, the textbook stack. Reported against the
nearest-neighbor construction baseline the agentic tsp task uses
(ratio > 1 beats nearest-neighbor), plus a random-tour floor.
"""
import argparse
import math
import random
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir))
from finch4.layers import Environment
from finch4.classic import (Breed, CapPopulation, Evaluate,
                                       Mutate, Populate, SortByFitness,
                                       inversion, order_crossover)


def tour_length(order, coords):
    return sum(math.dist(coords[order[i]], coords[order[(i + 1) % len(order)]])
               for i in range(len(order)))


def nearest_neighbor(coords):
    order, unvisited = [0], set(range(1, len(coords)))
    while unvisited:
        cur = coords[order[-1]]
        nxt = min(unvisited, key=lambda c: math.dist(coords[c], cur))
        order.append(nxt)
        unvisited.remove(nxt)
    return tour_length(order, coords)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", type=int, default=60)
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--live", action="store_true")
    a = p.parse_args()

    rng = random.Random(a.seed)
    coords = [(rng.random(), rng.random()) for _ in range(a.cities)]
    nn = nearest_neighbor(coords)

    def factory(r):
        order = list(range(a.cities))
        r.shuffle(order)
        return order

    def fitness(order):
        return -tour_length(order, coords)

    env = Environment([
        Populate(factory, 120),
        Breed(order_crossover, children=80, tournament=4),
        Mutate(inversion, rate=0.6),
        Evaluate(fitness, name="tsp"),
        SortByFitness(),
        CapPopulation(120),
    ], name="finch-tsp", live=a.live, seed=a.seed)

    env.compile()
    random_floor = -sum(fitness(factory(random.Random(i)))
                        for i in range(20)) / 20
    for block in range(10):
        env.evolve(generations=a.generations // 10)
        best = -env.best_ever["fitness"]
        print(f"gen {env.state['generation']:4d}  best tour {best:.4f}  "
              f"vs NN {nn:.4f}  ratio {nn / best:.4f}  "
              f"evals {env.state['evaluations']}", flush=True)
    best = -env.best_ever["fitness"]
    print(f"\nfinal: GA {best:.4f} | nearest-neighbor {nn:.4f} | "
          f"random ~{random_floor:.4f}")
    print(f"ratio vs NN: {nn / best:.4f} ({'beats' if best < nn else 'loses to'} "
          f"nearest-neighbor)")
    print("curve:", env.plot("benchmarks/finch_tsp_demo_fitness.svg"))


if __name__ == "__main__":
    main()
