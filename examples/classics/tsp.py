"""60-city traveling salesman with the classic permutation operators
(order crossover + segment inversion). Posts an SVG of the best tour to
the dashboard and saves it next to this script.

    python3 examples/classics/tsp.py
    python3 -m finch4.hub        # http://127.0.0.1:8800
"""
import math
import os
import random

from finch4 import (Breed, CapPopulation, Environment, Evaluate, Mutate,
                    Populate, SortByFitness, inversion, order_crossover)

rng0 = random.Random(3)
CITIES = [(rng0.random(), rng0.random()) for _ in range(60)]


def tour_length(order):
    return sum(math.dist(CITIES[order[i]],
                         CITIES[order[(i + 1) % len(order)]])
               for i in range(len(order)))


def random_tour(rng):
    order = list(range(len(CITIES)))
    rng.shuffle(order)
    return order


def tour_svg(order, size=420, pad=16):
    def pt(i):
        x, y = CITIES[i]
        return (pad + x * (size - 2 * pad), pad + y * (size - 2 * pad))
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                    [pt(i) for i in order] + [pt(order[0])])
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#b65c38"/>'
        for x, y in (pt(i) for i in order))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
            f'height="{size}" viewBox="0 0 {size} {size}">'
            f'<rect width="{size}" height="{size}" fill="#faf8f3"/>'
            f'<polyline points="{path}" fill="none" stroke="#5f7a3d" '
            f'stroke-width="1.6"/>{dots}</svg>')


env = Environment([
    Populate(random_tour, 120),
    Breed(order_crossover, children=80),
    Mutate(inversion, rate=0.6),
    Evaluate(lambda tour: -tour_length(tour), name="tsp"),
    SortByFitness(),
    CapPopulation(120),
], name="tsp-classic", live=True, seed=0)
env.evolve(generations=300)

best = env.best_ever
print(f"best tour length: {-best['fitness']:.4f} "
      f"({env.state['evaluations']} evaluations)")
svg = tour_svg(best["genome"])
env.report_media("best tour", svg=svg)
out = os.path.join(os.path.dirname(__file__), "tsp_tour.svg")
with open(out, "w") as f:
    f.write(svg)
print(f"tour drawing: {out}")
