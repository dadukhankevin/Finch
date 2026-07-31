"""Evolve a sentence letter by letter — the other classic. A tiny
custom layer posts the best sentence to the dashboard every generation,
so you can watch it come into focus.

    python3 examples/classics/evolve_string.py
    python3 -m finch4.hub        # http://127.0.0.1:8800
"""
import string

from finch4 import (Breed, CapPopulation, Environment, Evaluate, Layer,
                    Mutate, Populate, SortByFitness)

TARGET = "the universal genetic algorithm"
ALPHABET = string.ascii_lowercase + " "


def random_text(rng):
    return [rng.choice(ALPHABET) for _ in TARGET]


def one_point(a, b, rng):
    cut = rng.randrange(1, len(TARGET))
    return a[:cut] + b[cut:]


def retype(genome, rng):
    out = list(genome)
    out[rng.randrange(len(out))] = rng.choice(ALPHABET)
    return out


def matches(genome):
    return sum(c == t for c, t in zip(genome, TARGET))


class ShowBest(Layer):
    """Custom layers compose right into the stack — this one is pure
    telemetry."""

    def __call__(self, env):
        best = env.state.get("best_ever")
        if best:
            env.report_media("best sentence", text="".join(best["genome"]))


env = Environment([
    Populate(random_text, 80),
    Breed(one_point, children=60),
    Mutate(retype, rate=0.9),
    Evaluate(matches, name="letters"),
    SortByFitness(),
    CapPopulation(80),
    ShowBest(),
], name="evolve-string", live=True, seed=1)
env.evolve(generations=120)

best = env.best_ever
print(f"best: {''.join(best['genome'])!r} "
      f"({best['fitness']:.0f}/{len(TARGET)} letters, "
      f"{env.state['evaluations']} evaluations)")
