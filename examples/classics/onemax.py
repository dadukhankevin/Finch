"""OneMax — the hello world of genetic algorithms: maximize the number
of 1s in a bitstring. Every operator here is a plain function you could
swap; the stack is the whole algorithm.

    python3 examples/classics/onemax.py
    python3 -m finch4.hub        # watch it at http://127.0.0.1:8800
"""
from finch4 import (Breed, CapPopulation, Environment, Evaluate, Mutate,
                    Populate, SortByFitness)

BITS = 64


def random_bits(rng):
    return [rng.randint(0, 1) for _ in range(BITS)]


def one_point(a, b, rng):
    cut = rng.randrange(1, BITS)
    return a[:cut] + b[cut:]


def bit_flip(genome, rng):
    out = list(genome)
    i = rng.randrange(BITS)
    out[i] = 1 - out[i]
    return out


env = Environment([
    Populate(random_bits, 60),
    Breed(one_point, children=40),
    Mutate(bit_flip, rate=0.9),
    Evaluate(sum, name="ones"),
    SortByFitness(),
    CapPopulation(60),
], name="onemax", live=True, seed=0)
env.evolve(generations=60)

best = env.best_ever
print(f"best: {best['fitness']:.0f} of {BITS} ones "
      f"({env.state['evaluations']} evaluations)")
env.report_media("best bitstring", text="".join(map(str, best["genome"])))
