# Finch 4

*Evolution as composable layers — three engines, one dashboard, and AI
agents as first-class participants.*

Finch 4 is a complete rewrite of [Finch](https://github.com/dadukhankevin/Finch/tree/finch-3),
unified with the engines and evidence of a months-long research
campaign ([latentspace](https://github.com/dadukhankevin/latentspace) —
its `FINDINGS.md` is the full falsification-heavy record behind every
default here). Finch supplies the grammar: an **Environment** is a
stack of **Layers** over a population of typed individuals. The
campaign supplies the vetted sentences: engines whose mechanisms were
measured, ablated, and kept only when they survived.

```bash
pip install -e .
```

## Three engines, one surface

**1. The neural-decoder GA** — a universal engine that never mutates a
solution directly. Every individual is a set of **genes** (input to one
shared decoder network) plus **latents** (a small vector bending that
network for this individual alone); evolution operates only on those
numbers, so the same operators work for any modality. Multiple fitness
functions become species sharing one population and one decoder, with
fitness organized as **shares** (each species permanently owns an equal
slice of the fitness mass — no objective can be outcompeted out of
existence). On multi-function runs the decoder periodically
**distills**: gradient-trains its base to reproduce each species'
best-ever discovery, then decays every individual bending — evolution
vets, gradients consolidate (measured: 10/10 paired seeds, −30% error
on eight co-resident problems).

```python
import finch4

def fitness(phenotypes):   # batched: tensor in, one score per row
    return -((phenotypes - target) ** 2).flatten(1).mean(dim=1)

result = finch4.solve(fitness, output_shape=(32, 32, 3), epochs=1_500)
result.best_phenotype
```

**2. Classic layers** — the traditional GA as a stack, over plain
genomes (lists, permutations, strings, objects):

```python
from finch4 import (Environment, Populate, Breed, Mutate, Evaluate,
                    SortByFitness, CapPopulation, order_crossover,
                    inversion)

env = Environment([
    Populate(random_tour, 120),
    Breed(order_crossover, children=80),
    Mutate(inversion, rate=0.6),
    Evaluate(lambda tour: -tour_length(tour)),
    SortByFitness(),
    CapPopulation(120),
], seed=0)
env.evolve(generations=300)
env.best_ever
```

(`benchmarks/finch_tsp_demo.py`: 60-city TSP, ~24k evaluations in 1.4s,
beats nearest-neighbor construction by 12%.)

**3. The agentic substrate** — the decoder is an **AI agent**.
Individuals are text *methodologies* (one shared base playbook plus a
per-individual variation clause); an agent follows base+variation to
produce an artifact; a canonical scorer script — which agents run but
never edit — is the only source of truth for fitness. A small HTTP
server (`python3 -m finch4.serve`) holds the population laws (shares,
selection, capping, best-ever archives, consolidation cadence) while
agents report their own results the moment they finish. Consolidation
is distillation translated to text: a consolidator agent edits the
playbook so that an agent following it with an *empty* variation would
reproduce the audited wins, and every survivor then deepens its
variation or stands pat.

Two skills make agents native (in `.claude/skills/`): **agentic-ga**
(operate inside a run — decode, mutate, consolidate, audit) and
**evolution-author** (set up entirely new evolutionary problems:
representation choice, scorer constitution, launch, discipline). First
measured result: on a lossless-compression task with matched budgets
(31 agent jobs per side, same model), the population with its learned
playbook beat autoresearch-style solo keep/revert iteration on both the
practice and held-out slices.

## One dashboard for every run

```python
finch4.solve(fitness, output_shape=(64, 64), epochs=10_000,
             progress=finch4.live_progress())
```

```bash
python3 -m finch4.hub        # http://127.0.0.1:8800
```

Every run — any engine — reports to the same live page: fitness over
time, population state, event stream. The hub shows every run this
machine has served, live and finished, as one board. Warm paper theme;
no external assets; stdlib HTTP only.

## The trust layer (why agentic scores can be believed)

Agents self-report scores, so the substrate ships an immune system,
each part paid for by a real incident: canonical scorers with
**practice/surprise data splits**; scored bytes perturbed in memory so
answer-key embedding cannot round-trip; **audit-on-influence** (any
would-be parent or champion is re-scored by the orchestrator, exactly
for deterministic tasks, within a scorer-declared tolerance otherwise);
falsified best-evers are evictable; work logs are read against claims;
and lineage exhaustion lets an honest agent declare a dead end and
found fresh rather than grind out token variations.

## Where the evidence lives

This repository ships the engines; the research trail that produced
them — ~40 rounds of paired-seed ablations, the claims ledger, the
failures — lives in the
[latentspace repository](https://github.com/dadukhankevin/latentspace).
`tests/test_finch.py` here holds the seeded bit-identity tests proving
these engines reproduce the vetted originals exactly; that identity is
what lets the evidence transfer. Recompositions beyond the shipped
presets are new mechanisms — measure before trusting.

Finch 3 remains available at the
[`finch-3` tag](https://github.com/dadukhankevin/Finch/releases/tag/finch-3).

*By Daniel Losey, with Claude (Anthropic) as research and engineering
partner.*
