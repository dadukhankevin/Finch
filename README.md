# Finch 4

Finch 4 is the latest iteration of [Finch](https://github.com/dadukhankevin/Finch/tree/finch-3), combining Finch's modularity with concepts from genespace, [latentspace](https://github.com/dadukhankevin/latentspace), and [SCRISPR](https://github.com/dadukhankevin/SCRISPR).
The result is a library capable of very general evolution. For example, using latent decoders we can solve many kinds of problems with just one algorithm:

```python
import finch4

def fitness(phenotypes):   # batched: tensor in, one score per row
    return -((phenotypes - target) ** 2).flatten(1).mean(dim=1)

result = finch4.solve(fitness, output_shape=(32, 32, 3), epochs=1_500)
result.best_phenotype
```

Provided a fitness function and an output shape, Finch aims to search nearly any optimizable problem. This is a claim about generality, not universal superiority: as with universal function approximation, the right representation and decoder still matter.

![A photo of an apple, then snapshots of the evolved image sharpening over one run](docs/images/apple_evolution.png)

That strip is `examples/decoders/evolve_image.py` running on a photo of an apple: the target on the left, then the best individual at a few points during the run. The fitness function is plain pixel error; everything else is the shared decoder and evolution (24k evaluations, about 15 seconds on an M-series Mac).

Sometimes you might still want a traditional GA. Finch lets you (or your agent) express one as a compact stack of composable layers.

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

(That one is the 60-city traveling salesman demo in `benchmarks/finch_tsp_demo.py`: it took about 24k evaluations in 1.4 seconds and was 12% better than the nearest-neighbor algorithm.)

![The best tour found, drawn as a loop through the 60 cities](docs/images/tsp_tour.svg)

The tour drawing comes from `examples/classics/tsp.py`, which also posts it to the dashboard while it runs.

A new method called genetic auto-research (GAR) is also available within Finch 4. GAR keeps an evolving population of research lineages: each lineage is a persistent autoresearch trajectory (Karpathy's loop), while workers are compute assignments that may rotate without resetting it. Evolvers read and report through Finch directly. After trusted evaluation, the scored lineage writes its own cited interpretation; Finch atomically appends it to the same versioned `Decoder.md`, so parallel researchers cannot overwrite one another and later experiments inherit the latest findings. A task can give them a visible laboratory of concrete prediction/reference errors while reserving disjoint data for evolutionary selection and final confirmation. Finch can launch allocator-supplied native worker commands from a durable queue with per-lane concurrency limits; it schedules processes but never authors research. Separate judge agents compare exactly two immutable checkpoints at a time using fixed fitness criteria and evaluator evidence; their verdicts update lineage Elo. For translation, for example, chrF can ground a judgment while Elo remains the population-level selection record. Finch stores the jobs, pairings, verdicts, ratings, artifacts and communication, but never judges an individual itself. The decoder here is the LLM plus `Decoder.md`; the file is shared scientific memory rather than literal model parameters. Crossover normally happens when later evolvers naturally adopt, extend, contrast, avoid, and combine cited discoveries from that file. Reports use compact tokens such as `[L0003#2@8ddf8b41]`, which Finch parses into a searchable dashboard **Tree of Life**. GAR builds on my earlier work, [SCRISPR](https://github.com/dadukhankevin/SCRISPR), and targets the same iterative-research setting as Karpathy's autoresearch while maintaining several competing lineages. GAR won my first matched comparison against normal autoresearch, but that result is one run on one task under an earlier protocol; broader study is still needed.

```bash
RUN=benchmarks/agentic/runs/r2
python3 -m finch4.serve --run "$RUN" --tasks binpack &
until test -f "$RUN/server.json"; do sleep 0.1; done
PORT=$(python3 -c "import json;print(json.load(open('$RUN/server.json'))['port'])")
python3 -m finch4.evolver --run "$RUN" --task binpack \
    --tasks-dir benchmarks/agentic/tasks \
    --agent-cmd 'claude -p "$(cat {promptfile})"' \
    --population 6 --experiments 8 --serve-port "$PORT"
```

`finch4.evolver` owns the population. Each member is one auto-research
worker on the same artifact and the same `Decoder.md`. Finch injects
fitness and shared research; the allocator does not write per-worker
prompts. For extra process lanes, enqueue native command arrays in
`JobQueue(RUN)`:

```bash
python3 -m finch4.workers --root "$RUN" run --max-concurrency 14 \
  --lane-limit evolver=10 --lane-limit evaluator=2 --lane-limit judge=2
```

Finch schedules those processes and shows them on the dashboard; the task
adapter and allocator still decide what commands to queue.

In one small bin-packing run under the earlier protocol, the plain best-fit rule scored 0.9417; the GAR champion reached 0.9533 on the practice seeds and 0.9371 on held-out seeds. That is an illustrative run, not yet evidence of broad superiority.

Two skills in `.claude/skills/` make agents work well with Finch 4: one runs an agent-mediated GAR campaign and the other authors brand new evolutionary problems. The [agentic design](docs/high-agent.md) describes the split among evolvers, judge agents, the allocator, the evaluator, and Finch's communication and bookkeeping layer.

Agents can mix and match all of the above strategies, and all evolution is reported to one live dashboard:

```bash
python3 -m finch4.hub        # http://127.0.0.1:8800
```

All runs join the dashboard by passing `progress=finch4.live_progress()` to `solve()`. If what you are evolving is an image, the dashboard shows it evolving live, no extra code. Anything else (an SVG of the best tour, a snippet of the champion's code) should be posted with `progress.report_media(...)`. I'm actively working on improving the dashboard.

![The dashboard: one card per run, live and finished, with thumbnails of what each run evolved](docs/images/dashboard.png)

Finch 4 ranges from traditional GAs over problem-specific genomes to algorithms that separate genotype from phenotype. In the decoder engine, each individual carries continuous genes and per-individual latent modifiers, and one shared decoder maps them into the requested output shape. With a suitable decoder, the same solver can search many different problem classes; the decoder's representation and mutation locality determine how effective that search is. On multi-function runs, vetted discoveries can also be distilled back into the shared decoder.

In genetic auto-research, the decoder is the LLM plus Decoder.md. The document is the writable, shared part of that decoder: scored findings can be shared as scientific memory, while incorporating an audit-passed method as a default is analogous to a training step. Each lineage's running research state plays the role of the latents. The shared idea across both engines is the separation between an evolvable representation and the realized solution that fitness actually measures — plus one shared, learning decoder per run.

## More examples

After installation, everything in `examples/` runs on its own and reports to the dashboard.

Classics (`examples/classics/`): `onemax.py`, `evolve_string.py`, `knapsack.py`, and `tsp.py`. Same layer grammar as the TSP snippet above; each one defines its operators as plain functions.

Decoders (`examples/decoders/`): `evolve_image.py` (point it at any photo), `sine.py` (curve fitting), and `species.py`, which evolves four different images in one population. Fitness shares give each target an equal slice of the fitness mass, so no target can be outcompeted out of existence, and the shared decoder periodically distills each species' best discovery into its base:

![Four target images on top, the four evolved versions below](docs/images/species_grid.png)

![A dashed target sine wave with the evolved curve drawn on top of it](docs/images/sine_fit.svg)

Genetic auto-research (`examples/gar/`): `binpack.sh`, the unattended version of the GAR run described above.

Install:

```bash
pip install -e .
```

The research record behind every shipped default lives in the [latentspace repo](https://github.com/dadukhankevin/latentspace). Finch 3 remains available at the [finch-3 tag](https://github.com/dadukhankevin/Finch/releases/tag/finch-3).

This project is the result of six years of research into genetic algorithms and building better ones. I combined ideas from several of my earlier libraries with countless hours of experiments and collaboration with Claude and Codex in pursuit of a universal genetic algorithm. Research notes and reports from those conversations can be found at https://loseylabs.ai. The notes were written by these AI collaborators, but they document which ideas worked, which failed, and how the research developed. They also show why it remains important to trust your judgment as a human, ask good questions, and follow your curiosity.
