"""Finch 4 — evolution as composable layers, three engines, one
dashboard.

    import finch4

    # 1. The tensor engine: genes + latents through ONE shared decoder
    result = finch4.solve(fitness_fn, output_shape=(32, 32, 3),
                          epochs=1_500)
    result.best_phenotype
    # ...or the same engine stage by stage through layers (bit-identical;
    # solve() is sugar over TensorGA.STAGES):
    finch4.tensor_environment(fitness_fn, output_shape=(32, 32, 3),
                              epochs=1_500).evolve()

    # 2. Classic layers over plain genomes (permutations, strings, ...)
    from finch4.classic import (Populate, Breed, Mutate, Evaluate,
                                SortByFitness, CapPopulation)
    env = finch4.Environment([...layers...], seed=0)
    env.evolve(generations=300)

    # 3. Agent-mediated GAR: evolvers run parallel autoresearch trajectories;
    # judge agents decide pairs; Finch communicates, records, updates Elo,
    # and can launch allocator-supplied native jobs concurrently
    campaign = finch4.Campaign(tasks=["compress"])
    # from finch4.workers import JobQueue
    # pool = JobQueue("runs/compress")
    # (see docs/high-agent.md, the agentic-ga skill, finch4.serve)

    # Every run on one live dashboard:
    finch4.solve(..., progress=finch4.live_progress())
    # python3 -m finch4.hub   ->  http://127.0.0.1:8800

The engines carry the evidence of the research campaign they grew from
(github.com/dadukhankevin/latentspace — FINDINGS.md there is the full
falsification-heavy record); tests/test_finch.py holds the seeded
bit-identity tests that let that evidence transfer."""

from .agentic import Campaign
from .architectures import build_mlp, register_architecture, resolve
from .classic import (Breed, CapPopulation, Evaluate, Mutate, Populate,
                      SortByFitness, individual, inversion,
                      order_crossover, swap)
from .ga import (Distillation, GAResult, ProblemResult, TensorGA, Wave,
                 coin_flip_latent_inheritance, fitness_shares,
                 make_gaussian_mutation, make_random_speciation,
                 make_species_selection, one_point_gene_crossover,
                 register_substrate, share_selection, solve,
                 uniform_selection)
from .layers import (BreedWave, CullByShares,
                     Distill, Environment, EvolveDirections, Immigrate,
                     Layer, MutateWave, RecordProgress, ScoreWave,
                     Speciate, TensorStage, TuneDials,
                     tensor_environment, tensor_stack)
from .serve import live_progress, media_client, png_data_uri

__all__ = [
    "solve", "GAResult", "ProblemResult", "Campaign", "TensorGA", "Wave",
    "Distillation",
    "register_architecture", "register_substrate", "resolve", "build_mlp",
    "fitness_shares", "make_species_selection", "share_selection",
    "uniform_selection", "one_point_gene_crossover",
    "coin_flip_latent_inheritance", "make_gaussian_mutation",
    "make_random_speciation", "live_progress", "media_client",
    "png_data_uri",
    "Environment", "Layer",
    "tensor_environment", "tensor_stack",
    "TensorStage", "Immigrate", "BreedWave", "MutateWave", "ScoreWave",
    "TuneDials", "CullByShares", "Speciate", "Distill",
    "EvolveDirections", "RecordProgress",
    "Populate", "Breed", "Mutate", "Evaluate", "SortByFitness",
    "CapPopulation", "individual", "order_crossover", "inversion", "swap",
]
