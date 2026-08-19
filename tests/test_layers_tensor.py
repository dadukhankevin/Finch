"""The decomposed tensor stack: custom compositions, the Wave contract,
and the engine protocol Environment reads. The canonical-order identity
with solve() is structural (both paths run TensorGA.STAGES) and pinned
by test_finch.test_tensor_preset_is_bit_identical_to_solve; these tests
cover what the decomposition NEWLY allows — inserting layers, running in
chunks, inspecting waves — and that none of it changes the numbers."""
import numpy as np
import pytest
import torch

from finch4 import (Layer, TensorGA, TensorStage, Wave, solve,
                    tensor_environment, tensor_stack)
from finch4.layers import Environment


def _fitness(phenotypes):
    return -(phenotypes.flatten(1) ** 2).mean(dim=1)


ARGS = dict(output_shape=(8,), epochs=6, genes=4, latents=8, children=4,
            population_cap=8, founders=2, device="cpu", seed=21)


def test_observer_layer_sees_waves_and_numbers_do_not_move():
    """A custom layer between stages reads the in-flight Wave; because it
    consumes no randomness, the run stays bit-identical to solve()."""
    class Observer(Layer):
        def __init__(self):
            self.waves = []

        def __call__(self, env):
            self.waves.append(env.state["wave"])

    obs = Observer()
    stack = tensor_stack()
    stack.insert(4, obs)                     # right after score_and_adopt
    env = Environment(stack, name="observed")
    env.state["engine"] = TensorGA(_fitness, **ARGS)
    env.evolve()

    direct = solve(_fitness, **ARGS)
    layered = env.state["result"]
    assert layered.best_fitness == direct.best_fitness
    assert layered.evaluations == direct.evaluations
    assert ([h["mean_score"] for h in layered.history]
            == [h["mean_score"] for h in direct.history])

    assert len(obs.waves) == ARGS["epochs"]
    for wave in obs.waves:
        assert isinstance(wave, Wave)
        assert wave.score is not None        # scored by the time we look
        both = np.sort(np.concatenate([wave.gene_rows, wave.latent_rows]))
        # each child mutated exactly one space
        assert np.array_equal(both, np.arange(ARGS["children"]))


def test_evolving_in_chunks_matches_one_run():
    """generations= is free composition: 2 then 4 epochs equals 6."""
    env = tensor_environment(_fitness, **ARGS)
    env.evolve(generations=2)
    env.evolve(generations=4)
    chunked = env.state["result"]
    direct = solve(_fitness, **ARGS)
    assert chunked.best_fitness == direct.best_fitness
    assert chunked.evaluations == direct.evaluations
    assert len(chunked.history) == len(direct.history)


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        TensorStage("fold")                  # removed 2026-07-30, stays gone


def test_stack_covers_canonical_order():
    assert tuple(layer.stage for layer in tensor_stack()) == TensorGA.STAGES


def test_engine_protocol_tensor_campaign_and_classic():
    """Environment reads engines through best_summary/best_record only —
    no isinstance special-casing — and falls back to plain state for
    engineless (classic) stacks. The Campaign recorder speaks the same
    protocol so agentic runs share the dashboards."""
    env = tensor_environment(_fitness, **ARGS)
    env.evolve()
    assert set(env.best_scores()) == {"fn0"}
    assert env.best_ever.best_fitness == env.state["result"].best_fitness

    from finch4 import Campaign

    campaign = Campaign(["alpha"])
    line = campaign.found("alpha", "seed")
    campaign.report(line["id"], "gain", score=4.0, source="evaluator")
    holder = Environment([], name="campaign-holder")
    holder.state["engine"] = campaign
    assert holder.best_scores() == {"alpha": 4.0}
    assert holder.best_ever["lineage"] == line["id"]

    bare = Environment([], name="classic")
    bare.state["best"] = {"tsp": -1.0}
    bare.state["best_ever"] = {"genome": [0, 1], "fitness": -1.0}
    assert bare.best_scores() == {"tsp": -1.0}
    assert bare.best_ever["fitness"] == -1.0
