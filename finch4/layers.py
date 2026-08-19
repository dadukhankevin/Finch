"""Finch 4 — evolution as composable layers, with everything this
library already vetted living inside it EXACTLY as it was.

Finch's grammar (github.com/dadukhankevin/Finch): an Environment is a
stack of Layers executed in sequence over a population of typed
individuals — floats, strings, lists, arbitrary objects. Same surface
here, same method names:

    env = Environment([...layers...], name="my-run")
    env.compile()
    env.evolve(generations=100)
    env.best_ever
    env.plot()

What the research campaign behind this library (github.com/dadukhankevin/latentspace) contributes is not a competing grammar but the vetted
sentences: the tensor solver and the agentic substrate enter as
canonical presets whose evidence CARRIES because the code is identical
— enforced by seeded bit-identity tests (tests/test_finch.py), the
same acceptance rule that let the fold-removal rewrite inherit the old
engine's records. Recompositions beyond the presets are NEW mechanisms
and get new ledger rows; the presets are the defaults that earned
theirs.

Finch 3 -> Finch 4 correspondence (tensor wing):

    Populate            founding (founders per function) — engine-owned
    ParentNPoint        the gene_crossover slot (one_point_gene_
                        crossover default)
    GaussianMutation    make_gaussian_mutation + win-rate dials
    SortByFitness       fitness_shares (rank within function, equal
                        slice per function) — engine-owned law
    CapPopulation       population_cap culling, extinction allowed —
                        engine-owned law
    (new in 4)          Distill: decoder absorption on the
                        consolidation cadence
    (new in 4)          the reporting server, the hub, and media

The TENSOR engine owns its laws (selection, fitness shares, capping,
archives — this campaign's history says invariants drift when they
leave enforced code; the one-decoder rule drifted repeatedly even
inside code), and layers drive the engine one stage at a time. The
epoch's stages are the engine methods in TensorGA.STAGES; each
TensorStage layer runs one of them, and the default stack
(tensor_stack) runs them in the canonical order — the same order
solve() runs directly, one source of truth, so the two paths are
bit-identical by construction. Reordering, omitting, or interleaving
stages is expressible and runs fine; per the ledger rule it is a NEW
mechanism and inherits no evidence.

The AGENTIC wing has no layer adapter. Autonomous evolvers communicate through
finch4.agentic.Campaign, and named judge agents decide exact pairwise matches;
Finch records their verdicts and updates Elo mechanically. It is driven over
HTTP (finch4.serve), not by layers.

Engines plug into Environment through a small duck-typed protocol —
any object in state["engine"] may provide:

    best_summary()        {name: best score so far}         (required)
    best_record()         the best-ever record object       (required)
    result()              a final result object; stored in
                          state["result"] when evolve() ends (optional)
    default_generations   what one bare evolve() means      (optional)
"""
from __future__ import annotations

import os
import time

from .ga import TensorGA


class Layer:
    """One stage of an evolutionary step. Subclasses implement
    __call__(env); compile(env) is optional one-time setup."""

    def compile(self, env):
        pass

    def __call__(self, env):
        raise NotImplementedError

    @property
    def name(self):
        return type(self).__name__


class Environment:
    """Finch's surface: a named stack of layers over shared state.

    state is a plain dict every layer reads and writes; presets put
    their engine in state["engine"] (see the module docstring for the
    small protocol an engine may implement). live=True starts a
    telemetry-only reporting server (the same /progress page and hub
    registration every run gets); its run directory lives under
    ~/.finch4/runs/ — deliberately persistent, so the hub keeps showing
    the run after it finishes."""

    def __init__(self, layers, name="finch", live=False, seed=None,
                 run_dir=None):
        import random
        self.layers = list(layers)
        self.name = name
        self.state = {"generation": 0, "evaluations": 0, "history": [],
                      "rng": random.Random(seed)}
        self.url = None
        self._live = live
        self._run_dir = run_dir
        self._server = None
        self._compiled = False

    def compile(self):
        if self._live and self._server is None:
            import threading

            from .serve import serve
            run_dir = self._run_dir or os.path.expanduser(
                f"~/.finch4/runs/{self.name}-{os.getpid()}-"
                f"{int(time.time())}")
            self._server = serve(run_dir, port=0, telemetry_only=True)
            threading.Thread(target=self._server.serve_forever,
                             daemon=True).start()
            self.url = (f"http://127.0.0.1:"
                        f"{self._server.server_address[1]}/progress")
            print(f"[finch] {self.name}: {self.url}", flush=True)
        for layer in self.layers:
            layer.compile(self)
        self._compiled = True
        return self

    def evolve(self, generations=None):
        """Run the stack. generations=None means the engine's own idea
        of one full run (TensorGA: its epochs; anything else: 1)."""
        if not self._compiled:
            self.compile()
        engine = self.state.get("engine")
        if generations is None:
            generations = getattr(engine, "default_generations", 1)
        for _ in range(int(generations)):
            self.state["generation"] += 1
            for layer in self.layers:
                layer(self)
            best = self.best_scores()
            self.state["history"].append(
                {"generation": self.state["generation"],
                 "evaluations": self.state["evaluations"],
                 "best": best})
            if self._server is not None:
                self._server.service.handle("telemetry", {
                    "epoch": self.state["generation"],
                    "evaluations": self.state["evaluations"],
                    "best": best})
        if engine is not None and hasattr(engine, "result"):
            self.state["result"] = engine.result()
        return self

    def best_scores(self):
        engine = self.state.get("engine")
        if engine is not None and hasattr(engine, "best_summary"):
            return engine.best_summary()
        return dict(self.state.get("best", {}))

    @property
    def best_ever(self):
        engine = self.state.get("engine")
        if engine is not None and hasattr(engine, "best_record"):
            return engine.best_record()
        return self.state.get("best_ever")

    def report_media(self, name, image=None, svg=None, text=None):
        """Post media to this run's live dashboard (live=True; a quiet
        no-op otherwise, so layers can call it unconditionally). Exactly
        one of image= (numpy array), svg= or text=. Telemetry for the
        eyes, never evidence."""
        if self._server is None:
            return
        from .serve import _media_body
        self._server.service.handle("media", _media_body(
            name, image=image, svg=svg, text=text,
            epoch=self.state["generation"],
            evaluations=self.state["evaluations"]))

    def plot(self, path=None):
        """Write the run's fitness-over-time chart; returns the file
        path (and the live URL is always self.url when live=True)."""
        from .serve import curve_svg, telemetry_curves
        series = telemetry_curves(self.state["history"])
        svg = curve_svg(series, xlabel="evaluations")
        path = path or f"{self.name}_fitness.svg"
        with open(path, "w") as f:
            f.write(svg)
        return path


# -------------------------------------------------------- tensor layers


class TensorStage(Layer):
    """One stage of the tensor epoch: calls the engine method named
    `stage` (any name in TensorGA.STAGES), then mirrors the engine's
    in-flight Wave and evaluation count into env.state so neighboring
    custom layers can observe or amend them. The laws stay inside the
    engine; a layer chooses WHEN a stage runs, never HOW it works."""

    def __init__(self, stage):
        if stage not in TensorGA.STAGES:
            raise ValueError(f"unknown tensor stage {stage!r}; "
                             f"stages: {TensorGA.STAGES}")
        self.stage = stage

    def __call__(self, env):
        engine = env.state["engine"]
        getattr(engine, self.stage)()
        env.state["wave"] = engine.wave
        env.state["evaluations"] = engine.spent

    @property
    def name(self):
        return f"TensorStage({self.stage})"


class Immigrate(TensorStage):
    """Fresh random draws for stalled or extinct functions
    (immigrants="stall"; no-op otherwise)."""
    def __init__(self):
        super().__init__("immigrate")


class BreedWave(TensorStage):
    """Share-proportional selection, gene crossover, whole-latent
    inheritance -> a new Wave of children."""
    def __init__(self):
        super().__init__("breed")


class MutateWave(TensorStage):
    """Perturb each child in exactly one space, on that space's
    self-tuning dial."""
    def __init__(self):
        super().__init__("mutate")


class ScoreWave(TensorStage):
    """Score the wave; mixed-parent children adopt the function they
    beat their parent on by more."""
    def __init__(self):
        super().__init__("score_and_adopt")


class TuneDials(TensorStage):
    """Update the per-space mutation dials (and the round-50 memory)
    from the wave's successes."""
    def __init__(self):
        super().__init__("tune_dials")


class CullByShares(TensorStage):
    """Merge the wave into the population and cull to the cap by
    fitness share — extinction allowed."""
    def __init__(self):
        super().__init__("integrate")


class Speciate(TensorStage):
    """Move individuals between functions (honest re-score on arrival);
    no-op unless a speciation operator is configured."""
    def __init__(self):
        super().__init__("speciate")


class Distill(TensorStage):
    """Consolidation on its cadence: distill best-evers into the base
    decoder, decay every bending, re-score honestly."""
    def __init__(self):
        super().__init__("consolidate")


class EvolveDirections(TensorStage):
    """Trial a perturbation of the shared direction vocabulary
    ((1+1)-ES; "evolve" substrate only, no-op otherwise)."""
    def __init__(self):
        super().__init__("evolve_directions")


class RecordProgress(TensorStage):
    """Progress callback and the engine's own history; closes the
    epoch."""
    def __init__(self):
        super().__init__("record")


def tensor_stack():
    """The canonical tensor epoch as a fresh list of layers — one per
    stage of TensorGA.STAGES, in the engine's own order. This is the
    stack solve() is sugar for; edit a copy, not the canon."""
    return [TensorStage(stage) for stage in TensorGA.STAGES]


def tensor_environment(fitness_fns, output_shape, name="tensor",
                       live=False, **solve_kwargs):
    """The tensor engine as a Finch environment: env.evolve() runs the
    engine's full epoch budget through the canonical layer stack (or
    pass generations= to run fewer); the final GAResult lands in
    env.state["result"] and env.best_ever is the best ProblemResult.
    Construction founds and scores the initial population, exactly as
    solve() does."""
    env = Environment(tensor_stack(), name=name, live=live)
    env.state["engine"] = TensorGA(fitness_fns, output_shape,
                                   **solve_kwargs)
    return env


__all__ = ["Layer", "Environment", "tensor_environment", "TensorStage",
           "tensor_stack", "Immigrate", "BreedWave", "MutateWave",
           "ScoreWave", "TuneDials", "CullByShares", "Speciate",
           "Distill", "EvolveDirections", "RecordProgress"]
