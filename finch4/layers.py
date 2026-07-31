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

Finch 3 -> Finch 4 correspondence (phase one):

    Populate            founding (founders per task; assigned angles
                        in agent-space) — engine-owned
    ParentNPoint        the gene_crossover slot (one_point_gene_
                        crossover default) / smart crossover (agentic)
    GaussianMutation    make_gaussian_mutation + win-rate dials /
                        one-change, telephone, masked (agentic)
    SortByFitness       fitness_shares (rank within task, equal slice
                        per task) — engine-owned law
    CapPopulation       population_cap culling, extinction allowed —
                        engine-owned law
    (new in 4)          Consolidate: distillation (tensor) / playbook
                        absorption + rewrite (agentic)
    (new in 4)          the reporting server, the hub, audits, and the
                        two skill tiers (agents inside the loop;
                        agents authoring whole problems)

The selection/cap/shares laws stay ENGINE-owned in phase one — layers
drive the engine rather than reimplementing it — because this
campaign's history says invariants drift when they leave enforced code
(the one-decoder rule drifted repeatedly even inside code). Phase two
may open them into free layers, one at a time, each with a bit-identity
gate. The tensor epoch is wrapped whole for the same reason; its
decomposition into free stages is the known remaining step.
"""
from __future__ import annotations

from .agentic import AgenticGA


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
    their engine in state["engine"]. live=True starts a telemetry-only
    reporting server (the same /progress page and hub registration
    every run gets)."""

    def __init__(self, layers, name="finch", live=False, seed=None):
        import random
        self.layers = list(layers)
        self.name = name
        self.state = {"generation": 0, "evaluations": 0, "history": [],
                      "rng": random.Random(seed)}
        self._live = live
        self._server = None
        self._compiled = False

    def compile(self):
        if self._live and self._server is None:
            import tempfile

            from .serve import serve
            import threading
            run_dir = tempfile.mkdtemp(prefix=f"{self.name}-")
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

    def evolve(self, generations=1):
        if not self._compiled:
            self.compile()
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
        return self

    def best_scores(self):
        engine = self.state.get("engine")
        if isinstance(engine, AgenticGA):
            return {t: b["score"] for t, b in engine.best.items()
                    if b is not None}
        return dict(self.state.get("best", {}))

    @property
    def best_ever(self):
        engine = self.state.get("engine")
        if isinstance(engine, AgenticGA):
            best = {t: b for t, b in engine.best.items() if b is not None}
            if not best:
                return None
            top = max(best.values(), key=lambda b: b["score"])
            return top
        return self.state.get("best_ever")

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


# ------------------------------------------------------- agentic layers


class AskRun(Layer):
    """One breeding wave: ask the engine for jobs and run each through
    the provided runner — any callable(job) -> {variation, score,
    artifact?, contradicts_base?, log?} or None to abandon. The runner
    is where agents live (spawn a CLI, call the Agent tool, or a
    scripted fake in tests); the engine stays the law."""

    def __init__(self, runner):
        self.runner = runner

    def __call__(self, env):
        engine = env.state["engine"]
        for job in engine.ask():
            result = self.runner(job)
            if result is None:
                engine.abandon(job["job_id"])
                continue
            engine.tell(job["job_id"], result["variation"],
                        result["score"],
                        artifact=result.get("artifact"),
                        contradicts_base=result.get(
                            "contradicts_base", False),
                        log=result.get("log"))
            env.state["evaluations"] += 1


class Audit(Layer):
    """Audit-on-influence: run the auditor over each task's unaudited
    best-ever. auditor(record) -> bool (passed)."""

    def __init__(self, auditor=None):
        self.auditor = auditor

    def __call__(self, env):
        if self.auditor is None:
            return
        engine = env.state["engine"]
        for record in engine.consolidation_batch().values():
            full = engine.individuals[record["id"]]
            if not full["audited"]:
                engine.mark_audited(record["id"],
                                    bool(self.auditor(record)))


class Consolidate(Layer):
    """The consolidation event on the engine's cadence: consolidator
    (batch, env) edits the base playbook and returns True to proceed;
    rewriter(survivor, env) -> new variation text or None to stand
    pat."""

    def __init__(self, consolidator, rewriter=None):
        self.consolidator = consolidator
        self.rewriter = rewriter

    def __call__(self, env):
        engine = env.state["engine"]
        if not engine.consolidation_due():
            return
        batch = engine.consolidation_batch()
        if not self.consolidator(batch, env):
            return
        for survivor in engine.record_consolidation():
            if self.rewriter is None:
                continue
            new = self.rewriter(survivor, env)
            if new:
                engine.tell_rewrite(survivor["id"], new)


def agentic_environment(tasks, runner, consolidator=None, rewriter=None,
                        auditor=None, name="agentic", live=False,
                        **engine_kwargs):
    """The agentic substrate as a Finch environment — the exact
    AgenticGA engine driven by layers instead of a bespoke loop
    (bit-identity with the direct drive is tested)."""
    layers = [AskRun(runner)]
    if auditor is not None:
        layers.append(Audit(auditor))
    if consolidator is not None:
        layers.append(Consolidate(consolidator, rewriter))
    env = Environment(layers, name=name, live=live)
    env.state["engine"] = AgenticGA(tasks=tasks, **engine_kwargs)
    return env


# -------------------------------------------------------- tensor preset


class SolveWhole(Layer):
    """The tensor solver as one coarse layer: a call to evolve() runs
    the whole solve() with the stored arguments. Deliberately NOT
    decomposed in phase one — the epoch's stages stay inside the
    proven loop; opening them into free layers is the known next step,
    gated on bit-identity like everything else."""

    def __init__(self, fitness_fns, output_shape, **solve_kwargs):
        self.args = (fitness_fns, output_shape)
        self.kwargs = solve_kwargs

    def __call__(self, env):
        from . import solve
        result = solve(self.args[0], output_shape=self.args[1],
                       **self.kwargs)
        env.state["result"] = result
        env.state["best_ever"] = result.problems[
            max(range(len(result.problems)),
                key=lambda i: result.problems[i].best_fitness)]
        env.state["best"] = {f"fn{i}": p.best_fitness
                             for i, p in enumerate(result.problems)}
        env.state["evaluations"] += result.evaluations


def tensor_environment(fitness_fns, output_shape, name="tensor",
                       live=False, **solve_kwargs):
    """solve() wrapped as a Finch environment. env.evolve() runs one
    full solve; results land in env.state['result'] / env.best_ever."""
    return Environment([SolveWhole(fitness_fns, output_shape,
                                   **solve_kwargs)],
                       name=name, live=live)


__all__ = ["Layer", "Environment", "AskRun", "Audit", "Consolidate",
           "agentic_environment", "tensor_environment", "SolveWhole"]
