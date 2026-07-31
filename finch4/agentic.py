"""The agentic substrate's engine: the same laws as solve(), ask/tell form.

The tensor loop in ga.py owns evaluation (decoder forward passes are
cheap, so the loop calls them inline). This engine is for substrates where
evaluation is EXPENSIVE and EXTERNAL — the first being the agentic
decoder, where each individual is a text methodology executed by a
full agent run. The engine therefore treats genes as opaque payloads
(here: the per-individual VARIATION, the "methodology BUT <difference>"
clause appended to one shared base playbook) and never evaluates
anything itself. An orchestrator drives it:

    ga = AgenticGA(tasks=["binpack", "tsp"], seed=0)
    for job in ga.ask():            # {kind: found|mutate|crossover, ...}
        ...run an agent, get (variation, score, artifact)...
        ga.tell(job["job_id"], variation, score, artifact=path)
    if ga.consolidation_due():
        batch = ga.consolidation_batch()   # per-task best-evers
        ...consolidator agent edits the base playbook...
        for ind in ga.record_consolidation():
            ...rewrite agent: push the absorbed idea further, or keep...
            ga.tell_rewrite(ind["id"], new_variation)

What carries over from the tensor loop unchanged, because it is
substrate-free law, not implementation:

- fitness SHARES (imported from ga.py): each living task owns an equal
  slice of the total fitness mass; members split their slice by
  within-task rank. No task can be outcompeted out of existence.
- species breeding: parent one is share-proportional; a partner comes
  from the same task except a rare outcross.
- capped population culled by shares, extinction allowed, extinct tasks
  refounded on the next ask (the immigrant rule).
- best-ever archive per task as bookkeeping AND as the only input to
  consolidation — consolidation absorbs vetted winners, never "the top N
  ideas" (the arithmetic fold collected two weeks of unearned credit by
  exactly that kind of plausible merging; see FINDINGS.md).

What is deliberately DIFFERENT from the tensor loop:

- No automatic re-score after consolidation (Daniel's call, 2026-07-30:
  "i don't think we need to rescore, or only do it as needed"). Instead
  every individual records which base version its score was earned on;
  `stale` individuals are visible and the orchestrator MAY re-score via
  retell_score. Honesty is preserved by bookkeeping instead of budget.
- Rewrites after consolidation don't shrink toward zero (a zero
  variation is a clone of the base — in text, decay-to-nothing collapses
  the population's variance). The rewrite rule is Daniel's: take the
  absorbed part of your idea to the next level, or remain unchanged if
  consolidation didn't include it.
- Variations may CONTRADICT the base only through mutation (Daniel's
  rule). The engine records the flag; contradiction_report() shows
  whether contradictors are beating compliants — the one signal that a
  past consolidation wrote something wrong into the base and the base
  needs correcting.

ONE base playbook, ever (the one-decoder invariant). The base's content
lives on disk with the run; the engine tracks only its version number.
"""
from __future__ import annotations

import json
import random

import numpy as np

from .ga import fitness_shares


class AgenticGA:
    """Ask/tell population engine over opaque genes with external scoring.

    Scores must be HIGHER-better and raw-comparable only within one task
    (shares handle cross-task fairness, exactly as in solve())."""

    def __init__(self, tasks, population_cap=12, children=4, founders=2,
                 crossover_rate=0.1, outcross_rate=0.05,
                 consolidate_every=3, seed=None):
        if isinstance(tasks, str):
            tasks = [tasks]
        self.tasks = list(tasks)
        self.population_cap = int(population_cap)
        self.children = int(children)
        self.founders = int(founders)
        self.crossover_rate = float(crossover_rate)
        self.outcross_rate = float(outcross_rate)
        self.consolidate_every = int(consolidate_every)
        self.seed = seed
        self.rng = random.Random(seed)
        self.individuals = {}          # id -> record (living and dead)
        self.best = {t: None for t in self.tasks}   # per-task best-ever
        self.base_version = 0
        self.round = 0                 # ask() calls served
        self.last_consolidation = 0    # round of the last consolidation
        self.consolidations = 0
        self._next_id = 0
        self._next_job = 0
        self._open_jobs = {}           # job_id -> job dict
        self._founded = False

    # ------------------------------------------------------------ helpers

    def _living(self):
        return [i for i in self.individuals.values() if i["alive"]]

    def _weights(self, living):
        scores = np.asarray([i["score"] for i in living], dtype=np.float64)
        fn_idx = np.asarray([self.tasks.index(i["task"]) for i in living])
        return fitness_shares(scores, fn_idx)

    def _job(self, kind, task, parents=()):
        job_id = f"j{self._next_job:04d}"
        self._next_job += 1
        job = {"job_id": job_id, "kind": kind, "task": task,
               "parents": [self._public(p) for p in parents],
               "base_version": self.base_version}
        self._open_jobs[job_id] = job
        return job

    def _public(self, ind):
        return {k: ind[k] for k in
                ("id", "task", "variation", "score", "artifact",
                 "contradicts_base")}

    # ---------------------------------------------------------------- ask

    def ask(self):
        """Jobs for one round. The first call founds every task
        (founders each — the founding count is the run's coverage; see
        the sixteen-founders finding). Later calls breed `children` via
        share-proportional species selection, refounding extinct tasks
        first. Every job must be answered by tell() or abandon()."""
        self.round += 1
        if not self._founded:
            self._founded = True
            return [self._job("found", t)
                    for t in self.tasks for _ in range(self.founders)]
        jobs = []
        living = self._living()
        # a lineage its own agent declared exhausted stays in the
        # population (shares, cap, archive) but no longer breeds — the
        # agent's dead-end verdict is information (Daniel, 2026-07-30:
        # "if they can't take their idea any further, try something
        # else... killing the last one is fine cause it's now part of
        # the base"). A task whose every member is exhausted is
        # refounded like an extinct one.
        breedable = [i for i in living if not i.get("exhausted")]
        alive_tasks = {i["task"] for i in breedable}
        for t in self.tasks:
            if t not in alive_tasks:
                jobs.append(self._job("found", t))
        if breedable:
            weights = self._weights(breedable)
            while len(jobs) < self.children:
                a = self.rng.choices(breedable, weights=weights)[0]
                kin = [i for i in breedable
                       if i["task"] == a["task"] and i["id"] != a["id"]]
                pool = ([i for i in breedable if i["id"] != a["id"]]
                        if (self.rng.random() < self.outcross_rate
                            or not kin) else kin)
                if pool and self.rng.random() < self.crossover_rate:
                    kw = [max(w, 1e-12) for i, w in zip(breedable, weights)
                          if any(i["id"] == p["id"] for p in pool)]
                    b = self.rng.choices(pool, weights=kw)[0]
                    jobs.append(self._job("crossover", a["task"], (a, b)))
                else:
                    jobs.append(self._job("mutate", a["task"], (a,)))
        return jobs

    # --------------------------------------------------------------- tell

    def tell(self, job_id, variation, score, artifact=None,
             contradicts_base=False, log=None, fresh_start=False):
        """Report a finished job: the child's variation text, its score
        from the CANONICAL fitness script, and the artifact path that
        makes the score reproducible. Returns the new individual's id.
        Culls to the population cap by shares (extinction allowed).

        fresh_start=True is the agent's lineage-exhaustion verdict: it
        was given a parent to build on, judged the idea spent, and
        founded fresh instead. The new individual's origin is "refound"
        and every parent of the job is marked exhausted (it stops
        breeding; its best work survives in the archive and, if
        consolidated, in the base)."""
        job = self._open_jobs.pop(job_id)
        ind_id = f"i{self._next_id:04d}"
        self._next_id += 1
        origin = job["kind"]
        if fresh_start and job["kind"] != "found":
            origin = "refound"
            for parent in job["parents"]:
                self.individuals[parent["id"]]["exhausted"] = True
        record = {
            "id": ind_id, "task": job["task"], "variation": variation,
            "score": float(score), "artifact": artifact, "log": log,
            "origin": origin,
            "parents": [p["id"] for p in job["parents"]],
            "born_round": self.round,
            "scored_on_base": self.base_version,
            "contradicts_base": bool(contradicts_base),
            "audited": False, "alive": True,
        }
        self.individuals[ind_id] = record
        best = self.best[job["task"]]
        if best is None or record["score"] > best["score"]:
            self.best[job["task"]] = dict(record)
        self._cull()
        return ind_id

    def abandon(self, job_id):
        """Drop a job whose agent failed; nothing enters the population."""
        self._open_jobs.pop(job_id, None)

    def _cull(self):
        living = self._living()
        if len(living) <= self.population_cap:
            return
        weights = self._weights(living)
        order = np.argsort(-weights)
        for idx in order[self.population_cap:]:
            living[int(idx)]["alive"] = False

    # ------------------------------------------------------ consolidation

    def consolidation_due(self):
        return (any(b is not None for b in self.best.values())
                and self.round - self.last_consolidation
                >= self.consolidate_every)

    def consolidation_batch(self):
        """The ONLY sanctioned input to consolidation: each task's
        best-ever (its vetted winner), not the population's top N."""
        return {t: self._public(b) for t, b in self.best.items()
                if b is not None}

    def record_consolidation(self):
        """Call after the base playbook has been edited. Bumps the base
        version (all existing scores become visibly stale) and returns
        the living individuals, each of which owes a rewrite: push the
        absorbed part of its idea further, or stay unchanged."""
        self.base_version += 1
        self.consolidations += 1
        self.last_consolidation = self.round
        return [self._public(i) for i in self._living()]

    def tell_rewrite(self, ind_id, variation, contradicts_base=None):
        """Replace an individual's variation after consolidation. Its
        score is kept (Daniel: no automatic re-score) but remains marked
        stale until retell_score."""
        ind = self.individuals[ind_id]
        ind["variation"] = variation
        ind["origin"] = "rewrite"
        if contradicts_base is not None:
            ind["contradicts_base"] = bool(contradicts_base)

    def retell_score(self, ind_id, score, artifact=None):
        """Re-score as needed: refreshes the score and its base version.
        If the re-scored individual currently HOLDS the best-ever
        archive entry, the archive is rebuilt from all truthful scores —
        a downward correction (an audit exposing a false score) must be
        able to evict it, or consolidation would absorb a fraud (the
        compress run's embedded-slice cheater sat at 0.0 bpb in the
        archive after its correction until this path existed)."""
        ind = self.individuals[ind_id]
        ind["score"] = float(score)
        ind["scored_on_base"] = self.base_version
        if artifact is not None:
            ind["artifact"] = artifact
        best = self.best[ind["task"]]
        if best is not None and best["id"] == ind_id:
            members = [i for i in self.individuals.values()
                       if i["task"] == ind["task"]]
            self.best[ind["task"]] = dict(max(members,
                                              key=lambda i: i["score"]))
        elif best is None or ind["score"] > best["score"]:
            self.best[ind["task"]] = dict(ind)

    def mark_audited(self, ind_id, passed=True):
        self.individuals[ind_id]["audited"] = bool(passed)

    # ---------------------------------------------------------- reporting

    def stale(self):
        return [self._public(i) for i in self._living()
                if i["scored_on_base"] < self.base_version]

    def contradiction_report(self):
        """Mean score of base-contradicting vs base-compliant living
        individuals, per task. Contradictors consistently winning is the
        signal that a past consolidation wrote something wrong into the
        base."""
        out = {}
        for t in self.tasks:
            members = [i for i in self._living() if i["task"] == t]
            yes = [i["score"] for i in members if i["contradicts_base"]]
            no = [i["score"] for i in members if not i["contradicts_base"]]
            out[t] = {"contradicting": (float(np.mean(yes)) if yes else None),
                      "compliant": (float(np.mean(no)) if no else None)}
        return out

    def summary(self):
        living = self._living()
        return {
            "round": self.round, "base_version": self.base_version,
            "population": len(living),
            "tasks_alive": sorted({i["task"] for i in living}),
            "best": {t: (None if b is None else b["score"])
                     for t, b in self.best.items()},
            "stale": len(self.stale()),
            "open_jobs": len(self._open_jobs),
        }

    # -------------------------------------------------------- persistence

    def save(self, path):
        state = {k: v for k, v in self.__dict__.items()
                 if k not in ("rng", "_open_jobs")}
        state["rng_state"] = self.rng.getstate()
        state["_open_jobs"] = self._open_jobs
        with open(path, "w") as f:
            json.dump(state, f, indent=1, default=str)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            state = json.load(f)
        ga = cls(tasks=state["tasks"])
        rng_state = state.pop("rng_state")
        for k, v in state.items():
            setattr(ga, k, v)
        ga.rng = random.Random()
        ga.rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        return ga
