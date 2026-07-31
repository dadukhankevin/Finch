"""The agentic substrate, self-driving: one command, evolution happens.

    python3 -m finch4.drive --run benchmarks/agentic/runs/r2 \
        --tasks binpack tsp --tasks-dir benchmarks/agentic/tasks \
        --agent-cmd 'claude -p "$(cat {promptfile})"' --rounds 6

This is the answer to "how does solve() call its first agent" for the
agentic substrate: it doesn't — this driver does. It holds the engine,
serves the reporting API (agents POST /tell themselves the moment they
finish), renders one prompt per job from the templates in
agentic_prompts/, and shells out to the agent CLI per job with the
prompt path in {promptfile} and AGENTIC_PORT / AGENTIC_JOB /
AGENTIC_OUT in the environment.

Operator design credit where due — SCRISPR (Daniel's earlier
prompt-evolution project, github.com/dadukhankevin/SCRISPR) supplies
the mutation vocabulary, rotated per job so no single operator's bias
dominates:

- one-change: differ from the parent in ONE deliberate way
- telephone: read the parent's ARTIFACT, ignore its variation text,
  describe what the code actually does as a fresh clause, then follow
  it (breaks textual anchoring — SCRISPR's telephone mutation)
- masked: the parent's variation with ~25% of its words blanked; fill
  the gaps freshly (SCRISPR's masked mutation)

and crossover is SCRISPR's smart crossover (merge the beneficial
features of both parents). Founding jobs rotate through ASSIGNED
research angles — the round-38 smoke run measured founder proposals
collapsing to one idea per task when agents were merely told "be
distinct", so distinctness is now assigned, not requested.

Audit is mechanical here (re-run the canonical scorer on the shipped
artifact — must reproduce exactly — plus the held-out seeds, written to
audit.json); the legitimacy read of an agent's log stays with humans or
an orchestrating agent. Consolidation defaults to REVIEW: the
consolidator's proposed base edit is saved and the run stops cleanly
for a human decision; pass --auto-consolidate for overnight autonomy
(the proposal and every base version stay on disk either way).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import threading
import time

from .serve import serve

PROMPTS = os.path.join(os.path.dirname(__file__), "agentic_prompts")

FOUNDER_ANGLES = [
    "exploit the exact distribution the scorer's instance generator uses",
    "price the greedy rule's future cost into its present choice",
    "use global state across the whole solution, not per-step locality",
    "make thresholds adapt to live instance state instead of constants",
    "start from a different classical algorithm family than the obvious "
    "first choice",
    "add bounded lookahead or rollout to the decision rule",
]

MUTATION_STYLES = ["one_change", "telephone", "masked"]


def _template(name):
    with open(os.path.join(PROMPTS, name + ".md")) as f:
        return f.read()


def _mask(text, rng, rate=0.25):
    words = text.split()
    out = [("____" if rng.random() < rate and len(w) > 3 else w)
           for w in words]
    return " ".join(out)


class Driver:
    def __init__(self, run_dir, tasks=None, tasks_dir=None, agent_cmd=None,
                 rounds=4, max_parallel=4, agent_timeout=1200,
                 auto_consolidate=False, python="python3", port=0,
                 **ga_kwargs):
        self.run_dir = os.path.abspath(run_dir)
        self.tasks_dir = os.path.abspath(tasks_dir)
        self.agent_cmd = agent_cmd
        self.rounds = int(rounds)
        self.max_parallel = int(max_parallel)
        self.agent_timeout = float(agent_timeout)
        self.auto_consolidate = auto_consolidate
        self.python = python
        os.makedirs(os.path.join(self.run_dir, "jobs"), exist_ok=True)
        base = os.path.join(self.run_dir, "base_playbook.md")
        if not os.path.exists(base):
            shutil.copy(os.path.join(self.tasks_dir, os.pardir,
                                     "base_playbook.md"), base)
        self.server = serve(self.run_dir, port=port, tasks=tasks,
                            **ga_kwargs)
        self.port = self.server.server_address[1]
        self._log(f"live progress: http://127.0.0.1:{self.port}/progress")
        threading.Thread(target=self.server.serve_forever,
                         daemon=True).start()
        self._job_seq = 0
        self.rng = random.Random(ga_kwargs.get("seed"))
        # Finch ethos: the round is an assembly of stages, not a script.
        # Reorder, remove, or append callables to change the algorithm.
        self.stages = [self.stage_breed, self.stage_audit,
                       self.stage_consolidate]
        self._log(f"LIVE PROGRESS: http://127.0.0.1:{self.port}/progress")

    # ------------------------------------------------------------ plumbing

    def _call(self, name, body=None):
        import urllib.request
        url = f"http://127.0.0.1:{self.port}/{name}"
        req = (urllib.request.Request(url) if body is None else
               urllib.request.Request(url, data=json.dumps(body).encode(),
                                      method="POST"))
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def _log(self, msg):
        print(f"[drive] {msg}", flush=True)

    def _score_cmd(self, task, artifact, holdout=False):
        cmd = [self.python, os.path.join(self.tasks_dir, task, "score.py"),
               artifact]
        return cmd + (["--holdout"] if holdout else [])

    # ------------------------------------------------------------- prompts

    def _render(self, job):
        task = job["task"]
        fields = {
            "job_id": job["job_id"], "task": task, "port": self.port,
            "run_dir": self.run_dir,
            "base_playbook": os.path.join(self.run_dir, "base_playbook.md"),
            "scorer": os.path.join(self.tasks_dir, task, "score.py"),
            "out_dir": os.path.join(self.run_dir, "individuals",
                                    job["job_id"]),
        }
        if job["kind"] == "found":
            fields["angle"] = FOUNDER_ANGLES[self._job_seq
                                             % len(FOUNDER_ANGLES)]
            text = _template("found")
        elif job["kind"] == "crossover":
            a, b = job["parents"]
            fields.update(variation_a=a["variation"], score_a=a["score"],
                          variation_b=b["variation"], score_b=b["score"])
            text = _template("crossover")
        else:
            parent = job["parents"][0]
            style = MUTATION_STYLES[self._job_seq % len(MUTATION_STYLES)]
            fields.update(parent_variation=parent["variation"],
                          parent_score=parent["score"],
                          parent_artifact=parent["artifact"] or "(none)")
            if style == "masked":
                fields["masked_variation"] = _mask(parent["variation"],
                                                   self.rng)
            text = _template("mutate_" + style)
        self._job_seq += 1
        path = os.path.join(self.run_dir, "jobs", job["job_id"] + ".md")
        with open(path, "w") as f:
            f.write(text.format(**fields))
        return path

    # --------------------------------------------------------------- waves

    def _spawn(self, promptfile, job=None, extra_env=None):
        env = dict(os.environ, AGENTIC_PORT=str(self.port),
                   AGENTIC_PROMPT=promptfile,
                   AGENTIC_RUN=self.run_dir,
                   AGENTIC_TASKS_DIR=self.tasks_dir)
        if job is not None:
            env["AGENTIC_JOB"] = json.dumps(job)
            env["AGENTIC_OUT"] = os.path.join(self.run_dir, "individuals",
                                              job["job_id"])
            os.makedirs(env["AGENTIC_OUT"], exist_ok=True)
        if extra_env:
            env.update(extra_env)
        cmd = self.agent_cmd.format(promptfile=promptfile)
        return subprocess.Popen(cmd, shell=True, env=env,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)

    def _run_wave(self, jobs):
        """Launch one agent per job, at most max_parallel at once."""
        pending = [(j, self._render(j)) for j in jobs]
        procs, deadline = [], time.time() + self.agent_timeout
        while pending or procs:
            while pending and len(procs) < self.max_parallel:
                job, pf = pending.pop(0)
                procs.append((job, self._spawn(pf, job)))
                self._log(f"agent up: {job['job_id']} {job['kind']} "
                          f"{job['task']}")
            procs = [(j, p) for j, p in procs if p.poll() is None]
            if time.time() > deadline:
                for j, p in procs:
                    p.kill()
                    self._log(f"TIMEOUT {j['job_id']}")
                break
            time.sleep(0.3)
        # anything the agents failed to report is abandoned
        open_jobs = self._call("summary")["open_jobs"]
        for job in jobs:
            try:
                self._call("abandon", {"job_id": job["job_id"]})
            except Exception:
                pass
        if open_jobs:
            self._log(f"abandoned {open_jobs} unreported job(s)")

    # --------------------------------------------------------------- audit

    def _audit(self):
        """Mechanical audit of every unaudited per-task best-ever: the
        canonical score must REPRODUCE exactly; holdout is recorded."""
        for task, best in self._call("batch").items():
            if not best["artifact"]:
                self._call("audit", {"id": best["id"], "passed": False})
                self._log(f"audit {task} {best['id']}: NO ARTIFACT — fail")
                continue
            aud_path = os.path.join(os.path.dirname(best["artifact"]),
                                    "audit.json")
            if os.path.exists(aud_path):
                continue
            try:
                rerun = json.loads(subprocess.run(
                    self._score_cmd(task, best["artifact"]),
                    capture_output=True, text=True).stdout)
                holdout = json.loads(subprocess.run(
                    self._score_cmd(task, best["artifact"], holdout=True),
                    capture_output=True, text=True).stdout)
            except (json.JSONDecodeError, OSError) as e:
                self._call("audit", {"id": best["id"], "passed": False})
                self._log(f"audit {task} {best['id']}: scorer failed "
                          f"({e!r}) — fail")
                continue
            # each task's scorer declares its own reproduction tolerance
            # (deterministic tasks omit it -> exact; the lm task trains
            # on MPS, which is not bit-deterministic run to run)
            passed = (abs(rerun["score"] - best["score"])
                      <= rerun.get("tolerance", 1e-9))
            with open(aud_path, "w") as f:
                json.dump({"reported": best["score"],
                           "reproduced": rerun["score"],
                           "holdout": holdout["score"],
                           "passed": passed}, f, indent=1)
            self._call("audit", {"id": best["id"], "passed": passed})
            self._log(f"audit {task} {best['id']}: reproduce="
                      f"{'OK' if passed else 'FAIL'} "
                      f"holdout={holdout['score']:.4f}")
            if not passed:
                self._log(f"AUDIT FAILURE on {best['id']} — reported "
                          f"{best['score']} but scorer says "
                          f"{rerun['score']}; excluded from consolidation")

    # ------------------------------------------------------- consolidation

    def _consolidate(self):
        version = self._call("summary")["base_version"] + 1
        proposal = os.path.join(self.run_dir, f"proposed_base_v{version}.md")
        batch = {t: b for t, b in self._call("batch").items()}
        audits = {}
        for t, b in batch.items():
            ap = os.path.join(os.path.dirname(b["artifact"]), "audit.json")
            if os.path.exists(ap):
                audits[t] = json.load(open(ap))
        ctx = os.path.join(self.run_dir, "jobs", f"consolidate_v{version}.json")
        with open(ctx, "w") as f:
            json.dump({"batch": batch, "audits": audits}, f, indent=1)
        pf = os.path.join(self.run_dir, "jobs", f"consolidate_v{version}.md")
        with open(pf, "w") as f:
            f.write(_template("consolidate").format(
                base_playbook=os.path.join(self.run_dir, "base_playbook.md"),
                context=ctx, proposal=proposal, version=version))
        proc = self._spawn(pf, extra_env={"AGENTIC_PROPOSAL": proposal,
                                          "AGENTIC_CONTEXT": ctx})
        proc.wait(timeout=self.agent_timeout)
        if not os.path.exists(proposal):
            self._log("consolidator produced no proposal; skipping")
            return
        if not self.auto_consolidate:
            self._log(f"consolidation proposal saved: {proposal}")
            self._log("review it, apply to base_playbook.md, then POST "
                      f"/consolidated on port {self.port} — or rerun with "
                      "--auto-consolidate. Stopping here.")
            raise SystemExit(0)
        shutil.copy(proposal, os.path.join(self.run_dir, "base_playbook.md"))
        survivors = self._call("consolidated", {})
        self._log(f"base -> v{version}; {len(survivors)} rewrites owed")
        sv = os.path.join(self.run_dir, "jobs", f"survivors_v{version}.json")
        with open(sv, "w") as f:
            json.dump(survivors, f, indent=1)
        pf = os.path.join(self.run_dir, "jobs", f"rewrite_v{version}.md")
        with open(pf, "w") as f:
            f.write(_template("rewrite").format(
                base_playbook=os.path.join(self.run_dir, "base_playbook.md"),
                survivors=sv, port=self.port))
        self._spawn(pf, extra_env={"AGENTIC_SURVIVORS": sv}).wait(
            timeout=self.agent_timeout)

    # ------------------------------------------------- stages (composable)

    def stage_breed(self, r):
        jobs = self._call("ask", {})
        self._log(f"round {r + 1}/{self.rounds}: {len(jobs)} jobs")
        self._run_wave(jobs)

    def stage_audit(self, r):
        self._audit()
        summary = self._call("summary")
        self._log(f"population={summary['population']} "
                  f"best={summary['best']}")

    def stage_consolidate(self, r):
        if self._call("due")["due"]:
            self._consolidate()

    # ---------------------------------------------------------------- loop

    def run(self):
        for r in range(self.rounds):
            for stage in self.stages:
                stage(r)
        self._log(f"done: {self._call('summary')}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--tasks-dir", required=True)
    p.add_argument("--agent-cmd", required=True,
                   help="shell template; {promptfile} is substituted, "
                        "AGENTIC_PORT/AGENTIC_JOB/AGENTIC_OUT are in env")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--agent-timeout", type=float, default=1200)
    p.add_argument("--auto-consolidate", action="store_true")
    p.add_argument("--founders", type=int, default=2)
    p.add_argument("--children", type=int, default=4)
    p.add_argument("--population-cap", type=int, default=12)
    p.add_argument("--consolidate-every", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--port", type=int, default=0,
                   help="fixed port for the live /progress page")
    a = p.parse_args()
    Driver(a.run, tasks=a.tasks, tasks_dir=a.tasks_dir,
           agent_cmd=a.agent_cmd, rounds=a.rounds,
           max_parallel=a.max_parallel, agent_timeout=a.agent_timeout,
           auto_consolidate=a.auto_consolidate, port=a.port,
           founders=a.founders,
           children=a.children, population_cap=a.population_cap,
           consolidate_every=a.consolidate_every, seed=a.seed).run()


if __name__ == "__main__":
    main()
