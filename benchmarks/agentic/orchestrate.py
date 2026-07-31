"""Orchestrator helper for steady-state agentic runs.

    python3 benchmarks/agentic/orchestrate.py next --run <dir> --port N --n 5

Asks the engine for `--n` fresh jobs (one ask each — steady-state, no
round barrier), renders each job's prompt from the library templates
with the rotating mutation styles and assigned founder angles, and
prints one `job_id<TAB>promptfile` line per job for the orchestrating
agent to spawn. Jobs whose parent is disqualified (score <= -90, e.g.
an exposed cheater) are abandoned instead of rendered. The mutation-
style rotation counter persists in <run>/jobs/seq.txt so restarts don't
reset operator diversity."""
import argparse
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))
from finch4.drive import (FOUNDER_ANGLES, MUTATION_STYLES,
                                         _mask, _template)


def call(port, name, body=None):
    url = f"http://127.0.0.1:{port}/{name}"
    req = (urllib.request.Request(url) if body is None else
           urllib.request.Request(url, data=json.dumps(body).encode(),
                                  method="POST"))
    return json.loads(urllib.request.urlopen(req).read())


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    nxt = sub.add_parser("next")
    nxt.add_argument("--run", required=True)
    nxt.add_argument("--port", type=int, required=True)
    nxt.add_argument("--n", type=int, default=1)
    nxt.add_argument("--task-dir", default=None,
                     help="default: benchmarks/agentic/tasks/<task>")
    a = p.parse_args()
    run = os.path.abspath(a.run)
    seq_path = os.path.join(run, "jobs", "seq.txt")
    seq = int(open(seq_path).read()) if os.path.exists(seq_path) else 0
    for _ in range(a.n):
        for job in call(a.port, "ask", {}):
            parents = job["parents"]
            if any(p_["score"] <= -90 for p_ in parents):
                call(a.port, "abandon", {"job_id": job["job_id"]})
                print(f"{job['job_id']}\tABANDONED(disqualified parent)")
                continue
            task = job["task"]
            task_dir = a.task_dir or os.path.abspath(
                os.path.join("benchmarks", "agentic", "tasks", task))
            fields = dict(job_id=job["job_id"], task=task, port=a.port,
                          run_dir=run,
                          base_playbook=os.path.join(run,
                                                     "base_playbook.md"),
                          scorer=os.path.join(task_dir, "score.py"),
                          out_dir=os.path.join(run, "individuals",
                                               job["job_id"]))
            if job["kind"] == "found":
                fields["angle"] = FOUNDER_ANGLES[seq % len(FOUNDER_ANGLES)]
                tpl = _template("found")
            elif job["kind"] == "crossover":
                pa, pb = parents
                fields.update(variation_a=pa["variation"],
                              score_a=pa["score"],
                              variation_b=pb["variation"],
                              score_b=pb["score"])
                tpl = _template("crossover")
            else:
                parent = parents[0]
                style = MUTATION_STYLES[seq % len(MUTATION_STYLES)]
                fields.update(parent_variation=parent["variation"],
                              parent_score=parent["score"],
                              parent_artifact=parent["artifact"]
                              or "(none)")
                if style == "masked":
                    fields["masked_variation"] = _mask(
                        parent["variation"], random.Random(1000 + seq))
                tpl = _template("mutate_" + style)
            seq += 1
            pf = os.path.join(run, "jobs", job["job_id"] + ".md")
            with open(pf, "w") as f:
                f.write(tpl.format(**fields))
            print(f"{job['job_id']}\t{pf}")
    with open(seq_path, "w") as f:
        f.write(str(seq))


if __name__ == "__main__":
    main()
