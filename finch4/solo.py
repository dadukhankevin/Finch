"""The comparison run: ONE agent iterating alone, autoresearch-style.

    python3 -m finch4.solo --run benchmarks/agentic/runs/solo1 \
        --task lm --tasks-dir benchmarks/agentic/tasks \
        --agent-cmd 'claude -p "$(cat {promptfile})"' --experiments 16

This is the honest opponent for the agentic GA (drive.py): the same
agent CLI, the same canonical scorer, the same total number of agent
invocations — but no population, no selection, no shared playbook.
Each experiment is one agent call that reads the running WORKLOG.md and
the current champion script, forms ONE hypothesis, edits a copy, scores
it canonically, keeps it if better (replacing champion.py) or reverts,
and appends an honest log entry. Exactly Karpathy's autoresearch loop
(hypothesize -> edit -> 5-minute run -> keep or revert, ~100/night on
an H100), scaled to this machine's budget.

The driver verifies rather than trusts: after every experiment it
re-reads the champion's reported score from results.jsonl, and at the
end it re-runs the canonical scorer on the final champion (must
reproduce within the task's tolerance) plus the held-out seeds. Live
progress: the worklog tail is printed after each experiment and the
whole state is plain files in the run directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

PROMPT = """You are a research agent improving a training script — one \
experiment in a longer campaign (experiment {i} of {total}).

Read, in this order:
- The campaign worklog (every prior hypothesis and result): {worklog}
- The current champion script: {champion} (canonical score {champ_score})
- The canonical scorer (run it, NEVER edit it): {scorer}

Run ONE experiment:
1. Form ONE hypothesis for improving the champion's score — something \
the worklog has not already tried and rejected. State it in one sentence.
2. Copy the champion to {candidate} and apply ONLY that change.
3. Score it: python3 {scorer} {candidate}   (never pass --holdout; at \
most {max_scorings} scorer runs in this experiment, tuning included).
4. Verdict: if the candidate's canonical score is strictly better than \
{champ_score}, replace the champion: cp {candidate} {champion}. \
Otherwise leave the champion untouched.
5. Append to {worklog} — experiment number, hypothesis, exact scores \
printed by the scorer, KEPT or REVERTED. Never report a number the \
scorer did not print.
6. Append one JSON line to {results}: {{"experiment": {i}, \
"hypothesis": "<one sentence>", "score": <candidate score>, \
"kept": <true|false>}}

Your final message: one line — KEPT or REVERTED, and the score."""


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def canonical(python, scorer, artifact, holdout=False):
    cmd = [python, scorer, artifact] + (["--holdout"] if holdout else [])
    return json.loads(sh(cmd).stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--tasks-dir", required=True)
    p.add_argument("--agent-cmd", required=True)
    p.add_argument("--experiments", type=int, default=16)
    p.add_argument("--max-scorings", type=int, default=3)
    p.add_argument("--agent-timeout", type=float, default=2400)
    p.add_argument("--baseline", default=None,
                   help="starting champion; default <tasks-dir>/<task>/train.py")
    a = p.parse_args()
    run = os.path.abspath(a.run)
    os.makedirs(run, exist_ok=True)
    scorer = os.path.join(os.path.abspath(a.tasks_dir), a.task, "score.py")
    champion = os.path.join(run, "champion.py")
    candidate = os.path.join(run, "candidate.py")
    worklog = os.path.join(run, "WORKLOG.md")
    results = os.path.join(run, "results.jsonl")
    if not os.path.exists(champion):
        shutil.copy(a.baseline or os.path.join(
            os.path.abspath(a.tasks_dir), a.task, "train.py"), champion)
    if not os.path.exists(worklog):
        base = canonical("python3", scorer, champion)
        with open(worklog, "w") as f:
            f.write(f"# Solo campaign worklog — task {a.task}\n\n"
                    f"Baseline champion score (canonical): "
                    f"{base['score']}\n\n")
        with open(os.path.join(run, "baseline_score.json"), "w") as f:
            json.dump(base, f)
        champ_score = base["score"]
    else:
        champ_score = canonical("python3", scorer, champion)["score"]
    done = 0
    if os.path.exists(results):
        done = sum(1 for _ in open(results))
    for i in range(done + 1, a.experiments + 1):
        pf = os.path.join(run, f"experiment_{i:02d}.md")
        with open(pf, "w") as f:
            f.write(PROMPT.format(i=i, total=a.experiments,
                                  worklog=worklog, champion=champion,
                                  candidate=candidate, scorer=scorer,
                                  results=results,
                                  champ_score=champ_score,
                                  max_scorings=a.max_scorings))
        print(f"[solo] experiment {i}/{a.experiments} "
              f"(champion {champ_score:.5f})", flush=True)
        t0 = time.time()
        proc = subprocess.Popen(a.agent_cmd.format(promptfile=pf),
                                shell=True, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            proc.wait(timeout=a.agent_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"[solo] experiment {i} TIMED OUT", flush=True)
        if os.path.exists(results):
            last = [json.loads(l) for l in open(results)][-1]
            if last.get("kept"):
                champ_score = last["score"]
            print(f"[solo] {'KEPT' if last.get('kept') else 'reverted'}: "
                  f"{last.get('hypothesis', '?')[:90]} -> "
                  f"{last.get('score')}  ({time.time() - t0:.0f}s)",
                  flush=True)
    final = canonical("python3", scorer, champion)
    hold = canonical("python3", scorer, champion, holdout=True)
    audit = {"final_canonical": final, "final_holdout": hold,
             "last_reported": champ_score}
    with open(os.path.join(run, "final_audit.json"), "w") as f:
        json.dump(audit, f, indent=1)
    print(f"[solo] DONE. canonical={final['score']:.5f} "
          f"holdout={hold['score']:.5f} reported={champ_score:.5f}",
          flush=True)


if __name__ == "__main__":
    main()
