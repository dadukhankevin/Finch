"""One autoresearch loop: hypothesize, edit, score, keep or revert.

Standalone, this is the classic solo baseline. Under agent-mediated GAR
it is one member of an evolver's population:

    python3 -m finch4.solo --run <dir> --task binpack \
        --tasks-dir benchmarks/agentic/tasks \
        --agent-cmd 'claude -p "$(cat {promptfile})"' --experiments 8 \
        --decoder <campaign>/Decoder.md --idea "seed clause" \
        --lineage L0003 --serve-port 51234

The worker sees its own fitness and the current Decoder.md. Sealed
evaluation data stays with the driver. The driver restores the previous
champion even if a worker overwrites it, independently scores the
candidate, and alone decides keep or revert.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

from .prompts import (
    DECODER_SECTION,
    EMPTY_DECODER_SECTION,
    IDEA_SECTION,
    RESEARCHER_PROMPT,
)
from .workspace import (
    has_sealed,
    prepare_worker_workspace,
    refresh_decoder,
    task_dir,
)


def sh(cmd, env=None, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)


def canonical(python, scorer, artifact, holdout=False):
    cmd = [python, scorer, artifact]
    env = os.environ.copy()
    if has_sealed(os.path.dirname(scorer)):
        env["FINCH_SEALED_OK"] = "1"
        cmd.append("--holdout" if holdout else "--practice")
    elif holdout:
        cmd.append("--holdout")
    completed = sh(cmd, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"canonical scorer failed ({completed.returncode}): "
            f"{completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


def snapshot_artifact(run, artifact, label):
    """Immutable local identity for every driver-scored candidate."""
    with open(artifact, "rb") as f:
        payload = f.read()
    digest = hashlib.sha256(payload).hexdigest()
    directory = os.path.join(run, "artifacts")
    os.makedirs(directory, exist_ok=True)
    suffix = os.path.splitext(artifact)[1]
    target = os.path.join(directory, f"{label}-{digest[:16]}{suffix}")
    if not os.path.exists(target):
        shutil.copy2(artifact, target)
    return target, digest


def make_reporter(port, lineage):
    """POST /report to the campaign server; None if not streaming."""
    if port is None or lineage is None:
        return None

    def report(summary, score=None, source=None, evidence=None,
               kept=None, claimed_score=None, artifact=None,
               experiment_id=None, sealed_score=None, holdout_score=None):
        body = json.dumps({
            "lineage": lineage, "summary": summary, "score": score,
            "source": source, "evidence": evidence, "kept": kept,
            "claimed_score": claimed_score, "artifact": artifact,
            "experiment_id": experiment_id,
            "sealed_score": sealed_score,
            "holdout_score": holdout_score,
        }).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/report", data=body, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except Exception as error:
            print(f"[solo] report failed: {error!r}", flush=True)
            return None

    return report


def _jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _observe_holdout(python, scorer, artifact):
    if not has_sealed(os.path.dirname(scorer)):
        return None
    return float(canonical(python, scorer, artifact, holdout=True)["score"])


def ingest_claim(run, python, scorer, champion, candidate, claim, experiment,
                 champ_score, report):
    """Score a finished candidate. Practice decides keep. Holdout is
    recorded and posted, never used to pick."""
    scored = canonical(python, scorer, candidate)
    verified = float(scored["score"])
    holdout_score = _observe_holdout(python, scorer, candidate)
    previous = champ_score
    kept = verified > previous
    artifact, digest = snapshot_artifact(
        run, candidate, f"experiment-{experiment:04d}")
    if kept:
        shutil.copy2(artifact, champion)
        champ_score = verified
    extras = {
        key: scored[key]
        for key in (
            "right_on_humans", "right_on_breaks", "accuracy_by_break_type",
        )
        if key in scored
    }
    driver_record = {
        "experiment": experiment,
        "hypothesis": claim.get("hypothesis", f"experiment {experiment}"),
        "score": verified, "previous_score": previous,
        "kept": kept, "artifact": artifact,
        "artifact_sha256": digest,
        "claimed_score": claim.get("claimed_score", claim.get("score")),
        "holdout_score": holdout_score,
    }
    driver_record.update(extras)
    with open(os.path.join(run, "driver_results.jsonl"), "a",
              encoding="utf-8") as handle:
        handle.write(json.dumps(driver_record) + "\n")
    with open(os.path.join(run, "WORKLOG.md"), "a", encoding="utf-8") as handle:
        holdout_bit = ("" if holdout_score is None
                       else f" holdout {holdout_score:.5f} (observation)")
        handle.write(
            f"Driver verdict experiment {experiment}: canonical {verified} "
            f"vs {previous} — {'KEPT' if kept else 'REVERTED'}"
            f"{holdout_bit}\n\n")
    if report:
        evidence = {"previous_score": previous, "artifact_sha256": digest}
        evidence.update(extras)
        report(claim.get("hypothesis", f"experiment {experiment}"),
               score=verified,
               source="canonical scorer (driver rerun)",
               evidence=evidence,
               kept=kept,
               claimed_score=claim.get("claimed_score", claim.get("score")),
               artifact=artifact,
               experiment_id=f"experiment-{experiment:04d}",
               sealed_score=verified,
               holdout_score=holdout_score)
    print(f"[solo] {'KEPT' if kept else 'reverted'}: "
          f"{claim.get('hypothesis', '?')[:90]} -> {verified}",
          flush=True)
    return champ_score


def ingest_leftovers(run, python, scorer, champion, candidate, results_path,
                     driver_path, champ_score, report):
    """A worker result line is a claim, not an ingest. Resume any claim
    that never got a driver verdict, using the current candidate."""
    claims = _jsonl(results_path)
    ingested = {row.get("experiment") for row in _jsonl(driver_path)}
    pending = [row for row in claims
               if row.get("experiment") not in ingested]
    if not pending or not os.path.isfile(candidate):
        done = 0
        if claims:
            done = max(row.get("experiment", 0) for row in claims)
        if ingested:
            done = max(done, max(ingested))
        return champ_score, done
    last = pending[-1]
    experiment = int(last["experiment"])
    skipped = [row.get("experiment") for row in pending[:-1]]
    if skipped:
        print(f"[solo] skipping uningested claims {skipped} "
              f"(candidate is experiment {experiment})", flush=True)
    champ_score = ingest_claim(
        run, python, scorer, champion, candidate, last, experiment,
        champ_score, report)
    ingested.add(experiment)
    done = max(experiment, max(ingested) if ingested else 0)
    if claims:
        done = max(done, max(row.get("experiment", 0) for row in claims))
    return champ_score, done


def render_prompt(i, total, champ_score, decoder_text, idea):
    if decoder_text:
        decoder_section = DECODER_SECTION.format(decoder=decoder_text)
    else:
        decoder_section = EMPTY_DECODER_SECTION
    idea_section = IDEA_SECTION.format(idea=idea) if idea else ""
    return RESEARCHER_PROMPT.format(
        i=i, total=total, champ_score=f"{float(champ_score):.5f}",
        decoder_section=decoder_section, idea_section=idea_section)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--tasks-dir", required=True)
    p.add_argument("--agent-cmd", required=True)
    p.add_argument("--experiments", type=int, default=16)
    p.add_argument("--max-scorings", type=int, default=3)
    p.add_argument("--agent-timeout", type=float, default=2400)
    p.add_argument("--python", default=sys.executable,
                   help="Python used for the canonical scorer")
    p.add_argument("--baseline", default=None,
                   help="starting champion; default <tasks-dir>/<task>/train.py")
    p.add_argument("--decoder", default=None,
                   help="path to the campaign's Decoder.md")
    p.add_argument("--idea", default=None,
                   help="seed angle for this worker")
    p.add_argument("--lineage", default=None,
                   help="campaign lineage id to stream reports as")
    p.add_argument("--serve-port", type=int, default=None,
                   help="campaign server port for report streaming")
    p.add_argument("--final-holdout", action="store_true",
                   help="standalone comparison only: open holdout once at "
                        "the end; forbidden for campaign workers")
    a = p.parse_args()
    if a.final_holdout and (a.lineage is not None or
                            a.serve_port is not None):
        raise SystemExit("campaign evolvers cannot open holdout; the "
                         "independent evaluator owns audit and confirmation")
    run = os.path.abspath(a.run)
    prepare_worker_workspace(run, a.tasks_dir, a.task)
    scorer = str(task_dir(a.tasks_dir, a.task) / "score.py")
    champion = os.path.join(run, "champion.py")
    candidate = os.path.join(run, "candidate.py")
    worklog = os.path.join(run, "WORKLOG.md")
    results = os.path.join(run, "results.jsonl")
    driver_results = os.path.join(run, "driver_results.jsonl")
    report = make_reporter(a.serve_port, a.lineage)
    if not os.path.exists(champion):
        shutil.copy(a.baseline or os.path.join(
            os.path.abspath(a.tasks_dir), a.task, "train.py"), champion)
    if not os.path.exists(worklog):
        base = canonical(a.python, scorer, champion)
        with open(worklog, "w") as f:
            f.write(f"# Solo campaign worklog — task {a.task}\n\n"
                    f"Baseline champion score (canonical): "
                    f"{base['score']}\n\n")
        with open(os.path.join(run, "baseline_score.json"), "w") as f:
            json.dump(base, f)
        champ_score = base["score"]
        if report:
            artifact, _ = snapshot_artifact(run, champion, "baseline")
            report(f"baseline champion scored", score=base["score"],
                   source="canonical scorer (driver)", kept=True,
                   artifact=artifact, experiment_id="baseline")
    else:
        champ_score = canonical(a.python, scorer, champion)["score"]
    champ_score, done = ingest_leftovers(
        run, a.python, scorer, champion, candidate, results,
        driver_results, champ_score, report)
    for i in range(done + 1, a.experiments + 1):
        decoder_text = refresh_decoder(run, a.decoder)
        pf = os.path.join(run, f"experiment_{i:02d}.md")
        with open(pf, "w") as f:
            f.write(render_prompt(i, a.experiments, champ_score,
                                  decoder_text, a.idea))
        print(f"[solo] experiment {i}/{a.experiments} "
              f"(champion {champ_score:.5f})", flush=True)
        t0 = time.time()
        before = os.path.join(run, f"champion_before_{i:02d}.py")
        shutil.copy2(champion, before)
        if os.path.exists(candidate):
            os.remove(candidate)
        result_count = (sum(1 for _ in open(results))
                        if os.path.exists(results) else 0)
        proc = subprocess.Popen(
            a.agent_cmd.format(promptfile=os.path.basename(pf)),
            shell=True, cwd=run, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        try:
            proc.wait(timeout=a.agent_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"[solo] experiment {i} TIMED OUT", flush=True)
        shutil.copy2(before, champion)
        last = None
        if os.path.exists(results):
            lines = [json.loads(l) for l in open(results)]
            new = [line for line in lines[result_count:]
                   if line.get("experiment") == i]
            last = new[-1] if new else None
        if last is None or not os.path.isfile(candidate):
            if report:
                report(f"experiment {i} produced no result line",
                       kept=False)
            continue
        champ_score = ingest_claim(
            run, a.python, scorer, champion, candidate, last, i,
            champ_score, report)
        print(f"[solo] experiment {i} finished in {time.time() - t0:.0f}s",
              flush=True)
    final = canonical(a.python, scorer, champion)
    observed = any(
        row.get("holdout_score") is not None
        for row in _jsonl(driver_results))
    audit = {"final_canonical": final, "last_reported": champ_score,
             "holdout_status": "OBSERVED" if observed else "UNTOUCHED"}
    if a.final_holdout:
        audit["final_holdout"] = canonical(
            a.python, scorer, champion, holdout=True)
        audit["holdout_status"] = "OPENED_ONCE_STANDALONE"
    with open(os.path.join(run, "final_audit.json"), "w") as f:
        json.dump(audit, f, indent=1)
    suffix = (f" holdout={audit['final_holdout']['score']:.5f}"
              if "final_holdout" in audit else " holdout=UNTOUCHED")
    print(f"[solo] DONE. canonical={final['score']:.5f}{suffix} "
          f"reported={champ_score:.5f}", flush=True)


if __name__ == "__main__":
    main()
