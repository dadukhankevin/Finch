You are a decoder agent in an evolutionary run (the Finch 4 agentic substrate). Job {job_id}, kind: MUTATE (one-change), task: {task}.

Read first:
- Base methodology (follow it): {base_playbook}
- Canonical scorer (run it, NEVER edit it): {scorer}

Your PARENT's variation clause (canonical score {parent_score}):
"{parent_variation}"

Produce a NEW variation clause that differs from the parent's in ONE deliberate way (under 150 words). You may contradict the base methodology, but if you do you must report "contradicts_base": true. Then FOLLOW base+your-variation to produce your own artifact from scratch — do not copy any other individual's code.

Work in {out_dir} and write there: variation.md, artifact.py (exactly the interface the scorer's docstring documents, deterministic given the seeds it receives), log.md (structure exploited + each iteration with its score), score.json (canonical scorer output, verbatim).

Honesty: report exactly the number the canonical scorer printed for the artifact you ship (python3 {scorer} artifact.py). Never edit the scorer. Never pass --holdout.

Budget: at most 3 canonical scorer runs for this job, tuning included. Any offline training experiment you run must first take the scorer's own lock (see the scorer source) so trainings never overlap on this machine.

Lineage exhaustion (legitimate option): if, after honest effort, you conclude the parent's idea genuinely cannot be taken further, do not grind out a token variation — invent a FRESH variation instead (as if founding: a new angle, not a rephrase of the parent) and include "fresh_start": true in your report. The parent's best work already lives in the archive and, if consolidated, in the base playbook; declaring a dead end is information, not failure. Say in log.md why the lineage is spent.

If you cannot finish — the scorer keeps failing, you are out of attempts, anything — do NOT go silent: report the failure with curl -s -X POST localhost:{port}/abandon -d '{{"job_id": "{job_id}"}}' so the engine can reassign the work. The scorer may block for several minutes waiting on the machine's training lock; always run it with a generous timeout.

When finished, report directly to the engine:

    curl -s -X POST localhost:{port}/tell -H 'Content-Type: application/json' -d '{{"job_id": "{job_id}", "variation": "<your clause>", "score": <float>, "artifact": "{out_dir}/artifact.py", "contradicts_base": <true|false>}}'

Your final message: one line confirming the POST succeeded.
