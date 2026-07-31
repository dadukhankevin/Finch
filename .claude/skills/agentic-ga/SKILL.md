---
name: agentic-ga
description: Run the Finch 4 agentic substrate — a genetic algorithm whose decoder is an agent. Use when asked to evolve methodologies/specs with agent subagents, run the agentic GA, or continue an agentic run in benchmarks/agentic/runs/.
---

# Agentic GA — orchestration manual

The agentic substrate of Finch 4: individuals are text
**methodologies** (one shared **base playbook** + a per-individual
**variation** clause, "the base BUT <difference>"); the **decoder** is
an agent that follows base+variation to produce an **artifact**; fitness
comes from a **canonical scorer script**. Selection, fitness shares, the
population cap, best-ever archives, and consolidation cadence are owned
by `finch4.AgenticGA` (ask/tell, deterministic, seeded) —
never improvised by any agent.

## Invariants (fixed — these are NOT part of the evolvable playbook)

1. **One base playbook per run.** Individuals carry variations only
   (≤150 words). Content moves from variation to base only through
   consolidation.
2. **Never edit a canonical `score.py`.** Every reported score is the
   number the canonical scorer printed for the shipped artifact — the
   artifact is the evidence and must be saved. Audits re-run the scorer.
3. **Variations must not contradict the base**, except a mutation may
   deliberately contradict it and must say so (`contradicts_base=True`).
   Watch `ga.contradiction_report()`: contradictors consistently beating
   compliants means a past consolidation hurt the base — revert it.
4. **Audit on influence, not on score.** Before consolidation (and for
   any would-be record), re-run the canonical scorer AND `--holdout` on
   the artifact yourself, and read the subagent's log against its
   variation — quality that didn't come from the methodology must not be
   distilled into the base.
5. **Consolidation absorbs vetted winners only**: the engine's
   `consolidation_batch()` (per-task best-evers). Never "the top N
   ideas" — that is the arithmetic fold's failure mode (see FINDINGS.md).
6. **After consolidation**, every survivor is rewritten — push the
   absorbed part of its idea further, or keep it unchanged if it wasn't
   absorbed. No automatic re-score: the engine tracks staleness
   (`ga.stale()`); re-score individuals only as needed.

## Run layout

    benchmarks/agentic/runs/<run>/
      state.json           # AgenticGA.save / .load
      base_playbook.md     # copied from benchmarks/agentic/base_playbook.md
      individuals/<id>/    # variation.md, artifact.py, score.json, log.md

## The loop

PRIMARY MODE — orchestrator-run (Daniel's ruling, 2026-07-30): a live
agent session runs the GA with judgment, using the engine server for
the laws. Start the server (below), then: POST /ask for jobs, render
each job's prompt from finch4/agentic_prompts/ (reuse
`_template` and `FOUNDER_ANGLES` from finch4.drive),
and spawn one worker per job **with your runtime's NATIVE subagent
mechanism** — a Claude session uses its Agent tool, a Codex session
uses its own spawning. The server API is the only contract: any
process that can POST /tell is a valid worker; no cross-CLI plumbing
is needed when the orchestrator and workers share a runtime.
Cross-CLI shell-out is the exception, for unattended drive.py runs or
deliberately mixed fleets (verified codex string: `codex exec -m
gpt-5.6-luna --sandbox workspace-write -c
sandbox_workspace_write.network_access=true`).
Agents report themselves via POST /tell — including
`"fresh_start": true` when a breeding job's agent judged its parent
lineage exhausted and founded fresh instead (the engine marks the
parent non-breeding; its archive record survives). The orchestrator's judgment
calls: how many jobs run in parallel (measurement-locked tasks like lm
starve under contention — stagger or serialize; lock-free tasks like
compress parallelize fully), when to audit and what smells illegitimate,
when consolidation is worth its cost, retrying dead jobs, and reading
every consolidation proposal before applying it.

Self-driving (unattended/overnight): one command renders prompts,
spawns agents via any CLI, audits mechanically, and pauses for
consolidation review (--auto-consolidate to run overnight):

    python3 -m finch4.drive --run <run_dir> \
        --tasks ... --tasks-dir benchmarks/agentic/tasks \
        --agent-cmd 'claude -p "$(cat {promptfile})"' --rounds 6

Prompt templates live in finch4/agentic_prompts/
(SCRISPR-style operator rotation: one-change / telephone / masked
mutation, smart crossover, assigned founder angles). When orchestrating
manually instead, the same templates are the reference prompts.

Semi-manual: start the reporting server and let agents report themselves —

    python3 -m finch4.serve --run <run_dir> --tasks ... &

One process holds the engine; every request is lock-serialized and
state.json is saved after each mutation, so concurrent agents can POST
the moment they finish. The port is in the run's `server.json`. The
orchestrator drives rounds via `POST /ask`, `GET /due`, `GET /batch`,
`POST /consolidated`, `POST /rewrite`; each decoder-child ends its work
with `curl -s -X POST localhost:PORT/tell -d '{"job_id": ..., 
"variation": ..., "score": ..., "artifact": ...}'` (its final JSON
message then just confirms what it already reported). Routes are in the
serve.py docstring.

Fallback (no server): drive the engine from short python3 snippets —
`ask()` → spawn one subagent per job **in parallel** via the Agent tool
→ `tell()` each result (or `abandon()` failures) → `save()`. When
`consolidation_due()`: audit the batch, spawn the consolidator, apply
its edit to the run's `base_playbook.md`, `record_consolidation()`,
spawn rewrite jobs, `tell_rewrite()` each. Keep budgets honest:
founders 2–4 per task, children 4–8 per round, `consolidate_every` 2–4
— every job is a full agent run.

## Subagent prompts (each must be self-contained: absolute paths, no
references to this conversation)

**Decoder-child** (kinds found/mutate/crossover): give it the run's
`base_playbook.md` path, the canonical `score.py` path, its output
directory, and its job — *found*: "invent ONE distinct variation on the
base methodology, then follow base+variation"; *mutate*: parent's
variation + "produce a variation that differs in ONE deliberate way
(you may contradict the base, but flag it)"; *crossover*: both parents'
variations + scores + "merge their ideas into one variation". Require:
follow the methodology honestly; write `variation.md`, `artifact.py`,
`log.md` (what you actually did, plus the one-line structure-you-exploit
statement); run the canonical scorer; write its output verbatim to
`score.json`; final message = JSON `{"variation": ..., "score": ...,
"contradicts_base": ...}`.

**Consolidator**: give it the current `base_playbook.md`, the audited
`consolidation_batch()` (each task's best variation + score + log), and:
"edit the base so that an agent following the base with an EMPTY
variation would reproduce these wins — absorb what the winners actually
did, not a summary of every idea present; bump the version header;
change as little else as possible." Apply its edit only after reading it.

**Rewrite** (per survivor, after consolidation): give it old base
version, new base, its variation, and Daniel's rule: "if the new base
absorbed part of your idea, take that idea to the next level; if it
didn't, return your variation unchanged." Cheap job — batch several per
subagent if the population is large.

## What lives where

- Engine + laws: `finch4/agentic.py` (docstring = spec)
- Testbed tasks + evolvable playbook v0: `benchmarks/agentic/`
- The evolvable playbook must never absorb these invariants — if a
  consolidator's edit restates or overrides conduct rules, reject it.
