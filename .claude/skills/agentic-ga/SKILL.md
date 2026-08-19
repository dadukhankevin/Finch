---
name: agentic-ga
description: Run a Finch genetic auto-research campaign with autonomous evolver agents, pairwise judge agents, Elo selection, direct cited shared research, and lightweight orchestration. Use when asked to start, continue, or inspect GAR.
---

# Agent-mediated GAR

You are the allocator and campaign operator, not the population's shared
brain. Spawn diverse evolver agents, keep the experiment machinery running,
request useful comparisons, and allocate compute. Evolvers read and report
through Finch directly. Separate judge agents make pairwise selection
decisions. Finch records communication and performs Elo arithmetic; Finch
never judges.

The organism is a **research trajectory**, not its founding idea or most recent
experiment. A lineage may begin poorly and make a conceptual advance several
experiments later. A losing candidate is reverted; the lineage may continue.
Killing a lineage remains possible when its direction is falsified, redundant,
stagnant, or no longer worth compute.

## Invariants

1. **External evidence remains ground truth.** A trusted score names its
   evaluator. Worker claims use `claimed_score`. For translation, chrF belongs
   in the judge evidence even though Elo is the population rating.
2. **Agents judge; Finch does not.** A match contains exactly two immutable
   checkpoints, their trajectories to that point, plus frozen criteria and
   evidence. The named judge returns the winner or tie and rationale. Finch
   only validates and updates Elo.
3. **Evolvers communicate directly.** They read `GET /research` and publish
   `POST /report`. Finch appends each trusted scored report to `Decoder.md`.
   Do not continuously funnel their research through your own summaries.
   Periodic allocation is useful. Constant interpretive steering reduces
   diversity.
4. **Material influence is cited inline.** Use `[L0003#2@8ddf8b41]` when an
   experiment adopts, extends, contrasts, or combines that checkpoint. The
   prose is the lineage record; do not add a parallel influences object or
   ceremonial citations.
5. **One shared `Decoder.md`.** `share` records trusted findings, including
   failures. `incorporate` promotes only audit-passed methodology. `compact`
   can remove bloat but introduce no new source.
6. **Record reasons at decision time.** Found, pair, verdict, kill, revive,
   audit, and shared-memory changes all need honest rationales.

## Start a campaign

    python3 -m finch4.serve --run <run-dir> --tasks <task...>

Read `<run-dir>/server.json` for the port. Before tournament play, freeze the
criteria a judge should apply:

    curl -X POST http://127.0.0.1:$PORT/criteria -d '{
      "task":"translation",
      "criteria":{"primary":"paired chrF","also_consider":["cross-language robustness","feasibility","trajectory potential"]},
      "rationale":"use one explicit contract across judge agents"
    }'

Start the population with `python3 -m finch4.evolver`. Each member is an
auto-research worker (`finch4.solo`) on the same artifact. All of them
share the same `Decoder.md` and may cite each other. Finch
injects fitness and shared research. You may kill or insert a member;
you do not write their experiment prompts, file denylists, or a running
interpretation of the search.

Workers already have:

- their current fitness;
- the current `Decoder.md` (updated when a scored report lands);
- `GET /research?lineage=L####` and `POST /report`.

Do not add a second prompt that tells them which files not to open.
Those files are not in the worker workspace.

A general coding session on the Finch checkout or a task tree is not a
worker. That bypasses the workspace membrane. Launch researchers only
through `finch4.evolver` / `finch4.solo`, whose working directory is the
lineage folder Finch prepares.

## Run selection

Request a comparison between exact checkpoints:

    curl -X POST http://127.0.0.1:$PORT/pair -d '{
      "individuals":[["L0003",2],["L0007",1]],
      "requested_by":"allocator-1",
      "judge":"judge-4",
      "rationale":"compare two viable approaches for the next allocation"
    }'

Give only `GET /match?id=M####` to the assigned judge. Do not prime it with
the population leaderboard or your preferred theory. The judge inspects the
two artifacts, evaluator metrics and evidence under the supplied criteria,
then posts:

    curl -X POST http://127.0.0.1:$PORT/verdict -d '{
      "match":"M0002","winner":"L0007","judge":"judge-4",
      "rationale":"wins the primary metric and transfers across more cells",
      "evidence":{"paired_delta":1.7,"cells_won":"6/8"}
    }'

Use several judge agents by assigning different matches concurrently. Each
verdict is one named selection event, not a vote that Finch silently
aggregates. Read `GET /ratings` for Elo standings and `GET /matches` for the
auditable history. If a comparison was invalid, use `POST /void-match`; Finch
replays later valid Elo updates.

Pair near competitors often enough to learn ordering, but preserve novelty
and structurally different trajectories rather than turning every allocation
into exploitation of the current leader. Elo informs allocation; it does not
forbid a promising low-rated trajectory from receiving more experiments.

## Natural crossover and shared research

The usual crossover happens when evolvers read cited shared findings and
combine them naturally. When a report uses multiple source tokens, the Tree of
Life reconstructs that intellectual descent automatically. Use explicit
`parents=[[lineage, report], ...]` only when a new worker needs a deliberately
directed merge of exact artifacts. Create as many children as there are
meaningfully different promising combinations.

A trusted scored report appends itself to `Decoder.md`. Do not republish
those lines by hand. Use `POST /finding` only for extra cited prose, and
`POST /share` only when an agent is deliberately rewriting the full shared
Markdown.
After exact-artifact replication and audit, publish methods with
`POST /incorporate`. Always put the source token beside the claim it supports.
Do not turn a lucky screen into the population's default.

## Allocation guidance

- Maintain structural diversity; do not use your own feedback as a shared
  mutation prompt for every evolver.
- Reserve capacity for unrelated founders and eureka injections, especially
  when verified fitness plateaus.
- Prefer matched distributions over singleton stochastic scores.
- Keep local candidate acceptance separate from lineage survival: revert a
  losing experiment, then decide whether the trajectory deserves another.
- Read what an artifact actually does before recombining it; prose can be
  wrong.
- Tell the user plainly when verified progress stops. Activity and better
  measurement are not translation improvement.

## Where it lives

- Protocol and rationale: `docs/high-agent.md`
- Finch-owned prompts: `finch4/prompts.py`
- Record, match assignments, Elo: `finch4/agentic.py`
- HTTP communication layer and dashboard: `finch4/serve.py`
- Population evolver: `finch4/evolver.py`
- One auto-research worker: `finch4/solo.py`
