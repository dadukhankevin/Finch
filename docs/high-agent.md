# Agent-mediated genetic auto-research

Finch's agentic engine evolves research trajectories. A lineage is not one
proposal: it is a persistent autoresearch loop that forms hypotheses, changes
an artifact, runs experiments, and keeps or reverts each candidate. A poor
first experiment does not imply that the direction should die, and a strong
founder does not imply that its trajectory can keep improving.

The decoder is the LLM plus one shared `Decoder.md`. The file is the writable
scientific memory of the population. Later evolvers naturally adopt, extend,
contrast, avoid, and combine cited discoveries from it. That is the normal
form of crossover. Explicit checkpoint recombination remains available when
an allocator sees a particularly useful combination.

## Separation of roles

- **Evolver agents** own lineages. They read `Decoder.md` and its cited source
  reports directly from Finch, run their own experiments, and report results
  directly back. They do not need an orchestrator to summarize the population
  for them.
- **Judge agents** perform selection. Each assignment contains exactly two
  immutable report checkpoints, each lineage's trajectory to that point, the
  task's fixed fitness criteria, and the evaluator evidence attached to those
  checkpoints. The judge chooses a
  winner or tie and gives a rationale. For translation, chrF can be primary
  evidence without being the population's rating system.
- **The orchestrator or allocator** creates work, assigns compute, requests
  pairings, injects new ideas, and may kill or revive lineages. It should avoid
  continuously interpreting every experiment back to every evolver; that
  feedback loop centralizes the search and collapses diversity.
- **The evaluator** produces ground-truth measurements such as chrF. Trusted
  scores always name their source. An evolver's own claimed number remains a
  claim until an evaluator assigns it.
- **Finch** is the bookkeeper and communication layer. It stores lineages,
  reports, artifacts, citations, shared research, judge assignments,
  verdicts, Elo, audits, and decisions. It validates identities and performs
  standard Elo arithmetic after a verdict. Its generic worker runner may also
  launch durable, allocator-supplied command jobs under explicit concurrency
  limits. Process scheduling is not semantic authority: Finch never
  interprets chrF, reads an artifact semantically, authors an idea, or chooses
  a winner.

This keeps semantic authority in agents while making the process concurrent,
inspectable, and recoverable.

## Empirical feedback without evaluation leakage

An autoresearch lineage needs more than an aggregate score. A task adapter may
provide a research-visible laboratory containing concrete inputs, candidate
outputs, baseline outputs, references, per-example measurements, and execution
artifacts. The evolver uses that bundle to determine whether its hypothesis was
wrong, its mechanism never activated, or the downstream agent ignored useful
evidence.

Laboratory feedback is training data, not trusted fitness. A worker may
see a description and at most one synthetic example of the call shape.
Evaluation languages, verses, leftover cells, and Bible files stay in
the driver's workspace, not the worker's. A corpus-aware task may inject
the rest of that language's parallel text at score time, with evaluation
verses stripped and no verse ids. The outer Finch loop scores snapshots
on hidden held-in data and may keep on that number. Holdout is scored as
often as held-in, graphed beside it, and never used for keep, best, or
Elo — observation only. Finch need not understand either payload: it
schedules the laboratory/evaluator commands and records their artifacts,
while evolver and judge agents retain semantic authority. This gives each
lineage a Karpathy-style edit–run–inspect loop inside the population-level
evolutionary search.

## Native worker pool

`finch4.workers` is the optional execution half of the communication layer.
The allocator queues an argv list, working directory, lane, timeout, and retry
limit. Finch launches it without a shell, records every attempt in `jobs.json`,
captures logs, and exposes the queue on the dashboard. Evolver, evaluator, and
judge lanes can have different concurrency limits.

The queue is deliberately generic. A task adapter decides what command means
“advance lineage L0003” or “judge match M0007”; Finch only runs that command and
records whether the process succeeded. This permits ten native Codex evolvers,
for example, without embedding Codex, translation, prompts, or an allocation
policy in the library. A different orchestrator can supply worktrees, another
agent runtime, or no worker pool at all.

## Native population

`finch4.evolver` is a population of auto-research workers on one artifact
and one `Decoder.md`. Finch starts, stops, and replaces members. Each
member sees its own fitness and the current shared research. Isolation of
sealed evaluation is environmental: those files are not in the worker
workspace, and worker prompts do not name them. The allocator does not
author per-worker denylist prompts or continuously interpret experiments
back into the population. A general coding session on the Finch checkout
or a task tree is not a worker; launch researchers only through
`finch4.evolver` / `finch4.solo`.

## Pairwise selection and Elo

Elo rates lineages because the evolving organism is the research trajectory.
A judge nevertheless compares exact report checkpoints, so every verdict can
be reconstructed against the artifacts and evidence that actually existed.

1. Freeze task criteria with `POST /criteria`.
2. Request a match with `POST /pair`, naming exactly two `[lineage, report]`
   checkpoints. Finch snapshots the criteria and evidence into `M####`.
3. Give `GET /match?id=M####` to one judge agent. This narrow payload contains
   the pair, not the population standings.
4. The judge returns `POST /verdict` with a lineage ID or `tie`, its agent
   identity, rationale, and optional evidence.
5. Finch applies the fixed Elo update and exposes standings at `GET /ratings`.

Several judge agents can work concurrently on different assignments. Finch
does not aggregate votes or infer a result: every rating change traces to one
named agent verdict. A bad assignment or invalid verdict can be append-only
voided; Finch replays later valid matches so the standings are repaired.

Elo is the selection record, not a replacement for measurement. The judge can
use chrF, robustness across languages, audit evidence, novelty, feasibility,
or trajectory potential exactly as the frozen criteria specify. Confidence is
recorded but does not secretly alter the Elo K-factor.

## Direct research communication

`GET /research` returns the current shared Markdown and resolves every inline
citation to its source report. `GET /research?lineage=L0003` additionally
returns that evolver's own trajectory. Evolvers publish experiments through
`POST /report` themselves.

A trusted scored report with an artifact appends itself to the one versioned
`Decoder.md`. Finch writes the citation token, keep/revert, score, and the
worker's own summary. No allocator publishes that line. Concurrent reports
serialize through the campaign lock, so workers cannot erase one another's
discoveries. `POST /finding` remains for optional extra prose; it is not how
the population shares scores. `python -m finch4.research read|append` is the
small agent-facing CLI for those two operations.

One ordinary Markdown token names intellectual inheritance:

    [L0003#2@8ddf8b41]

The first part locates the lineage checkpoint and the hash prefix identifies
its immutable artifact. When an evolver materially uses an earlier idea, it
writes that token in its report. Cite actual adoption, extension, contrast, or
combination—not everything read. Finch parses the prose and `Decoder.md`
history into the dashboard's searchable **Tree of Life**. No parallel lineage
schema or citation tool call is required.

## The evolutionary loop

There are no mandatory rounds and no fixed number of children.

- **found / inject** starts a genuinely distinct research trajectory. The
  allocator may inject a eureka idea at any time.
- **report** records every experiment, including losses. A rejected candidate
  is reverted to the lineage's local champion; it does not automatically kill
  the research direction.
- **reflect / finding** returns trusted evidence to that lineage, which appends
  its cited interpretation to shared research before later experiments plan.
- **pair / verdict** obtains pairwise agent judgment and updates lineage Elo.
- **kill / revive** reallocates compute. Killing a lineage remains possible,
  but should reflect its trajectory rather than one weak experiment.
- **directed recombination** starts as many children as there are genuinely
  promising ways to combine exact parent checkpoints. Natural cited crossover
  through shared research is expected to be more common.
- **share** adds a trusted, immutable result—including a useful failure—to the
  shared scientific record without declaring it a default method.
- **incorporate** makes an audit-passed method inherited behavior in
  `Decoder.md`.
- **compact** rewrites shared research for density without introducing new
  sources.

The orchestrator can allocate based on Elo, diversity, trajectory promise,
novelty, or available compute. Finch records those choices but does not embed
one mandatory allocation policy. A lineage can remain promising before it is
fit, and it can still be killed when further work is no longer justified.

## Evidence membranes

Reports may contain both `claimed_score` and trusted `score`; only the latter
has a named evaluator source. Stochastic measurements should be compared on
matched hidden units and replicated before a checkpoint is declared verified.
Singleton improvements remain visible but do not become truth through prose.

Natural crossover and shared methodology have different gates:

- any non-void immutable checkpoint may represent its trajectory in judgment,
  including a reverted loss;
- trusted immutable checkpoints may be explicit recombination parents;
- trusted results may enter `Decoder.md` as clearly labelled findings;
- only audit-passed results may enter as inherited methodology.

A failed organism can still contain a useful gene. Its citation carries the
failure context so a later evolver can preserve the mechanism without
inheriting the unsupported claim.

## HTTP surface

    GET  /research[?lineage=L####]   shared file, sources, own trajectory
    POST /report                     one evolver experiment; scored
                                     artifacts append themselves to
                                     Decoder.md
    POST /criteria                   freeze per-task judge criteria
    POST /pair                       exact two-checkpoint assignment
    GET  /match?id=M####             narrow judge payload
    POST /verdict                    named agent decision; Finch updates Elo
    POST /void-match                 invalidate and replay the rating history
    GET  /ratings                    population standings
    GET  /matches                    assignment and verdict history
    GET  /jobs                       durable native-worker queue state
    POST /found /kill /revive        lineage lifecycle
    POST /assign                     compute ownership
    POST /audit /revise-score        evidence correction
    POST /finding                    atomic agent-authored Markdown append
    POST /share /incorporate         full-file edit / audited methodology
    GET  /tree /summary /decisions   inspection and dashboard data

Evolutionary state lives in `finch4.agentic.Campaign`; durable process state
lives beside it in `jobs.json`. Both are mirrored by `finch4.serve`.
`finch4.evolver` owns the population; `finch4.solo` is one auto-research
member. `finch4.workers` can run many task-supplied jobs concurrently. The
tensor engine is unchanged; its fast, non-semantic substrate retains its
engine-owned selection laws.
