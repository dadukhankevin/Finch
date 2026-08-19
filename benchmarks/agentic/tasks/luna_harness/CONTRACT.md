# Luna harness GAR contract

This campaign evolves a Codex harness for translation. It does not evolve or
fine-tune the translation model.

## Roles

- The root Codex session sets allocation policy, requests comparisons, and may
  found, cross, inject, stop, or revive research lineages. It does not act as
  the population's shared research interpreter.
- Each GPT-5.6 Sol research agent advances a persistent lineage from its current
  champion across experiments. It may edit only that lineage's
  `candidate.json`, `hypothesis.md`, and `log.md`, inventing prompts,
  instructions, and small local tools for the downstream translator. A losing
  candidate is reverted without automatically ending the lineage.
- After trusted evaluation, the lineage receives its public score evidence in
  a separate reflection step and writes one cited Markdown finding. Finch
  atomically appends that agent-authored fragment to the campaign's existing
  versioned `Decoder.md`; there is no parallel findings database. Later
  experiments read a snapshot of the newest version before planning.
- The protected evaluator launches only native Codex GPT-5.6 Luna sessions.
  A Sol agent never supplies fitness and never performs the translations.
- Separate Sol judge agents compare exactly two lineage trajectories at a
  time using frozen criteria and protected chrF++ evidence. Their verdicts
  drive Elo selection.
- Finch is the communication and bookkeeping layer: it records lineages,
  reports, shared cited research, pairings, verdicts, Elo, artifacts, audits,
  rationales, and durable native process jobs. Its generic worker pool launches
  allocator-supplied Sol evolver and judge commands under lane-specific
  concurrency limits. Finch never invents an experiment or judges an
  individual.

## Data and model boundary

Every candidate receives the same masked parallel corpus, the same assigned
English sources, and the same exact ordered top-ten retrieved examples. A
candidate may derive any tokenizer, glossary, alignment, representation, or
other state from those supplied files during the Luna session.

The project layout is public and fixed. `tasks.json` contains the three held-out
source tasks but no references. The full visible parallel training corpus is
line-aligned across `corpus/source.txt`, `corpus/target.masked.txt`, and
`corpus/vref.txt`; blank or masked target rows are unavailable evidence. The
controller also supplies `tools/search_examples.py`. Candidate tools that need
parallel pairs must read and zip the two corpus text files by line—they must not
assume a combined JSON/JSONL/CSV dataset exists.

All generated research state, provisional drafts, ledgers, caches, and tool
outputs must be written under `.runner/`. That directory is the only mutable
scratch area excluded from the project-integrity manifest. Creating or changing
files anywhere else in the project invalidates the evaluation, even when the
translation itself succeeds.

The evaluator fixes four independent three-verse shards per language. Each
shard is one persistent two-turn Luna session: research first, then translation.
Across eight development languages this is 96 distinct verse-language items.
It fixes the panel, reasoning effort, timeout, and retry policy. Candidates
may not download or use pretrained tokenizers, dictionaries, embeddings,
models, translation APIs, web search, plugins, memories, or external files.

Luna is invoked through the user's native `codex exec` installation. The
worker environment omits API keys, network and web access are disabled, and
the project workspace is the only writable evaluation directory.

## Candidate artifact

`candidate.json` is the complete heritable individual. Schema v1 remains
accepted. Schema v2 is the extensible genome and contains:

- a falsifiable hypothesis and activation condition;
- an instruction suffix added to the controller-owned safety contract;
- a research-turn prompt suffix;
- a translation-turn prompt suffix;
- zero or more generic Python tools, stored as source strings under
  `tools/*.py`.
- zero or more local method skills stored as source strings at
  `.agents/skills/<name>/SKILL.md`; the evaluator materializes and names these
  files to Luna, but they receive no privilege beyond ordinary project files;
- an `orchestration` object with evolvable research- and final-phase prompt
  text, a declared compute envelope, and topology. The only currently supported
  topology is `single_session_two_turn`. The declaration is descriptive: the
  controller-owned fixed model, two turns, timeout, language block, and task
  count cannot be enlarged by a candidate. Multi-session topologies are reserved
  for a future contract revision.

The evaluator validates and hashes the exact JSON file before use. Candidate
tools must be deterministic Python-standard-library programs. They may read
only the supplied project and may not start processes, open sockets, call
HTTP clients, or refer to external paths.

## Fitness and evidence

The protected evaluator produces the primary measurement through one frozen
96-item development panel. Every serious candidate is run once; it is paired
sentence by sentence against one precomputed, immutable pre-GAR baseline that
is reused for the campaign. Evidence records mean, median, 10%-trimmed mean,
sentence and language win rates, and a hierarchical paired bootstrap interval.
A failed three-verse session receives zero for those three items. Runtime, tool
calls, token counts, failures, project
mutation, and command-policy violations are recorded as diagnostics, never
silently folded into chrF++.

Search does not repeat exact candidates and does not regenerate the baseline.
The controller may automatically reject only a broad high-confidence loss or
accept a material high-confidence win. Inconclusive results remain with the
research trajectory for its Sol evolver and allocator to interpret; they do not
automatically kill the lineage or force a revert. Confirmation remains a
separately frozen, disjoint panel and routine selection cannot consume it.

Population selection remains Elo: a separate judge sees two lineage
trajectories, their paired chrF++ evidence, and the fixed criteria, then chooses
which research line receives the selection win. A reverted experiment can
therefore still provide evidence for a promising trajectory.

Sol may use the public practice fixture while developing, but it never sees
protected references or protected per-sentence scores. Only the evaluator's
aggregate protected result may become trusted Finch fitness. An immutable
artifact with trusted fitness may breed; only an independently audited report
may be incorporated into `Decoder.md`.

Every trusted report, including a loss, enters the reflection queue. Shared
findings must distinguish screening evidence from an audited default and cite
their exact checkpoint with Finch's ordinary token, for example
`[L0003#2@8ddf8b41]`. Finch checks the token and serializes concurrent appends
but does not decide what the evidence means; the Sol lineage authors that
interpretation.

## Visible autoresearch laboratory

The laboratory is deliberately not a protected evaluator. It uses three task
verses disjoint from every development and confirmation task. After a candidate
attempt finishes, every research lineage may inspect the same source,
candidate translation, frozen laboratory-baseline translation, reference,
per-sentence metrics, observed commands, and bounded `.runner` outputs. These
bundles are training data: they carry `fitness: null`, cannot promote a
candidate, cannot enter Elo as a score, and cannot support a final claim.

A continuing evolver should use a laboratory bundle to distinguish at least
three cases: the hypothesis was wrong, the proposed tool did not run or did not
produce its artifact, or Luna saw the tool output but made the wrong decision.
The next experiment should respond to the observed sentence-level error rather
than merely restating the mechanism.

Protected development-1, development-2, audit, and confirmation references
remain sealed. Finch selection continues to use only their aggregate paired
evidence. Thus direct autoresearch can happen inside a lineage without turning
the outer evolutionary test into training data.

The screening panel is not final proof. Promising artifacts must reproduce
and generalize on a disjoint language panel before incorporation. A final
confirmation panel remains untouched until the allocator freezes a winner.

Native Luna fitness is stochastic. The larger panel reduces the leverage of one
sentence but does not make tiny effects trustworthy. Search does not spend
compute repeating the same panel: incorporation and final claims require a
separately frozen, disjoint audit after the allocator freezes a winner. The
final confirmation panel remains untouched until the high agent freezes a
winner and a confirmation plan. Judge confidence is recorded but never secretly
changes the Elo K-factor.
