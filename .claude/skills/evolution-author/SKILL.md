---
name: evolution-author
description: Set up and run a NEW evolutionary problem on this library (Finch 4) — choose the representation, write a canonical scorer, launch tensor or agentic evolution, watch it on the hub. Use when asked to "evolve X", "set up an evolutionary run", or apply the library to a fresh problem. For operating INSIDE an existing agentic run, use the agentic-ga skill instead.
---

# Evolution author — agents setting up their own evolutionary problems

You are the author tier: you design the problem, the library runs it.
Three decisions, in order.

## 1. Pick the representation (this decides everything else)

- **Tensor phenotype** (image, curve, vector, weights): use `solve()`.
  One call, cheap evaluations, proven defaults (FINDINGS ledger).
- **Text/code/methodology** (heuristics, scripts, prompts, specs): use
  the agentic substrate — individuals are methodology variations, an
  agent is the decoder. Follow the `agentic-ga` skill to run it.
- **Anything else / custom composition**: `finch4.layers` —
  `Environment([...layers...])`, engine-backed presets
  (`tensor_environment`, `agentic_environment`). New compositions are
  NEW mechanisms: they get a FINDINGS ledger row and earn defaults by
  measurement; the presets already have theirs.

## 2. Write the canonical scorer (the constitution of the run)

One script: `python3 score.py artifact [--holdout]`, printing one JSON
line `{"task", "score", ...extras, "errors"}`. Higher-better score.
Hard-won rules (each one paid for — see FINDINGS rounds 38-39):

- **Practice vs surprise**: score on fixed practice data; keep disjoint
  surprise data for audits only. Tuning on surprise data is forbidden.
- **Score what cannot be faked**: if the scored data is derivable from
  anything on disk, an optimizer WILL ship the answer key (one founder
  base64-embedded the whole practice slice). Perturb scored data in
  memory (seeded substitutions) so only real modeling round-trips.
- **Declare nondeterminism**: if scoring is not bit-reproducible (GPU
  training), include `"tolerance": <x>` in the output; audits honor it.
  Deterministic scorers omit it and audits demand exact reproduction.
- **Enforce the budget in the scorer** (wall-clock caps, forbidden
  imports, purity), never in prompts alone. If evaluation monopolizes
  hardware, serialize with a lock file INSIDE the scorer; if not, say
  so — parallel agents are then safe.

## 3. Launch and watch

- Tensor: `solve(fns, shape, epochs=..., progress=live_progress())`.
- Agentic: start `python3 -m finch4.serve --run <dir>
  --tasks ...`, then orchestrate per the `agentic-ga` skill (or
  `python3 -m finch4.drive ...` unattended).
- Finch: `Environment(..., live=True).evolve(generations=N)`.

Every server registers itself; `python3 -m finch4.hub`
(port 8800) shows ALL runs — live and finished — on one page. Watch
there when several problems evolve at once.

## Discipline (what makes a result a finding)

- **A comparison or it didn't happen**: the standing opponent is one
  agent iterating alone (`finch4.solo`) at the same
  agent-call budget, or the library's tensor baselines. One paired run
  is suggestive; rates over repeated runs are evidence.
- **Audit on influence**: re-run the scorer yourself on anything that
  becomes a parent, a champion, or consolidation input; check surprise
  data; read the work log against the claim. Record verdicts via the
  engine's audit route so falsified bests are evicted.
- **Write the FINDINGS row**: mechanism, direct evidence, status.
  Unmeasured things never become defaults.
- **Budget judgment**: match concurrency to the scorer's contention
  (lock-bound: ~1-2 agents; free: population-cap x 1-2) and say what a
  run will cost before spending it.
