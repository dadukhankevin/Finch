# Finch 4 — working conventions

This is the primary development repo. It grew from the latentspace
research campaign (github.com/dadukhankevin/latentspace), whose
FINDINGS.md is the evidence behind every shipped default. If a
`research/` directory exists locally it holds that record plus run
evidence — it is intentionally gitignored; the public tree ships the
library only.

## The rules that are the project

- **Measured defaults only.** No mechanism becomes a default without
  direct evidence against its own absence, on the current substrate.
  New findings get a row in the claims ledger (research/FINDINGS.md):
  mechanism, direct evidence, status. Evidence attaches to code
  identity, not to slot names — a swapped implementation inherits
  nothing (tests/test_finch.py's bit-identity tests are how ports
  carry their records).
- **One shared substrate per run, ever** (decoder weights or base
  playbook); individuals are small bendings of it. Selection, fitness
  shares, capping, and archives are ENGINE-owned laws — layers drive
  the engine, never reimplement it.
- **Canonical scorers are constitutions**: agents run them, never edit
  them; practice/surprise splits; budgets and purity enforced in the
  scorer, not in prompts; audit-on-influence before anything breeds or
  consolidates.
- **Comparisons or it didn't happen**: single runs are suggestive;
  success rates over repeats are evidence (agent decoding is
  nondeterministic — paired seeds don't exist there).

## Practicalities

- Tests: `python3 -m pytest tests/` — keep it green; bit-identity
  failures mean a port changed behavior and its evidence no longer
  transfers.
- Dashboards: `python3 -m finch4.hub` (port 8800); any run joins via
  `finch4.live_progress()` or the agentic server. Light, earthy theme
  — no dark mode.
- Skills: `.claude/skills/agentic-ga` (operate inside an agentic run),
  `.claude/skills/evolution-author` (author new evolutionary problems).
- Write results in plain words: define every measurement term at first
  use (practice/surprise passage, not canonical/holdout), no invented
  shorthand, every back-reference self-contained.
