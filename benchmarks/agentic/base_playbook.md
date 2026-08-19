# Base playbook — version 0

This file is the EVOLVABLE shared methodology — the agentic substrate's
"base decoder weights." Every individual in the population is this
playbook plus one variation clause. Consolidation edits this file (and
bumps the version above); nothing else may. The rules of conduct live in
the `agentic-ga` skill and are NOT part of this file — they cannot be
evolved away.

## Methodology

1. Read `TASK.md` and the current champion. Finch injects your fitness
   and the shared `Decoder.md`.
2. State, in one or two sentences in your work log, what structure of
   the problem your heuristic will exploit and why.
3. Write the simplest version of that heuristic first. Pure function,
   numpy only, no I/O, no randomness, fast enough to run hundreds of
   times.
4. Write the candidate. The driver scores it and keeps it when fitness
   rises.
5. Change one general predicate at a time, but the artifact you ship
   must already include working instruments you adopted from shared
   research.
6. Ship the artifact. Report only a number you actually produced. The
   driver owns campaign fitness.
