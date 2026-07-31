#!/bin/sh
# Genetic auto-research (GAR) on the binpack task, unattended: one
# command renders prompts, spawns one agent per job, audits
# mechanically, and pauses for consolidation review (add
# --auto-consolidate to run overnight). Watch it live on the dashboard:
#
#     python3 -m finch4.hub        # http://127.0.0.1:8800
#
# An orchestrating agent session can run the same loop natively instead
# (its own subagents as workers, the server API as the only contract):
# see .claude/skills/agentic-ga.
python3 -m finch4.drive --run benchmarks/agentic/runs/gar-binpack \
    --tasks binpack --tasks-dir benchmarks/agentic/tasks \
    --agent-cmd 'claude -p "$(cat {promptfile})"' \
    --rounds 6 --founders 2 --children 4
