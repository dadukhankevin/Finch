#!/bin/sh
# Genetic auto-research (GAR) on binpack: an allocator starts Finch and
# parallel evolvers; separate judge agents compare exact pairs and drive Elo.
# Evolvers read/report through Finch and exchange cited findings in Decoder.md.
# Watch it live:
#
#     python3 -m finch4.hub        # http://127.0.0.1:8800
claude -p 'Use the agentic-ga skill: run an agent-mediated GAR campaign on
the binpack task (tasks dir benchmarks/agentic/tasks) in
benchmarks/agentic/runs/gar-binpack. Found 3 lineages with distinct
angles, run their evolvers for 6 experiments each, freeze judge criteria,
have separate judge agents compare exact checkpoints, use Elo for selection,
and incorporate any verified finding into Decoder.md.'
