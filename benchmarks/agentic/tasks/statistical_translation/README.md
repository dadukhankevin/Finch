# Statistical translation GAR

This task evolves generic non-neural batch statistical translation programs. See
[`CONTRACT.md`](CONTRACT.md) for the frozen information boundary and fitness
policy.

The current high-agent campaign gives every candidate the complete approved
parallel training set—there is no example limit. It starts every independent
lineage from a neutral scaffold, invites creative mechanisms and explicit
eureka injections, records one protected macro-chrF++ score per evaluable
experiment, and requires report-level generalization audit before Decoder.md
incorporation. A report must pass the protected compositional feasibility gate
before it can become a lineage checkpoint or a whole-organism seed. Failed
artifacts remain available only as disclosed gene donors, so a useful mechanism
can be repaired without silently making its exploit the default. Complete-target
retrieval remains a comparison control, never the shared default.

```bash
python native_control.py init \
  --run ../../../runs/statistical-translation-high-agent-v3 \
  --corpus-dir /path/to/EBibleBenchmarks/corpus \
  --python /path/to/EBibleBenchmarks/.venv/bin/python
python native_control.py server \
  --run ../../../runs/statistical-translation-high-agent-v3 --port 8769
```

The root agent then uses `found`, dispatches the generated `prompt.md` to one
native subagent per lineage, and calls `score` after each returned experiment.
It chooses when to `audit`, kill, inject, or create crossover children; the
control script does not schedule or select evolution.

Crossover workers start from the strongest composition-eligible, audit-passed
parent when available. If every parent is ineligible, they start neutral and
receive the parent artifacts only as evidence/gene donors.
Their first candidate competes against that inherited protected score; a loss
is recorded but automatically restored rather than becoming the child's local
checkpoint. Failed or inconclusive parents remain usable as explicit sources
of partial mechanisms.

Open the per-run progress URL printed by the campaign or start the Finch hub:

```bash
python -m finch4.hub
```

Workers may inspect and tune against `practice.json`. Private selection,
generalization-development, and confirmation datasets are reconstructed only
inside the protected controller; the run manifest stores commitments rather
than contents or seeds. Confirmation remains untouched until the candidate is
frozen.
