# Founder v2 specification

## Heritable envelope

This proposal keeps the current candidate fields and adds optional fields that
a future campaign adapter could materialize without changing the protected
evaluator's scientific boundary:

```json
{
  "schema_version": 2,
  "name": "...",
  "hypothesis": "...",
  "activation_condition": "...",
  "fallback": "...",
  "falsifier": "...",
  "agent_instructions": "...",
  "research_prompt": "...",
  "translation_prompt": "...",
  "skill_md": "...",
  "phase_policy": {"phases": [], "gates": [], "budgets": {}},
  "evidence_schema": {},
  "fallback_policy": {},
  "tools": {"tools/name.py": "..."},
  "trace_contract": {}
}
```

`skill_md` is candidate-local text exposed within the supplied project. It may
describe a workflow and local tools but must not reference installed skills,
external paths, memories, packages, or the network. Tool source remains Python
standard library, deterministic, and bound to supplied corpus/task files and
`.runner/` output.

## Common evidence envelope

Every family writes one JSONL decision record per verse:

```json
{
  "task_id": "...",
  "family": "...",
  "activated": true,
  "inputs": [{"row": 12, "span": [3, 5], "kind": "parallel_evidence"}],
  "alternatives": [{"id": "a", "target": "...", "support_rows": [12, 90]}],
  "decision": "a",
  "changed_from_disabled": true,
  "unresolved": ["polarity_scope"],
  "fallback_used": false,
  "reason_codes": ["two_origin_support", "stage_match"]
}
```

Free-form rationale may accompany this record, but it cannot substitute for
`alternatives`, `decision`, and `changed_from_disabled`.

## Test protocol

For each founder, run these comparisons only when an authorized campaign is
later launched:

1. exact pre-GAR staged harness;
2. founder enabled;
3. founder with its family gene disabled but all other text/tools unchanged;
4. fallback-forced mode;
5. repeated deterministic-tool run with output hashes compared.

Record activation count, output-change count, fallback count, failures,
runtime, tool calls, and protected aggregate fitness. A founder that never
changes a decision is inert even if its notes are insightful. A founder whose
fallback does not reproduce its declared baseline is malformed. Fitness still
comes only from the protected evaluator.

## Family-specific observables

- Speech act: force inventory, clause boundaries, chosen force, competing
  realizations, and coverage of polarity/modality/participants.
- Register: payload-derived field/tenor/mode feature values, neighbor rows,
  competing candidates, and which register dimension broke the tie.
- Genre: induced stage sequence, boundary evidence, stage-conditioned
  candidate ranking, and transition surprise.
- Constraint/program: variables/rules, support provenance, violations, chosen
  solution/program, and deterministic tie order.
- Contrast/edit/lattice: explicit counterexample or edit/path structure and the
  exact candidate it rejected or replaced.
- Orchestration: pre-compute uncertainty/opportunity signals, budget decision,
  phase exit reason, and fallback route.

## Safety and anti-theater rules

- SFL labels are latent payload summaries, not universal facts imported from a
  textbook. A founder must infer its inventory from recurrent contrasts in the
  supplied English/target pairs and may use neutral identifiers when labels are
  uncertain.
- No family may generate target tokens unsupported by supplied targets unless
  the unchanged pre-GAR Luna harness already permits that behavior; the trace
  must distinguish attested bundles from Luna-composed surface text.
- An analysis step must expose downstream authority. If deleting its output
  cannot change candidate ranking, generation, checking, or fallback, delete
  the step.
- Expensive corpus-wide models run only after a query-level opportunity gate,
  extending the verified compute discipline [L0048#1@e644f21b] and explicitly
  avoiding the eager failure [L0048#0@e9dda937].
