# Luna harness founders v2

This directory proposes a deliberately diverse founder population for evolving
the Luna translation harness. It is a design artifact only: nothing here starts
a campaign, changes the evaluator, or claims fitness.

The unit of inheritance is broader than a prompt. A founder may vary six gene
surfaces together:

1. controller-appended agent instructions;
2. research- and translation-turn prompts;
3. local `SKILL.md` content loaded by Luna from the candidate workspace;
4. deterministic standard-library tools that read only supplied files and
   write only `.runner/`;
5. evidence representations and their provenance/uncertainty fields;
6. phase policy, arbitration, activation gates, and exact fallbacks.

Every founder remains inside the Luna contract: identical supplied data, no
pretrained models or language resources, no network, no external files, and no
learning from protected references. The proposed `SKILL.md` files are local
instructions, not installed skills and not external memory.

## Why these are new families

The first three families target linguistic decisions mostly absent from the
current retrieval/alignment decoder:

- **Speech-act realization** predicts clause mood, polarity, modality,
  participant address, and discourse-function wording before lexical filling.
- **SFL register** treats field, tenor, and mode as payload-induced conditioning
  variables that select among attested realizations.
- **SFL genre/staging** induces recurrent passage stages and uses stage
  transitions to constrain how a verse functions in its local sequence.

These are not invitations to write richer analyses. A representation earns
authority only when it chooses between at least two concrete target candidates,
records the choice, and differs from its disabled ablation. Otherwise the
founder returns its exact fallback.

The remaining families are structurally distinct: constraint solving,
minimum-description-length program induction, contrastive counterexample
testing, retrieval-as-edit-script, target construction lattices, and sequential
risk budgeting. They provide diversity beyond SFL and beyond variations on one
glossary or retrieval prompt.

## Population plan

`population.json` contains 12 founders in six families. Start with all 12 rather
than collapsing them into one omnibus organism:

| Family | Founders | Main changed decision |
|---|---:|---|
| speech act | SA1, SA2 | clause force and its target realization |
| SFL register | RG1, RG2 | realization conditioned on field/tenor/mode |
| SFL genre | GN1, GN2 | stage label and stage-conditioned candidate choice |
| constraint/program induction | CS1, MD1 | globally consistent bundle or induced rewrite program |
| contrast/edit/lattice | CT1, ED1, LT1 | reject unsupported choice, edit donor, or select construction path |
| orchestration | OR1 | allocate research effort and fallback by measured uncertainty |

A first wave should hold the population structure fixed and compare every
founder against the pre-GAR staged harness and its own disabled ablation. Do
not cross founders until each has produced an activation report. Preserve at
least one founder from each family through the first selection decision even if
another family leads; structural diversity is the point of founding.

Promote a founder only when all of the following are observable:

- its mechanism activated on at least one assigned verse;
- the evidence ledger identifies the alternatives and the chosen candidate;
- disabling only the family-specific gene changes that decision, or proves the
  gene inert;
- its fallback reproduces a named baseline behavior exactly;
- deterministic tools reproduce byte-identical outputs on a rerun;
- runtime and token use leave margin for the fixed two-turn session.

Crossovers should be interaction hypotheses, not prompt concatenations. The
most plausible second-wave pairs are SA1×RG1 (force realized under tenor),
RG2×GN2 (register transitions at genre boundaries), GN1×LT1 (stage-conditioned
construction path), and CT1×OR1 (counterexample risk controls effort). Each
child must retain separately switchable parent ablations.

## Lessons inherited from the current Decoder

- Require sparse edits and preserve unsupported context; exact-cover editing
  was unreachable, while sparse provenance-gated surgery worked
  [L0018#0@b63b70c1]. ED1 extends that lesson from token surgery to explicit
  source-change edit scripts.
- Apply repair only after a structural winner and only to unresolved content
  [L0024#0@4d5a4e2d]. SA2 and LT1 contrast with peer competition by freezing a
  semantic/structural plan before surface repair.
- Do not trust extra-neighbor agreement merely because it sounds robust: the
  two-neighbor consensus extension regressed [L0014#1@5a0254f5]. CT1 therefore
  asks for a concrete counterexample and decision flip, not generic consensus.
- Keep weak representations in narrow roles. Spectral evidence hurt as a
  direct gate and helped only in retrieval geometry [L0034#0@464849d2]. RG2
  and GN2 may rank evidence neighborhoods but cannot emit unattested text.
- Gate expensive alternatives before constructing them. Eager bidirectional
  runner work timed out on unseen data [L0048#0@e9dda937], whereas the
  opportunity-gated direct runner survived [L0048#1@e644f21b]. OR1 makes
  pre-compute opportunity gating a first-class gene.
- A structurally plausible branch can still be causally inert: reverse-only
  runners did not change output [L0049#0@e63fdac5]. Every v2 founder therefore
  requires a decision trace and disabled ablation.

## Files

- `population.json`: machine-readable founder briefs and genome deltas.
- `SPEC.md`: candidate schema extension, common test protocol, and trace
  requirements.
