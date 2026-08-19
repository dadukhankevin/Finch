# Statistical translation GAR: campaign contract v3

This continuation campaign searches for a generic non-neural statistical
translation mechanism. Research workers may be language models; submitted
translators may not contain, call, train, or consume features from a neural
network.

## Question and primary outcome

Can GAR discover productive non-neural translation mechanisms when complete
target retrieval is available as a control but cannot satisfy compositional
eligibility?

Primary selection fitness is equal-language, equal-condition macro sentence
chrF++. No alignment, shuffled-pair, copying, speed, or data-quantity term is
part of selection fitness.

Selection is constrained by a separate feasibility boundary. Before a report
may become a lineage checkpoint, a default organism, or a whole-organism seed,
it must pass protected novel-conjunction episodes. Each episode combines two
supplied source examples while the correct target combines their two targets,
so no single complete training target can solve it.
Eligibility requires zero failures, at most 5% exact complete-target copies,
at least 3 chrF++ improvement over frozen single retrieval, and no losing
language/condition cell. This gate is not a weighted fitness term.

The harness also reports, without optimizing them directly:

- chrF++ improvement over the frozen v0 retrieval-copy mechanism;
- exact training-target copy rate;
- output/reference length ratio;
- on generalization audits, chrF++ after training targets are deliberately
  paired with the wrong sources.

The wrong-pair condition is diagnostic evidence about source conditioning. It
is not selection fitness.

## Information boundary

For each language and natural/cipher condition, one candidate process receives:

1. the complete approved parallel training set as `training`;
2. all held-out source strings as `queries`.

The candidate may use every training pair, retrieve a subset internally,
train a statistical model once for the batch, or combine these approaches.
There is no `K`, example-count penalty, or target-reveal stage.

Every language-dependent representation or parameter must be derived during
that process from the supplied payload. Nothing learned may be loaded from an
external or pretrained asset.

Allowed mechanisms include counts, word/character n-grams, Markov and
variable-order models, IBM-style alignment, EM, HMMs, phrase tables,
finite-state methods, BM25, TF-IDF, edit distance, dynamic programming, beam
search, minimum-description-length models, and non-neural co-occurrence or SVD
representations fitted solely to the supplied training pairs.

Forbidden inputs and mechanisms:

- neural inference or neural training of any kind;
- pretrained tokenizers, embeddings, lexicons, dictionaries, or model files;
- external corpora, translation services, network access, or subprocesses;
- files or state from another evaluation batch;
- hard-coded bilingual entries, queries, references, or evaluation indices.

Generic tokenizer algorithms and Unicode normalization are allowed. Any fitted
vocabulary or tokenizer state must come from the current payload.

## Candidate protocol

The artifact is one deterministic `candidate.py` command:

```text
python candidate.py translate
```

It reads one JSON object from stdin:

```json
{"training": [{"source": "...", "target": "..."}],
 "queries": ["...", "..."]}
```

It writes one JSON object as its final stdout line:

```json
{"translations": ["...", "..."]}
```

The translation list must match query order and length. Natural and cipher
conditions run in separate clean processes. Cipher mode independently replaces
every source and target token with opaque episode codes, so the algorithm must
remain valid without pretrained lexical knowledge.

## Frozen partitions

All partitions are disjoint by normalized English source across both training
and query rows.

- **Public practice:** visible ordinary and novel-conjunction queries and
  references; debugging and causal intervention checks only.
- **Private development selection:** fixed common data for evolutionary
  fitness; aggregate results only.
- **Private generalization development:** distinct languages and sources used
  for explicit GAR audits and possible distillation decisions.
- **Private confirmation:** distinct languages and sources, evaluated once
  only after the final artifact and analysis are frozen.

Natural and cipher scores for one source query are paired observations, not
independent samples. Audit/confirmation uncertainty uses a paired query-cluster
bootstrap stratified by language.

## Audit and success criteria

A generalization-development audit passes only if all are true:

1. admissible with zero batch failures;
2. its private selection score reproduces exactly;
3. its ordinary-query chrF++ is non-inferior to the frozen baseline: the paired
   95% query-cluster bootstrap interval stays above a -1 chrF margin;
4. no language/condition cell loses more than 3 chrF++ against baseline.
5. it independently passes novel-conjunction eligibility on the
   generalization languages.

Audit outcomes are passed, inconclusive, or failed. An ordinary-query interval
that crosses the non-inferiority margin is inconclusive and may continue as a
lineage checkpoint but cannot be distilled. An artifact that fails the
composition feasibility gate is recorded, but cannot be kept as a checkpoint
or inherited wholesale. It may appear only as an explicitly disclosed gene
donor whose failure is part of the child's brief. Only audit-passed candidates
may become shared methodology. The frozen
retrieval-copy mechanism is a control, never an initial or shared default;
founders begin from a neutral protocol scaffold.

The final confirmation candidate is the highest-selection-fitness candidate
with a passed audit. Confirmation repeats the same ordinary non-inferiority and
novel-composition superiority rules. It is run once; it cannot be used to
replace the candidate or resume evolution.

## Campaign identity

This is a **continuation**, not an independent replication, because its frozen
baseline and seeded hypotheses use knowledge from v0. Fresh private seeds and
sources prevent direct data reuse, but they do not erase methodological
inheritance.

The actual worker model, orchestrator, prompt hashes, evaluator hash, corpus
hashes, dataset commitments, candidate hashes, retries, and interventions are
recorded in the run manifest/evidence. The worker model named in the manifest
must match the model actually dispatched.
