# Translation anomaly GAR (corpus)

Evolve a language-agnostic

`flag(source, candidate, examples, corpus) -> bool`.

Workers see the public task brief, their champion, their fitness, and
the shared `Decoder.md`. Finch scores hidden practice and injects a
corpus with evaluation verses removed.

## Interface

```
source:     English
candidate:  a target-language string (human gold or a model draft)
examples:   list of {source, gold} from the same few-shot memory
corpus:     list of {source, gold} from the rest of that language
            (no verse ids, no language names)
return:     True to flag the candidate as a bad draft
```

The corpus is the parallel text for the same language with the
evaluation verses removed. It is injected at score time. You may
build counts from it and cache on `id(corpus)` — the object is stable
per language. Do not assume a size. Subsample if you need to.

Fitness: `flags(drafts)/n - flags(gold)/n`. Flag drafts; leave gold alone.

Driver verdicts report integer counts, not just rates: drafts flagged /
missed, and gold flagged (false positives). A keep that adds one draft
and one gold is a wash. An arm that never fires is 0 new flags.

Drafts are model outputs at several keep rates. Gold is the human verse
with the same memory. A full-keep draft is still a draft.

## The one example you may study

Invented text, not an evaluation language:

```
source = "The child sat down."
examples = [
    {"source": "The dog ran.", "gold": "dog-run."},
    {"source": "The woman stood.", "gold": "woman-stand."},
]
corpus = [
    {"source": "The man walked.", "gold": "man-walk."},
    {"source": "The bird flew.", "gold": "bird-fly."},
    {"source": "The fish swam.", "gold": "fish-swim."},
    {"source": "The fire burned.", "gold": "fire-burn."},
]
# a broken draft (should be easy to flag):
candidate = "xxx child sit down sit."
# a clean gold (must not flag):
candidate = "child-sit."
```

`python3 score.py artifact` runs only this smoke pair. That printed
score is an interface check, not campaign fitness.

## What to write

One general predicate that can use `corpus`. No verse names, no
language names, no bands fitted to a cell you have not been shown.
